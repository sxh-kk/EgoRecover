import copy
import os
import sys
import IPython
import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from loguru import logger
from tqdm.auto import tqdm
from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_DataModule, careful_collate_fn
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from module.ema import apply_ema_weights_from_checkpoint
from module.uem_module import UEM_Module
from utils.torch_utils import to_device


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo")
    device = torch.device("cuda", local_rank)
    pl.seed_everything(62 + rank, workers=True)
    sys.argv = sys.argv + ["TRAIN.ONLY_VALIDATE", "True"]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert os.path.exists(cfg.TRAIN.EXP_PATH)
    assert cfg.TRAIN.EVAL_TASK in ["recon", "gen", "fore"]
    ds_name = "ee4d"
    split = "val"
    save_path = f"{cfg.TRAIN.EXP_PATH}/preds_{ds_name}_{cfg.TRAIN.EVAL_TASK}{cfg.TRAIN.EVAL_SUFFIX}.pkl"
    if os.path.exists(save_path):
        if rank == 0:
            print(f"Preds already computed at {save_path}")
        if distributed:
            dist.destroy_process_group()
        return
    ds_class = {"ee4d": EE4D_Motion_Dataset}[ds_name]
    ds = ds_class(
        data_dir=cfg.DATA.DATA_DIR,
        split=split,
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
    )
    ckpt_path = cfg.MODEL.CKPT_PATH
    if cfg.MODEL.CKPT_PATH == "last_ckpt":
        ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
    logger.info(f"Loading model from {ckpt_path}")
    model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if apply_ema_weights_from_checkpoint(model.model, checkpoint):
        logger.info("Using EMA weights stored in the checkpoint for evaluation.")
    del checkpoint
    model = model.to(device).eval()

    def process_batch(pred_batch, all_preds):
        B = len(pred_batch)
        pred_batch = careful_collate_fn(pred_batch)
        with torch.inference_mode():
            y = to_device(pred_batch["y"], device)
            x = model.sample(y, B, cond_scale=cfg.TRAIN.COND_SCALE, return_all_pred_xstart=False)
            pred_batch["pred"]["motion"] = to_device(x, "cpu")
        pred_mdata = ds.ret_to_full_sequence(pred_batch)
        pred_mdata = to_device(pred_mdata, "cpu")
        for i in range(B):
            seq_name = pred_batch["misc"]["seq_name"][i]
            start_idx = pred_batch["misc"]["start_idx"][i] // 3
            k = f"{seq_name}_start_{start_idx}"
            assert k not in all_preds
            all_preds[k] = {
                "smpl_params": pred_mdata["smpl_params_full"][i],
                "aria_traj_T": pred_mdata["aria_traj_T"][i],
            }
        return all_preds

    all_preds = {}
    batch_size = cfg.EVAL.BATCH_SIZE
    batch = []
    eval_indices = list(range(0, len(ds), 10))
    if cfg.EVAL.NUM_SAMPLES > 0:
        eval_indices = eval_indices[: cfg.EVAL.NUM_SAMPLES]
    local_indices = eval_indices[rank::world_size]
    logger.info(f"Rank {rank}/{world_size}: evaluating {len(local_indices)} of {len(eval_indices)} samples")
    for idx in tqdm(local_indices, disable=rank != 0):
        sample = ds[idx]
        sample = ds.process_sample_for_task(sample, cfg.TRAIN.EVAL_TASK)
        batch.append(sample)
        if len(batch) >= batch_size:
            all_preds = process_batch(batch, all_preds)
            batch = []
    if len(batch) > 0:
        all_preds = process_batch(batch, all_preds)
        batch = []
    shard_path = f"{save_path}.rank-{rank:02d}-of-{world_size:02d}.part"
    joblib.dump(all_preds, shard_path)
    if distributed:
        dist.barrier()
    if rank == 0:
        merged_preds = {}
        shard_paths = [f"{save_path}.rank-{r:02d}-of-{world_size:02d}.part" for r in range(world_size)]
        for path in shard_paths:
            shard = joblib.load(path)
            duplicate_keys = merged_preds.keys() & shard.keys()
            if duplicate_keys:
                raise RuntimeError(f"Duplicate prediction keys across ranks: {sorted(duplicate_keys)[:5]}")
            merged_preds.update(shard)
        if len(merged_preds) != len(eval_indices):
            raise RuntimeError(f"Expected {len(eval_indices)} predictions, got {len(merged_preds)}")
        tmp_save_path = save_path + ".tmp"
        joblib.dump(merged_preds, tmp_save_path)
        os.replace(tmp_save_path, save_path)
        for path in shard_paths:
            os.remove(path)
        logger.info(f"Saved {len(merged_preds)} predictions at {save_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
