import os
import torch
import numpy as np
import copy
from tqdm.auto import tqdm
import joblib
from loguru import logger
import argparse

from dataset.smpl_utils import get_smpl, evaluate_smpl
from eval.metrics import compute_metrics, reconstruction_error
from utils.torch_utils import to_device, to_tensor, to_numpy


DEFAULT_KEY_JOINT_INDICES = (0, 4, 5, 7, 8, 10, 11, 15, 18, 19, 20, 21)


def _validate_key_joint_indices(key_joint_indices, num_joints):
    key_joint_indices = tuple(key_joint_indices)
    if not key_joint_indices:
        raise ValueError("At least one key joint index is required")
    if len(set(key_joint_indices)) != len(key_joint_indices):
        raise ValueError(f"Key joint indices must be unique, got {key_joint_indices}")

    invalid_indices = [idx for idx in key_joint_indices if idx < 0 or idx >= num_joints]
    if invalid_indices:
        raise ValueError(
            f"Key joint indices {invalid_indices} are outside the available joint range [0, {num_joints - 1}]"
        )
    return key_joint_indices


def compute_key_joint_metrics(gt_kp3d, pred_kp3d, key_joint_indices):
    """Compute positional metrics on an explicit subset of SMPL-X joints."""
    if gt_kp3d.shape != pred_kp3d.shape:
        raise ValueError(f"GT/predicted keypoints must have the same shape, got {gt_kp3d.shape} and {pred_kp3d.shape}")

    key_joint_indices = _validate_key_joint_indices(key_joint_indices, gt_kp3d.shape[1])
    key_joint_indices_t = torch.as_tensor(key_joint_indices, dtype=torch.long, device=gt_kp3d.device)
    gt_key_kp3d = gt_kp3d.index_select(1, key_joint_indices_t)
    pred_key_kp3d = pred_kp3d.index_select(1, key_joint_indices_t)

    mpjpe_key = (pred_key_kp3d - gt_key_kp3d).square().sum(dim=-1).sqrt().mean().item()
    mpjpe_key_pa = reconstruction_error(pred_key_kp3d.numpy(), gt_key_kp3d.numpy()).item()
    root_trans_error = torch.norm(pred_kp3d[:, 0] - gt_kp3d[:, 0], dim=-1).mean().item()
    return {
        "mpjpe_key": mpjpe_key,
        "mpjpe_key_pa": mpjpe_key_pa,
        "root_trans_error": root_trans_error,
    }


def get_clip_metrics(
    gt_aria_traj_T,
    pred_aria_traj_T,
    gt_smpl_params,
    pred_smpl_params,
    smpl,
    st=None,
    en=None,
    key_joint_indices=None,
):
    # deepcopy everything
    gt_aria_traj_T = copy.deepcopy(gt_aria_traj_T)
    pred_aria_traj_T = copy.deepcopy(pred_aria_traj_T)
    gt_smpl_params = copy.deepcopy(gt_smpl_params)
    pred_smpl_params = copy.deepcopy(pred_smpl_params)

    if gt_aria_traj_T is not None:
        gt_aria_traj_T = gt_aria_traj_T[st:en]
    if pred_aria_traj_T is not None:
        pred_aria_traj_T = pred_aria_traj_T[st:en]

    gt_smpl_params = {k: v[st:en] for k, v in gt_smpl_params.items()}
    pred_smpl_params = {k: v[st:en] for k, v in pred_smpl_params.items()}
    if key_joint_indices is not None:
        gt_kp3d = evaluate_smpl(smpl, gt_smpl_params, return_joints_only=True)
        pred_kp3d = evaluate_smpl(smpl, pred_smpl_params, return_joints_only=True)
        gt_kp3d = to_device(gt_kp3d, torch.device("cpu"))
        pred_kp3d = to_device(pred_kp3d, torch.device("cpu"))
        return compute_key_joint_metrics(gt_kp3d, pred_kp3d, key_joint_indices)

    gt_kp3d, gt_verts, gt_full_pose = evaluate_smpl(smpl, gt_smpl_params)
    mdata_i = {
        "kp3d": gt_kp3d,
        "verts": gt_verts,
        "full_pose": gt_full_pose,
        "smpl_params": gt_smpl_params,
        "aria_traj_T": gt_aria_traj_T,
    }

    pred_kp3d, pred_verts, pred_full_pose = evaluate_smpl(smpl, pred_smpl_params)
    pred_mdata_i = {
        "kp3d": pred_kp3d,
        "verts": pred_verts,
        "full_pose": pred_full_pose,
        "smpl_params": pred_smpl_params,
        "aria_traj_T": pred_aria_traj_T,
    }

    mdata_i = to_device(mdata_i, torch.device("cpu"))
    pred_mdata_i = to_device(pred_mdata_i, torch.device("cpu"))

    return compute_metrics(mdata_i, pred_mdata_i, smpl)


def main(exp_path, task, eval_suffix, data_dir, key_joints_only=False, key_joint_indices=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment path: {exp_path}")

    # save path
    metric_set_suffix = "_keyjoints" if key_joints_only else ""
    save_path = f"{exp_path}/metrics_3d_ee4d_{task}{eval_suffix}{metric_set_suffix}.pkl"
    if os.path.exists(save_path):
        print(f"Metrics already computed at {save_path}")
        return

    if key_joints_only:
        key_joint_indices = _validate_key_joint_indices(key_joint_indices or DEFAULT_KEY_JOINT_INDICES, 22)
        logger.info(f"Evaluating key SMPL22 joints only: {key_joint_indices}")

    smpl = get_smpl()
    gt = joblib.load(f"{data_dir}/uniegomotion/ee_val_gt_for_evaluation.pkl")
    gt = to_device(gt, device)
    smpl = smpl.to(device)

    preds = joblib.load(f"{exp_path}/preds_ee4d_{task}{eval_suffix}.pkl")
    preds = to_device(preds, device)
    assert not os.path.exists(save_path), f"File already exists: {save_path}"

    missing_gt_keys = preds.keys() - gt.keys()
    if missing_gt_keys:
        raise KeyError(f"Predictions have {len(missing_gt_keys)} keys absent from ground truth")
    logger.info(f"Computing metrics for {len(preds)} predicted samples")

    all_metrics = None
    for seq_k in tqdm(list(preds.keys())):

        gt_aria_traj_T = gt[seq_k]["aria_traj_T"]
        gt_smpl_params = gt[seq_k]["smpl_params"]
        nf = len(gt_smpl_params["global_orient"])

        pred_aria_traj_T = preds[seq_k]["aria_traj_T"]
        pred_smpl_params = preds[seq_k]["smpl_params"]
        if pred_aria_traj_T is None:
            pred_aria_traj_T = gt_aria_traj_T.clone()

        if key_joints_only:
            pred_lengths = {key: len(value) for key, value in pred_smpl_params.items()}
            too_short = {key: length for key, length in pred_lengths.items() if length < nf}
            if too_short:
                raise ValueError(f"Prediction {seq_k} is shorter than its {nf}-frame ground truth: {too_short}")
            pred_smpl_params = {key: value[:nf] for key, value in pred_smpl_params.items()}
            gt_kp3d = evaluate_smpl(smpl, gt_smpl_params, return_joints_only=True)
            pred_kp3d = evaluate_smpl(smpl, pred_smpl_params, return_joints_only=True)
            gt_kp3d = to_device(gt_kp3d, torch.device("cpu"))
            pred_kp3d = to_device(pred_kp3d, torch.device("cpu"))
            metrics = compute_key_joint_metrics(gt_kp3d, pred_kp3d, key_joint_indices)
        else:
            metrics = get_clip_metrics(
                gt_aria_traj_T, pred_aria_traj_T, gt_smpl_params, pred_smpl_params, smpl, st=0, en=nf,
            )

        # metrics of every 20 frames
        for seg_start in range(0, nf, 20):
            seg_end = min(seg_start + 20, nf)
            if key_joints_only:
                metrics_seg = compute_key_joint_metrics(
                    gt_kp3d[seg_start:seg_end],
                    pred_kp3d[seg_start:seg_end],
                    key_joint_indices,
                )
            else:
                metrics_seg = get_clip_metrics(
                    gt_aria_traj_T, pred_aria_traj_T, gt_smpl_params, pred_smpl_params, smpl,
                    st=seg_start, en=seg_end,
                )
            for k, v in metrics_seg.items():
                metrics[k + f"_seg_{seg_start}"] = v

        if all_metrics is None:
            all_metrics = {k: [] for k in metrics.keys()}
        for k, v in metrics.items():
            all_metrics[k].append(v)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    assert not os.path.exists(save_path), "Metrics file already exists"
    if not os.path.exists(save_path):
        joblib.dump(all_metrics, save_path)
    logger.info(f"Metrics saved at {save_path}")


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion")
    parser.add_argument("--EXP_PATH", type=str, required=True)
    parser.add_argument("--EVAL_SUFFIX", type=str, default="")
    parser.add_argument("--EVAL_TASK", type=str, required=True)
    parser.add_argument(
        "--KEY_JOINTS_ONLY",
        action="store_true",
        help="Only compute MPJPE, PA-MPJPE, and root error on selected key joints.",
    )
    parser.add_argument(
        "--KEY_JOINT_INDICES",
        type=int,
        nargs="+",
        default=list(DEFAULT_KEY_JOINT_INDICES),
        help="SMPL22 joint indices used by --KEY_JOINTS_ONLY.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.EXP_PATH), f"Experiment path does not exist: {args.EXP_PATH}"
    assert args.EVAL_TASK in ["recon", "gen", "fore"], f"Task {args.EVAL_TASK} not supported"
    main(
        args.EXP_PATH,
        args.EVAL_TASK,
        args.EVAL_SUFFIX,
        args.DATA_DIR,
        key_joints_only=args.KEY_JOINTS_ONLY,
        key_joint_indices=args.KEY_JOINT_INDICES,
    )
