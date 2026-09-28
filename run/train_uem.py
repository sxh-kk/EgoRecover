import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
from module.uem_module import UEM_Module
from module.ema import EMA

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# torch.multiprocessing.set_sharing_strategy("file_system")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    pl.seed_everything(62, workers=True)
    torch.set_float32_matmul_precision("high")
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None

    os.makedirs(cfg.TRAIN.EXP_PATH, exist_ok=True)

    datamodule = EE4D_Motion_DataModule(cfg)
    model = UEM_Module(cfg)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(cfg.TRAIN.EXP_PATH, name="", version="", default_hp_metric=False, flush_secs=60)

    # Setup checkpoint saving and lr monitor callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # type: ignore
        dirpath=cfg.TRAIN.EXP_PATH,
        # every_n_train_steps=cfg.TRAIN.SAVE_EVERY_N_STEPS,
        every_n_epochs=cfg.TRAIN.SAVE_EVERY_N_EPOCHS,
        save_top_k=-1,
        # monitor="train/loss",
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor("step")
    pbar_callback = TQDMProgressBar(refresh_rate=getattr(cfg.TRAIN, "PROGRESS_REFRESH_RATE", 10))
    ema_callback = EMA(getattr(cfg.TRAIN, "EMA_DECAY", 0.999))
    callbacks = [
        ema_callback,
        checkpoint_callback,
        lr_monitor_callback,
        pbar_callback,
    ]
    if getattr(cfg.TRAIN, "EARLY_STOP_PATIENCE", 0) > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                min_delta=getattr(cfg.TRAIN, "EARLY_STOP_MIN_DELTA", 1.0e-4),
                patience=cfg.TRAIN.EARLY_STOP_PATIENCE,
                strict=False,
            )
        )

    # Setup PyTorch Lightning Trainer
    strategy = "auto"
    if cfg.TRAIN.NUM_GPUS > 1:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=cfg.TRAIN.DDP_STATIC_GRAPH)
    trainer = pl.Trainer(
        default_root_dir=cfg.TRAIN.EXP_PATH,
        logger=logger,
        devices=cfg.TRAIN.NUM_GPUS,
        accelerator="gpu",
        strategy=strategy,
        precision=getattr(cfg.TRAIN, "PRECISION", "32-true"),
        num_sanity_val_steps=getattr(cfg.TRAIN, "NUM_SANITY_VAL_STEPS", 2),
        log_every_n_steps=cfg.TRAIN.LOG_EVERY_N_STEPS,
        callbacks=callbacks,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        max_steps=getattr(cfg.TRAIN, "MAX_STEPS", -1),
        check_val_every_n_epoch=cfg.TRAIN.CHECK_VAL_EVERY_N_EPOCHS,
        accumulate_grad_batches=getattr(cfg.TRAIN, "ACCUMULATE_GRAD_BATCHES", 1),
        gradient_clip_algorithm="norm",
        gradient_clip_val=getattr(cfg.TRAIN, "GRADIENT_CLIP_VAL", 1.0),
        benchmark=getattr(cfg.TRAIN, "CUDNN_BENCHMARK", False),
        # limit_train_batches=20,
        # limit_val_batches=10,
        # profiler="advanced",
    )

    # Set checkpoint path if needed
    ckpt_path = cfg.MODEL.CKPT_PATH
    if cfg.MODEL.CKPT_PATH == "last_ckpt":
        ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")

    # If resuming the same experiment, set appropriate global step
    if ckpt_path is not None and ckpt_path.startswith(cfg.TRAIN.EXP_PATH):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        global_step_offset = checkpoint["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        del checkpoint

    if cfg.TRAIN.ONLY_VALIDATE:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        return

    # model = UEM_Module.load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
    # ckpt_path = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
