"""E7-only Lightning module: x0 Flow Matching, dense motion, EMA-compatible."""

import math
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR, StepLR
from model.uniegomotion import UniEgoMotion
from mydiffusion.flow_matching import FlowMatching
from module.utils import cfg_to_dict
from config.defaults import validate_e7


class UEM_Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        validate_e7(cfg)
        if not cfg.TRAIN.ONLY_VALIDATE:
            self.save_hyperparameters(cfg_to_dict(cfg))
        self.cfg = cfg
        self.window = cfg.DATA.WINDOW
        self.model = UniEgoMotion(cfg)
        self.flow = FlowMatching(
            num_steps=cfg.FLOW.NUM_STEPS,
            solver=cfg.FLOW.SOLVER,
            beta_alpha=cfg.FLOW.BETA_ALPHA,
            beta_beta=cfg.FLOW.BETA_BETA,
            t_min=cfg.FLOW.T_MIN,
            prediction_type=cfg.FLOW.PREDICTION_TYPE,
            global_weight=cfg.FLOW.GLOBAL_WEIGHT,
            global_feature_start=cfg.FLOW.GLOBAL_FEATURE_START,
            global_feature_end=cfg.FLOW.GLOBAL_FEATURE_END,
            repaint_enabled=cfg.FLOW.REPAINT_ENABLED,
        )

    def configure_optimizers(self):
        fused = getattr(self.cfg.TRAIN, "FUSED_ADAMW", False) and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            fused=fused,
        )

        scheduler_name = getattr(self.cfg.TRAIN, "SCHEDULER", "step").lower()
        if scheduler_name == "cosine_warmup":
            warmup_epochs = max(0, getattr(self.cfg.TRAIN, "WARMUP_EPOCHS", 0))
            total_epochs = getattr(self.cfg.TRAIN, "SCHEDULER_TOTAL_EPOCHS", 0)
            if total_epochs <= 0:
                total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            min_lr_ratio = getattr(self.cfg.TRAIN, "MIN_LR_RATIO", 0.1)

            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = StepLR(optimizer, max(1, self.cfg.TRAIN.NUM_EPOCHS - 50), gamma=0.1)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        ]

    def on_train_start(self):
        if self.cfg.MODEL.CKPT_PATH is None:
            return
        if self.cfg.TRAIN.USE_CKPT_LR:
            logger.warning("Using LR from checkpoint.")
            return
        logger.warning("Discarding LR of optimizer dict and using config LR.")
        for g in self.optimizers().param_groups:
            g["lr"] = self.cfg.TRAIN.LR
        for g in self.optimizers().param_groups:
            g["weight_decay"] = self.cfg.TRAIN.WEIGHT_DECAY

    def training_step(self, batch, batch_idx, mode="train"):
        x = batch["misc"]["motion"]
        losses = self.flow.training_losses(self.model, x, model_kwargs={"y": batch["y"]})
        for name in ("loss", "local_mse", "global_mse"):
            self.log(
                f"{mode}/{name}",
                losses[name].mean(),
                on_step=mode == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
        return losses["loss"].mean()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False, *, noise=None, repaint_enabled=None):
        for key, value in y.items():
            if key in {"repaint_mask", "repaint_value"}:
                continue  # The sampler validates constraints only when enabled.
            if len(value) != B:
                raise ValueError(f"y[{key}] must have batch size {B}.")
        return self.flow.sample_loop(
            self.model,
            (B, self.window, self.model.input_feats),
            model_kwargs={"y": y, "cond_scale": cond_scale},
            noise=noise,
            device=self.device,
            return_all_pred_xstart=return_all_pred_xstart,
            repaint_enabled=repaint_enabled,
        )
