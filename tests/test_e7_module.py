import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from config.defaults import get_cfg_defaults
from module.ema import EMA
from module.uem_module import UEM_Module


class SyntheticMotion(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(62)
        return {
            "misc": {"motion": torch.randn(4, 243, generator=g)},
            "y": {
                "traj": torch.randn(4, 18, generator=g),
                "img_embs": torch.randn(4, 1024, generator=g),
                "valid_frames": torch.ones(4, dtype=torch.long),
                "valid_img_embs": torch.ones(4, dtype=torch.long),
            },
        }


def test_lightning_train_validation_ema_and_sampling(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg = get_cfg_defaults()
        cfg.DATA.WINDOW = 4
        cfg.DATA.BATCH_SIZE = 1
        cfg.TRAIN.EXP_PATH = str(tmp_path)
        model = UEM_Module(cfg)
        before = model.model.output_process.weight.detach().clone()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[EMA(cfg.TRAIN.EMA_DECAY)],
            default_root_dir=tmp_path,
        )
        loader = DataLoader(SyntheticMotion(), batch_size=1)
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        assert trainer.global_step == 1
        assert not torch.equal(before, model.model.output_process.weight)
        assert torch.isfinite(trainer.callback_metrics["train/loss"])
        assert torch.isfinite(trainer.callback_metrics["val/loss"])
        model.eval()
        y = next(iter(loader))["y"]
        noise = torch.randn(1, 4, 243)
        value = torch.randn_like(noise)
        mask = torch.zeros_like(noise, dtype=torch.bool)
        mask[:, :2] = True
        y.update(repaint_mask=mask, repaint_value=value)
        keep = model.sample(y, noise=noise, repaint_enabled=True)
        release = model.sample(y, noise=noise, repaint_enabled=False)
        assert torch.equal(keep[mask], value[mask])
        assert torch.isfinite(keep).all() and torch.isfinite(release).all()
    finally:
        torch.set_num_threads(old_threads)
