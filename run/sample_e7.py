"""Sample E7 from prepared, normalized conditions with optional history constraints."""

import argparse
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from module.ema import apply_ema_weights_from_checkpoint
from module.uem_module import UEM_Module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True, help="torch.save(y): normalized input tensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/e7.yaml")
    parser.add_argument("--repaint", choices=("on", "off"), default=None, help="Override FLOW.REPAINT_ENABLED")
    parser.add_argument("--noise", type=Path, help="Optional saved tensor shared by keep/release calls")
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.config))
    cfg.TRAIN.ONLY_VALIDATE = True
    cfg.MODEL.CKPT_PATH = str(args.checkpoint)
    enabled = cfg.FLOW.REPAINT_ENABLED if args.repaint is None else args.repaint == "on"
    device = torch.device(args.device)
    y = torch.load(args.conditioning, map_location=device, weights_only=True)
    if not isinstance(y, dict) or any(not isinstance(v, torch.Tensor) for v in y.values()):
        raise ValueError("Conditioning must be a dictionary of tensors.")
    if "valid_frames" not in y or y["valid_frames"].ndim != 2:
        raise ValueError("Conditioning requires a [batch, frames] valid_frames tensor.")
    batch, frames = y["valid_frames"].shape
    if frames != cfg.DATA.WINDOW:
        raise ValueError("Conditioning frames must match DATA.WINDOW.")
    if enabled and ("repaint_mask" not in y or "repaint_value" not in y):
        raise ValueError("--repaint on requires repaint_mask and repaint_value in the conditioning file.")
    model = UEM_Module.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location="cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    used_ema = apply_ema_weights_from_checkpoint(model.model, checkpoint)
    del checkpoint
    model = model.to(device).eval()
    if args.noise is not None:
        noise = torch.load(args.noise, map_location=device, weights_only=True)
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(batch, frames, 243, device=device, generator=generator)
    with torch.inference_mode():
        motion = model.sample(y, B=batch, noise=noise, repaint_enabled=enabled, cond_scale=cfg.TRAIN.COND_SCALE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "motion": motion.cpu(),
            "normalized": True,
            "repaint_enabled": enabled,
            "ema_applied": bool(used_ema),
            "seed": args.seed,
            "noise_file": str(args.noise) if args.noise else None,
        },
        args.output,
    )
    print(f"Saved normalized E7 motion {tuple(motion.shape)} to {args.output}; repaint={enabled}")


if __name__ == "__main__":
    main()
