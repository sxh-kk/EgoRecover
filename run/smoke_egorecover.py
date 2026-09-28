"""Exercise the actual 12-layer history G without datasets or SMPL assets."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from egorecover.actions import Action
from egorecover.checkpoint import load_e7_weights
from egorecover.conditioning import build_conditioning
from egorecover.history_flow import HistoryFlow
from model.history_uniegomotion import HistoryUniEgoMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--source-mode", choices=("history", "gaussian"), default="history")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Optional JSON report; no tensors/checkpoints are saved.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = HistoryUniEgoMotion(get_cfg_defaults())
    migration = None
    if args.checkpoint is not None:
        migration = asdict(load_e7_weights(model, args.checkpoint, weight_source=args.weight_source))
    elif args.weight_source == "ema":
        parser.error("--weight-source ema requires --checkpoint")
    model.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def random(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    inputs = dict(
        history_motion=random(1, 20, 243),
        history_valid=torch.ones(1, 20, device=device, dtype=torch.bool),
        prior_mu=random(1, 1, 243),
        traj=random(1, 1, 18),
        img_embs=random(1, 1, 1024),
        img_available=torch.ones(1, 1, device=device, dtype=torch.bool),
        traj_available=torch.ones(1, 1, device=device, dtype=torch.bool),
    )
    history_before = inputs["history_motion"].clone()
    epsilon, target = random(1, 1, 243), random(1, 1, 243)
    flow = HistoryFlow(source_mode=args.source_mode, sigma=args.sigma)
    model.train()
    y = build_conditioning(**inputs)
    losses = flow.training_losses(model, target, y, epsilon=epsilon, t=torch.tensor([0.5], device=device))
    losses["loss"].mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    model.zero_grad(set_to_none=True)
    model.eval()
    outputs = {}
    for action in Action:
        sample = flow.sample(model, build_conditioning(**inputs, action=action), epsilon=epsilon)
        outputs[action.name.lower()] = {"shape": list(sample.shape), "finite": bool(torch.isfinite(sample).all())}
    report = {
        "scope": "synthetic tensors, actual 12-layer G; no accuracy evaluation or optimizer update",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "source_mode": args.source_mode,
        "sigma": args.sigma,
        "nfe_per_action": flow.flow.num_steps,
        "migration": migration,
        "loss": float(losses["loss"].mean().detach()),
        "loss_finite": bool(torch.isfinite(losses["loss"]).all()),
        "gradients_finite": gradient_finite,
        "history_unchanged": torch.equal(history_before, inputs["history_motion"]),
        "actions": outputs,
    }
    report["passed"] = (
        report["loss_finite"]
        and gradient_finite
        and report["history_unchanged"]
        and all(value["finite"] and value["shape"] == [1, 1, 243] for value in outputs.values())
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
