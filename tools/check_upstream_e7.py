"""Compare this E7 extraction against a separate, pinned upstream checkout.

python -m tools.check_upstream_e7 --upstream ../.codebase-audit/UEM-update
Uses synthetic weights/data; does not validate pretrained accuracy or SMPL-X.
"""

import argparse
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from config.defaults import get_cfg_defaults
from module.uem_module import UEM_Module


def import_upstream(root):
    prefixes = ("config", "model", "module", "mydiffusion", "utils", "dataset")

    def relevant(name):
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    saved = {k: v for k, v in sys.modules.items() if relevant(k)}
    for key in list(saved):
        del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        old_defaults = importlib.import_module("config.defaults")
        old_module = importlib.import_module("module.uem_module")
        return old_defaults.get_cfg_defaults, old_module.UEM_Module
    finally:
        sys.path.pop(0)
        for key in list(sys.modules):
            if relevant(key):
                del sys.modules[key]
        sys.modules.update(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("verification/upstream_e7_parity.json"))
    args = parser.parse_args()
    source = args.upstream.resolve()
    old_defaults, OldModule = import_upstream(source)
    old_cfg = old_defaults()
    old_cfg.merge_from_file(str(source / "ablation/configs/e7_x0_global_w8_u84k.yaml"))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "config/e7.yaml"))
    old_cfg.TRAIN.ONLY_VALIDATE = cfg.TRAIN.ONLY_VALIDATE = True
    torch.set_num_threads(2)
    torch.manual_seed(62)
    old = OldModule(old_cfg).eval()
    new = UEM_Module(cfg).eval()
    status = new.load_state_dict(old.state_dict(), strict=True)
    assert [n for n, p in old.model.named_parameters() if p.requires_grad] == [
        n for n, p in new.model.named_parameters() if p.requires_grad
    ]
    result = {
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "torch": torch.__version__,
        "device": "cpu",
        "weights": "synthetic random upstream E7 state_dict",
        "strict_state_dict": str(status),
        "optimizer_parameter_order_equal": True,
        "checks": {},
    }

    def equal(label, a, b):
        error = (a - b).abs().max().item()
        result["checks"][label] = {"max_abs_error": error, "bitwise_equal": torch.equal(a, b)}
        if not torch.equal(a, b):
            raise AssertionError((label, error))

    noise = torch.randn(1, 80, 243)
    y = {
        "traj": torch.randn(1, 80, 18),
        "img_embs": torch.randn(1, 80, 1024),
        "valid_frames": torch.ones(1, 80, dtype=torch.long),
        "valid_img_embs": torch.ones(1, 80, dtype=torch.long),
    }
    with torch.inference_mode():
        for task in ("recon", "fore", "gen"):
            cond = copy.deepcopy(y)
            cond["traj_mask"] = torch.zeros(1, 80, dtype=torch.long)
            cond["img_mask"] = torch.zeros(1, 80, dtype=torch.long)
            if task == "fore":
                cond["traj_mask"][:, 20:] = cond["img_mask"][:, 20:] = 1
            elif task == "gen":
                cond["traj_mask"][:] = 1
                cond["img_mask"][:, 1:] = 1
            t = torch.tensor([0.65])
            equal(task + "_forward", old.model(noise, t, copy.deepcopy(cond)), new.model(noise, t, copy.deepcopy(cond)))
        equal(
            "guided_forward",
            old.model(noise, torch.tensor([0.7]), copy.deepcopy(y), cond_scale=1.5),
            new.model(noise, torch.tensor([0.7]), copy.deepcopy(y), cond_scale=1.5),
        )
        baseline = old.flow.sample_loop(old.model, noise.shape, {"y": copy.deepcopy(y)}, noise=noise)
        equal("euler10_80frames", baseline, new.sample(copy.deepcopy(y), noise=noise))
        repaint = copy.deepcopy(y)
        repaint["repaint_mask"] = torch.zeros_like(noise)
        repaint["repaint_mask"][:, :70] = 1
        repaint["repaint_value"] = torch.randn_like(noise)
        equal("disabled_repaint_80frames", baseline, new.sample(repaint, noise=noise, repaint_enabled=False))
        keep = new.sample(repaint, noise=noise, repaint_enabled=True)
        equal("enabled_final_history", keep[:, :70], repaint["repaint_value"][:, :70])
        assert torch.isfinite(keep).all()

    # Include training dropout, random condition masks, weighted loss and gradients.
    old.train()
    new.train()
    short_y = {k: v[:, :4].clone() for k, v in y.items()}
    target, short_noise, t = torch.randn(1, 4, 243), noise[:, :4].clone(), torch.tensor([0.65])
    for obj, key in ((old, "old"), (new, "new")):
        torch.manual_seed(123)
        terms = obj.flow.training_losses(obj.model, target, {"y": copy.deepcopy(short_y)}, noise=short_noise, t=t)
        terms["loss"].mean().backward()
        values = {name: value.detach().clone() for name, value in terms.items()}
        values["output_gradient"] = obj.model.output_process.weight.grad.clone()
        values["input_gradient"] = obj.model.input_process.weight.grad.clone()
        if key == "old":
            old_values = values
        else:
            for name in ("loss", "local_mse", "global_mse", "output_gradient", "input_gradient"):
                equal("training_" + name, old_values[name], values[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
