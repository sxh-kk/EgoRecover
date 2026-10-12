"""Synthetic end-to-end functionality check. Its metrics are not research results."""

import argparse
from pathlib import Path

import torch

from .config import Config
from .data import Episodes, Normalizer, save_json, save_torch, sha256
from .evaluation import evaluate
from .geometry import axis_angle_to_matrix, fk, head_features, matrix_to_rotation_6d
from .train import load_bundle, run_training


def tiny_config():
    return Config(
        image_dim=12,
        latent_dim=8,
        history=4,
        bootstrap=4,
        target_frames=4,
        tae_width=24,
        tae_dilations=(1, 1),
        width=32,
        heads=4,
        layers=2,
        fusion_layers=1,
        flow_width=48,
        flow_layers=2,
        checkpoint_layers=True,
        batch_size=2,
        flow_steps=3,
        total_steps=4,
        eval_every=2,
        precision="fp32",
    )


def synthetic_offsets():
    return torch.tensor(
        [
            [0, 0, 0],
            [-0.09, 0, -0.08],
            [0.09, 0, -0.08],
            [0, 0, 0.12],
            [0, 0, -0.40],
            [0, 0, -0.40],
            [0, 0, 0.13],
            [0, 0, -0.40],
            [0, 0, -0.40],
            [0, 0, 0.13],
            [0, 0.12, -0.04],
            [0, 0.12, -0.04],
            [0, 0, 0.15],
            [-0.08, 0, 0.07],
            [0.08, 0, 0.07],
            [0, 0, 0.12],
            [-0.12, 0, 0],
            [0.12, 0, 0],
            [-0.26, 0, 0],
            [0.26, 0, 0],
            [-0.24, 0, 0],
            [0.24, 0, 0],
        ]
    )


def make_dataset(output, cfg, frames=24):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Synthetic dataset output must be fresh.")
    generator = torch.Generator().manual_seed(123)
    records = []
    for number, split in enumerate(("train", "train", "dev", "test")):
        time = torch.arange(frames) / cfg.fps
        axis = torch.randn(frames, 21, 3, generator=generator) * 0.02
        axis[:, 0, 0] += 0.15 * torch.sin(time * 2)
        pose = torch.cat(
            (
                matrix_to_rotation_6d(axis_angle_to_matrix(axis)).flatten(1),
                torch.randn(frames, 24, generator=generator) * 0.02,
                torch.ones(frames, 2),
            ),
            -1,
        )
        root_rotation = axis_angle_to_matrix(torch.stack((time * 0, time * 0, 0.1 * time), -1))
        root = torch.cat(
            (matrix_to_rotation_6d(root_rotation), torch.stack((0.15 * time, time * 0, time * 0 + 0.98), -1)), -1
        )
        offsets = synthetic_offsets()
        joints, rotations = fk(pose, root, offsets)
        head_transforms = torch.eye(4).expand(frames, 4, 4).clone()
        head_transforms[:, :3, :3], head_transforms[:, :3, 3] = rotations[:, 15], joints[:, 15]
        iv, hv = torch.ones(frames, dtype=torch.bool), torch.ones(frames, dtype=torch.bool)
        iv[cfg.bootstrap + 2 :: 5] = False
        hv[cfg.bootstrap + 3 :: 7] = False
        images = torch.randn(frames, cfg.image_dim, generator=generator) * 0.1
        images[:, :3] += root[:, 6:9]
        images[~iv] = float("nan")  # Explicit missing-payload masking exercise.
        head = head_features(head_transforms, torch.eye(4), hv)
        startup_root = root[: cfg.bootstrap].clone()
        startup_root[:, 6:] += torch.randn(cfg.bootstrap, 3, generator=generator) * 0.01
        packet = {
            "pose": pose,
            "root": root,
            "joints": joints,
            "images": images,
            "head": head,
            "valid": torch.ones(frames, dtype=torch.bool),
            "image_valid": iv,
            "head_valid": hv,
            "beta": torch.zeros(10),
            "offsets": offsets,
            "anchor": torch.eye(4),
            "startup_pose": pose[: cfg.bootstrap] + torch.randn(cfg.bootstrap, 152, generator=generator) * 0.01,
            "startup_root": startup_root,
            "startup_source": "synthetic_predicted",
        }
        eid = f"synthetic_{number}"
        path = output / "episodes" / f"{eid}.pt"
        save_torch(path, packet)
        records.append({"id": eid, "take": eid, "split": split, "path": f"episodes/{eid}.pt", "sha256": sha256(path)})
    save_json(
        output / "manifest.json",
        {
            "version": 1,
            "fps": cfg.fps,
            "backend": "synthetic_demo",
            "records": records,
            "warning": "Functionality fixture only; no real motion accuracy claim.",
        },
    )
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Demo requires a fresh independent output directory.")
    cfg = tiny_config()
    output.mkdir(parents=True, exist_ok=True)
    config = output / "tiny.json"
    cfg.save(config)
    manifest = make_dataset(output / "data", cfg)
    tae = run_training(stage="tae", config=config, manifest=manifest, output=output / "tae", device_name=args.device)
    flow = run_training(
        stage="flow", config=config, manifest=manifest, output=output / "flow", device_name=args.device, tae_path=tae
    )
    _, norm, autoencoder, model, _ = load_bundle(flow, args.device)
    episodes = Episodes(manifest, cfg)
    results = {
        action: evaluate(model, autoencoder, norm, episodes, "test", action, seed=cfg.seed)
        for action in ("a11", "a10", "a01", "a00")
    }
    save_json(output / "smoke_results.json", {"purpose": "synthetic functionality only", "results": results})
    print(f"Synthetic smoke finished: {output / 'smoke_results.json'}")


if __name__ == "__main__":
    main()
