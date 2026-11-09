"""Build split-verified frozen-G histories for P training or validation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from egorecover.data import keyed_noise
from egorecover.evaluate import load_evaluation_context
from egorecover.fk import FixedShapeFK
from egorecover.io import file_sha256, save_checkpoint, save_json
from egorecover.prior_data import concatenate_examples, prior_example, reconstructed_history_count
from egorecover.state import state_map


@torch.no_grad()
def main(args):
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    if selection["split"] not in ("train", "val"):
        raise ValueError("Only official train/val splits are supported.")
    context = load_evaluation_context(
        SimpleNamespace(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            split=selection["split"],
            device="cuda",
            synthetic=False,
            data_format="official",
            prior_mode="constant_velocity",
            prior_checkpoint=None,
            limit=None,
        )
    )
    if context.system.config.source != "gaussian" or context.system.predictor is not None:
        raise ValueError("History producer must be the fixed Gaussian/CV control.")
    if context.data.digest != selection["manifest_sha256"]:
        raise ValueError("Bank data identity changed.")
    if any(
        context.data.records[i]["episode_id"] != name for i, name in zip(selection["indices"], selection["episode_ids"])
    ):
        raise ValueError("Bank selection no longer matches the official ordering.")
    args.output.mkdir(parents=True, exist_ok=False)
    fk = FixedShapeFK(context.smpl, torch.zeros(10, device="cuda"))
    examples, rows = [], []
    indices = selection["indices"][args.shard :: args.shards]
    for completed, index in enumerate(indices, 1):
        episode = context.data.load(index)
        cache = episode.cache()
        anchors = [frame for frame in selection["anchors"] if frame < len(episode)]
        for frame in range(20, max(anchors) + 1):
            if frame in anchors:
                truth = state_map(episode.target(frame), lambda value: value[:, -1])
                generated = reconstructed_history_count(
                    frame - 20, context.system.config.window, context.system.config.revise_recent
                )
                candidates = [("predicted", cache.body, generated)]
                if selection["split"] == "val" and context.system.config.window == 1:
                    history = state_map(episode.target(frame - 1, 20), lambda value: value.detach())
                    candidates.append(("clean", history, None))
                for kind, history, generated in candidates:
                    example = prior_example(context.system.codec, history, truth, generated_frames=generated)
                    cv = context.system.codec.decode_current(example["base"][:, 0], example["reference"])
                    cv_error = float((fk(cv) - example["truth_xyz"]).norm(dim=-1).mean() * 1000)
                    examples.append({key: value.cpu() for key, value in example.items()})
                    rows.append(
                        {
                            "episode": episode.identifier,
                            "take": episode.take,
                            "frame": frame,
                            "kind": kind,
                            "window": context.system.config.window if kind == "predicted" else 0,
                            "cv_body_mm": cv_error,
                        }
                    )
            if frame < max(anchors):
                noise = keyed_noise(episode.identifier, 1062, frame, context.system.config.window, "cuda")
                cache = context.system.online(context.model, cache, episode.observations(frame), noise).cache.detach()
        save_json(
            args.output / "progress.json", {"phase": "building", "clips": completed, "target_clips": len(indices)}
        )
        print(
            json.dumps(
                {
                    "window": context.system.config.window,
                    "split": selection["split"],
                    "clips": completed,
                    "target": len(indices),
                }
            ),
            flush=True,
        )
    packet = {
        "format": "egorecover_prior_bank_v1",
        "split": selection["split"],
        "manifest_sha256": context.data.digest,
        "selection_sha256": selection["sha256"],
        "producer_checkpoint_sha256": file_sha256(args.checkpoint),
        "producer_window": context.system.config.window,
        "rows": rows,
        "tensors": concatenate_examples(examples),
    }
    save_checkpoint(args.output / "bank.pt", packet)
    save_json(args.output / "progress.json", {"phase": "complete", "clips": len(indices), "samples": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "data-root", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    main(parser.parse_args())
