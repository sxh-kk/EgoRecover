"""Collect TRAINING-only predicted histories with a verified E7 startup cache.

The optional GT-start mode is a legacy diagnostic and is rejected by formal
FK training. No development or holdout take enters this cache.
"""

import argparse
import json
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from egorecover.annotations import body_states_from_supervision
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.calibration import estimate_bootstrap_floor
from egorecover.codec import BodyState, MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.evaluation_protocol import DEFAULT_SPLIT_MANIFEST, file_sha256, load_fixed_split
from egorecover.history import HistoryBuffer
from egorecover.history_flow import HistoryFlow
from egorecover.prior import HistoryPrior
from model.history_uniegomotion import HistoryUniEgoMotion
from run.engineering_pilot import sha256


def slice_state(state, index):
    return BodyState(*(getattr(state, name)[index] for name in ("joints", "reference", "auxiliary")))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["clean", "freeze_3s", "drift_0p03mps"])
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--bootstrap-cache", type=Path)
    parser.add_argument("--allow-gt-bootstrap", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--group", choices=("train", "dev"), default="train")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid take shard.")
    if (args.bootstrap_cache is None) == (not args.allow_gt_bootstrap):
        parser.error("Supply --bootstrap-cache, or explicitly choose --allow-gt-bootstrap for a diagnostic.")
    if args.output.exists():
        parser.error("Choose a new output directory.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    experiment = json.loads((args.experiment / "report.json").read_text())
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, handoff = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    splits = load_fixed_split(args.split_manifest, dataset)
    if experiment["splits"] != splits:
        raise ValueError("Experiment takes differ from the frozen split manifest.")
    selected_takes = splits[args.group][args.shard_index::args.shard_count]
    if not selected_takes:
        raise ValueError("This take shard is empty.")
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False)).to(device)
    bootstraps = (
        ModelBootstrapShapes(
            args.bootstrap_cache,
            allowed_takes=splits["train"] + splits["dev"] + splits["holdout"],
            stats_sha256=file_sha256(stats_path),
            split_manifest_sha256=file_sha256(args.split_manifest),
            dataset_spec_sha256=file_sha256(dataset.root / "spec.json"),
        )
        if args.bootstrap_cache
        else None
    )
    if bootstraps and bootstraps.identity.get("reference_mode") != codec.reference_mode:
        raise ValueError("Bootstrap cache and replay use different reference conventions.")
    if bootstraps and (
        experiment.get("e7_checkpoint_sha256") != bootstraps.identity["e7_checkpoint_sha256"]
        or experiment.get("e7_weight_source") != bootstraps.identity["e7_weight_source"]
    ):
        raise ValueError("The frozen G experiment and startup cache use different E7 weights.")
    prior = HistoryPrior().to(device)
    prior.load_state_dict(
        torch.load(args.experiment / "prior.pt", map_location=device, weights_only=True)["state_dict"]
    )
    prior.freeze()
    cache = {
        key: []
        for key in (
            "history_motion",
            "base_mu",
            "velocity_mu",
            "target",
            "traj",
            "img_embs",
            "img_available",
            "traj_available",
            "previous_reference",
            "target_joints",
            "beta_boot",
            "beta_boot_is_model",
            "floor_estimate_m",
        )
    }
    metadata = {key: [] for key in ("record_ids", "time_indices", "take_names", "generator_modes")}
    summaries = []
    for mode_index, mode in enumerate(("gaussian", "history")):
        checkpoint = torch.load(args.experiment / f"g_{mode}.pt", map_location=device, weights_only=True)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=mode, sigma=checkpoint["sigma"])
        for variant_index, variant in enumerate(args.variants):
            all_records = [
                r
                for r in dataset.records
                if r["base_take_name"] in splits[args.group] and r["variant_name"] == variant
            ]
            selected_indices = [i for i, record in enumerate(all_records) if record["base_take_name"] in selected_takes]
            records = [all_records[i] for i in selected_indices]
            if not records or any(r["num_frames"] != 200 or r["bootstrap_frames"] != 20 for r in records):
                raise ValueError("Expected 200-frame train episodes with 20-frame bootstrap.")
            observed, truth, buffers, floors = [], [], [], []
            for record in records:
                packet = {
                    k: v.to(device)
                    for k, v in dataset.observations(record["variant_id"], as_of=199, history_frames=200).items()
                }
                clean = {
                    k: v.to(device)
                    for k, v in dataset.observations(record["clean_variant_id"], as_of=199, history_frames=200).items()
                }
                floor = estimate_bootstrap_floor(packet, codec, 20)
                supervision = dataset.supervision(record["variant_id"])
                states = body_states_from_supervision(
                    supervision,
                    clean["aria_traj_obs"],
                    floor_height=floor,
                    contact_floor_height=float(supervision["floor_height"]),
                )
                if bootstraps:
                    startup = bootstraps.startup_for_record(record)
                    codes = startup["normalized_motion"].to(device)
                    initial_reference = startup["initial_reference"].to(device)
                    if abs(float(startup["floor_estimate_m"]) - floor) > 1e-5:
                        raise ValueError("Bootstrap cache floor differs from the legal prefix estimate.")
                else:
                    prefix = slice_state(states, slice(0, 20))
                    codes = codec.encode_history(
                        BodyState(*(getattr(prefix, k)[None] for k in ("joints", "reference", "auxiliary"))),
                        states.reference[0:1],
                    )[0]
                    initial_reference = states.reference[0]
                buffer = HistoryBuffer.from_bootstrap(codec, codes, initial_reference)
                if bootstraps and not torch.allclose(buffer.beta_boot.cpu(), startup["beta_boot"], atol=1e-5):
                    raise ValueError("Cached E7 startup body and beta_boot disagree.")
                if bootstraps:
                    references = torch.stack([state.reference for state in buffer.states]).cpu()
                    world_joints = torch.stack([state.joints[..., :3, 3] for state in buffer.states]).cpu()
                    world_joints[..., 2] += floor
                    if not torch.allclose(references, startup["references"], atol=1e-5, rtol=0) or not torch.allclose(
                        world_joints, startup["world_joints"], atol=1e-4, rtol=0
                    ):
                        raise ValueError("Cached E7 startup references/world joints disagree with its motion.")
                buffers.append(buffer)
                observed.append(packet)
                truth.append(states)
                floors.append(floor)
            errors = []
            print(f"Collect {mode}/{variant}: {len(records)} {args.group} episodes", flush=True)
            for t in range(20, 200):
                conditions = [
                    buffer.conditions({k: v[t : t + 1] for k, v in packet.items()}, floor_height=floor)
                    for buffer, packet, floor in zip(buffers, observed, floors)
                ]
                y = {key: torch.cat([condition[key] for condition in conditions]) for key in conditions[0]}
                previous = torch.stack([buffer.previous_reference for buffer in buffers])
                target_state = BodyState(
                    *(
                        torch.stack([getattr(state, key)[t] for state in truth])
                        for key in ("joints", "reference", "auxiliary")
                    )
                )
                row = {
                    "history_motion": y["history_motion"],
                    "base_mu": y["prior_mu"],
                    "velocity_mu": torch.stack(
                        [codec.constant_velocity_prior(buffer.states[-2], buffer.states[-1]) for buffer in buffers]
                    )[:, None],
                    "target": codec.encode_current(target_state, previous)[:, None],
                    "previous_reference": previous,
                    "target_joints": target_state.joints[..., :3, 3],
                    "beta_boot": torch.stack([buffer.beta_boot for buffer in buffers]),
                    "beta_boot_is_model": torch.full(
                        (len(records),), bootstraps is not None, device=device, dtype=torch.bool
                    ),
                    "floor_estimate_m": torch.tensor(floors, device=device),
                    "traj": y["traj"],
                    "img_embs": y["img_embs"],
                    "img_available": ~y["img_mask"],
                    "traj_available": ~y["traj_mask"],
                }
                y["prior_mu"] = prior(y["history_motion"], y["history_valid"], y["prior_mu"])
                epsilon_all = torch.randn(
                    len(all_records),
                    1,
                    243,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + 100000 * mode_index + 1000 * variant_index + t
                    ),
                )
                epsilon = epsilon_all[selected_indices]
                prediction = flow.sample(model, y, epsilon=epsilon)[:, 0]
                states = [buffer.commit(pred, t) for buffer, pred in zip(buffers, prediction)]
                error = (
                    torch.stack([state.joints[..., :3, 3] for state in states]) - row["target_joints"]
                ).double().norm(dim=-1).mean(-1) * 1000
                errors.extend(error.cpu().tolist())
                for key, value in row.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Nonfinite training cache {mode}/{variant}/{t}/{key}")
                    cache[key].append(value.cpu())
                metadata["record_ids"].extend(r["variant_id"] for r in records)
                metadata["take_names"].extend(r["base_take_name"] for r in records)
                metadata["time_indices"].extend([t] * len(records))
                metadata["generator_modes"].extend([mode] * len(records))
            summaries.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "episodes": len(records),
                    "mean_dense22_mm": sum(errors) / len(errors),
                    "max_dense22_mm": max(errors),
                }
            )
            print(summaries[-1], flush=True)
        del model, checkpoint
    identity = {
        "scope": ("training_only_predicted_histories" if args.group == "train"
                  else "development_diagnostic_predicted_histories"),
        "reference_mode": codec.reference_mode,
        "bootstrap": (
            "frozen E7 clean first 20 frames on TRAINING takes; no resets"
            if bootstraps
            else "GT first 20 TRAINING frames; diagnostic only; no resets"
        ),
        "bootstrap_is_model": bootstraps is not None,
        "bootstrap_cache_sha256": file_sha256(args.bootstrap_cache) if bootstraps else None,
        "bootstrap_identity": bootstraps.identity if bootstraps else None,
        "policy": "a11 frozen P/G; pooled gaussian/history policies",
        "generator_experiment": str(args.experiment),
        "g_sha256": {mode: sha256(args.experiment / f"g_{mode}.pt") for mode in ("gaussian", "history")},
        "p_sha256": sha256(args.experiment / "prior.pt"),
        "stats_sha256": sha256(stats_path),
        "dataset_spec_sha256": file_sha256(dataset.root / "spec.json"),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "train_takes": experiment["splits"]["train"],
        "group": args.group,
        "group_takes": splits[args.group],
        "selected_takes": selected_takes,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "variants": args.variants,
        "seed": args.seed,
        "sigma": experiment["sigma"],
        "nfe": 10,
    }
    output = args.output / "frames.pt"
    torch.save({"identity": identity, "tensors": {k: torch.cat(v) for k, v in cache.items()}, **metadata}, output)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "frames": len(metadata["record_ids"]),
                "episodes": summaries,
                "cache_sha256": sha256(output),
                "completed": True,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
