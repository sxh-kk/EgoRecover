"""Offline same-state four-action FK diagnostic on reachable predicted histories."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from dataset.smpl_utils import get_smpl
from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.engineering import conditions_from_batch
from egorecover.evaluation_protocol import event_window, file_sha256, load_fixed_split
from egorecover.fk import FixedShapeFK
from egorecover.utility import generate_utility_labels
from run.evaluate_paired_development import load_model


def fixed_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def state_phase(record, time):
    window = event_window(record)
    if window is None:
        return min(((50, "early"), (110, "middle"), (170, "late")), key=lambda item: abs(time - item[0]))[1]
    start, end = window
    if time < start:
        return "pre_fault"
    if time >= end:
        return "recovery"
    return "late_fault" if end >= record["num_frames"] and time >= (start + end) / 2 else "fault"


def summarize_actions(rows, draws):
    """Distinguish per-draw oracle from choosing after averaging the same K draws."""
    states = defaultdict(list)
    for row in rows:
        states[row["take"], row["mode"], row["variant"], row["time"]].append(row)
    averaged = []
    for key, items in sorted(states.items()):
        if sorted(row["draw"] for row in items) != list(range(draws)):
            raise ValueError("Every state needs exactly K distinct draws.")
        if any(row["canonical_actions"] != items[0]["canonical_actions"] for row in items):
            raise ValueError("Canonical actions differ across draws.")
        errors = [sum(row["errors_smpl22_fk_mm"][action] for row in items) / draws for action in range(4)]
        averaged.append({
            "take": key[0], "mode": key[1], "variant": key[2], "time": key[3],
            "phase": items[0]["phase"], "errors_smpl22_fk_mm": errors,
            "oracle_gain_mm": errors[0] - min(errors),
            "oracle_action": items[0]["canonical_actions"][errors.index(min(errors))],
            "per_draw_oracle_gain_mm": sum(row["oracle_gain_mm"] for row in items) / draws,
        })
    groups = defaultdict(list)
    for row in averaged:
        for key in (f"{row['mode']}/{row['variant']}",
                    f"{row['mode']}/{row['variant']}/{row['phase']}"):
            groups[key].append(row)
    result = {}
    for key, items in sorted(groups.items()):
        by_take = defaultdict(list)
        for row in items:
            by_take[row["take"]].append(row)
        per_take = {}
        for take, samples in sorted(by_take.items()):
            per_take[take] = {
                "mean_action_error_mm": [sum(row["errors_smpl22_fk_mm"][i] for row in samples) / len(samples)
                                         for i in range(4)],
                "oracle_gain_after_draw_mean_mm": sum(row["oracle_gain_mm"] for row in samples) / len(samples),
                "per_draw_oracle_gain_mm": sum(row["per_draw_oracle_gain_mm"] for row in samples) / len(samples),
            }
        result[key] = {
            "states": len(items), "take_count": len(per_take), "per_take": per_take,
            "macro_action_error_mm": [sum(value["mean_action_error_mm"][i] for value in per_take.values())
                                      / len(per_take) for i in range(4)],
            "macro_oracle_gain_after_draw_mean_mm": sum(value["oracle_gain_after_draw_mean_mm"]
                                                        for value in per_take.values()) / len(per_take),
            "macro_per_draw_oracle_gain_mm": sum(value["per_draw_oracle_gain_mm"]
                                                 for value in per_take.values()) / len(per_take),
            "oracle_action_counts_after_draw_mean": {
                str(i): sum(row["oracle_action"] == i for row in items) for i in range(4)
            },
        }
    return {"states": averaged, "groups": result}


def select_frames(payload, records, takes, mode):
    lookup = defaultdict(dict)
    for index, (record_id, time, generator) in enumerate(
        zip(payload["record_ids"], payload["time_indices"], payload["generator_modes"])
    ):
        if generator == mode and records[record_id]["base_take_name"] in takes:
            lookup[record_id][time] = index
    selected = []
    for record_id, times in sorted(lookup.items()):
        ordered = sorted(times)
        window = event_window(records[record_id])
        if window is None:
            chosen = [min(ordered, key=lambda time: abs(time - target)) for target in (50, 110, 170)]
        else:
            start, end = window
            before = [time for time in ordered if time < start]
            fault = [time for time in ordered if start <= time < end]
            recovery = [time for time in ordered if time >= end]
            if not before or len(fault) < 2:
                raise ValueError(f"Fault event lacks sampled pre/fault states: {record_id}.")
            chosen = [before[len(before) // 2], fault[len(fault) // 3]]
            chosen.append(recovery[len(recovery) // 2] if recovery else fault[2 * len(fault) // 3])
        if len(set(chosen)) != 3:
            raise ValueError(f"Phase samples are not distinct for {record_id}.")
        selected.extend(times[time] for time in chosen)
    if not selected:
        raise ValueError("No reachable frames selected.")
    return selected


@torch.no_grad()
def diagnose(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, _ = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    splits = load_fixed_split(args.split_manifest, dataset)
    records = {record["variant_id"]: record for record in dataset.records}
    takes = splits[args.group][:args.max_takes]
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats_sha = file_sha256(stats_path)
    e7_sha = file_sha256(args.checkpoint)
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False)).to(device)
    payload = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    identity = payload["identity"]
    expected_scope = ("training_only_predicted_histories" if args.group == "train"
                      else "development_diagnostic_predicted_histories")
    if identity["scope"] != expected_scope or identity.get("group") != args.group:
        raise ValueError("Predicted-history cache has the wrong group/scope.")
    for key, expected in (("split_manifest_sha256", file_sha256(args.split_manifest)),
                          ("stats_sha256", stats_sha),
                          ("dataset_spec_sha256", file_sha256(dataset.root / "spec.json"))):
        if identity[key] != expected:
            raise ValueError(f"Predicted-history cache has a different {key}.")
    if not set(payload["take_names"]).issubset(splits[args.group]) or not set(takes).issubset(payload["take_names"]):
        raise ValueError("Predicted-history cache is missing or mixes group takes.")
    if identity["g_sha256"] != {
        mode: file_sha256(args.experiment / f"g_{mode}.pt") for mode in ("gaussian", "history")
    } or identity["p_sha256"] != file_sha256(args.experiment / "prior.pt"):
        raise ValueError("Predicted histories came from another P/G experiment.")
    if identity["bootstrap_identity"]["e7_checkpoint_sha256"] != e7_sha or not identity["bootstrap_is_model"]:
        raise ValueError("Diagnostic requires model-generated E7 startup.")
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    rows = []
    for mode in ("gaussian", "history"):
        model, flow, prior, _ = load_model(
            args.experiment, mode, device=device, stats_sha256=stats_sha,
            e7_sha256=e7_sha, reference_mode=codec.reference_mode,
        )
        selected = select_frames(payload, records, set(takes), mode)
        for start in range(0, len(selected), args.batch_size):
            indices = selected[start:start + args.batch_size]
            batch = {key: value[indices].to(device) for key, value in payload["tensors"].items()}
            if not bool(batch["beta_boot_is_model"].all()):
                raise ValueError("Every diagnostic state needs an E7 bootstrap shape.")
            conditions = conditions_from_batch(batch, prior=prior, action="a11")
            fk = FixedShapeFK(smpl, batch["beta_boot"])

            def fk_error(prediction):
                state = codec.decode_current(prediction[:, 0], batch["previous_reference"])
                predicted, _ = fk.project(state)
                return (predicted - batch["target_joints"]).norm(dim=-1).mean(-1) * 1000

            for draw in range(args.draws):
                epsilon = torch.stack([
                    torch.randn(
                        1, 243, device=device,
                        generator=torch.Generator(device=device).manual_seed(
                            fixed_seed(args.seed, mode, payload["record_ids"][index],
                                       payload["time_indices"][index], draw)
                        ),
                    )
                    for index in indices
                ])
                labels = generate_utility_labels(
                    model, flow, conditions, epsilon, fk_error,
                    checkpoint_id=identity["g_sha256"][mode],
                )
                for offset, index in enumerate(indices):
                    record = records[payload["record_ids"][index]]
                    errors = labels["errors"][offset].cpu().tolist()
                    canonical = labels["canonical_actions"][offset].cpu().tolist()
                    rows.append({
                        "take": record["base_take_name"],
                        "variant": record["variant_name"],
                        "mode": mode,
                        "time": payload["time_indices"][index],
                        "event_window": list(event_window(record)) if event_window(record) else None,
                        "phase": state_phase(record, payload["time_indices"][index]),
                        "draw": draw,
                        "errors_smpl22_fk_mm": errors,
                        "canonical_actions": canonical,
                        "oracle_gain_mm": errors[0] - min(errors),
                        "oracle_action": canonical[errors.index(min(errors))],
                    })
            print(f"{mode}: {min(start + args.batch_size, len(selected))}/{len(selected)} states", flush=True)
        del model, prior
        if device.type == "cuda":
            torch.cuda.empty_cache()
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"], row["variant"]].append(row)
    summary = {
        f"{mode}/{variant}": {
            "states_times_draws": len(items),
            "mean_oracle_gain_mm": sum(row["oracle_gain_mm"] for row in items) / len(items),
            "oracle_action_counts": {
                str(action): sum(row["oracle_action"] == action for row in items) for action in range(4)
            },
            "mean_action_error_mm": [
                sum(row["errors_smpl22_fk_mm"][action] for row in items) / len(items) for action in range(4)
            ],
        }
        for (mode, variant), items in sorted(groups.items())
    }
    return {
        "completed": True,
        "group": args.group,
        "takes": takes,
        "sampling": "three states per record: pre/fault/recovery or late-fault; clean early/middle/late",
        "draws": args.draws,
        "metric": "world SMPL22 FK MPJPE with model-generated E7 beta_boot, mm",
        "experiment": str(args.experiment),
        "cache": str(args.cache),
        "cache_sha256": file_sha256(args.cache),
        "summary": summary,
        "draw_averaged": summarize_actions(rows, args.draws),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--group", choices=("train", "dev"), required=True)
    parser.add_argument("--max-takes", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_takes < 1 or args.batch_size < 1 or args.draws < 1 or args.output.exists():
        parser.error("Positive counts and a new output path are required.")
    report = diagnose(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(report['rows'])} same-state labels to {args.output}")


if __name__ == "__main__":
    main()
