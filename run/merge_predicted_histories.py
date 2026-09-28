"""Merge complete, disjoint take shards of predicted-history frames."""

import argparse
import json
from pathlib import Path

import torch

from egorecover.evaluation_protocol import file_sha256


def merge(shard_dirs, output):
    reports = [json.loads((path / "report.json").read_text()) for path in shard_dirs]
    if not reports or not all(report["completed"] for report in reports):
        raise ValueError("All shard reports must be complete.")
    identities = [report["identity"] for report in reports]
    count = identities[0]["shard_count"]
    indices = [identity["shard_index"] for identity in identities]
    if len(reports) != count or sorted(indices) != list(range(count)):
        raise ValueError("Expected exactly one shard per shard index.")
    varying = {"selected_takes", "shard_index"}
    common = {key: value for key, value in identities[0].items() if key not in varying}
    if any({key: value for key, value in identity.items() if key not in varying} != common for identity in identities):
        raise ValueError("Shard identities disagree.")
    selected = [take for identity in identities for take in identity["selected_takes"]]
    if len(selected) != len(set(selected)):
        raise ValueError("Take shards overlap.")
    if set(selected) != set(common["group_takes"]):
        raise ValueError("Take shards do not cover the entire fixed group.")
    payloads = []
    for path, report in zip(shard_dirs, reports):
        frames = path / "frames.pt"
        if file_sha256(frames) != report["cache_sha256"]:
            raise ValueError(f"Shard cache differs from its report: {path}")
        payload = torch.load(frames, map_location="cpu", weights_only=True)
        if payload["identity"] != report["identity"] or len(payload["record_ids"]) != report["frames"]:
            raise ValueError(f"Shard payload differs from its report: {path}")
        payloads.append(payload)
    keys = set(payloads[0]["tensors"])
    if any(set(payload["tensors"]) != keys for payload in payloads):
        raise ValueError("Shard tensor fields disagree.")
    tuples = [
        (record_id, time, mode)
        for payload in payloads
        for record_id, time, mode in zip(payload["record_ids"], payload["time_indices"], payload["generator_modes"])
    ]
    if len(tuples) != len(set(tuples)):
        raise ValueError("Shard frame keys overlap.")
    expected = len(selected) * len(common["variants"]) * 2 * 180
    if len(tuples) != expected:
        raise ValueError(f"Expected {expected} frames, found {len(tuples)}.")
    identity = {key: value for key, value in common.items() if key != "shard_count"}
    identity["selected_takes"] = selected
    merged = {
        "identity": identity,
        "tensors": {key: torch.cat([payload["tensors"][key] for payload in payloads]) for key in keys},
    }
    for key in ("record_ids", "time_indices", "take_names", "generator_modes"):
        merged[key] = [item for payload in payloads for item in payload[key]]
    if not set(merged["take_names"]) == set(selected):
        raise ValueError("Merged frame takes differ from selected takes.")
    output.mkdir(parents=True)
    frames = output / "frames.pt"
    torch.save(merged, frames)
    report = {
        "identity": identity,
        "frames": len(tuples),
        "episodes": [episode for source in reports for episode in source["episodes"]],
        "shards": [str(path) for path in shard_dirs],
        "cache_sha256": file_sha256(frames),
        "completed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    report = merge(args.shard, args.output)
    print(f"Saved {report['frames']} frames to {args.output}")


if __name__ == "__main__":
    main()
