"""Audit source integrity, paired labels, corruption replay and causal observations."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .corruptions import apply_operations, stable_seed, variant_operations
from .dataset import MismatchDataset
from .source import annotation_digest, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_dataset(root, *, data_root=None, source=None, verify_source=True):
    dataset = MismatchDataset(root, data_root=data_root, source=source)
    source, spec = dataset.source, dataset.spec
    require(len(dataset) == spec["num_variants"], "Manifest count mismatch")
    require(spec["purpose"] != "development" or source.split == "train", "Validation data used for development")
    if verify_source:
        require(source.fingerprints() == spec["source_files"], "Source file checksums changed")
    split_manifest = json.loads((dataset.root / "split_manifest.json").read_text())
    split_map = split_manifest["take_to_split"]
    scope = spec.get("episode_scope", "sampled_takes")
    require(scope in {"sampled_takes", "all_source_sequences"}, "Unknown episode scope")
    adaptive = scope == "all_source_sequences"
    expected_full = {}
    if adaptive:
        expected_episodes, inventory = source.select_all_sequences(spec["seed"], spec["bootstrap_frames"],
                                                                   spec["recovery_observation_frames"])
        expected_full = {episode["episode_id"]: episode for episode in expected_episodes}
        require(len(expected_full) == inventory["source_sequences"] == spec["num_episodes"],
                "The full corpus does not cover every source sequence")
    episodes, families, splits = defaultdict(list), Counter(), defaultdict(set)
    whitelist = {"img_feats", "aria_traj_obs", "img_available", "traj_available", "frame_id_30fps"}
    require(set(spec["online_fields"]) == whitelist, "Online schema changed")
    for record_number, record in enumerate(dataset.records, 1):
        name = record["variant_id"]
        take = record["base_take_name"]
        require(record["base_seq_name"].rsplit("___", 2)[0] == take, "Take/sequence mismatch")
        require(record["base_split"] == source.split, "Base split mismatch")
        require(split_map[take] == record["split"], "Take split leakage")
        if adaptive:
            require(record["episode_id"] in expected_full, "Unexpected full-corpus episode")
            for field, value in expected_full[record["episode_id"]].items():
                if field != "seed":
                    require(record[field] == value, f"Full-corpus episode metadata changed: {field}")
        else:
            require(record["num_frames"] == spec["frames"], "Episode length mismatch")
            require(record["bootstrap_frames"] == spec["bootstrap_frames"], "Bootstrap mismatch")
        require(record["pair_id"] == record["episode_id"], "Pair identity mismatch")
        require(record["seed"] == stable_seed(spec["seed"], name), "Generation seed mismatch")
        expected_variants = variant_operations(record["fault_onset"], record["num_frames"],
                                              stable_seed(spec["seed"], record["episode_id"]), spec["profile"],
                                              adaptive=adaptive)
        require(record["operations"] == expected_variants[record["variant_name"]], "Variant parameters changed")
        arrays = dataset.arrays(name, verify_hash=True)
        clean = source.clean(record)
        replay, provenance = apply_operations(clean, record["operations"], record["seed"], record["bootstrap_frames"])
        require(provenance == record["provenance"], "Provenance replay mismatch")
        for field in arrays:
            require(np.array_equal(arrays[field], replay[field]), f"Corruption replay mismatch: {name}/{field}")
            require(np.array_equal(arrays[field][:record["bootstrap_frames"]], clean[field][:record["bootstrap_frames"]]),
                    f"Bootstrap changed: {name}/{field}")
        for field in ("seq_motion_idx", "frame_id_30fps", "eval_mask"):
            require(np.array_equal(arrays[field], clean[field]), f"Physical time changed: {name}/{field}")
        require(annotation_digest(dataset.supervision(name)) == record["supervision_sha256"], "Physical labels changed")
        features = source.visual_payload(take, arrays["img_source_idx"], arrays["img_available"])
        require(bool(torch.isfinite(features).all()), "Non-finite feature payload")
        require(bool((features[~torch.from_numpy(arrays["img_available"])] == 0).all()), "Missing visual placeholder is not zero")
        # Exercise the online API at boot, onset, during the fault and at recovery.
        points = {0, record["bootstrap_frames"]-1, record["fault_onset"], record["num_frames"]-1}
        for operation in record["operations"]:
            points.update((operation["start"], operation["end"]-1, min(operation["end"], record["num_frames"]-1)))
        for frame in sorted(points):
            current = dataset.observations(name, as_of=frame, history_frames=8)
            first = max(0, frame-7)
            require(set(current) == whitelist, "Offline metadata leaked into online observation fields")
            require(torch.equal(current["img_feats"], features[first:frame+1]), "Causal visual slice mismatch")
            for field in whitelist - {"img_feats"}:
                require(np.array_equal(current[field].numpy(), arrays[field][first:frame+1]), "Causal payload slice mismatch")
        episodes[record["pair_id"]].append(record)
        families[record["variant_name"]] += 1
        splits[take].add(record["split"])
        if record_number % 2000 == 0 or record_number == len(dataset):
            print(f"Audited {record_number}/{len(dataset)} variants", flush=True)
    require(len(episodes) == spec["num_episodes"], "Episode count mismatch")
    require(set(splits) == set(split_map), "Take list mismatch")
    if adaptive:
        require(set(episodes) == set(expected_full), "Some source sequences are missing from the full corpus")
        require(set(splits) == {name.rsplit("___", 2)[0] for name in source.motion},
                "Some source takes are missing from the full corpus")
    require(all(len(values) == 1 for values in splits.values()), "A take crosses dataset splits")
    for paired in episodes.values():
        clean = [record for record in paired if record["variant_name"] == "clean"]
        require(len(clean) == 1, "Each pair must have exactly one clean variant")
        require(len(paired) == len(variant_operations(paired[0]["fault_onset"], paired[0]["num_frames"],
                                                      0, spec["profile"], adaptive=adaptive)),
                "Incomplete variant group")
        for record in paired:
            require(record["clean_variant_id"] == clean[0]["variant_id"], "Wrong clean pair reference")
            for field in ("base_seq_name", "episode_start_motion_idx", "num_frames", "supervision_sha256", "split"):
                require(record[field] == clean[0][field], f"Paired supervision differs: {field}")
    return dict(
        status="passed", completed_at=datetime.now(timezone.utc).isoformat(),
        source_hashes_verified=verify_source, num_episodes=len(episodes), num_variants=len(dataset),
        num_takes=len(splits), source_sequences_covered=len({r["base_seq_name"] for r in dataset.records}),
        physical_frames_covered=sum(paired[0]["num_frames"] for paired in episodes.values()),
        total_variant_frames=sum(r["num_frames"] for r in dataset.records),
        variants=dict(sorted(families.items())),
        checks=["array_sha256", "exact_operator_replay", "clean_bootstrap", "unchanged_physical_labels",
                "unchanged_target_time", "paired_clean_reference", "take_split_isolation", "source_index_bounds",
                "no_future_sources", "finite_missing_payloads", "online_field_whitelist", "causal_observation_slices"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--skip-source-hashes", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_dataset(args.dataset, data_root=args.data_root, verify_source=not args.skip_source_hashes)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
