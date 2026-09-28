"""Build reproducible paired episodes from the official processed EE4D files."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .corruptions import MODALITY_ORDER, apply_operations, stable_seed, variant_operations
from .source import EE4DSource, annotation_digest, file_sha256, write_json


def build_dataset(source, output, *, num_takes=12, frames=200, seed=20260923,
                  bootstrap=20, recovery=80, profile="pilot", purpose="engineering",
                  dev_fraction=0.25, holdout_fraction=0.0, all_sequences=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new version directory: {output}")
    if purpose not in {"engineering", "development"}:
        raise ValueError("Purpose must be engineering or development")
    if purpose == "development" and source.split != "train":
        raise ValueError("Development datasets must come from official train; val is engineering-only")
    if purpose == "development" and not (0 < dev_fraction < 1 and 0 <= holdout_fraction < 1 and
                                          dev_fraction + holdout_fraction < 1):
        raise ValueError("Development fractions must leave a nonempty training group")
    if purpose != "development" and holdout_fraction:
        raise ValueError("Holdout fraction is only valid for development datasets")
    if all_sequences:
        episodes, inventory = source.select_all_sequences(seed, bootstrap, recovery)
    else:
        episodes, inventory = source.select_episodes(num_takes, frames, seed, bootstrap, recovery)
    print(json.dumps({"inventory": inventory}, ensure_ascii=False), flush=True)
    print("Fingerprinting source files...", flush=True)
    fingerprints = source.fingerprints()
    take_names = sorted({item["base_take_name"] for item in episodes})
    selected_takes = len(take_names)
    if purpose == "development" and selected_takes < 2:
        raise ValueError("Development needs at least two distinct takes")
    np.random.default_rng(stable_seed(seed, "split")).shuffle(take_names)
    dev_count = max(1, min(selected_takes-1, round(selected_takes*dev_fraction))) if purpose == "development" else 0
    holdout_count = max(1, round(selected_takes*holdout_fraction)) if holdout_fraction else 0
    if selected_takes - dev_count - holdout_count < 1:
        raise ValueError("Development split must retain a training take")
    dev_takes = set(take_names[:dev_count])
    holdout_takes = set(take_names[dev_count:dev_count + holdout_count])
    split_map = {take: ("dev" if take in dev_takes else "holdout" if take in holdout_takes else "train")
                 if purpose == "development" else "engineering"
                 for take in take_names}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        (staging / "arrays").mkdir(parents=True)
        (staging / "audit").mkdir()
        records, summaries = [], []
        for episode_number, episode in enumerate(episodes, 1):
            episode = dict(episode, split=split_map[episode["base_take_name"]])
            clean = source.clean(episode)
            truth_hash = annotation_digest(source.supervision(episode))
            variants = variant_operations(episode["fault_onset"], episode["num_frames"], episode["seed"],
                                          profile, adaptive=all_sequences)
            clean_id = episode["episode_id"] + "__clean"
            for name, operations in variants.items():
                variant_id = episode["episode_id"] + "__" + name
                generation_seed = stable_seed(seed, variant_id)
                arrays, provenance = apply_operations(clean, operations, generation_seed,
                                                      episode["bootstrap_frames"])
                # Load real features, including each missing/frozen/delayed combination.
                source.visual_payload(episode["base_take_name"], arrays["img_source_idx"], arrays["img_available"])
                relative = f"arrays/{variant_id}.npz"
                np.savez_compressed(staging / relative, **arrays)
                record = dict(
                    episode, variant_id=variant_id, variant_name=name, pair_id=episode["episode_id"],
                    clean_variant_id=clean_id, seed=generation_seed,
                    index_unit="episode_motion_index_10fps", interval_convention="[start,end)",
                    array_file=relative, array_sha256=file_sha256(staging / relative),
                    supervision_sha256=truth_hash, operations=operations, provenance=provenance,
                )
                records.append(record)
                difference = np.linalg.norm(arrays["aria_traj_obs"][:, 6:9] - clean["aria_traj_obs"][:, 6:9], axis=-1)
                usable = arrays["traj_available"]
                summaries.append(dict(
                    variant_id=variant_id, take=episode["base_take_name"], variant=name,
                    injected_frames=arrays["corruption_mask_gt"].sum(axis=0).tolist(),
                    changed_visual_source_frames=int(np.sum(arrays["img_source_idx"] != clean["img_source_idx"])),
                    max_available_translation_change_m=float(difference[usable].max()) if usable.any() else None,
                ))
            if annotation_digest(source.supervision(episode)) != truth_hash:
                raise RuntimeError("Building variants mutated source supervision")
            if episode_number % 100 == 0 or episode_number == len(episodes):
                print(f"Built {episode_number}/{len(episodes)} episodes, {len(records)} variants", flush=True)
        manifests = []
        for split in sorted(set(split_map.values())):
            name = split + ".jsonl"
            manifests.append(name)
            (staging / name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                               for r in records if r["split"] == split))
        code = hashlib.sha256()
        for name in ("build.py", "source.py", "corruptions.py"):
            code.update(name.encode())
            code.update(Path(__file__).with_name(name).read_bytes())
        spec = dict(
            schema_version="ee4d-mismatch-v1", purpose=purpose, source_root=str(source.root),
            base_split=source.split, source_files=fingerprints, generator_sha256=code.hexdigest(),
            manifests=manifests, seed=seed, profile=profile, frames=None if all_sequences else frames,
            episode_scope="all_source_sequences" if all_sequences else "sampled_takes",
            frame_policy="full_original_sequence" if all_sequences else "fixed_clip",
            short_sequence_policy="adaptive_bootstrap_and_clipped_fault" if all_sequences else None,
            bootstrap_frames=bootstrap,
            recovery_observation_frames=recovery, motion_fps=10, original_fps=30, feature_fps=5,
            index_unit="episode_motion_index_10fps", interval_convention="[start,end)",
            trajectory_format="rotation_6d_first_two_rows_then_xyz", translation_unit="metre",
            coordinate_system="official_processed_Aria_world", rotation_perturbation="world_frame_left_multiply",
            visual_token=0, visual_dimension=1024, missing_source_index=-1,
            corruption_mask_columns=list(MODALITY_ORDER), operation_composition="ordered_stream_transforms",
            rng="numpy.PCG64_with_SHA256_seed_derivation", online_fields=[
                "img_feats", "aria_traj_obs", "img_available", "traj_available", "frame_id_30fps"],
            num_episodes=len(episodes), num_variants=len(records),
            source_labels="unchanged_physical_SMPLX_and_dense_joints_referenced_from_original_sequence",
            evaluation_note="Engineering selection is recorded; exclude these takes from final results if used for tuning.",
        )
        write_json(staging / "spec.json", spec)
        write_json(staging / "split_manifest.json", dict(
            take_to_split=split_map, base_split=source.split, purpose=purpose,
            reserved_engineering_takes=take_names if purpose == "engineering" else [],
        ))
        write_json(staging / "audit" / "inventory.json", inventory)
        write_json(staging / "audit" / "construction_summary.json", dict(
            episodes=episodes, variants=summaries, tasks=dict(Counter(e["task_name"] for e in episodes)),
        ))
        # Validate the entire staged artifact before publishing its final directory.
        from .audit import audit_dataset
        report = audit_dataset(staging, source=source, verify_source=False)
        write_json(staging / "audit" / "validation.json", report)
        os.rename(staging, output)
    print(f"Completed {output}: {len(episodes)} episodes, {len(records)} variants", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-split", choices=("train", "val"), default="val")
    parser.add_argument("--num-takes", type=int, default=12)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--recovery-frames", type=int, default=80)
    parser.add_argument("--profile", choices=("pilot", "extended"), default="pilot")
    parser.add_argument("--purpose", choices=("engineering", "development"), default="engineering")
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--all-sequences", action="store_true",
                        help="Cover every source sequence at its original length, including short ones")
    args = parser.parse_args()
    source = EE4DSource(args.data_root, args.base_split)
    build_dataset(source, args.output, num_takes=args.num_takes, frames=args.frames, seed=args.seed,
                  bootstrap=args.bootstrap, recovery=args.recovery_frames, profile=args.profile,
                  purpose=args.purpose, dev_fraction=args.dev_fraction,
                  holdout_fraction=args.holdout_fraction, all_sequences=args.all_sequences)


if __name__ == "__main__":
    main()
