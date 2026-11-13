"""Read-only CPU diagnosis of P01 sampling, bank inputs and target alignment.

Uses existing banks/checkpoints. It never produces G histories or trains P.
Run with PYTHONPATH pointing at the P01 frozen source for exact codec/FK math.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from egorecover.codec import MotionCodec, rigid_inverse, transform_from_9d
from egorecover.fk import FixedShapeFK, load_smpl
from egorecover.io import file_sha256, save_json
from egorecover.official import SMPL22_PARENTS
from egorecover.prior_data import TENSOR_FIELDS, reconstructed_history_count
from egorecover.rotations import rotation_6d_to_matrix

CODE_GROUPS = {
    "body_rotation6d": [9 * joint + channel for joint in range(22) for channel in range(6)],
    "root_translation": list(range(6, 9)),
    "nonroot_dense_translation": [9 * joint + channel for joint in range(1, 22) for channel in range(6, 9)],
    "reference_rotation6d": list(range(198, 204)),
    "reference_translation": list(range(204, 207)),
    "hand_PCA": list(range(207, 231)),
    "contact": [231, 232],
    "beta": list(range(233, 243)),
}
EARLY_L = {0, 1, 4, 8, 12, 19}
ACTIVITIES = ("basketball", "soccer", "bike", "dance", "cooking", "cpr", "covid", "climbing", "knitting", "golf")


def activity(take):
    return next((token for token in ACTIVITIES if token in take.lower().split("_")), "other")


def summary(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"n": int(len(array)), "nonfinite": int(len(array))}
    return {
        "n": int(len(array)),
        "nonfinite": int(len(array) - len(finite)),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "p50": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def groups_for_sampler(rows):
    groups = {True: defaultdict(lambda: defaultdict(list)), False: defaultdict(lambda: defaultdict(list))}
    for index, row in enumerate(rows):
        groups[row["frame"] >= 40][row["take"]][row["source"]].append(index)
    return {
        phase: [
            (take, [(source, sorted(ids)) for source, ids in sorted(sources.items())])
            for take, sources in sorted(takes.items())
        ]
        for phase, takes in groups.items()
    }


def sample_indices(groups, generator, count):
    result = []
    for phase, amount in ((True, 3 * count // 4), (False, count // 4)):
        takes = groups[phase]
        for _ in range(amount):
            _, sources = takes[int(torch.randint(len(takes), (), generator=generator))]
            _, anchors = sources[int(torch.randint(len(sources), (), generator=generator))]
            result.append(anchors[int(torch.randint(len(anchors), (), generator=generator))])
    return result


def sampling_diagnosis(bank, saved, traces):
    rows, groups = bank["rows"], groups_for_sampler(bank["rows"])
    ranks, step, seed = len(saved["rank_rng_states"]), saved["step"], saved["identity"]["seed"]
    generators = [torch.Generator().manual_seed(seed + 1 + rank) for rank in range(ranks)]
    counts = np.zeros((ranks, len(rows)), dtype=np.int64)
    mismatches = []
    for update in range(1, step + 1):
        drawn = []
        for rank, generator in enumerate(generators):
            indices = sample_indices(groups, generator, 8) + sample_indices(groups, generator, 8)
            drawn.append(indices)
            np.add.at(counts[rank], indices, 1)
        digest = hashlib.sha256(json.dumps(drawn).encode()).hexdigest()
        if digest != traces[update]["trace_hash"]:
            mismatches.append(update)
    per_rank = []
    for rank, generator in enumerate(generators):
        state = saved["rank_rng_states"][rank]
        visited = set(np.flatnonzero(counts[rank]).tolist())
        expected_counts = Counter()
        for index, count in enumerate(counts[rank]):
            if count:
                expected_counts["mature" if rows[index]["frame"] >= 40 else "early"] += int(count)
                expected_counts[f"L{rows[index]['frame'] - 20}"] += int(count)
        per_rank.append(
            {
                "rank": rank,
                "draws": int(counts[rank].sum()),
                "final_sample_rng_exact": torch.equal(generator.get_state(), state["sample_rng"]),
                "visited_indices_exact": visited == set(state["visited_indices"]),
                "all_L_and_phase_counters_exact": dict(expected_counts) == state["sampling_counts"],
            }
        )
    probabilities = np.zeros(len(rows), dtype=np.float64)
    for phase, takes in groups.items():
        for _, sources in takes:
            for _, anchors in sources:
                probabilities[anchors] = (0.75 if phase else 0.25) / len(takes) / len(sources) / len(anchors)
    combined = counts.sum(0)
    observed = np.flatnonzero(combined)
    expected_unique = sum(
        1
        - (1 - probabilities[index] / (0.75 if row["frame"] >= 40 else 0.25))
        ** (96000 if row["frame"] >= 40 else 32000)
        for index, row in enumerate(rows)
    )
    result = {
        "method": "independent scalar-randint replay of frozen sampler, seed63..66, two micro8 per rank per update",
        "step": step,
        "total_draws": int(combined.sum()),
        "trace_hash_mismatch_steps": mismatches,
        "all_2000_step_trace_hashes_match": not mismatches,
        "per_rank_recovery_checks": per_rank,
        "unique_anchors": len(observed),
        "unique_sources": len({rows[index]["source"] for index in observed}),
        "unique_takes": len({rows[index]["take"] for index in observed}),
        "expected_unique_anchors_under_declared_hierarchy": expected_unique,
        "anchor_probability": summary(probabilities),
        "anchor_probability_max_min_ratio": float(probabilities.max() / probabilities.min()),
        "effective_number_of_uniformly_weighted_anchors": float(1 / np.square(probabilities).sum()),
        "phase": {},
    }
    for phase, name in ((False, "early"), (True, "mature")):
        indices = [index for index, row in enumerate(rows) if (row["frame"] >= 40) == phase]
        histogram = Counter()
        take_visits, source_visits = Counter(), Counter()
        for index in indices:
            row, count = rows[index], int(combined[index])
            histogram[str(row["L"])] += count
            take_visits[row["take"]] += count
            source_visits[row["source"]] += count
        total = sum(histogram.values())
        result["phase"][name] = {
            "draws": total,
            "bank_rows": len(indices),
            "unique_visited_rows": int((combined[indices] > 0).sum()),
            "eligible_takes": len(groups[phase]),
            "eligible_sources": len(source_visits),
            "take_draw_counts": summary(list(take_visits.values())),
            "source_draw_counts": summary(list(source_visits.values())),
            "L_counts": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            "L_greater_than59_fraction": sum(count for value, count in histogram.items() if int(value) > 59) / total,
            "five_long_context_anchor_L_fraction": sum(histogram[str(value)] for value in (20, 50, 80, 109, 139))
            / total,
        }
    return result, probabilities


def bank_checks(bank):
    rows, flags = bank["rows"], bank["tensors"]["body_valid"]
    L = torch.tensor([row["L"] for row in rows])
    counts = torch.tensor([reconstructed_history_count(int(value), 3, True) for value in L])
    expected = torch.arange(20)[None] >= 20 - counts[:, None]
    metadata_ok = all(
        row["L"] == row["frame"] - 20
        and row["source_query"] == row["context_start"] + row["frame"]
        and row["window"] == 3
        and row["kind"] == "predicted"
        for row in rows
    )
    return {
        "format": bank["format"],
        "cohort": bank["cohort"],
        "rows": len(rows),
        "producer": bank["producer"],
        "selection_sha256": bank["selection_sha256"],
        "manifest_sha256": bank["manifest_sha256"],
        "local_query_source_query_metadata_valid": metadata_ok,
        "all_bank_body_valid_flags_match_local_L_W3_rewrites": torch.equal(flags, expected),
        "mature_all_flags_true": bool(flags[L >= 20].all()),
        "early_flag_true_counts": dict(
            sorted(Counter(str(int(value)) for value in flags[L < 20].sum(-1)).items(), key=lambda item: int(item[0]))
        ),
        "mature_L_range": [int(L[L >= 20].min()), int(L.max())],
        "nonzero_context_start_rows": sum(row["context_start"] != 0 for row in rows),
        "takes": len({row["take"] for row in rows}),
        "sources": len({row["source"] for row in rows}),
    }


@torch.no_grad()
def row_metrics(bank, indices, codec, fk):
    weights = torch.ones(243)
    weights[198:207] = 8
    metrics = defaultdict(list)
    for start in range(0, len(indices), 256):
        ids = indices[start : start + 256]
        tensors = {key: bank["tensors"][key][ids] for key in TENSOR_FIELDS}
        base, target = tensors["base"][:, 0], tensors["target"][:, 0]
        error = (target - base).square() * weights
        total = error.sum(-1) / weights.sum()
        metrics["weighted_code_CV_error"].append(total)
        for group, channels in CODE_GROUPS.items():
            metrics["code_contribution_" + group].append(error[:, channels].sum(-1) / weights.sum())
        metrics["normalized_base_max_abs"].append(base.abs().amax(-1))
        metrics["normalized_target_max_abs"].append(target.abs().amax(-1))
        metrics["normalized_history_max_abs"].append(tensors["history"].abs().flatten(1).amax(-1))
        cv = codec.decode_current(base, tensors["reference"])
        truth_state = codec.decode_current(target, tensors["reference"])
        cv_fk, truth_fk = fk(cv), fk(truth_state)
        positions = tensors["truth_xyz"]
        metrics["CV_FK_MPJPE_mm"].append((cv_fk - positions).norm(dim=-1).mean(-1) * 1000)
        metrics["GT_target_FK_MPJPE_mm"].append((truth_fk - positions).norm(dim=-1).mean(-1) * 1000)
        metrics["CV_dense_MPJPE_mm"].append((cv.joints[..., :3, 3] - positions).norm(dim=-1).mean(-1) * 1000)
        metrics["GT_decode_truth_xyz_max_abs_mm"].append(
            (truth_state.joints[..., :3, 3] - positions).abs().flatten(1).amax(-1) * 1000
        )
        metrics["recorded_CV_FK_error_difference_mm"].append(
            (torch.tensor([bank["rows"][index]["cv_body_mm"] for index in ids]) - metrics["CV_FK_MPJPE_mm"][-1]).abs()
        )
        raw_base, raw_target = codec.denormalize(base), codec.denormalize(target)
        metrics["CV_beta_norm"].append(raw_base[:, -10:].norm(dim=-1))
        metrics["GT_beta_norm"].append(raw_target[:, -10:].norm(dim=-1))
        metrics["beta_error_norm"].append((raw_base[:, -10:] - raw_target[:, -10:]).norm(dim=-1))
        for prefix, raw in (("CV", raw_base), ("GT_label", raw_target)):
            delta = transform_from_9d(raw[:, 198:207])
            metrics[prefix + "_reference_delta_xy_mm"].append(delta[:, :2, 3].norm(dim=-1) * 1000)
            metrics[prefix + "_reference_delta_yaw_deg"].append(
                torch.atan2(delta[:, 1, 0], delta[:, 0, 0]).abs() * (180 / math.pi)
            )
        relative = cv.joints[..., :3, :3].transpose(-1, -2) @ truth_state.joints[..., :3, :3]
        angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
        metrics["CV_body_rotation_mean_deg"].append(angle.mean(-1) * (180 / math.pi))
        metrics["CV_root_position_error_mm"].append((cv_fk[:, 0] - positions[:, 0]).norm(dim=-1) * 1000)
        last_raw = codec.denormalize(tensors["history"][:, -1])
        before_raw = codec.denormalize(tensors["history"][:, -2])
        last_delta = codec.canonical_reference(transform_from_9d(last_raw[:, 198:207]))
        before_delta = codec.canonical_reference(transform_from_9d(before_raw[:, 198:207]))
        before_ref = codec.canonical_reference(tensors["reference"] @ rigid_inverse(last_delta))
        before_before_ref = codec.canonical_reference(before_ref @ rigid_inverse(before_delta))
        last = codec.decode_current(tensors["history"][:, -1], before_ref)
        before = codec.decode_current(tensors["history"][:, -2], before_before_ref)
        recomputed = codec.constant_velocity_prior(before, last)
        recomputed_state = codec.decode_current(recomputed, last.reference)
        metrics["CV_from_stored_history_dense_max_abs_mm"].append(
            (recomputed_state.joints[..., :3, 3] - cv.joints[..., :3, 3]).abs().flatten(1).amax(-1) * 1000
        )
        metrics["CV_from_stored_history_code_max_abs"].append((recomputed - base).abs().amax(-1))
    return {key: torch.cat(values).numpy() for key, values in metrics.items()}


def summarize_rows(bank, ids, metrics):
    rows = [bank["rows"][index] for index in ids]
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "early": np.array([row["L"] < 20 for row in rows]),
        "mature": np.array([row["L"] >= 20 for row in rows]),
        "mature_L20_59": np.array([20 <= row["L"] <= 59 for row in rows]),
        "mature_L60_139": np.array([row["L"] >= 60 for row in rows]),
        "early_at_training_Ls": np.array([row["L"] in EARLY_L for row in rows]),
        "L20": np.array([row["L"] == 20 for row in rows]),
        "L50": np.array([row["L"] == 50 for row in rows]),
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        subrows = [row for row, keep in zip(rows, mask) if keep]
        average_error = float(metrics["weighted_code_CV_error"][mask].mean())
        result[name] = {
            "rows": len(subrows),
            "takes": len({row["take"] for row in subrows}),
            "unique_anchors": len({(row["episode"], row["frame"]) for row in subrows}),
            "L": summary([row["L"] for row in subrows]),
            "activity_proxy_counts": dict(Counter(activity(row["take"]) for row in subrows)),
            "metrics": {key: summary(values[mask]) for key, values in metrics.items()},
            "code_error_fraction_by_group": {
                key: float(metrics["code_contribution_" + key][mask].mean()) / max(average_error, 1e-30)
                for key in CODE_GROUPS
            },
        }
    top = np.argsort(metrics["weighted_code_CV_error"])[-12:][::-1]
    result["largest_sample_code_residual_rows"] = [
        {
            **rows[index],
            "weighted_code_error": float(metrics["weighted_code_CV_error"][index]),
            "largest_group": max(CODE_GROUPS, key=lambda key: metrics["code_contribution_" + key][index]),
            "reference_code_fraction": float(
                (
                    metrics["code_contribution_reference_rotation6d"][index]
                    + metrics["code_contribution_reference_translation"][index]
                )
                / max(metrics["weighted_code_CV_error"][index], 1e-30)
            ),
            "GT_target_FK_MPJPE_mm": float(metrics["GT_target_FK_MPJPE_mm"][index]),
        }
        for index in top
    ]
    return result


def weighted_summary(values, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    order = np.argsort(values)
    values, weights = np.asarray(values, dtype=np.float64)[order], weights[order]
    cumulative = weights.cumsum()
    result = {"n": len(values), "mean": float((values * weights).sum()), "max": float(values[-1])}
    for quantile in (0.5, 0.9, 0.95, 0.99):
        result[f"p{round(100 * quantile)}"] = float(
            values[min(int(np.searchsorted(cumulative, quantile)), len(values) - 1)]
        )
    top1 = cumulative >= 0.99
    result["upper_one_percent_weighted_energy_fraction"] = float(
        (values[top1] * weights[top1]).sum() / max((values * weights).sum(), 1e-30)
    )
    return result


def weighted_correlation(first, second, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    first = first - (first * weights).sum()
    second = second - (second * weights).sum()
    return float(
        (weights * first * second).sum() / max(np.sqrt((weights * first**2).sum() * (weights * second**2).sum()), 1e-30)
    )


@torch.no_grad()
def full_code_scan(bank, codec, *, probabilities=None):
    """Read base/target only (~218MB train), not the2.2GB history tensor."""
    vectors = defaultdict(list)
    weights = torch.ones(243)
    weights[198:207] = 8
    for start in range(0, len(bank["rows"]), 4096):
        base = bank["tensors"]["base"][start : start + 4096, 0]
        target = bank["tensors"]["target"][start : start + 4096, 0]
        error = (target - base).square() * weights
        vectors["code"].append(error.sum(-1).numpy() / weights.sum().item())
        vectors["reference_code"].append(error[:, 198:207].sum(-1).numpy() / weights.sum().item())
        for name, value in (("base", base), ("target", target)):
            raw = value[:, 198:207] * codec.motion_std[198:207] + codec.motion_mean[198:207]
            vectors[name + "_delta_yaw"].append(torch.atan2(raw[:, 3], raw[:, 0]).numpy())
    arrays = {key: np.concatenate(values).astype(np.float64) for key, values in vectors.items()}
    rows = bank["rows"]
    yaw_difference = arrays["target_delta_yaw"] - arrays["base_delta_yaw"]
    arrays["CV_GT_reference_yaw_error_deg"] = (
        np.abs(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference))) * 180 / math.pi
    )
    arrays["CV_body_mm"] = np.array([row["cv_body_mm"] for row in rows])
    masks = {
        "early": np.array([row["L"] < 20 for row in rows]),
        "mature": np.array([row["L"] >= 20 for row in rows]),
        "mature_L20_59": np.array([20 <= row["L"] <= 59 for row in rows]),
        "mature_L60_139": np.array([row["L"] >= 60 for row in rows]),
    }
    p = np.ones(len(rows)) if probabilities is None else probabilities
    result = {
        "scope": "Full base+target code scan only; physical history inputs/FK remain sampled.",
        "weighting": (
            "training take->source->anchor probability"
            if probabilities is not None
            else "uniform dense D128 rows, equal clip lengths"
        ),
    }
    for name, mask in masks.items():
        if not mask.any():
            continue
        subp = p[mask] / p[mask].sum()
        result[name] = {
            "code_error": weighted_summary(arrays["code"][mask], subp),
            "reference_yaw_error_deg": weighted_summary(arrays["CV_GT_reference_yaw_error_deg"][mask], subp),
            "CV_body_mm": weighted_summary(arrays["CV_body_mm"][mask], subp),
            "reference_fraction_of_expected_code_energy": float(
                np.sum(subp * arrays["reference_code"][mask]) / np.sum(subp * arrays["code"][mask])
            ),
            "code_vs_reference_yaw_error_Pearson": weighted_correlation(
                arrays["code"][mask], arrays["CV_GT_reference_yaw_error_deg"][mask], subp
            ),
            "code_vs_CV_body_error_Pearson": weighted_correlation(
                arrays["code"][mask], arrays["CV_body_mm"][mask], subp
            ),
            "probability_reference_yaw_error_above45deg": float(
                subp[arrays["CV_GT_reference_yaw_error_deg"][mask] > 45].sum()
            ),
        }
    outliers = np.argsort(arrays["code"])[-12:][::-1]
    result["largest_full_bank_code_rows"] = [
        {
            **rows[index],
            "weighted_code_error": float(arrays["code"][index]),
            "reference_fraction": float(arrays["reference_code"][index] / arrays["code"][index]),
            "expected_training_probability": float(p[index]) if probabilities is not None else None,
        }
        for index in outliers
    ]
    if bank["cohort"] == "D128":
        reference = bank["tensors"]["reference"]
        previous_yaw = torch.atan2(reference[:, 1, 0], reference[:, 0, 0]).numpy()
        gt_yaw = previous_yaw + arrays["target_delta_yaw"]
        lookup = {(row["episode"], row["frame"]): index for index, row in enumerate(rows)}
        cache_errors, observed_steps = [], []
        for index, row in enumerate(rows):
            if row["L"] < 20:
                continue
            before = lookup[(row["episode"], row["frame"] - 1)]
            gap = previous_yaw[index] - gt_yaw[before]
            change = gt_yaw[index] - gt_yaw[before]
            cache_errors.append(abs(math.atan2(math.sin(gap), math.cos(gap))) * 180 / math.pi)
            observed_steps.append(abs(math.atan2(math.sin(change), math.cos(change))) * 180 / math.pi)
        result["mature_heading_labels"] = {
            "previous_G_reference_vs_previous_recorded_GT_head_deg": summary(cache_errors),
            "recorded_GT_head_one_step_change_deg": summary(observed_steps),
        }
    return result


def merged_chunk_spots(bank, directory, *, samples=24):
    generator = np.random.default_rng(4062)
    ids = generator.choice(len(bank["rows"]), size=min(samples, len(bank["rows"])), replace=False).tolist()
    by_context = defaultdict(list)
    for index in ids:
        by_context[bank["rows"][index]["context_index"]].append(index)
    files = {}
    for complete in Path(directory).glob("shard*/complete.json"):
        manifest = json.loads(complete.read_text())
        for chunk in manifest["chunks"]:
            if any(index in by_context for index in chunk["context_indices"]):
                path = complete.parent / chunk["file"]
                if path.is_file():
                    files[path] = chunk["context_indices"]
    checked, mismatches = 0, []
    for path, contexts in files.items():
        packet = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        lookup = {(row["context_index"], row["frame"]): index for index, row in enumerate(packet["rows"])}
        for context in contexts:
            for index in by_context.get(context, []):
                row = bank["rows"][index]
                original = lookup[(context, row["frame"])]
                if packet["rows"][original] != row or any(
                    not torch.equal(bank["tensors"][key][index], packet["tensors"][key][original])
                    for key in TENSOR_FIELDS
                ):
                    mismatches.append(index)
                checked += 1
    return {
        "requested": len(ids),
        "checked": checked,
        "mismatched_bank_indices": mismatches,
        "scope": "metadata and all6 tensors compared to original generated chunks; no G regeneration",
    }


@torch.no_grad()
def raw_spots(bank, raw_file, codec, *, samples):
    if not samples:
        return {"checked": 0, "scope": "Raw dataset spot-check disabled."}
    raw = torch.load(raw_file, weights_only=False, map_location="cpu", mmap=True)
    ids = (
        np.random.default_rng(5062)
        .choice(len(bank["rows"]), size=min(samples, len(bank["rows"])), replace=False)
        .tolist()
    )
    differences = defaultdict(list)
    for index in ids:
        row = bank["rows"][index]
        original = raw[row["source"]]
        query = row["source_query"]
        floor = float(original["floor_height"])
        xyz = original["kp3d"][query, :22].float().clone()
        xyz[:, 2] -= floor
        differences["truth_xyz_max_abs_mm"].append(
            float((xyz - bank["tensors"]["truth_xyz"][index]).abs().max()) * 1000
        )
        timestamp = (original["start_idx"] + 3 * query) / 30
        differences["timestamp_abs_seconds"].append(abs(timestamp - row["timestamp_s"]))
        target = bank["tensors"]["target"][index : index + 1, 0]
        raw_target = codec.denormalize(target)[0]
        differences["beta_max_abs"].append(float((raw_target[-10:] - original["smpl_params"]["betas"][0]).abs().max()))
        state = codec.decode_current(target, bank["tensors"]["reference"][index : index + 1])
        parameters = original["smpl_params"]
        local = rotation_6d_to_matrix(
            torch.cat((parameters["global_orient"][query : query + 1], parameters["body_pose"][query]), dim=0).float()
        )
        global_rotation = []
        for joint, parent in enumerate(SMPL22_PARENTS):
            global_rotation.append(local[joint] if parent < 0 else global_rotation[parent] @ local[joint])
        differences["GT_global_rotation_max_abs"].append(
            float((state.joints[0, :, :3, :3] - torch.stack(global_rotation)).abs().max())
        )
        head = original["aria_traj"][query].float().clone()
        head[8] -= floor
        expected_reference = codec.canonical_reference(transform_from_9d(head[None]))
        differences["GT_reference_max_abs"].append(float((state.reference - expected_reference).abs().max()))
    return {
        "checked": len(ids),
        "bank_indices": ids,
        "differences": {key: summary(values) for key, values in differences.items()},
        "scope": "Only selected rows in memory-mapped trusted official data; no full source/data audit",
    }


@torch.no_grad()
def reference_outlier_diagnosis(bank, raw_file, codec, outliers):
    """Separate recorded head turns, cache heading error and normalization."""
    raw = torch.load(raw_file, weights_only=False, map_location="cpu", mmap=True)
    lookup = {(row["episode"], row["frame"]): index for index, row in enumerate(bank["rows"])}

    def yaw(rotation):
        return float(torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])) * 180 / math.pi

    def angular_distance(first, second):
        return abs((first - second + 180) % 360 - 180)

    rows = []
    for outlier in outliers:
        index = lookup[(outlier["episode"], outlier["frame"])]
        row = bank["rows"][index]
        query = row["source_query"]
        head = transform_from_9d(raw[row["source"]]["aria_traj"][query - 2 : query + 1])
        rotation = head[:, :3, :3]
        angles = ((rotation[:-1].transpose(-1, -2) @ rotation[1:]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        yaws = [yaw(value) for value in rotation]
        reference = bank["tensors"]["reference"][index : index + 1]
        base = bank["tensors"]["base"][index : index + 1, 0]
        target = bank["tensors"]["target"][index : index + 1, 0]
        cv = codec.decode_current(base, reference)
        truth = codec.decode_current(target, reference)
        previous_yaw = yaw(reference[0, :3, :3])
        rows.append(
            {
                "episode": row["episode"],
                "take": row["take"],
                "local_frame": row["frame"],
                "source_query": query,
                "code_loss": outlier["weighted_code_error"],
                "reference_code_fraction": outlier["reference_code_fraction"],
                "CV_MPJPE_mm": row["cv_body_mm"],
                "recorded_head_yaws_last3_deg": yaws,
                "recorded_head_3D_step_deg": (angles.clamp(-1, 1).acos() * 180 / math.pi).tolist(),
                "recorded_head_horizontal_projection_norm": head[:, :2, 0].norm(dim=-1).tolist(),
                "predicted_previous_reference_yaw_deg": previous_yaw,
                "CV_reference_yaw_deg": yaw(cv.reference[0, :3, :3]),
                "GT_reference_yaw_deg": yaw(truth.reference[0, :3, :3]),
                "previous_reference_vs_recorded_previous_head_error_deg": angular_distance(previous_yaw, yaws[-2]),
                "recorded_latest_head_yaw_step_deg": angular_distance(yaws[-1], yaws[-2]),
                "normalized_reference_target_minus_CV": (target[0, 198:207] - base[0, 198:207]).tolist(),
            }
        )
    return {
        "rows": rows,
        "reference_normalization_std": codec.motion_std[198:207].tolist(),
        "scope": "Highest12 code-error D128 rows, not a full motion/observation audit.",
        "interpretation_limit": "Large rotations are recorded aria_traj observations; original video/SLAM reliability was not independently checked.",
    }


def existing_activity_diagnosis(report):
    result = {}
    for phase in ("all", "mature", "startup"):
        grouped = defaultdict(list)
        for row in report["rows"]:
            if row["phase"] == phase:
                grouped[activity(row["take"])].append(row)
        result[phase] = {
            name: {
                "clips": len(rows),
                "P_minus_CV_MPJPE_mm": sum(row["P"]["mpjpe_mm"] - row["CV"]["mpjpe_mm"] for row in rows) / len(rows),
                "CV_MPJPE_mm": sum(row["CV"]["mpjpe_mm"] for row in rows) / len(rows),
                "worse_clips": sum(row["P"]["mpjpe_mm"] > row["CV"]["mpjpe_mm"] for row in rows),
            }
            for name, rows in grouped.items()
        }
    result["classification_scope"] = (
        "Take-name keyword proxy, not official activity labels; other includes uncategorized names."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--raw-spots", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    run = Path(plan["run_directory"])
    saved = torch.load(run / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
    train = torch.load(plan["train_bank"], weights_only=True, map_location="cpu", mmap=True)
    val = torch.load(plan["val_bank"], weights_only=True, map_location="cpu", mmap=True)
    traces = {
        row["step"]: row
        for line in (run / "train.jsonl").read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    print("Loaded frozen banks and2000-step native sampling state.", flush=True)
    sampling, probabilities = sampling_diagnosis(train, saved, traces)
    train_ids = sample_indices(
        groups_for_sampler(train["rows"]), torch.Generator().manual_seed(3062), args.train_samples
    )
    val_ids = list(range(len(val["rows"])))
    os.environ["SMPLX_MODEL_PATH"] = plan["smpl_directory"]
    codec = MotionCodec(train["stats"]).cpu()
    fk = FixedShapeFK(load_smpl("cpu"), torch.zeros(10), shape_mode="predicted")
    print("Exact sampler replay complete; computing sampled CPU code/FK diagnostics.", flush=True)
    train_metrics = row_metrics(train, train_ids, codec, fk)
    val_metrics = row_metrics(val, val_ids, codec, fk)
    checks = {"train": bank_checks(train), "val": bank_checks(val)}
    chunk_checks = {
        "train": merged_chunk_spots(train, Path(plan["train_bank"]).parent),
        "val": merged_chunk_spots(val, Path(plan["val_bank"]).parent),
    }
    print("Code/FK and merged/chunk diagnostics complete; checking a few raw label rows.", flush=True)
    original = Path(plan["data_root"]) / "uniegomotion"
    raw_checks = {
        "train": raw_spots(train, original / "ee_train.pt", codec, samples=args.raw_spots),
        "val": raw_spots(val, original / "ee_val.pt", codec, samples=args.raw_spots),
    }
    train_summary = summarize_rows(train, train_ids, train_metrics)
    val_summary = summarize_rows(val, val_ids, val_metrics)
    result = {
        "scope": "P01 bank/sampling/representation diagnosis only; no new training or G-history generation",
        "sampling": sampling,
        "bank_metadata_and_flags": checks,
        "producer_dicts_identical": train["producer"] == val["producer"],
        "train_val_takes_disjoint": not ({row["take"] for row in train["rows"]} & {row["take"] for row in val["rows"]}),
        "bank_and_chunk_alignment": chunk_checks,
        "raw_label_spots": raw_checks,
        "statistical_sample": {
            "seed": 3062,
            "train_draws": args.train_samples,
            "train_scheme": "same75:25 take->source->anchor hierarchy, independent diagnostic draws, with replacement",
            "val_rows": len(val_ids),
            "validation_clip_equal_equals_take_equal": len({row["episode"] for row in val["rows"]})
            == len({row["take"] for row in val["rows"]}),
        },
        "train": train_summary,
        "val": val_summary,
        "full_code_scan": {
            "train": full_code_scan(train, codec, probabilities=probabilities),
            "val": full_code_scan(val, codec),
        },
        "existing_step2000_validation_activity_proxy": existing_activity_diagnosis(
            json.loads((run / f"validation-step{saved['step']}.json").read_text())
        ),
        "reference_outliers": (
            reference_outlier_diagnosis(
                val, original / "ee_val.pt", codec, val_summary["largest_sample_code_residual_rows"]
            )
            if args.raw_spots
            else {"checked": 0, "scope": "Raw reference outlier checks disabled."}
        ),
        "causal_code_evidence": {
            "frozen_prior_data_sha256": file_sha256(Path(plan["source_snapshot"]) / "egorecover/prior_data.py"),
            "prior_inputs": "Reads past body history only; current truth enters target/truth_xyz only",
            "native_initial_health": json.loads((run / "health_checks.json").read_text()),
            "cache_producer": train["producer"],
            "limitations": "No new G replay; bank generation causality is code/identity evidence plus existing preflight and sampled target checks.",
        },
        "interpretation_limits": [
            "Training and validation takes differ; aggregate moment differences alone do not prove harmful distribution shift.",
            "MatureL20..139 versusL20..59 is a planned support difference; compare overlappingL and actual residual statistics before attributing failure.",
            "Weighted code residual and FK position error measure different aspects of a dense243-channel representation.",
            "Sampled train statistics are not a full raw-dataset audit; no optimizer or predictor was modified.",
        ],
    }
    output = args.output or root / "diagnosis/distribution.json"
    save_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "all_sampling_hashes_match": sampling["all_2000_step_trace_hashes_match"],
                "unique_anchors": sampling["unique_anchors"],
                "train_mature_code": result["train"]["mature"]["metrics"]["weighted_code_CV_error"],
                "val_mature_code": result["val"]["mature"]["metrics"]["weighted_code_CV_error"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
