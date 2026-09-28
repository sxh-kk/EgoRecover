"""Frozen experiment split and event-derived offline evaluation windows."""

import hashlib
import json
import math
from pathlib import Path

DEFAULT_SPLIT_MANIFEST = Path(__file__).resolve().parents[1] / "config/egorecover_pilot_split_v1.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_split(path, dataset):
    """A seed-independent, take-disjoint pilot split bound to the dataset spec."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "egorecover-fixed-split-v1":
        raise ValueError("Unsupported EgoRecover split manifest.")
    if manifest.get("base_split") != dataset.spec["base_split"]:
        raise ValueError("Split manifest and dataset base split differ.")
    if manifest.get("dataset_spec_sha256") != file_sha256(dataset.root / "spec.json"):
        raise ValueError("Split manifest was frozen for a different dataset spec.")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "dev", "holdout"}:
        raise ValueError("Split manifest requires train/dev/holdout groups.")
    groups = []
    for key in ("train", "dev", "holdout"):
        value = splits[key]
        if not isinstance(value, list) or not value or any(not isinstance(take, str) or not take for take in value):
            raise ValueError(f"Invalid {key} take list.")
        groups.extend(value)
    available = {record["base_take_name"] for record in dataset.records}
    if len(groups) != len(set(groups)) or set(groups) != available:
        raise ValueError("Split manifest must partition every pilot take exactly once.")
    return {key: list(splits[key]) for key in ("train", "dev", "holdout")}


def _operation_window(record):
    operations = record.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("Record operations must be a list.")
    if not operations:
        return None
    bounds = sorted((int(op["start"]), int(op["end"])) for op in operations)
    count = int(record["num_frames"])
    if any(not 0 <= start < end <= count for start, end in bounds):
        raise ValueError("Fault operation is outside the episode.")
    covered_until = bounds[0][1]
    for start, end in bounds[1:]:
        if start > covered_until:
            raise ValueError("Disjoint fault intervals need an explicit multi-event evaluation protocol.")
        covered_until = max(covered_until, end)
    return bounds[0][0], covered_until


def event_window(record, selected_siblings=()):
    """Use manifest operation times, never an assumed 30-frame duration.

    Clean has no event of its own. It gets a reference window only when all
    selected corrupt siblings agree; otherwise paired per-frame analysis must
    specify the corrupt variant's window explicitly.
    """
    own = _operation_window(record)
    if own is not None:
        return own
    peers = [sibling for sibling in selected_siblings if sibling["base_take_name"] == record["base_take_name"]]
    windows = {_operation_window(sibling) for sibling in peers} - {None}
    return next(iter(windows)) if len(windows) == 1 else None


def phase_indices(frame_indices, window):
    if window is None:
        return {"pre_fault": [], "fault": [], "recovery": []}
    start, end = window
    return {
        "pre_fault": [index for index, frame in enumerate(frame_indices) if frame < start],
        "fault": [index for index, frame in enumerate(frame_indices) if start <= frame < end],
        "recovery": [index for index, frame in enumerate(frame_indices) if frame >= end],
    }


def paired_fault_delta(fault_error_mm, clean_error_mm, frame_indices, window, *, fps=30, recovery_tolerance_mm=None):
    """Pair identical frames from one policy/seed/take, in physical millimetres."""
    if (
        window is None
        or fps <= 0
        or len(fault_error_mm) != len(clean_error_mm)
        or len(frame_indices) != len(fault_error_mm)
    ):
        raise ValueError("Paired fault analysis needs matching frames, an event window and positive FPS.")
    delta = [float(fault) - float(clean) for fault, clean in zip(fault_error_mm, clean_error_mm)]
    if not all(math.isfinite(value) for value in delta):
        raise ValueError("Nonfinite paired fault delta.")
    sections = phase_indices(frame_indices, window)
    fault = [delta[index] for index in sections["fault"]]
    if not fault:
        raise ValueError("Selected rollout does not overlap the fault interval.")
    if recovery_tolerance_mm is not None and (not math.isfinite(recovery_tolerance_mm) or recovery_tolerance_mm < 0):
        raise ValueError("Recovery tolerance must be finite and nonnegative.")
    recovery_seconds = None
    recovery_status = "tolerance_not_predeclared"
    if recovery_tolerance_mm is not None:
        recovery_status = "right_censored"
        width = round(fps)
        recovered = sections["recovery"]
        for offset in range(0, len(recovered) - width + 1):
            items = recovered[offset : offset + width]
            if sum(delta[index] for index in items) / width <= recovery_tolerance_mm:
                recovery_seconds = (frame_indices[items[-1]] - window[1] + 1) / fps
                recovery_status = "observed"
                break
    return {
        "fps": fps,
        "fault_window": list(window),
        "fault_signed_mean_mm": sum(fault) / len(fault),
        "fault_positive_auc_mm_s": sum(max(value, 0) for value in fault) / fps,
        "fault_peak_increase_mm": max(fault),
        "per_frame_delta_mm": delta,
        "recovery_time_s": recovery_seconds,
        "recovery_time_status": recovery_status,
        "recovery_tolerance_mm": recovery_tolerance_mm,
    }
