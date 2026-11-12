"""Deterministic take-balanced validation subsets without data mutation."""

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

ALGORITHM_VERSION = "take_balanced_sha256_round_robin_v1"


def select_training_dev_records(records, count=None, seed=1062, *, selection_file=None):
    """Bind checkpoint selection to an explicit, physically verified clip list."""
    if selection_file is None:
        return select_dev_records(records, count, seed)
    indices, selection = load_fixed_selection(records, selection_file)
    if count is not None and count != len(indices):
        raise ValueError("dev-limit must match the explicit development selection.")
    return indices, {
        **selection,
        "mode": "fixed_physical_clips",
        "selection_file_sha256": hashlib.sha256(Path(selection_file).read_bytes()).hexdigest(),
        "primary_aggregation": "clip_equal",
    }


def load_fixed_selection(records, path):
    """Validate and reuse explicit physical clips without selecting a new subset."""
    selection = json.loads(Path(path).read_text())
    selected = selection["records"]
    if not selected or len(selected) != selection["clips"]:
        raise ValueError("Fixed selection count disagrees with its records.")
    indices, seen = [], set()
    for row in selected:
        index = row["record_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError("Fixed selection has an invalid source index.")
        actual = records[index]
        expected = (row["episode_id"], row["source"], row["take"], row["start"], row["frames"])
        physical = tuple(
            actual[key]
            for key in ("episode_id", "base_seq_name", "base_take_name", "episode_start_motion_idx", "num_frames")
        )
        if physical != expected or row["episode_id"] in seen:
            raise ValueError("Fixed selection physical clip changed or was duplicated.")
        seen.add(row["episode_id"])
        indices.append(index)
    if len({row["take"] for row in selected}) != selection["takes"]:
        raise ValueError("Fixed selection take count disagrees with its records.")
    return indices, selection


def _sha256(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_dev_records(records, count=None, seed=1062):
    """Return source indices and a JSON-safe, hashed subset protocol.

    ``count=None`` preserves the full published order. A positive count uses a
    seeded hash order for takes and for each take's episodes, then visits takes
    round-robin. The selected episode IDs do not depend on source list order.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("Validation selection seed must be an integer, excluding bool.")
    seed = int(seed)
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("Validation selection count must be a positive integer or None.")
        count = int(count)
    records = list(records)
    groups, population, seen = {}, [], set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("Validation records must provide episode_id and base_take_name.")
        identifier, take = record.get("episode_id"), record.get("base_take_name")
        if not isinstance(identifier, str) or not identifier or not isinstance(take, str) or not take:
            raise ValueError("Validation episode_id and base_take_name must be nonempty strings.")
        if identifier in seen:
            raise ValueError("Validation episode identifiers must be unique.")
        seen.add(identifier)
        population.append({"episode_id": identifier, "base_take_name": take})
        groups.setdefault(take, []).append(index)
    if count is None:
        indices = list(range(len(records)))
    else:

        def rank(kind, value):
            return _sha256([ALGORITHM_VERSION, seed, kind, value]), value

        takes = sorted(groups, key=lambda take: rank("take", take))
        ordered = [
            sorted(groups[take], key=lambda index: rank("episode", [take, population[index]["episode_id"]]))
            for take in takes
        ]
        indices, offset, wanted = [], 0, min(count, len(records))
        while len(indices) < wanted:
            for group in ordered:
                if offset < len(group):
                    indices.append(group[offset])
                    if len(indices) == wanted:
                        break
            offset += 1
    protocol = {
        "algorithm": ALGORITHM_VERSION,
        "version": 1,
        "mode": "all_original_order" if count is None else "take_balanced_subset",
        "seed": seed,
        "requested_count": count,
        "count": len(indices),
        "available_count": len(records),
        "available_takes": len(groups),
        "selected_takes": len({population[index]["base_take_name"] for index in indices}),
        "indices": indices,
        "episode_ids": [population[index]["episode_id"] for index in indices],
        "population_sha256": _sha256(population),
    }
    protocol["sha256"] = _sha256(protocol)
    return indices, protocol
