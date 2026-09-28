"""Protocol checks that prevent seed-dependent splits and invented fault spans."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from egorecover.evaluation_protocol import event_window, load_fixed_split, paired_fault_delta, phase_indices


def test_fixed_split_uses_manifest_not_training_seed(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    spec = b'{"base_split":"val"}'
    (dataset_root / "spec.json").write_bytes(spec)
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "egorecover-fixed-split-v1",
                "base_split": "val",
                "dataset_spec_sha256": hashlib.sha256(spec).hexdigest(),
                "splits": {"train": ["a"], "dev": ["b"], "holdout": ["c"]},
            }
        )
    )
    dataset = SimpleNamespace(
        root=dataset_root,
        spec={"base_split": "val"},
        records=[{"base_take_name": name} for name in ("a", "b", "c")],
    )
    assert load_fixed_split(manifest, dataset) == {"train": ["a"], "dev": ["b"], "holdout": ["c"]}
    dataset.records.append({"base_take_name": "d"})
    with pytest.raises(ValueError, match="partition every pilot take"):
        load_fixed_split(manifest, dataset)


def test_event_bounds_come_from_operations_and_ambiguous_clean_has_no_phases():
    clean = {"base_take_name": "a", "num_frames": 100, "operations": []}
    short = {
        "base_take_name": "a",
        "num_frames": 100,
        "operations": [{"start": 30, "end": 40}],
    }
    long = {
        "base_take_name": "a",
        "num_frames": 100,
        "operations": [{"start": 30, "end": 60}],
    }
    assert event_window(short) == (30, 40)
    assert event_window(clean, [clean, short]) == (30, 40)
    assert event_window(clean, [clean, short, long]) is None
    assert phase_indices([20, 30, 39, 40, 60], event_window(short)) == {
        "pre_fault": [0],
        "fault": [1, 2],
        "recovery": [3, 4],
    }
    assert all(not items for items in phase_indices([20, 30, 40], None).values())


def test_overlapping_events_merge_but_disjoint_events_fail_explicitly():
    record = {
        "base_take_name": "a",
        "num_frames": 100,
        "operations": [{"start": 30, "end": 70}, {"start": 35, "end": 40}, {"start": 50, "end": 80}],
    }
    assert event_window(record) == (30, 80)
    record["operations"][2]["start"] = 81
    record["operations"][2]["end"] = 90
    with pytest.raises(ValueError, match="Disjoint fault intervals"):
        event_window(record)


def test_paired_fault_delta_uses_actual_event_and_preserves_signed_change():
    summary = paired_fault_delta([100, 98, 105, 110, 95], [100] * 5, [20, 21, 22, 23, 24], (21, 24), fps=2)
    assert summary["fault_signed_mean_mm"] == pytest.approx(13 / 3)
    assert summary["fault_positive_auc_mm_s"] == pytest.approx(7.5)
    assert summary["fault_peak_increase_mm"] == 10
    assert summary["per_frame_delta_mm"] == [0, -2, 5, 10, -5]
    recovered = paired_fault_delta(
        [0, 4, 6, 1, 1], [0] * 5, [20, 21, 22, 23, 24], (21, 23), fps=2, recovery_tolerance_mm=2
    )
    assert recovered["recovery_time_status"] == "observed"
    assert recovered["recovery_time_s"] == 1.0
