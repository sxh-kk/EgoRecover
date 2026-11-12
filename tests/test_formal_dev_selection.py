import json
from pathlib import Path

import pytest

from egorecover.cli import build_parser
from egorecover.selection import select_training_dev_records


def fixture(tmp_path):
    records = [
        {
            "episode_id": f"take{i}_start0",
            "base_seq_name": f"take{i}___0___237",
            "base_take_name": f"take{i}",
            "episode_start_motion_idx": 0,
            "num_frames": 80,
        }
        for i in range(3)
    ]
    rows = [
        {
            "episode_id": records[i]["episode_id"],
            "source": records[i]["base_seq_name"],
            "take": records[i]["base_take_name"],
            "start": 0,
            "frames": 80,
            "record_index": i,
        }
        for i in (2, 0)
    ]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"clips": 2, "takes": 2, "records": rows}))
    return records, path


def test_explicit_clips_override_seed_and_bind_physical_order_and_file_content(tmp_path):
    records, path = fixture(tmp_path)
    indices, identity = select_training_dev_records(records, selection_file=path)
    assert indices == [2, 0]
    assert identity["primary_aggregation"] == "clip_equal"
    assert select_training_dev_records(records, seed=5, selection_file=path) == (indices, identity)
    payload = json.loads(path.read_text())
    payload["purpose"] = "different protocol identity"
    path.write_text(json.dumps(payload))
    assert select_training_dev_records(records, selection_file=path)[1] != identity


def test_explicit_selection_rejects_a_conflicting_limit_and_changed_physical_source(tmp_path):
    records, path = fixture(tmp_path)
    with pytest.raises(ValueError, match="dev-limit"):
        select_training_dev_records(records, count=1, selection_file=path)
    records[2]["episode_start_motion_idx"] = 1
    with pytest.raises(ValueError, match="physical clip changed"):
        select_training_dev_records(records, selection_file=path)


def test_train_cli_accepts_the_frozen_dev_manifest():
    args = build_parser().parse_args(["train", "--output", "unused", "--dev-selection-file", "fixed.json"])
    assert args.dev_selection_file == Path("fixed.json")


def test_formal_manifest_is_a_balanced_complete_subset_of_the_previous400():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/reports/data/formal_dev128_20261112.json"
    selection = json.loads(path.read_text())
    assert len(selection["records"]) == selection["clips"] == 128
    assert len({row["take"] for row in selection["records"]}) == selection["takes"] == 128
    assert all(row["frames"] == 80 for row in selection["records"])
    assert all(sum(row["shard"] == i for row in selection["records"]) == 32 for i in range(4))
