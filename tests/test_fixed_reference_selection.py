import json

import pytest

from egorecover.selection import load_fixed_selection


def test_fixed_selection_verifies_physical_interval_and_order(tmp_path):
    records = [
        {
            "episode_id": f"e{i}",
            "base_seq_name": "s",
            "base_take_name": "t",
            "episode_start_motion_idx": i * 20,
            "num_frames": 80,
        }
        for i in range(3)
    ]
    selection = {
        "clips": 2,
        "takes": 1,
        "records": [
            {"episode_id": f"e{i}", "source": "s", "take": "t", "start": i * 20, "frames": 80, "record_index": i}
            for i in (2, 0)
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    assert load_fixed_selection(records, path) == ([2, 0], selection)
    selection["records"][0]["start"] += 1
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="physical clip"):
        load_fixed_selection(records, path)
    selection["records"][0] = selection["records"][1]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="duplicated"):
        load_fixed_selection(records, path)
