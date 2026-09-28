import json

import pytest

from run.summarize_closed_loop_pilot import summarize


def test_pilot_summary_pairs_takes_and_rejects_duplicate_reports(tmp_path):
    paths = []
    for index, take in enumerate(("take_a", "take_b")):
        report = {
            "completed": True,
            "group": "holdout",
            "take": take,
            "results": [
                {"source_mode": "history", "variant": "clean",
                 "metrics": {"mpjpe_body_m": 0.1 + index * 0.1}},
                {"source_mode": "history", "variant": "freeze_3s",
                 "metrics": {"mpjpe_body_m": 0.12 + index * 0.1},
                 "paired_fault_delta": {"fault_signed_mean_mm": 20 + index * 10,
                                        "fault_positive_auc_mm_s": 2 + index}},
            ],
        }
        path = tmp_path / f"{take}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    result = summarize(paths, [])
    assert result["take_count"] == 2
    rows = {row["variant"]: row for row in result["macro_by_take"]}
    assert rows["clean"]["mean_smpl22_mpjpe_mm"] == pytest.approx(150)
    assert rows["freeze_3s"]["mean_fault_signed_delta_mm"] == pytest.approx(25)
    with pytest.raises(ValueError, match="Duplicate take"):
        summarize([paths[0], paths[0]], [])
