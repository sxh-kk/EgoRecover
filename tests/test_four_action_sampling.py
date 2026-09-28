import pytest

from run.diagnose_four_actions import select_frames, state_phase, summarize_actions


def test_phase_sampling_respects_fault_window_and_late_fault():
    times = list(range(20, 200))
    records = {
        "freeze": {"base_take_name": "take", "num_frames": 200,
                   "operations": [{"start": 61, "end": 91}]},
        "drift": {"base_take_name": "take", "num_frames": 200,
                  "operations": [{"start": 61, "end": 200}]},
    }
    payload = {
        "record_ids": [record_id for record_id in records for _ in times],
        "time_indices": times * len(records),
        "generator_modes": ["gaussian"] * (len(times) * len(records)),
    }
    selected = select_frames(payload, records, {"take"}, "gaussian")
    picked = {record_id: [] for record_id in records}
    for index in selected:
        picked[payload["record_ids"][index]].append(payload["time_indices"][index])
    assert picked["freeze"][0] < 61
    assert 61 <= picked["freeze"][1] < 91
    assert picked["freeze"][2] >= 91
    assert picked["drift"][0] < 61
    assert 61 <= picked["drift"][1] < picked["drift"][2] < 200


def test_draw_mean_oracle_does_not_choose_a_different_action_for_each_draw():
    rows = []
    for draw, errors in enumerate(([10., 0., 30., 40.], [10., 30., 0., 40.])):
        rows.append({"take": "a", "mode": "history", "variant": "freeze_3s", "time": 70,
                     "phase": "fault", "draw": draw, "canonical_actions": [0, 1, 2, 3],
                     "errors_smpl22_fk_mm": errors, "oracle_gain_mm": 10.})
    group = summarize_actions(rows, 2)["groups"]["history/freeze_3s/fault"]
    assert group["macro_per_draw_oracle_gain_mm"] == 10.
    assert group["macro_oracle_gain_after_draw_mean_mm"] == 0.
    assert group["oracle_action_counts_after_draw_mean"]["0"] == 1
    with pytest.raises(ValueError, match="distinct draws"):
        summarize_actions(rows[:1], 2)


def test_phase_names_cover_recovery_and_persistent_fault():
    record = {"num_frames": 200, "operations": [{"start": 61, "end": 91}]}
    assert state_phase(record, 40) == "pre_fault"
    assert state_phase(record, 70) == "fault"
    assert state_phase(record, 130) == "recovery"
    record["operations"][0]["end"] = 200
    assert state_phase(record, 160) == "late_fault"
