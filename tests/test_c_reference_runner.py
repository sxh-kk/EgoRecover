import json

import pytest

from scripts.run_c_reference import collect_packets, command


def test_c_diagnostic_command_explicitly_preserves_frozen_prior_and_fixed_cohort(tmp_path):
    plan = {
        "python": "python",
        "checkpoint": "C.pt",
        "prior_checkpoint": "P.pt",
        "data_root": "data",
        "selection_file": "400.json",
        "baseline_root": "previous",
    }
    cmd = command(plan, tmp_path, 3)
    for key, value in (
        ("--prior-checkpoint", "P.pt"),
        ("--prior-mode", "learned"),
        ("--selection-file", "400.json"),
        ("--aggregation", "episode"),
        ("--shard-count", "8"),
        ("--shard-index", "3"),
    ):
        assert cmd[cmd.index(key) + 1] == value
    assert "--limit" not in cmd and "--episode-limit" not in cmd
    assert "--verify-sham" in cmd
    assert command(plan, tmp_path, 0, pilot=True)[-2:] == ["--episode-limit", "2"]


def test_collector_requires_all_paired_conditions_and_baseline_checks(tmp_path):
    records = [{"episode_id": f"e{i}", "take": f"t{i}"} for i in range(16)]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"sha256": "cohort", "takes": 16, "records": records}))
    plan = {
        "output": str(tmp_path),
        "selection_file": str(path),
        "checkpoint_sha256": "G",
        "prior_checkpoint_sha256": "P",
        "selection_file_sha256": "S",
        "code_identity": {"code": "frozen"},
    }
    identity = {k: plan[k] for k in ("checkpoint_sha256", "prior_checkpoint_sha256", "selection_file_sha256")}
    identity.update(selection={"sha256": "cohort"}, code=plan["code_identity"])
    pilot = tmp_path / "pilot"
    episodes = pilot / "episodes"
    episodes.mkdir(parents=True)
    (pilot / "identity.json").write_text(json.dumps(identity))
    for record in records[::8]:
        rows = [
            {
                "episode": record["episode_id"],
                "take": record["take"],
                "intervention": mode,
                "anchor": anchor,
                "horizon": horizon,
                "draw": draw,
            }
            for mode in ("body_pose", "full_state")
            for anchor in (20, 40, 60)
            for horizon in (1, 2, 4, 10)
            for draw in (1062, 1063, 1064)
        ]
        packet = {
            "episode": record["episode_id"],
            "take": record["take"],
            "frames": 80,
            "rows": rows,
            "baseline_sha256": "original",
            "verification": {"sham_checks": 3, "baseline_comparisons": 71},
        }
        (episodes / (record["episode_id"] + ".json")).write_text(json.dumps(packet))
    assert len(collect_packets(plan, pilot=True)) == 2
    with pytest.raises(ValueError, match="complete fixed400"):
        collect_packets(plan)
    packet["rows"][0]["draw"] = 999
    (episodes / "e8.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="paired conditions"):
        collect_packets(plan, pilot=True)
