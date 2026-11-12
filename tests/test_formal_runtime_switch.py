import json

import pytest
import torch

from egorecover.io import file_sha256
from scripts.formal_runtime_migration import PACKED_PACKAGE_CHANGES, assert_same_state
from scripts.schedule_formal_runtime import rate, restore_archived_jobs, upgrade, verified_source


def test_real_file_migration_and_failure_rollback_preserve_optimizer_rng_and_logs(tmp_path):
    old_source, new_source = tmp_path / "old", tmp_path / "new"
    for source, content in ((old_source, "old"), (new_source, "new")):
        for relative in PACKED_PACKAGE_CHANGES:
            if source == old_source and relative == "rollin.py":
                continue
            path = source / "egorecover" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def plan(source):
        return {
            "output": str(tmp_path / "run"),
            "source_snapshot": str(source),
            "source_hashes": {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")},
            "jobs": [{"name": "W1"}, {"name": "W3"}],
        }

    original, prepared = plan(old_source), plan(new_source)
    old_code = verified_source(original)
    originals = {}
    for job in original["jobs"]:
        directory = tmp_path / "run" / job["name"]
        directory.mkdir(parents=True)
        saved = {
            "identity": {"code": old_code, "batch": 16},
            "step": 2000,
            "model": {"weight": torch.randn(4)},
            "optimizer": {0: {"step": torch.tensor(2000), "exp_avg": torch.randn(4)}},
            "rng": torch.get_rng_state(),
        }
        originals[job["name"]] = saved
        torch.save(saved, directory / "resume.pt")
        (directory / "identity.json").write_text(json.dumps(saved["identity"]))
        (directory / "train.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in (1999, 2000, 2001)))
    journal = tmp_path / "archive"
    rows = upgrade(original, prepared, 2000, journal)
    assert len(rows) == 2 and all(row["all_other_values_bit_exact"] for row in rows)
    for job in original["jobs"]:
        directory = tmp_path / "run" / job["name"]
        restored = torch.load(directory / "resume.pt", weights_only=True)
        assert_same_state(originals[job["name"]], restored)
        assert restored["identity"]["code"] == verified_source(prepared)
        assert json.loads((directory / "progress.json").read_text())["step"] == 2000
        assert "2001" not in (directory / "train.jsonl").read_text()
        assert "2001" in (journal / job["name"] / "train.jsonl").read_text()
        restored["model"]["weight"].add_(1)
        torch.save(restored, directory / "resume.pt")
    restore_archived_jobs(original, journal, 2000)
    for job in original["jobs"]:
        restored = torch.load(tmp_path / "run" / job["name"] / "resume.pt", weights_only=True)
        assert_same_state(originals[job["name"]], restored)
        assert restored["identity"]["code"] == old_code


def test_throughput_uses_saved_update_timestamps_and_rejects_nonadvancing_steps():
    assert rate([{"step": 2010, "mtime_ns": 0}, {"step": 2040, "mtime_ns": 60_000_000_000}])["seconds_per_update"] == 2
    with pytest.raises(ValueError):
        rate([{"step": 2010, "mtime_ns": 0}])
