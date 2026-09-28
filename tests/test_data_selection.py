"""A growing handoff must not silently change a frozen experiment dataset."""

import json

import pytest

from egorecover.data import read_handoff, select_handoff_dataset
from egorecover import data
from egorecover.evaluation_protocol import file_sha256


def test_duplicate_pilot_profiles_require_the_exact_frozen_spec():
    handoff = {
        "datasets": [
            {"profile": "pilot", "spec_sha256": "frozen", "path": "old"},
            {"profile": "pilot", "spec_sha256": "new", "path": "new"},
        ]
    }
    assert select_handoff_dataset(handoff, profile="pilot", spec_sha256="frozen")["path"] == "old"
    with pytest.raises(ValueError, match="Expected one dataset"):
        select_handoff_dataset(handoff, profile="pilot")
    with pytest.raises(ValueError, match="Expected one dataset"):
        select_handoff_dataset(handoff, profile="pilot", spec_sha256="missing")


def test_handoff_requires_verified_or_explicitly_accepted_source_hashes(tmp_path):
    root = tmp_path / "dataset"
    (root / "audit").mkdir(parents=True)
    (root / "spec.json").write_text("{}")
    (root / "train.jsonl").write_text("")
    audit_path = root / "audit" / "validation.json"
    signal = tmp_path / "ready.json"
    signal.write_text(json.dumps({
        "status": "complete", "tests": {"failed": 0},
        "datasets": [{"path": str(root), "manifest": str(root / "train.jsonl"),
                      "spec_sha256": file_sha256(root / "spec.json"),
                      "manifest_sha256": file_sha256(root / "train.jsonl")}],
    }))
    audit_path.write_text(json.dumps({"status": "passed", "source_hashes_verified": False}))
    with pytest.raises(ValueError, match="audit is not complete"):
        read_handoff(signal)
    audit_path.write_text(json.dumps({"status": "passed", "source_hashes_verified": False,
                                      "source_hashes_accepted_by_user": True}))
    assert read_handoff(signal)["status"] == "complete"


def test_moved_handoff_keeps_spec_hash_checks_and_records_runtime_paths(tmp_path, monkeypatch):
    original = tmp_path / "old_project"
    current = tmp_path / "current_project"
    root = current / "data" / "dataset"
    (root / "audit").mkdir(parents=True)
    (current / "data_pipeline").mkdir()
    (current / "data" / "source").mkdir()
    (root / "spec.json").write_text("{}")
    (root / "train.jsonl").write_text("")
    (root / "audit/validation.json").write_text(json.dumps(
        {"status": "passed", "source_hashes_accepted_by_user": True}))
    signal = current / "ready.json"
    value = {"status": "complete", "tests": {"failed": 0},
             "code_root": str(original / "data_pipeline"), "source_root": str(original / "data/source"),
             "datasets": [{"path": str(original / "data/dataset"),
                           "manifest": str(original / "data/dataset/train.jsonl"),
                           "spec_sha256": file_sha256(root / "spec.json"),
                           "manifest_sha256": file_sha256(root / "train.jsonl")}]}
    signal.write_text(json.dumps(value))
    monkeypatch.setattr(data, "PROJECT_ROOT", current)
    moved = read_handoff(signal)
    assert moved["datasets"][0]["path"] == str(root)
    assert moved["source_root"] == str(current / "data/source")
    assert moved["runtime_path_overrides"][str(original / "data/dataset")] == str(root)
    assert json.loads(signal.read_text()) == value
    (root / "spec.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Handoff hash mismatch"):
        read_handoff(signal)
