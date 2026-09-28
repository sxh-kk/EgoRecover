"""Verify the explicit local handoff before opening the independent dataset."""

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL = PROJECT_ROOT / "data/EE4D_MISMATCH_READY.json"


def relocate_handoff(value):
    """Resolve missing paths after moving a checkout, without rewriting data identities."""
    if not value.get("code_root"):
        return value
    recorded_project = Path(value["code_root"]).parent
    overrides = {}

    def resolve(raw):
        path = Path(raw)
        if path.exists():
            return raw
        try:
            relative = path.relative_to(recorded_project)
        except ValueError:
            return raw
        candidate = PROJECT_ROOT / relative
        if not candidate.exists():
            return raw
        overrides[raw] = str(candidate)
        return str(candidate)

    resolved = dict(value)
    for key in ("code_root", "source_root"):
        if key in resolved:
            resolved[key] = resolve(resolved[key])
    resolved["datasets"] = [
        {key: resolve(item) if key in ("path", "manifest", "audit_report") else item
         for key, item in entry.items()} for entry in value["datasets"]
    ]
    if overrides:
        resolved["runtime_path_overrides"] = overrides
    return resolved


def read_handoff(signal=DEFAULT_SIGNAL):
    signal = Path(signal)
    value = json.loads(signal.read_text())
    if value.get("status") != "complete" or not value.get("datasets"):
        raise ValueError("Dataset handoff is not complete.")
    if value.get("tests", {}).get("failed") != 0:
        raise ValueError("Dataset handoff does not declare passing tests.")
    value = relocate_handoff(value)
    for item in value["datasets"]:
        root = Path(item["path"])
        audit = json.loads((root / "audit/validation.json").read_text())
        source_accepted = (audit.get("source_hashes_verified") is True or
                           audit.get("source_hashes_accepted_by_user") is True)
        if audit.get("status") != "passed" or not source_accepted:
            raise ValueError(f"Dataset audit is not complete: {root}")
        for file, key in ((root / "spec.json", "spec_sha256"), (Path(item["manifest"]), "manifest_sha256")):
            if hashlib.sha256(file.read_bytes()).hexdigest() != item[key]:
                raise ValueError(f"Handoff hash mismatch: {file}")
    return value


def select_handoff_dataset(handoff, *, profile, spec_sha256=None):
    """Select an exact dataset identity when one handoff has multiple profiles."""
    entries = [entry for entry in handoff["datasets"] if entry["profile"] == profile]
    if spec_sha256 is not None:
        entries = [entry for entry in entries if entry["spec_sha256"] == spec_sha256]
    if len(entries) != 1:
        raise ValueError(f"Expected one dataset for profile {profile!r} and spec {spec_sha256!r}.")
    return entries[0]


def open_dataset(*, profile="pilot", signal=DEFAULT_SIGNAL, spec_sha256=None):
    handoff = read_handoff(signal)
    if profile == "pilot" and spec_sha256 is None:
        manifest = Path(__file__).resolve().parents[1] / "config/egorecover_pilot_split_v1.json"
        if manifest.is_file():
            spec_sha256 = json.loads(manifest.read_text())["dataset_spec_sha256"]
    entry = select_handoff_dataset(handoff, profile=profile, spec_sha256=spec_sha256)
    # The dataset producer is a separate sibling package, not part of UEM.
    project = Path(handoff["code_root"]).parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from data_pipeline.ee4d_mismatch.dataset import MismatchDataset

    source = handoff.get("source_root")
    if source:
        spec = json.loads((Path(entry["path"]) / "spec.json").read_text())
        for relative, expected in spec.get("source_files", {}).items():
            if (Path(source) / relative).stat().st_size != expected["bytes"]:
                raise ValueError(f"Source file size differs from the audited spec: {relative}")
    return MismatchDataset(entry["path"], data_root=source), handoff
