"""Wait for the official validation download, then build and audit both pilot sets.

This creates verified datasets; the final cross-dialogue readiness signal is
published separately after tests and the resulting artifacts have been reviewed.
"""

import json
from pathlib import Path
import time

import torch

from .ee4d_mismatch.audit import audit_dataset
from .ee4d_mismatch.build import build_dataset
from .ee4d_mismatch.preview import preview
from .ee4d_mismatch.source import EE4DSource, write_json


def main():
    torch.set_num_threads(2)
    data = Path(__file__).resolve().parents[1] / "data"
    status_path = data / ".ee4d_validation_download" / "status.json"
    deadline = time.monotonic() + 7200
    while True:
        status = json.loads(status_path.read_text())
        if status["phase"] == "complete":
            break
        if status["phase"] == "error":
            raise RuntimeError(f"Download failed: {status.get('error')}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Download did not finish within two hours")
        print(f"Waiting: phase={status['phase']} progress={status.get('progress', 0):.1%}", flush=True)
        time.sleep(15)
    source = EE4DSource(data / "ee4d_motion_uniegomotion")
    _, inventory = source.inventory()
    print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
    first_feature = next(iter(source.features.values()))
    print(f"Actual feature shape: {tuple(first_feature.shape)}; dtype: {first_feature.dtype}", flush=True)
    fingerprints = source.fingerprints()
    for file in status["verified_files"]:
        relative = str(Path(file["path"]).relative_to(source.root))
        if fingerprints[relative] != {"bytes": file["bytes"], "sha256": file["sha256"]}:
            raise ValueError(f"Source does not match verified download: {relative}")
    results = []
    for name, takes, profile in (("ee4d_mismatch_smoke_v0", 2, "extended"),
                                 ("ee4d_mismatch_pilot_v0", 12, "pilot")):
        output = data / name
        if output.exists():
            spec = json.loads((output / "spec.json").read_text())
            if (spec["num_episodes"], spec["profile"], spec["seed"], spec["frames"], spec["purpose"]) != (
                    takes, profile, 20260923, 200, "engineering"):
                raise ValueError(f"Existing dataset has a different configuration: {output}")
        else:
            build_dataset(source, output, num_takes=takes, profile=profile)
        report = audit_dataset(output, source=source, verify_source=True)
        write_json(output / "audit" / "validation.json", report)
        preview(output, output / "audit" / "input_changes.png", source=source)
        results.append(dict(path=str(output), profile=profile, report=report))
        print(f"FULL AUDIT PASSED: {output}", flush=True)
    write_json(data / "ee4d_mismatch_build_results.json", dict(download_completed_at=status["completed_at"], datasets=results))
    print("Both real-data datasets passed full audit. Ready for final handoff review.", flush=True)


if __name__ == "__main__":
    main()
