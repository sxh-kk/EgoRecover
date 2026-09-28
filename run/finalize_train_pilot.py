"""Freeze an audited official-train development split and handoff."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from data_pipeline.ee4d_mismatch.source import file_sha256, write_json
from egorecover.data import read_handoff
from egorecover.evaluation_protocol import load_fixed_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-out", type=Path, required=True)
    parser.add_argument("--handoff-out", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--split-counts", type=int, nargs=3, metavar=("TRAIN", "DEV", "HOLDOUT"),
                        default=(8, 2, 2))
    parser.add_argument("--accept-source-hashes", action="store_true",
                        help="Record explicit user acceptance when source hashes were not rechecked")
    args = parser.parse_args()
    if (args.tests_passed < 1 or any(count < 1 for count in args.split_counts) or
            args.split_out.exists() or args.handoff_out.exists()):
        parser.error("Tests and split counts must be positive, and split/handoff outputs must be new files.")
    root = args.dataset.resolve()
    spec = json.loads((root / "spec.json").read_text())
    audit = json.loads((root / "audit/validation.json").read_text())
    total = sum(args.split_counts)
    if ((spec["base_split"], spec["purpose"], spec["profile"]) != ("train", "development", "pilot") or
            spec["num_episodes"] != total or spec["num_variants"] != total * 7 or
            audit.get("status") != "passed"):
        raise ValueError(f"Expected an audited {total}-take official-train pilot with {total * 7} variants.")
    if audit.get("source_hashes_verified") is not True:
        if not args.accept_source_hashes:
            raise ValueError("Source hashes were not rechecked; explicit acceptance is required.")
        audit["source_hashes_accepted_by_user"] = True
    split_map = json.loads((root / "split_manifest.json").read_text())["take_to_split"]
    splits = {name: sorted(take for take, group in split_map.items() if group == name)
              for name in ("train", "dev", "holdout")}
    if [len(splits[name]) for name in ("train", "dev", "holdout")] != list(args.split_counts):
        raise ValueError(f"Expected a {'/'.join(map(str, args.split_counts))} take-disjoint split.")
    if audit.get("source_hashes_accepted_by_user") is True:
        write_json(root / "audit/validation.json", audit)
    manifest = {
        "schema_version": "egorecover-fixed-split-v1",
        "profile": "pilot",
        "base_split": "train",
        "dataset_spec_sha256": file_sha256(root / "spec.json"),
        "source": f"Frozen from the audited {total}-take official-train pilot; no reseeding during training.",
        "splits": splits,
    }
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.split_out, manifest)
    handoff = {
        "status": "complete",
        "message": f"Audited official-train {total}-take P/G development dataset; official val remains unused.",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "code_root": str((Path(__file__).resolve().parents[1] / "data_pipeline").resolve()),
        "source_root": spec["source_root"],
        "datasets": [{
            "path": str(root), "profile": "pilot", "scope": "sampled_takes",
            "purpose": "development", "base_split": "train", "num_takes": total,
            "num_sequences": total, "num_variants": total * 7, "physical_frames": total * spec["frames"],
            "manifest": str(root / spec["manifests"][0]),
            "audit_report": str(root / "audit/validation.json"),
            "spec_sha256": manifest["dataset_spec_sha256"],
            "manifest_sha256": file_sha256(root / spec["manifests"][0]),
        }],
        "tests": {"passed": args.tests_passed, "failed": 0,
                  "scope": "data_pipeline and EgoRecover local test suites"},
        "usage": "Official train development pilot only; holdout is withheld until settings are frozen.",
    }
    args.handoff_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.handoff_out, handoff)
    verified = read_handoff(args.handoff_out)
    records = [json.loads(line) for name in spec["manifests"]
               for line in (root / name).read_text().splitlines() if line]
    dataset = SimpleNamespace(root=root, spec=spec, records=records)
    if load_fixed_split(args.split_out, dataset) != splits:
        raise ValueError("Frozen split verification failed.")
    print(json.dumps({"handoff": str(args.handoff_out), "split": str(args.split_out),
                      "splits": splits}, indent=2))


if __name__ == "__main__":
    main()
