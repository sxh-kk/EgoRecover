"""Wait for a verified dataset handoff; emits a completion message and exits 0."""

import argparse
import json
from pathlib import Path
import time


def ready(signal):
    if not signal.is_file():
        return None
    value = json.loads(signal.read_text())
    if value.get("status") != "complete":
        return None
    for artifact in value["datasets"]:
        root = Path(artifact["path"])
        report = json.loads((root / "audit" / "validation.json").read_text())
        if report.get("status") != "passed" or not report.get("source_hashes_verified"):
            raise ValueError(f"Dataset has not passed full audit: {root}")
        if not (root / "spec.json").is_file():
            raise FileNotFoundError(root / "spec.json")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "EE4D_MISMATCH_READY.json")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.timeout < 0 or args.poll_seconds <= 0:
        parser.error("timeout must be nonnegative and poll-seconds positive")
    deadline = time.monotonic() + args.timeout
    print(f"Waiting for {args.signal}", flush=True)
    while True:
        value = ready(args.signal)
        if value is not None:
            print("数据集构建完成", flush=True)
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
            return
        if time.monotonic() >= deadline:
            raise SystemExit("Dataset is not ready; timed out without emitting a completion signal")
        time.sleep(min(args.poll_seconds, max(0, deadline-time.monotonic())))


if __name__ == "__main__":
    main()
