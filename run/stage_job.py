"""Run one queued experiment and persist its exit status independently of the queue."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from egorecover.evaluation_resume import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    status = {"command": spec["command"], "gpu": spec["gpu"], "pid": os.getpid(),
              "started_at": now(), "exit_code": None}
    path = args.spec.with_suffix(".status.json")
    atomic_json(path, status)
    try:
        # Only the queue's selected card is visible in this process.
        probe = subprocess.run([sys.executable, "-c",
                                "import torch; x=torch.ones(2,device='cuda'); "
                                "assert x.sum().item()==2; torch.cuda.synchronize()"], check=False)
        if probe.returncode:
            status["exit_code"] = probe.returncode
            status["error"] = "CUDA preflight failed"
        else:
            child = subprocess.Popen(spec["command"])
            status["child_pid"] = child.pid
            atomic_json(path, status)
            status["exit_code"] = child.wait()
    except Exception as error:
        status["exit_code"] = 1
        status["error"] = repr(error)
    finally:
        status["ended_at"] = now()
        atomic_json(path, status)
    raise SystemExit(status["exit_code"])


if __name__ == "__main__":
    main()
