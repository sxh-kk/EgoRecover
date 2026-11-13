"""CPU process-lifecycle checks; no production runtime or CUDA use."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import formal_prior_processes as managed

REPO = Path(__file__).resolve().parents[1]


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def await_file(path, process):
    deadline = time.monotonic() + 8
    while not path.exists():
        if process.poll() is not None:
            raise AssertionError(process.communicate()[0].decode())
        if time.monotonic() > deadline:
            raise AssertionError("CPU child failed to publish its identity")
        time.sleep(0.02)


def controller(tmp_path, *, nested=False):
    root = tmp_path / "runtime"
    root.mkdir()
    pid_file = root / "worker.pid"
    code = (
        f"import os,time;from pathlib import Path;Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    plan = {
        "output": str(root),
        "source_snapshot": str(REPO),
        "smpl_directory": str(REPO / "body_models/smplx"),
        "minimum_free_GiB": 0,
    }
    spec = {"name": "CPU-worker", "directory": str(root / "worker"), "gpus": "", "argv": [sys.executable, "-c", code]}
    if nested:
        child = {**plan, "output": str(root / "A"), "pair_managed": str(root / "plan.json")}
        child_code = (
            f"from scripts.run_formal_prior import run_processes;run_processes({child!r},[{spec!r}],'nested-worker')"
        )
        spec = {
            "name": "CPU-nested-controller",
            "directory": str(root / "A"),
            "gpus": "",
            "argv": [sys.executable, "-c", child_code],
        }
    launch = f"from scripts.run_formal_prior import run_processes;run_processes({plan!r},[{spec!r}],'outer-worker')"
    env = {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""}
    process = subprocess.Popen(
        [sys.executable, "-c", launch],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    await_file(pid_file, process)
    return plan, process, int(pid_file.read_text())


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_term_and_int_clean_up_owned_cpu_worker(tmp_path, signum):
    plan, process, worker = controller(tmp_path)
    try:
        process.send_signal(signum)
        process.wait(timeout=12)
        assert process.returncode == 128 + signum
        assert not alive(worker)
        assert not managed.owned_processes(plan)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        managed.cleanup_owned_jobs(plan, grace_seconds=0.2)


@pytest.mark.parametrize("nested", [False, True])
def test_sigkill_orphans_are_safely_cleaned_before_restart(tmp_path, nested):
    plan, process, worker = controller(tmp_path, nested=nested)
    try:
        process.kill()
        process.wait(timeout=3)
        assert alive(worker)
        rows = managed.owned_processes(plan)
        assert any(row["pid"] == worker for row in rows)
        if nested:
            assert len(rows) >= 2
        report = managed.cleanup_owned_jobs(plan, reason="test_restart_before_gpu_wait", grace_seconds=2)
        assert report["owned_processes"] and not report["remaining"]
        assert not alive(worker)
        assert not managed.owned_processes(plan)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        managed.cleanup_owned_jobs(plan, grace_seconds=0.2)


def test_unrelated_process_is_never_signalled_by_a_stale_pid_record(tmp_path):
    plan = {"output": str(tmp_path)}
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        token = "f" * 32
        record = {
            "format": managed.FORMAT,
            "token": token,
            "runtime": str(tmp_path.resolve()),
            "scope": str(tmp_path.resolve()),
            "leader": managed.process_info(unrelated.pid),
        }
        managed.atomic_json(tmp_path / "process-journal" / f"{token}.json", record)
        report = managed.cleanup_owned_jobs(plan, grace_seconds=0.1)
        assert report["owned_processes"] == []
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_mismatched_leader_start_is_refused_without_a_signal(tmp_path):
    plan, process, worker = controller(tmp_path)
    process.kill()
    process.wait(timeout=3)
    journal = next((Path(plan["output"]) / "process-journal").glob("*.json"))
    original = json.loads(journal.read_text())
    changed = json.loads(journal.read_text())
    changed["leader"]["start_ticks"] += 1
    managed.atomic_json(journal, changed)
    try:
        with pytest.raises(RuntimeError, match="changed process identity"):
            managed.cleanup_owned_jobs(plan, grace_seconds=0.1)
        assert alive(worker)
    finally:
        managed.atomic_json(journal, original)
        managed.cleanup_owned_jobs(plan, grace_seconds=0.2)
