"""Journal-owned process cleanup without signalling unrelated GPU jobs."""

import contextlib
import ctypes
import hashlib
import json
import os
import platform
import signal
import time
import uuid
from pathlib import Path

FORMAT = "egorecover_owned_process_v1"
ROOT_ENV = "EGORECOVER_PROCESS_RUNTIME"
TOKEN_ENV = "EGORECOVER_PROCESS_TOKEN"
_signal_depth = 0
_shutdown_signal = None


class SupervisorInterrupted(SystemExit):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(128 + signum)


@contextlib.contextmanager
def termination_signals():
    """Turn TERM/INT into normal stack unwinding; repeated signals allow cleanup."""
    global _signal_depth, _shutdown_signal
    if not _signal_depth:
        _shutdown_signal = None
    _signal_depth += 1
    previous = {}

    def stop(signum, frame):
        global _shutdown_signal
        if _shutdown_signal is None:
            _shutdown_signal = signum
            raise SupervisorInterrupted(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _signal_depth -= 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_info(pid):
    """Return only ownership fields, never another process's environment."""
    root = Path("/proc") / str(pid)
    try:
        text = (root / "stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        command = (root / "cmdline").read_bytes()
        environment = {}
        for item in (root / "environ").read_bytes().split(b"\0"):
            key, _, value = item.partition(b"=")
            if key in (ROOT_ENV.encode(), TOKEN_ENV.encode()):
                environment[key.decode()] = value.decode()
        return {
            "pid": int(pid),
            "start_ticks": int(fields[19]),
            "pgid": int(fields[2]),
            "session_id": int(fields[3]),
            "cmdline_sha256": hashlib.sha256(command).hexdigest(),
            "runtime": environment.get(ROOT_ENV),
            "token": environment.get(TOKEN_ENV),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError, OSError):
        return None


def runtime_root(plan):
    return Path(plan["pair_managed"]).resolve().parent if plan.get("pair_managed") else Path(plan["output"]).resolve()


def journal_job(plan, spec):
    """Publish the nonce before Popen, including the spawn/record crash window."""
    token = uuid.uuid4().hex
    path = Path(plan["output"]) / "process-journal" / (token + ".json")
    record = {
        "format": FORMAT,
        "token": token,
        "runtime": str(runtime_root(plan)),
        "scope": str(Path(plan["output"]).resolve()),
        "name": spec["name"],
        "directory": str(Path(spec["directory"]).resolve()),
        "argv": list(map(str, spec["argv"])),
        "gpus": spec["gpus"],
        "created_ns": time.time_ns(),
        "leader": None,
        "state": "spawning",
    }
    atomic_json(path, record)
    return path, record


def bind_job(path, record, process):
    info = process_info(process.pid)
    if info is not None:
        if info["runtime"] != record["runtime"] or info["token"] != record["token"]:
            raise RuntimeError("The spawned process does not carry its journal ownership identity.")
        if info["pgid"] != process.pid or info["session_id"] != process.pid:
            raise RuntimeError("A managed job must own its separate process group and session.")
    record.update(pid=process.pid, leader=info, state="running", started_ns=time.time_ns())
    atomic_json(path, record)
    return record


def finish_job(path, record, exit_code):
    record.update(state="exited", exit_code=exit_code, finished_ns=time.time_ns())
    atomic_json(path, record)
    return record


def _journal_records(plan):
    scope = Path(plan["output"]).resolve()
    expected_runtime = str(runtime_root(plan))
    records = {}
    for path in scope.rglob("process-journal/*.json"):
        if path.is_symlink():
            continue
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            continue
        if record.get("format") != FORMAT or record.get("runtime") != expected_runtime:
            continue
        token = record.get("token")
        if not isinstance(token, str) or len(token) != 32 or path.stem != token:
            continue
        if any(character not in "0123456789abcdef" for character in token):
            continue
        if Path(record.get("scope", "")).resolve() != path.parent.parent.resolve():
            continue
        records[token] = (path, record)
    return records


def owned_processes(plan):
    """Find journal nonce inheritors, including orphaned ranks after SIGKILL."""
    records = _journal_records(plan)
    expected_runtime = str(runtime_root(plan))
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = process_info(int(entry.name))
        if not info or info["runtime"] != expected_runtime or info["token"] not in records:
            continue
        path, record = records[info["token"]]
        if info["pid"] == os.getpid():
            raise RuntimeError("Refusing to treat this supervisor as its own child.")
        leader = record.get("leader")
        if leader and info["pid"] == leader["pid"]:
            if any(info[key] != leader[key] for key in ("start_ticks", "pgid", "session_id", "cmdline_sha256")):
                raise RuntimeError("A journal leader changed process identity; refusing to signal it.")
        # Bind even a child forked between Popen and leader publication to its
        # nonce and current PID/start/argv/group before any signal is allowed.
        found.append({**info, "journal": str(path)})
    return sorted(found, key=lambda row: (-len(Path(row["journal"]).parts), row["pid"]))


def _signal_owned(row, signum):
    """pidfd prevents a verified PID being recycled between check and signal."""
    try:
        descriptor = _pidfd_open(row["pid"])
    except ProcessLookupError:
        return False
    try:
        actual = process_info(row["pid"])
        if actual is None:
            return False
        keys = ("pid", "start_ticks", "pgid", "session_id", "cmdline_sha256", "runtime", "token")
        if any(actual[key] != row[key] for key in keys):
            raise RuntimeError("Owned process changed identity before signal; refusing cleanup.")
        _pidfd_send_signal(descriptor, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def _pidfd_syscall(number, *arguments):
    # This conda Python/glibc predates the wrappers, while the actual5.4 kernel
    # supports pidfds. These two architectures share the Linux syscall numbers.
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Safe cleanup requires Linux pidfd support; no unverified signals were sent.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return result


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _pidfd_send_signal(descriptor, signum):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(descriptor, signum)
    return _pidfd_syscall(424, ctypes.c_int(descriptor), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0))


def cleanup_owned_jobs(plan, *, reason="startup_recovery", grace_seconds=10):
    """Terminate only this scope's nonce-bound jobs, before waiting for GPUs."""
    initial = owned_processes(plan)
    if not initial:
        return {"reason": reason, "owned_processes": [], "signals": [], "remaining": []}
    output = Path(plan["output"])
    path = output / f"process-recovery-{time.time_ns()}.json"
    report = {"reason": reason, "owned_processes": initial, "signals": [], "remaining": initial}
    # Persist the exact PID/start/argv/group identities before sending signals.
    atomic_json(path, report)
    deadline = time.monotonic() + grace_seconds
    sent = set()
    while True:
        remaining = owned_processes(plan)
        if not remaining:
            break
        signum = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
        for row in remaining:
            identity = (row["pid"], row["start_ticks"], signum)
            if identity not in sent and _signal_owned(row, signum):
                sent.add(identity)
                report["signals"].append({"pid": row["pid"], "start_ticks": row["start_ticks"], "signal": signum})
        report["remaining"] = remaining
        atomic_json(path, report)
        if time.monotonic() > deadline + 10:
            raise RuntimeError("Owned jobs did not exit after verified cleanup; GPU launch remains blocked.")
        time.sleep(0.05)
    report["remaining"] = []
    report["finished_ns"] = time.time_ns()
    atomic_json(path, report)
    return report
