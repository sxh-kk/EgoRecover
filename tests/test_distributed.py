import json
import os
import socket
import subprocess
import sys

import pytest
import torch

from egorecover.distributed import DistributedContext


def test_single_cpu_is_a_noop_and_does_not_initialize_cuda_or_a_process_group(monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("A standalone CPU context must not initialize CUDA or a process group")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    model = torch.nn.Linear(1, 1)
    before = torch.get_rng_state().clone()
    context = DistributedContext.from_environment("cpu")
    assert (context.rank, context.world_size, context.local_rank, context.is_primary) == (0, 1, 0, True)
    assert context.device == torch.device("cpu")
    assert context.wrap(model) is model and context.unwrap(model) is model
    value = {"rng": before}
    assert context.all_gather_object(value)[0] is value
    assert context.gather_rows([{"take": "one"}]) == [{"take": "one"}]
    assert context.sync_scalar_dict({"loss": 3.0, "count": 2}) == {"loss": 3.0, "count": 2.0}
    context.barrier()
    context.close()
    context.close()
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="closed"):
        context.barrier()


@pytest.mark.parametrize(
    "name,value", (("WORLD_SIZE", "0"), ("WORLD_SIZE", "bad"), ("RANK", "1"), ("LOCAL_RANK", "-1"))
)
def test_invalid_torchrun_environment_is_rejected_before_initialization(monkeypatch, name, value):
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        DistributedContext.from_environment("cpu")


def test_two_real_gloo_workers_average_gradients_and_gather_rank_rng_without_leaking_processes():
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This PyTorch build does not provide the Gloo CPU backend")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    script = """
import json
import torch
from egorecover.distributed import DistributedContext

torch.set_num_threads(1)
before = torch.get_rng_state().clone()
context = DistributedContext.from_environment('cpu', timeout_seconds=20)
try:
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.distributed.get_backend() == 'gloo'
    assert context.device.type == 'cpu'
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    wrapped = context.wrap(model)
    assert wrapped.broadcast_buffers is False
    assert context.unwrap(wrapped) is model
    wrapped(torch.tensor([[float(context.rank + 1)]])).sum().backward()
    gradient = model.weight.grad.item()
    assert abs(gradient - 1.5) < 1e-7
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    optimizer.step()
    assert abs(model.weight.item() - 0.85) < 1e-6
    averaged = context.sync_scalar_dict({'loss': context.rank + 1, 'aux': context.rank + 3})
    assert averaged == {'loss': 1.5, 'aux': 3.5}
    torch.manual_seed(100 + context.rank)
    states = context.all_gather_object({'rank': context.rank, 'rng': torch.get_rng_state()})
    assert [entry['rank'] for entry in states] == [0, 1]
    for rank, entry in enumerate(states):
        expected = torch.Generator().manual_seed(100 + rank).get_state()
        assert torch.equal(entry['rng'], expected)
    rows = context.gather_rows([{'rank': context.rank, 'row': row} for row in (0, 1)])
    assert rows == [{'rank': 0, 'row': 0}, {'rank': 0, 'row': 1}, {'rank': 1, 'row': 0}, {'rank': 1, 'row': 1}]
    borrowed = DistributedContext.from_environment('cpu', timeout_seconds=20)
    borrowed.close()
    assert torch.distributed.is_initialized()
    context.barrier()
    result = {'rank': context.rank, 'world_size': context.world_size, 'gradient': gradient, 'gathered_rows': len(rows)}
finally:
    context.close()
assert not torch.distributed.is_initialized()
result['closed'] = True
print(json.dumps(result), flush=True)
"""
    processes = []
    try:
        for rank in range(2):
            environment = os.environ.copy()
            environment.update(
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                WORLD_SIZE="2",
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        results = []
        for process in processes:
            try:
                stdout, stderr = process.communicate(timeout=45)
            except subprocess.TimeoutExpired:
                pytest.fail("Two-worker Gloo integration exceeded its 45-second process timeout")
            assert process.returncode == 0, f"Gloo worker failed:\n{stdout}\n{stderr}"
            results.append(json.loads(next(line for line in reversed(stdout.splitlines()) if line.startswith("{"))))
        assert [entry["rank"] for entry in results] == [0, 1]
        assert all(entry["gradient"] == 1.5 and entry["gathered_rows"] == 4 and entry["closed"] for entry in results)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
