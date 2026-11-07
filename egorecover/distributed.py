"""Small torchrun/DDP context with explicit rank ownership and collectives."""

import math
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _environment_integer(name, default, minimum):
    text = os.environ.get(name, str(default))
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer >= {minimum}.") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    _owns_group: bool = False
    _closed: bool = False

    @classmethod
    def from_environment(cls, device, *, timeout_seconds=600):
        """Initialize torchrun's group only for WORLD_SIZE > 1.

        CUDA workers use LOCAL_RANK and NCCL; CPU workers use Gloo. An already
        initialized matching group is borrowed and is never destroyed here.
        The timeout applies to group initialization and collective operations.
        """
        world_size = _environment_integer("WORLD_SIZE", 1, 1)
        rank = _environment_integer("RANK", 0, 0)
        local_rank = _environment_integer("LOCAL_RANK", 0, 0)
        if rank >= world_size:
            raise ValueError("RANK must be smaller than WORLD_SIZE.")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Distributed timeout_seconds must be finite and positive.")
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Distributed training supports CPU/Gloo or CUDA/NCCL.")
        if not dist.is_available():
            if world_size > 1:
                raise RuntimeError("This PyTorch build does not provide distributed support.")
            return cls(rank, world_size, local_rank, target)
        initialized = dist.is_initialized()
        if initialized and (dist.get_rank() != rank or dist.get_world_size() != world_size):
            raise ValueError("The existing process group disagrees with torchrun rank/world-size environment.")
        if world_size == 1:
            return cls(rank, world_size, local_rank, target)
        backend = "nccl" if target.type == "cuda" else "gloo"
        if target.type == "cuda":
            torch.cuda.set_device(local_rank)
            target = torch.device("cuda", local_rank)
        if initialized:
            if dist.get_backend() != backend:
                raise ValueError("The existing process group backend disagrees with the selected device.")
        else:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )
        return cls(rank, world_size, local_rank, target, _owns_group=not initialized)

    @property
    def is_primary(self):
        return self.rank == 0

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The distributed context is closed.")
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError("The distributed process group is not initialized.")

    def wrap(self, model):
        self._check_open()
        if self.world_size == 1 or isinstance(model, DistributedDataParallel):
            return model
        options = {"broadcast_buffers": False}
        if self.device.type == "cuda":
            options.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(model, **options)

    @staticmethod
    def unwrap(model):
        return model.module if isinstance(model, DistributedDataParallel) else model

    def barrier(self):
        self._check_open()
        if self.world_size > 1:
            dist.barrier()

    def all_gather_object(self, value):
        self._check_open()
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def gather_rows(self, rows):
        """Return rank-ordered, flattened rows on every worker."""
        return [row for partition in self.all_gather_object(rows) for row in partition]

    def sync_scalar_dict(self, data):
        """Mean scalars across workers; every worker must supply the same keys."""
        self._check_open()
        keys = sorted(data)
        values = [float(data[key]) for key in keys]
        if self.world_size > 1 and keys:
            tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            values = (tensor / self.world_size).cpu().tolist()
        return dict(zip(keys, values))

    def close(self):
        if self._closed:
            return
        if self._owns_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
