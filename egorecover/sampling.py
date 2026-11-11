"""Replay a reference worker's CPU query/time stream across more workers."""

import math
from numbers import Integral, Real

import torch


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def reference_microbatch_plan(
    data,
    generator,
    *,
    rank,
    world_size,
    reference_world_size,
    batch_size,
    microbatch_size,
    rollin_frames,
    endpoint_probability,
    horizon=2,
):
    """Replay all reference chunks, retaining this worker's contiguous slice.

    The caller seeds each worker's CPU generator with ``seed + 1 + rank //
    (world_size // reference_world_size)``. Every worker consumes its entire
    reference batch: query draws, then tau draws, then endpoint draws, once per
    reference microbatch. Only retained query descriptors are returned; this
    helper never loads episodes. Paired physical workers therefore have the
    same sample RNG state at step boundaries while owning disjoint queries.
    """
    rank = _integer(rank, "rank", 0)
    world_size = _integer(world_size, "world_size")
    reference_world_size = _integer(reference_world_size, "reference_world_size")
    batch_size = _integer(batch_size, "batch_size")
    microbatch_size = _integer(microbatch_size, "microbatch_size")
    rollin_frames = _integer(rollin_frames, "rollin_frames", 0)
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size.")
    if world_size % reference_world_size:
        raise ValueError("world_size must be divisible by reference_world_size.")
    if batch_size % world_size or batch_size % reference_world_size:
        raise ValueError("batch_size must be divisible by both worker counts.")
    local_batch = batch_size // world_size
    reference_batch = batch_size // reference_world_size
    if local_batch % microbatch_size or reference_batch % microbatch_size:
        raise ValueError("Both local and reference batches must be divisible by microbatch_size.")
    if (
        isinstance(endpoint_probability, bool)
        or not isinstance(endpoint_probability, Real)
        or not math.isfinite(endpoint_probability)
        or not 0 <= endpoint_probability <= 1
    ):
        raise ValueError("endpoint_probability must be finite and within [0, 1].")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Reference sampling requires a CPU torch.Generator.")
    workers_per_reference = world_size // reference_world_size
    reference_rank = rank // workers_per_reference
    first = (rank % workers_per_reference) * local_batch
    last = first + local_batch
    plan = []
    for logical_offset in range(0, reference_batch, microbatch_size):
        kwargs = {} if horizon == 2 else {"horizon": horizon}
        queries, frame = data.sample_queries(generator, microbatch_size, rollin_frames, **kwargs)
        if len(queries) != microbatch_size:
            raise ValueError("sample_queries must return one query per requested sample.")
        tau = 0.001 + 0.999 * torch.rand(microbatch_size, generator=generator, device="cpu")
        tau[torch.rand(microbatch_size, generator=generator, device="cpu") < endpoint_probability] = 1.0
        if first <= logical_offset < last:
            plan.append(
                {
                    "offset": logical_offset - first,
                    "global_offset": reference_rank * reference_batch + logical_offset,
                    "queries": queries,
                    "frame": frame,
                    "tau": tau,
                    "count": microbatch_size,
                }
            )
    return plan
