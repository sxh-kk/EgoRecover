"""Batch detached eval roll-ins without changing logical training microbatches."""

from dataclasses import dataclass

import torch

from egorecover.data import TrainingBatch
from egorecover.state import WindowCache, state_map


@dataclass(frozen=True)
class RollinChunk:
    batch: TrainingBatch
    frame: int
    global_offset: int
    rollin_frames: int


def packed_rollin_caches(system, model, chunks, *, seed):
    """Return caches in the original logical chunk order.

    Only chunks with the same target frame and history length share an eval
    call. Query sampling, tau, endpoint dropout seeds and the gradient-bearing
    training calls remain owned by the unchanged logical microbatch loop.
    Separate noise generation per chunk retains even non-contiguous logical
    offsets when several chunks share a history length.
    """
    groups = {}
    for index, chunk in enumerate(chunks):
        groups.setdefault((chunk.frame, chunk.rollin_frames), []).append((index, chunk))
    result = [None] * len(chunks)
    model.eval()
    for (frame, rollin_frames), group in groups.items():
        batch = TrainingBatch(tuple(episode for _, chunk in group for episode in chunk.batch.episodes))
        first = max(20, frame - rollin_frames)
        cache = batch.cache(first)
        for t in range(first, frame):
            epsilon = torch.cat(
                [
                    chunk.batch.noise(seed, t, system.config.window, tag="rollin", offset=chunk.global_offset)
                    for _, chunk in group
                ]
            )
            cache = system.online(model, cache, batch.observations(t), epsilon).cache.detach()
        start = 0
        for index, chunk in group:
            end = start + len(chunk.batch)
            result[index] = WindowCache(
                state_map(cache.body, lambda value: value[start:end]),
                cache.beta[start:end],
                cache.anchor[start:end],
                cache.frame,
                cache.generated_frames,
            )
            start = end
    return result
