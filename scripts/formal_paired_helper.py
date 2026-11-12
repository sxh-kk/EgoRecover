"""Clip-equal contrasts with a paired take-cluster bootstrap.

This preserves the fixed-400 collector's weighting, 2000 draws and seed1062.
Repeated clips from the same take stay together in every bootstrap sample.
"""

from collections import defaultdict


def paired_cluster(values):
    import numpy as np

    grouped = defaultdict(list)
    for take, value in values:
        grouped[take].append(value)
    takes = sorted(grouped)
    sums = np.array([sum(grouped[take]) for take in takes])
    counts = np.array([len(grouped[take]) for take in takes])
    rng = np.random.default_rng(1062)
    indices = rng.integers(len(takes), size=(2000, len(takes)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "delta": float(sums.sum() / counts.sum()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "valid_clips": int(counts.sum()),
        "takes": len(takes),
    }
