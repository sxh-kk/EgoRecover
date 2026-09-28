"""Explicit calibration from the legal clean startup prefix only."""

import torch


def estimate_bootstrap_floor(observations, codec, frames=20):
    """Heuristic: median startup head z minus the training mean head height.

    This is a deployment-available estimate, not the scene's annotated floor.
    Its accuracy must be evaluated separately. No body labels are accepted.
    """
    if len(observations["aria_traj_obs"]) < frames or not bool(observations["traj_available"][:frames].all()):
        raise ValueError("Floor initialization requires the complete available startup prefix.")
    heights = observations["aria_traj_obs"][:frames, 8]
    if not bool(torch.isfinite(heights).all()):
        raise ValueError("Nonfinite startup head heights.")
    return float(heights.median() - codec.traj_mean[8].to(heights))
