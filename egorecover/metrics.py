"""Explicit EE4D reconstruction metrics in a common SMPL22 world frame.

Position inputs are in meters. Similarity alignment is evaluation-only and
never changes the emitted world-frame prediction or the model's cache.
"""

import torch

METRIC_DIRECTIONS = {
    "mpjpe_world_mm": "lower",
    "pa_mpjpe_mm": "lower",
    "seq_sim_mpjpe_mm": "lower",
    "head_joint_pe_mm": "lower",
    "wrist_pe_mm": "lower",
    "mpjre_f": "lower",
    "head_rot_f": "lower",
    "root_mean_initial_translation_mm": "lower",
    "root_final_initial_translation_mm": "lower",
    "velocity_error_mm_s": "lower",
    "acceleration_error_mm_s2": "lower",
    "foot_slide_uniegomotion_mm": "lower",
    "gt_foot_slide_uniegomotion_mm": "diagnostic",
    "diagnostic_jerk_peak_mm_s3": "diagnostic",
    "gt_diagnostic_jerk_peak_mm_s3": "diagnostic",
    "diagnostic_jerk_peak_gap_mm_s3": "lower",
    "diagnostic_jerk_auc_mm_s2": "diagnostic",
    "gt_diagnostic_jerk_auc_mm_s2": "diagnostic",
    "diagnostic_jerk_auc_gap_mm_s2": "lower",
}

METRIC_UNITS = {
    **{name: "mm" for name in METRIC_DIRECTIONS if name.endswith("_mm")},
    "mpjre_f": "dimensionless Frobenius norm",
    "head_rot_f": "dimensionless Frobenius norm",
    "velocity_error_mm_s": "mm/s",
    "acceleration_error_mm_s2": "mm/s^2",
    **{name: "mm/s^3" for name in METRIC_DIRECTIONS if name.endswith("_mm_s3")},
    **{name: "mm/s^2" for name in METRIC_DIRECTIONS if name.endswith("_mm_s2")},
}


def comparison_protocol():
    """Machine-readable definitions; shared names alone do not imply parity."""
    return {
        "version": "ee4d_smpl22_comparison_v1",
        "input_position_units": "meters",
        "coordinate_frame": "z-up world frame with floor at z=0",
        "joint_indices": list(range(22)),
        "head_joint": 15,
        "wrists": [20, 21],
        "scope": "current-frame first emissions after startup; no revised historical outputs",
        "units": METRIC_UNITS.copy(),
        "directions": METRIC_DIRECTIONS.copy(),
        "metrics": {
            name: {"units": METRIC_UNITS[name], "direction": direction} for name, direction in METRIC_DIRECTIONS.items()
        },
        "alignment": {
            "mpjpe_world_mm": "none",
            "pa_mpjpe_mm": "proper rotation, positive uniform scale and translation fitted separately each frame",
            "seq_sim_mpjpe_mm": "one proper similarity transformation fitted to all sequence joints together",
            "root_mean_initial_translation_mm": "translate predicted root to GT root at first evaluated frame only",
            "root_final_initial_translation_mm": "same initial translation; error at final evaluated frame",
        },
        "rotation": "mean world/global joint rotation-matrix Frobenius difference; head uses SMPL joint 15",
        "foot_slide": {
            "source": "https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/eval/metrics.py",
            "joint_indices": [7, 10, 8, 11],
            "height_thresholds_m": [0.08, 0.04, 0.08, 0.04],
            "definition": "height-weighted horizontal displacements; sum/T then average four joints; not a speed",
        },
        "temporal": {
            "sampling": "strictly increasing uniform timestamps; actual mean dt used",
            "derivatives": "forward finite differences on positions, with physical dt at each order",
            "jerk_trace": "mean joint norm of third finite difference / dt^3, expressed in mm/s^3",
            "jerk_peak": "maximum jerk trace sample",
            "jerk_auc": "trapezoidal integral over jerk trace sample times, expressed in mm/s^2",
            "short_sequences": "unavailable derivatives or integrals are null",
            "status": "diagnostic jerk definitions; not verified as EgoForce PJ/AUJ implementation",
        },
        "paper_boundaries": {
            "excluded_by_scope": ["finger joints", "fine hand reconstruction", "MPJPE-H"],
            "egoprimo": "v2 reconstruction uses Nymeria and 21 common joints; same alignment family only",
            "head_position": "SMPL15 head-joint PE is not UniEgoMotion's left-eye-joint position error",
            "mpjre_f": "global joint rotation definition must match external evaluator before comparing paper numbers",
            "pending": {
                "tmr_semantic_similarity": "requires identical TMR checkpoint, preprocessing and clip set",
                "tmr_fid": "requires identical TMR checkpoint, preprocessing and population-level aggregation",
                "egoforce_pj_auj": "requires verified author definitions and normalization",
            },
        },
    }


def _similarity_align(prediction, truth):
    """Align (...,N,3) point sets with scale and a proper rotation."""
    source_mean, target_mean = prediction.mean(-2, keepdim=True), truth.mean(-2, keepdim=True)
    source, target = prediction - source_mean, truth - target_mean
    covariance = source.transpose(-1, -2) @ target
    u, singular, vh = torch.linalg.svd(covariance)
    signs = torch.ones_like(singular)
    signs[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
    rotation = (u * signs[..., None, :]) @ vh
    variance = source.square().sum((-2, -1))
    scale = (singular * signs).sum(-1) / variance.clamp_min(torch.finfo(source.dtype).eps)
    return scale[..., None, None] * (source @ rotation) + target_mean


def foot_slide_uniegomotion(joints):
    """UniEgoMotion's published-code displacement metric, returned in mm."""
    feet = joints[:, [7, 10, 8, 11]]
    height = feet[:-1, :, 2]
    thresholds = height.new_tensor([0.08, 0.04, 0.08, 0.04])
    displacement = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1)
    # Evaluate powers only for selected contacts to avoid overflow aloft.
    contact = height < thresholds
    weight = 2 - torch.exp2(torch.where(contact, height / thresholds, 0))
    weighted = torch.where(contact, (displacement * weight).abs(), 0)
    return float(weighted.sum(0).mean() * (1000 / len(joints)))


def _jerk_values(positions, dt):
    if len(positions) < 4:
        return None, None
    trace = (torch.diff(positions, n=3, dim=0) / dt**3).norm(dim=-1).mean(-1) * 1000
    peak = float(trace.max())
    area = float(torch.trapezoid(trace, dx=dt)) if len(trace) > 1 else None
    return peak, area


def _rotation_pair(predicted, truth, frames):
    if predicted is None and truth is None:
        return None
    if predicted is None or truth is None:
        raise ValueError("Provide both predicted and GT rotations, or neither.")
    pair = [torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64) for value in (predicted, truth)]
    for value in pair:
        if value.shape != (frames, 22, 3, 3) or not bool(torch.isfinite(value).all()):
            raise ValueError("Rotations must be finite [T,22,3,3] matrices.")
        identity = torch.eye(3, dtype=value.dtype)
        if not torch.allclose(value.transpose(-1, -2) @ value, identity.expand_as(value), atol=1e-4, rtol=1e-4):
            raise ValueError("Rotation matrices must be orthogonal.")
        determinants = torch.linalg.det(value)
        if not torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4, rtol=1e-4):
            raise ValueError("Rotation matrices must have determinant +1.")
    return pair


@torch.no_grad()
def comparison_metrics(prediction, truth, timestamps, *, predicted_rotations=None, truth_rotations=None):
    """Compute a flat, JSON-safe metric row from emitted SMPL22 trajectories."""
    prediction, truth = [
        torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64) for value in (prediction, truth)
    ]
    if (
        prediction.ndim != 3
        or prediction.shape[1:] != (22, 3)
        or not len(prediction)
        or truth.shape != prediction.shape
    ):
        raise ValueError("Prediction and GT positions must have matching nonempty [T,22,3] shapes.")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(truth).all()):
        raise ValueError("Position trajectories must be finite.")
    timestamps = torch.as_tensor(timestamps).detach().to(device="cpu", dtype=torch.float64)
    if timestamps.shape != (len(prediction),) or not bool(torch.isfinite(timestamps).all()):
        raise ValueError("Timestamps must be finite [T] seconds.")
    dt = None
    if len(timestamps) > 1:
        intervals = torch.diff(timestamps)
        if not bool((intervals > 0).all()):
            raise ValueError("Timestamps must be strictly increasing.")
        if not torch.allclose(intervals, intervals.mean().expand_as(intervals), rtol=2e-5, atol=1e-8):
            raise ValueError("Temporal metrics require uniform timestamps.")
        dt = float(intervals.mean())
    error = (prediction - truth).norm(dim=-1) * 1000
    aligned = _similarity_align(prediction, truth)
    sequence_aligned = _similarity_align(prediction.reshape(-1, 3), truth.reshape(-1, 3)).reshape_as(prediction)
    root_difference = (prediction[:, 0] - prediction[0, 0]) - (truth[:, 0] - truth[0, 0])
    root_error = root_difference.norm(dim=-1) * 1000
    result = {
        "mpjpe_world_mm": float(error.mean()),
        "pa_mpjpe_mm": float((aligned - truth).norm(dim=-1).mean() * 1000),
        "seq_sim_mpjpe_mm": float((sequence_aligned - truth).norm(dim=-1).mean() * 1000),
        "head_joint_pe_mm": float(error[:, 15].mean()),
        "wrist_pe_mm": float(error[:, [20, 21]].mean()),
        "mpjre_f": None,
        "head_rot_f": None,
        "root_mean_initial_translation_mm": float(root_error.mean()),
        "root_final_initial_translation_mm": float(root_error[-1]),
        "velocity_error_mm_s": None,
        "acceleration_error_mm_s2": None,
        "foot_slide_uniegomotion_mm": foot_slide_uniegomotion(prediction),
        "gt_foot_slide_uniegomotion_mm": foot_slide_uniegomotion(truth),
    }
    rotation_pair = _rotation_pair(predicted_rotations, truth_rotations, len(prediction))
    if rotation_pair is not None:
        difference = (rotation_pair[0] - rotation_pair[1]).norm(dim=(-2, -1))
        result.update(mpjre_f=float(difference.mean()), head_rot_f=float(difference[:, 15].mean()))
    for order, name in ((1, "velocity_error_mm_s"), (2, "acceleration_error_mm_s2")):
        if len(prediction) > order:
            difference = torch.diff(prediction - truth, n=order, dim=0) / dt**order
            result[name] = float(difference.norm(dim=-1).mean() * 1000)
    predicted_jerk = _jerk_values(prediction, dt)
    true_jerk = _jerk_values(truth, dt)
    for index, name in enumerate(("diagnostic_jerk_peak_mm_s3", "diagnostic_jerk_auc_mm_s2")):
        result[name], result[f"gt_{name}"] = predicted_jerk[index], true_jerk[index]
        gap_name = name.replace("_mm_", "_gap_mm_")
        result[gap_name] = abs(predicted_jerk[index] - true_jerk[index]) if predicted_jerk[index] is not None else None
    return result
