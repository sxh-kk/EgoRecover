"""Deterministic corruptions on absolute device poses and feature source indices.

This module has no dependency on the E7 model or its data loader.
"""

import hashlib
import json

import numpy as np


MODALITY_ORDER = ("traj", "img")
KINDS = {
    "video_delay", "video_freeze", "video_missing", "traj_delay", "traj_missing",
    "head_translation_drift", "head_yaw_drift", "trajectory_jitter", "trajectory_jump",
}


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def rotation_6d_to_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    first, second = value[..., :3], value[..., 3:6]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-8):
        raise ValueError("Non-finite or degenerate 6D rotation")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Collinear 6D rotation axes")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def rotvec_to_matrix(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vectors, axis=-1)
    skew = np.zeros(vectors.shape[:-1] + (3, 3), dtype=np.float64)
    x, y, z = np.moveaxis(vectors, -1, 0)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    # sinc remains well-defined at zero; this also handles exactly 180 degrees.
    a = np.sinc(angle / np.pi)[..., None, None]
    b = (0.5 * np.sinc(angle / (2 * np.pi)) ** 2)[..., None, None]
    return np.eye(3) + a * skew + b * (skew @ skew)


def clean_arrays(trajectory, seq_start_frame30, episode_start, bootstrap_frames):
    trajectory = np.array(trajectory, dtype=np.float32, copy=True)
    if trajectory.ndim != 2 or trajectory.shape[1] != 9 or not np.isfinite(trajectory).all():
        raise ValueError("Absolute trajectory must be a finite [T,9] array")
    rotation_6d_to_matrix(trajectory[:, :6])
    length = len(trajectory)
    if not 0 < bootstrap_frames < length:
        raise ValueError("Bootstrap must leave at least one evaluation frame")
    sequence_indices = episode_start + np.arange(length, dtype=np.int64)
    frames = seq_start_frame30 + 3 * sequence_indices
    evaluate = np.arange(length) >= bootstrap_frames
    return {
        "aria_traj_obs": trajectory,
        "img_source_idx": frames // 3 // 2,
        "traj_source_idx": sequence_indices.copy(),
        "seq_motion_idx": sequence_indices,
        "img_available": np.ones(length, dtype=np.bool_),
        "traj_available": np.ones(length, dtype=np.bool_),
        "frame_id_30fps": frames,
        "corruption_mask_gt": np.zeros((length, 2), dtype=np.bool_),
        "eval_mask": evaluate,
    }


def _finite_number(op, key, default=None, minimum=None):
    number = op.get(key, default)
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return float(number)


def _vector(op, key):
    value = np.asarray(op[key], dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must have three finite components")
    return value


def apply_operations(clean, operations, seed, bootstrap_frames=20, fps=10):
    """Apply ordered operators to an episode, never to independently sampled windows.

    Delays read a snapshot of the preceding operator's stream. Freezes latch the
    last available preceding payload. Source indices and fault masks are audit
    information; only payloads and actual availability belong at model inputs.
    """
    if fps != 10:
        raise ValueError("This dataset version uses the EE4D 10 FPS motion grid")
    result = {key: value.copy() for key, value in clean.items()}
    length = len(result["aria_traj_obs"])
    provenance = []
    for position, op in enumerate(operations):
        kind = op.get("type")
        if kind not in KINDS:
            raise ValueError(f"Unknown corruption {kind!r}")
        start, end = op.get("start"), op.get("end")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end)):
            raise ValueError("Event boundaries must be integer motion indices")
        if not bootstrap_frames <= start < end <= length:
            raise ValueError("Event must follow bootstrap and lie inside the episode")
        recovery = op.get("recovery", "snap")
        if recovery not in {"snap", "none"} or (recovery == "none" and end != length):
            raise ValueError("Use snap recovery, or none for an event ending at the episode boundary")
        span = slice(start, end)
        visual = kind.startswith("video_")
        column = 1 if visual else 0
        result["corruption_mask_gt"][span, column] = True
        rng = np.random.default_rng(stable_seed(seed, position, kind))
        detail = {"operation_index": position, "type": kind}
        if kind in {"video_delay", "traj_delay"}:
            lag = op.get("lag_motion_steps")
            if isinstance(lag, bool) or not isinstance(lag, int) or not 0 < lag <= start:
                raise ValueError("Delay must be positive and have enough already observed history")
            source = np.arange(start, end) - lag
            fields = ("img_source_idx", "img_available") if visual else (
                "aria_traj_obs", "traj_source_idx", "traj_available",
            )
            for field in fields:
                result[field][span] = result[field][source].copy()
        elif kind == "video_freeze":
            available = np.flatnonzero(result["img_available"][:start])
            if not len(available):
                raise ValueError("Cannot freeze without an already available payload")
            previous = int(available[-1])
            source = int(result["img_source_idx"][previous])
            result["img_source_idx"][span] = source
            result["img_available"][span] = True
            detail.update(latched_observation_index=previous, latched_feature_index=source)
        elif kind == "video_missing":
            result["img_source_idx"][span] = -1
            result["img_available"][span] = False
        elif kind == "traj_missing":
            result["traj_available"][span] = False
            result["traj_source_idx"][span] = -1
            result["aria_traj_obs"][span] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        else:
            elapsed = np.arange(end - start, dtype=np.float64) / fps
            count = end - start
            translation = np.zeros((count, 3))
            angles = np.zeros((count, 3))
            if kind == "head_translation_drift":
                velocity = _vector(op, "velocity_m_s")
                if velocity[2] != 0:
                    raise ValueError("head_translation_drift is horizontal in this version")
                translation = elapsed[:, None] * velocity
            elif kind == "head_yaw_drift":
                angles[:, 2] = np.deg2rad(_finite_number(op, "yaw_rate_deg_s")) * elapsed
            elif kind == "trajectory_jitter":
                translation = rng.normal(size=(count, 3)) * _finite_number(op, "translation_std_m", minimum=0)
                angles = rng.normal(size=(count, 3)) * np.deg2rad(_finite_number(op, "rotation_std_deg", minimum=0))
            elif kind == "trajectory_jump":
                translation[:] = _vector(op, "translation_m")
                angles[:, 2] = np.deg2rad(_finite_number(op, "yaw_deg", default=0))
            usable = result["traj_available"][span]
            rows = np.arange(start, end)[usable]
            if len(rows):
                result["aria_traj_obs"][rows, 6:9] += translation[usable].astype(np.float32)
                if np.any(angles[usable]):
                    original = rotation_6d_to_matrix(result["aria_traj_obs"][rows, :6])
                    changed = rotvec_to_matrix(angles[usable]) @ original
                    result["aria_traj_obs"][rows, :6] = changed[:, :2, :].reshape(-1, 6).astype(np.float32)
        provenance.append(detail)
    validate_arrays(result)
    return result, provenance


def validate_arrays(arrays, feature_rows=None):
    length = len(arrays["aria_traj_obs"])
    expected = {
        "aria_traj_obs": (length, 9), "corruption_mask_gt": (length, 2),
        **{key: (length,) for key in ("img_source_idx", "traj_source_idx", "seq_motion_idx",
                                     "img_available", "traj_available", "frame_id_30fps", "eval_mask")},
    }
    if set(arrays) != set(expected):
        raise ValueError("Unexpected dataset array fields")
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid shape for {key}: {arrays[key].shape}, expected {shape}")
        dtype = np.float32 if key == "aria_traj_obs" else (
            np.bool_ if key in {"corruption_mask_gt", "img_available", "traj_available", "eval_mask"} else np.int64
        )
        if arrays[key].dtype != dtype:
            raise ValueError(f"Invalid dtype for {key}: {arrays[key].dtype}")
    if not np.isfinite(arrays["aria_traj_obs"]).all():
        raise ValueError("Non-finite trajectory payload")
    rotation_6d_to_matrix(arrays["aria_traj_obs"][:, :6])
    if np.any(np.diff(arrays["seq_motion_idx"]) != 1) or np.any(np.diff(arrays["frame_id_30fps"]) != 3):
        raise ValueError("Episode indices must be a continuous 10 FPS sequence")
    for modality in ("img", "traj"):
        source, available = arrays[modality + "_source_idx"], arrays[modality + "_available"]
        if np.any(source[available] < 0) or np.any(source[~available] != -1):
            raise ValueError(f"Invalid {modality} availability/source sentinel")
        if modality == "img":
            if np.any(source[available] * 6 > arrays["frame_id_30fps"][available]):
                raise ValueError("Visual source is in the future")
            if feature_rows is not None and np.any(source[available] >= feature_rows):
                raise ValueError("Visual source exceeds the original take cache")
        elif np.any(source[available] > arrays["seq_motion_idx"][available]):
            raise ValueError("Trajectory source is in the future")


def variant_operations(onset, frames, seed, profile="pilot", *, adaptive=False):
    """Seven pilot variants, with an extended diagnostic set for all operators."""
    if profile not in {"pilot", "extended"}:
        raise ValueError("Unknown variant profile")
    angle = np.random.default_rng(stable_seed(seed, "direction")).uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])

    def event(kind, duration=30, **parameters):
        end = min(frames, onset + duration) if adaptive else onset + duration
        operation = dict(type=kind, start=onset, end=end, recovery="snap", **parameters)
        if adaptive:
            operation["requested_duration_motion_steps"] = duration
        return operation

    slow = event("head_translation_drift", velocity_m_s=(direction * 0.01).tolist())
    fast = event("head_translation_drift", velocity_m_s=(direction * 0.03).tolist())
    variants = {
        "clean": [],
        "freeze_1s": [event("video_freeze", 10)],
        "freeze_3s": [event("video_freeze")],
        "delay_0p2s": [event("video_delay", lag_motion_steps=2)],
        "delay_0p4s": [event("video_delay", lag_motion_steps=4)],
        "drift_0p01mps": [slow],
        "drift_0p03mps": [fast],
    }
    if profile == "extended":
        variants.update({
            "delay_0p3s": [event("video_delay", lag_motion_steps=3)],
            "video_missing": [event("video_missing")],
            "traj_missing": [event("traj_missing")],
            "traj_delay_0p4s": [event("traj_delay", lag_motion_steps=4)],
            "yaw_3dps": [event("head_yaw_drift", yaw_rate_deg_s=3)],
            "jitter": [event("trajectory_jitter", translation_std_m=0.02, rotation_std_deg=2)],
            "pulse": [event("trajectory_jump", 1, translation_m=[0.10, 0, 0], yaw_deg=5)],
            "jump": [event("trajectory_jump", translation_m=[0.15, 0, 0], yaw_deg=5)],
            "freeze_and_drift": [event("video_freeze"), fast],
            "delay_then_freeze": [event("video_delay", lag_motion_steps=4),
                                  dict(type="video_freeze", start=min(frames-1, onset+5) if adaptive else onset+5,
                                       end=min(frames, onset+15) if adaptive else onset+15, recovery="snap")],
            "persistent_freeze": [dict(type="video_freeze", start=onset, end=frames, recovery="none")],
        })
    return variants
