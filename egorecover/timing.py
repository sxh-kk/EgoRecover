"""Validate the prepared source's frame grid and standalone episode timebase.

Source timestamps are derived from official 30 Hz frame indices. They are not
measurements of a sensor clock, and this exporter does not resample observations.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

SOURCE_FPS = 30
SOURCE_FRAME_STRIDE = 3


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _fps(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("FPS must be a finite positive number.")
    return float(value)


def validate_timing(timestamps_s, fps, length):
    """Require a finite, uniform, increasing timeline at the declared FPS.

    Prefer float64 for absolute take times: low precision timestamps may not
    retain enough resolution to demonstrate the sampling interval.
    """
    fps = _fps(fps)
    length = _integer(length, "Timeline length", minimum=1)
    if not isinstance(timestamps_s, torch.Tensor) or tuple(timestamps_s.shape) != (length,):
        raise ValueError(f"timestamps_s must have shape ({length},).")
    if not timestamps_s.is_floating_point() or not torch.isfinite(timestamps_s).all():
        raise ValueError("timestamps_s must contain finite floating-point values.")
    if length > 1:
        intervals = timestamps_s.double().diff()
        if not (intervals > 0).all():
            raise ValueError("timestamps_s must be strictly increasing.")
        expected = torch.full_like(intervals, 1.0 / fps)
        if not torch.allclose(intervals, expected, rtol=1e-5, atol=1e-8):
            raise ValueError("timestamps_s must have constant spacing consistent with FPS.")


def source_timing(packet, cfg):
    """Return ``(frame_ids, timestamps_s, provenance)`` for a prepared episode.

    The sequence name stores inclusive first/last original 30 Hz frame indices.
    Prepared motion samples advance three original frames per sample. Validate
    the record and payload lengths before deriving the episode's timebase.
    """
    if not isinstance(packet, Mapping) or not isinstance(packet.get("record"), Mapping):
        raise ValueError("Source timing requires the prepared episode record.")
    record = packet["record"]
    required = ("base_seq_name", "episode_start_motion_idx", "num_frames")
    if any(key not in record for key in required):
        raise ValueError(f"Source timing record requires {', '.join(required)}.")
    truth = packet.get("truth")
    joints = truth.get("joints") if isinstance(truth, Mapping) else None
    if not isinstance(joints, torch.Tensor) or joints.ndim < 1:
        raise ValueError("Source timing requires truth joints with a frame dimension.")
    length = len(joints)
    if length < 1:
        raise ValueError("Source timing requires at least one frame.")
    count = _integer(record["num_frames"], "Source num_frames", minimum=1)
    if count != length:
        raise ValueError("Source num_frames does not match truth joints length.")
    for key in ("images", "head"):
        value = packet.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != length:
            raise ValueError(f"Source {key} length does not match truth joints.")
    name = record["base_seq_name"]
    try:
        take, first_text, last_text = name.rsplit("___", 2)
        first, last = int(first_text), int(last_text)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Source base_seq_name must end with ___first30___last30.") from None
    if not take or first < 0 or last < first or (last - first) % SOURCE_FRAME_STRIDE:
        raise ValueError("Source sequence frame bounds must follow the 30 Hz grid with stride 3.")
    start = _integer(record["episode_start_motion_idx"], "Source episode_start_motion_idx")
    sequence_length = (last - first) // SOURCE_FRAME_STRIDE + 1
    if start + length > sequence_length:
        raise ValueError("Source episode exceeds the sequence frame bounds.")
    inferred_fps = SOURCE_FPS / SOURCE_FRAME_STRIDE
    if not math.isclose(_fps(cfg.fps), inferred_fps, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Configured FPS differs from source frame grid FPS ({inferred_fps:g}).")
    final = first + SOURCE_FRAME_STRIDE * (start + length - 1)
    if final > torch.iinfo(torch.int64).max:
        raise ValueError("Source frame indices exceed int64 range.")
    frame_ids = first + SOURCE_FRAME_STRIDE * (start + torch.arange(length, dtype=torch.int64))
    if "frame_id_30fps" in packet:
        try:
            supplied = torch.as_tensor(packet["frame_id_30fps"])
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("Source frame_id_30fps must contain integer frame indices.") from None
        if (
            supplied.shape != frame_ids.shape
            or supplied.dtype == torch.bool
            or supplied.is_floating_point()
            or supplied.is_complex()
            or not torch.equal(supplied.cpu().to(torch.int64), frame_ids)
        ):
            raise ValueError("Source frame_id_30fps differs from the canonical episode frame grid.")
    timestamps_s = frame_ids.double() / SOURCE_FPS
    validate_timing(timestamps_s, inferred_fps, length)
    provenance = {
        "timestamp_source": "derived_official_30hz_frame_grid",
        "measured_sensor_timestamps": False,
        "source_fps": SOURCE_FPS,
        "source_frame_stride": SOURCE_FRAME_STRIDE,
        "fps": inferred_fps,
        "resampled": False,
        "sequence_first_frame_30fps": first,
        "sequence_last_frame_30fps_inclusive": last,
        "episode_start_motion_idx": start,
        "supplied_frame_ids_verified": "frame_id_30fps" in packet,
    }
    return frame_ids, timestamps_s, provenance
