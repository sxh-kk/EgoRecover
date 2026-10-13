from types import SimpleNamespace

import pytest
import torch

from motionstreamer_flow.time_protocol import source_timing, validate_timing


def source_packet(frames=5):
    return {
        "record": {
            "base_seq_name": "take___30___87",
            "episode_start_motion_idx": 5,
            "num_frames": frames,
        },
        "truth": {"joints": torch.zeros(frames, 22, 4, 4)},
        "images": torch.zeros(frames, 12),
        "head": torch.zeros(frames, 9),
    }


def test_source_grid_uses_sequence_and_episode_offsets_and_labels_derived_time():
    frame_ids, timestamps, provenance = source_timing(source_packet(), SimpleNamespace(fps=10))
    torch.testing.assert_close(frame_ids, torch.tensor([45, 48, 51, 54, 57]))
    torch.testing.assert_close(timestamps, torch.tensor([1.5, 1.6, 1.7, 1.8, 1.9], dtype=torch.float64))
    assert frame_ids.dtype == torch.int64 and timestamps.dtype == torch.float64
    assert provenance["timestamp_source"] == "derived_official_30hz_frame_grid"
    assert provenance["measured_sensor_timestamps"] is False
    assert provenance["resampled"] is False and provenance["fps"] == 10
    assert provenance["source_fps"] == 30 and provenance["source_frame_stride"] == 3
    assert provenance["supplied_frame_ids_verified"] is False


@pytest.mark.parametrize("fps", [5, 20, 0, -10, float("nan"), float("inf"), True])
def test_source_rejects_forged_or_invalid_fps(fps):
    with pytest.raises(ValueError, match="FPS"):
        source_timing(source_packet(), SimpleNamespace(fps=fps))


@pytest.mark.parametrize("key", ["record", "truth", "images", "head"])
def test_source_requires_record_and_all_timed_payloads(key):
    packet = source_packet()
    del packet[key]
    with pytest.raises(ValueError):
        source_timing(packet, SimpleNamespace(fps=10))


@pytest.mark.parametrize("key", ["base_seq_name", "episode_start_motion_idx", "num_frames"])
def test_source_requires_time_fields(key):
    packet = source_packet()
    del packet["record"][key]
    with pytest.raises(ValueError, match="requires"):
        source_timing(packet, SimpleNamespace(fps=10))


@pytest.mark.parametrize(
    "name", [None, "take", "take___30___x", "___30___87", "take___-3___87", "take___30___29", "take___30___88"]
)
def test_source_rejects_invalid_sequence_frame_bounds(name):
    packet = source_packet()
    packet["record"]["base_seq_name"] = name
    with pytest.raises(ValueError):
        source_timing(packet, SimpleNamespace(fps=10))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("episode_start_motion_idx", -1),
        ("episode_start_motion_idx", 16),
        ("episode_start_motion_idx", 5.0),
        ("episode_start_motion_idx", True),
        ("num_frames", 4),
        ("num_frames", 0),
        ("num_frames", "5"),
    ],
)
def test_source_rejects_out_of_bounds_or_inconsistent_episode_metadata(key, value):
    packet = source_packet()
    packet["record"][key] = value
    with pytest.raises(ValueError):
        source_timing(packet, SimpleNamespace(fps=10))


@pytest.mark.parametrize("key", ["images", "head"])
def test_source_rejects_payload_length_mismatch(key):
    packet = source_packet()
    packet[key] = packet[key][:-1]
    with pytest.raises(ValueError, match="length"):
        source_timing(packet, SimpleNamespace(fps=10))


def test_source_accepts_verified_supplied_frame_indices():
    packet = source_packet()
    packet["frame_id_30fps"] = torch.tensor([45, 48, 51, 54, 57])
    _, _, provenance = source_timing(packet, SimpleNamespace(fps=10))
    assert provenance["supplied_frame_ids_verified"] is True


@pytest.mark.parametrize(
    "indices",
    [
        None,
        torch.tensor([45, 48, 48, 54, 57]),
        torch.tensor([45, 51, 48, 54, 57]),
        torch.tensor([45, 48, 51, 54, 60]),
        torch.tensor([45, 48, 51, 54]),
        torch.tensor([45, 48, 51, 54, 57], dtype=torch.float64),
    ],
)
def test_source_rejects_duplicate_reordered_missing_or_noninteger_frame_indices(indices):
    packet = source_packet()
    packet["frame_id_30fps"] = indices
    with pytest.raises(ValueError, match="frame_id_30fps"):
        source_timing(packet, SimpleNamespace(fps=10))


def test_validate_timing_accepts_uniform_absolute_time_and_declared_fps():
    validate_timing(1000 + torch.arange(200, dtype=torch.float64) / 10, 10, 200)


@pytest.mark.parametrize(
    "timestamps",
    [
        torch.arange(5),
        torch.zeros(5, 1),
        torch.tensor([0.0, 0.1, float("nan"), 0.3, 0.4]),
        torch.tensor([0.0, 0.1, 0.1, 0.3, 0.4]),
        torch.tensor([0.0, 0.2, 0.1, 0.3, 0.4]),
        torch.tensor([0.0, 0.1, 0.3, 0.4, 0.5]),
        torch.arange(5, dtype=torch.float64) / 5,
    ],
)
def test_validate_timing_rejects_invalid_or_inconsistent_timestamps(timestamps):
    with pytest.raises(ValueError):
        validate_timing(timestamps, 10, 5)


def test_validate_timing_rejects_inconsistent_declared_length():
    with pytest.raises(ValueError, match="shape"):
        validate_timing(torch.arange(5, dtype=torch.float64) / 10, 10, 4)
