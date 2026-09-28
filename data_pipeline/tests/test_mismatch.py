import json

import numpy as np
import pytest
import torch

from data_pipeline.ee4d_mismatch.audit import audit_dataset
from data_pipeline.ee4d_mismatch.build import build_dataset
from data_pipeline.ee4d_mismatch.corruptions import (
    apply_operations, clean_arrays, rotation_6d_to_matrix, rotvec_to_matrix, variant_operations,
)
from data_pipeline.ee4d_mismatch.dataset import MismatchDataset
from data_pipeline.ee4d_mismatch.source import EE4DSource, annotation_digest, write_json
from data_pipeline.wait_for_dataset import ready


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trajectory(length=200):
    array = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32), (length, 1))
    array[:, 6] = np.arange(length) / 10
    array[:, 7] = 1
    return array


@pytest.fixture
def clean():
    return clean_arrays(trajectory(), 33, 5, 20)


def op(kind, **kw):
    return dict(type=kind, start=30, end=60, **kw)


def test_clock_freeze_and_odd_delay(clean):
    np.testing.assert_array_equal(clean["frame_id_30fps"], 48+3*np.arange(200))
    np.testing.assert_array_equal(clean["img_source_idx"], (48+3*np.arange(200))//6)
    delayed, _ = apply_operations(clean, [op("video_delay", lag_motion_steps=3)], 1)
    np.testing.assert_array_equal(delayed["img_source_idx"][30:60], clean["img_source_idx"][27:57])
    frozen, detail = apply_operations(clean, [op("video_freeze")], 1)
    assert np.all(frozen["img_source_idx"][30:60] == clean["img_source_idx"][29])
    assert frozen["img_available"].all()
    assert detail[0]["latched_observation_index"] == 29
    assert frozen["img_source_idx"][60] == clean["img_source_idx"][60]


def test_composition_uses_observed_payload(clean):
    operations = [op("video_delay", lag_motion_steps=4), dict(type="video_freeze", start=35, end=45)]
    changed, detail = apply_operations(clean, operations, 1)
    assert np.all(changed["img_source_idx"][35:45] == clean["img_source_idx"][30])
    assert detail[1]["latched_feature_index"] == clean["img_source_idx"][30]


def test_drift_units_and_snap(clean):
    changed, _ = apply_operations(clean, [op("head_translation_drift", velocity_m_s=[0.03, 0, 0])], 1)
    np.testing.assert_allclose(changed["aria_traj_obs"][30:60, 6]-clean["aria_traj_obs"][30:60, 6],
                               np.arange(30)*0.003, atol=5e-7)
    np.testing.assert_array_equal(changed["aria_traj_obs"][60:], clean["aria_traj_obs"][60:])
    assert changed["traj_available"].all()


def test_world_yaw_and_exact_half_turn(clean):
    original = rotvec_to_matrix(np.array([[0.3, 0.1, 0]]))[0]
    clean["aria_traj_obs"][:, :6] = original[:2].reshape(6)
    changed, _ = apply_operations(clean, [op("trajectory_jump", translation_m=[0, 0, 0], yaw_deg=180)], 1)
    expected = np.diag([-1, -1, 1]) @ original
    np.testing.assert_allclose(rotation_6d_to_matrix(changed["aria_traj_obs"][30, :6]), expected, atol=1e-7)
    np.testing.assert_array_equal(changed["aria_traj_obs"][:, 6:], clean["aria_traj_obs"][:, 6:])
    assert np.isfinite(changed["aria_traj_obs"]).all()


def test_seed_repeatability_and_source_immutability(clean):
    before = {key: value.copy() for key, value in clean.items()}
    operations = [op("trajectory_jitter", translation_std_m=0.02, rotation_std_deg=2)]
    a, _ = apply_operations(clean, operations, 10)
    b, _ = apply_operations(clean, operations, 10)
    c, _ = apply_operations(clean, operations, 11)
    for key in clean:
        np.testing.assert_array_equal(clean[key], before[key])
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["aria_traj_obs"], c["aria_traj_obs"])


@pytest.mark.parametrize("operation", [
    dict(type="video_freeze", start=19, end=40), dict(type="video_freeze", start=30, end=201),
    op("video_delay", lag_motion_steps=-1), op("video_delay", lag_motion_steps=3.0),
    op("video_delay", lag_motion_steps=31), op("not_supported"),
    op("trajectory_jitter", translation_std_m=-1, rotation_std_deg=2),
    op("head_translation_drift", velocity_m_s=[0, 0, 1]),
    op("head_yaw_drift", yaw_rate_deg_s=float("nan")), op("video_freeze", recovery="none"),
])
def test_invalid_operations_fail(clean, operation):
    with pytest.raises(ValueError):
        apply_operations(clean, [operation], 1)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "official"
    (root / "uniegomotion").mkdir(parents=True)
    (root / "annotations").mkdir()
    motion, features, takes = {}, {}, []
    for index in range(4):
        name = f"take_{index}"
        length, start30 = 240, 33
        motion[f"{name}___{start30}___{start30+3*(length-1)}"] = dict(
            start_idx=start30, end_idx=start30+3*(length-1), num_frames=length,
            aria_traj=torch.from_numpy(trajectory(length)),
            smpl_params=dict(global_orient=torch.ones(length, 6), body_pose=torch.ones(length, 21, 6),
                             transl=torch.arange(length*3).reshape(length, 3).float(), betas=torch.ones(1, 10),
                             left_hand_pose=torch.zeros(length, 12), right_hand_pose=torch.zeros(length, 12)),
            kp3d=torch.ones(length, 76, 3), body_root_offset=torch.zeros(3), floor_height=np.float64(0),
        )
        feature = torch.arange(140).reshape(-1, 1, 1).expand(-1, 5, 1024).clone().float()
        feature[:, 1:] += 1000  # Token 0 must be used.
        features[name] = feature
        takes.append(dict(take_name=name, parent_task_name=f"task_{index%2}"))
    for split in ("train", "val"):
        torch.save(motion, root / "uniegomotion" / f"ee_{split}.pt")
        torch.save({"feats": features}, root / "uniegomotion" / f"egoview_dinov2_{split}.pt")
    write_json(root / "takes.json", takes)
    write_json(root / "annotations" / "splits.json", {"train": [], "val": list(features)})
    return EE4DSource(root)


def test_missing_never_indexes_minus_one(source, clean):
    source.features["take_0"][-1] = float("nan")
    payload = source.visual_payload("take_0", np.array([3, -1, 4]), np.array([True, False, True]))
    assert payload.shape == (3, 1024)
    assert torch.isfinite(payload).all()
    assert torch.all(payload[0] == 3) and torch.all(payload[1] == 0)
    changed, _ = apply_operations(clean, [op("video_missing"), op("traj_missing")], 1)
    assert np.all(changed["img_source_idx"][30:60] == -1)
    assert np.all(changed["traj_source_idx"][30:60] == -1)
    assert np.isfinite(changed["aria_traj_obs"]).all()
    assert not changed["traj_available"][30:60].any()


def test_extended_real_format_roundtrip(source, tmp_path):
    output = tmp_path / "extended"
    before = annotation_digest(source.motion)
    report = build_dataset(source, output, num_takes=2, profile="extended")
    assert report["num_variants"] == 36
    assert annotation_digest(source.motion) == before
    assert audit_dataset(output, source=source)["source_hashes_verified"]
    dataset = MismatchDataset(output, source=source)
    freeze = next(r for r in dataset.records if r["variant_name"] == "persistent_freeze")
    arrays = dataset.arrays(freeze["variant_id"])
    assert arrays["corruption_mask_gt"][freeze["fault_onset"]:, 1].all()
    assert not arrays["corruption_mask_gt"][:20].any()
    current = dataset.observations(freeze["variant_id"], as_of=45, history_frames=15)
    later = dataset.observations(freeze["variant_id"], as_of=50, history_frames=15)
    for key in current:
        assert torch.equal(current[key][5:], later[key][:-5])
    assert set(current) == set(dataset.spec["online_fields"])
    labels = dataset.supervision(0)
    labels["kp3d"][:] = -999
    assert annotation_digest(dataset.supervision(0)) == dataset.record(0)["supervision_sha256"]
    with pytest.raises(FileExistsError):
        build_dataset(source, output, num_takes=2)


def test_reproducible_selection_build_and_tamper_detection(source, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    build_dataset(source, first, num_takes=2)
    build_dataset(source, second, num_takes=2)
    a, b = MismatchDataset(first, source=source), MismatchDataset(second, source=source)
    assert a.records == b.records
    target = first / a.record(0)["array_file"]
    with target.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        audit_dataset(first, source=source)
    source.metadata_path.write_text(source.metadata_path.read_text() + " ")
    with pytest.raises(ValueError, match="Source file checksums"):
        audit_dataset(second, source=source)


def test_development_split_isolation(source, tmp_path):
    with pytest.raises(ValueError, match="official train"):
        build_dataset(source, tmp_path / "invalid", num_takes=4, purpose="development")
    train_source = EE4DSource(source.root, "train")
    output = tmp_path / "development"
    build_dataset(train_source, output, num_takes=4, purpose="development")
    dataset = MismatchDataset(output, source=train_source)
    train = {r["base_take_name"] for r in dataset.records if r["split"] == "train"}
    dev = {r["base_take_name"] for r in dataset.records if r["split"] == "dev"}
    assert train and dev and not train & dev


def test_development_holdout_is_take_disjoint(source, tmp_path):
    train_source = EE4DSource(source.root, "train")
    output = tmp_path / "three_way"
    build_dataset(train_source, output, num_takes=4, purpose="development",
                  dev_fraction=0.25, holdout_fraction=0.25)
    dataset = MismatchDataset(output, source=train_source)
    groups = {name: {r["base_take_name"] for r in dataset.records if r["split"] == name}
              for name in ("train", "dev", "holdout")}
    assert [len(groups[name]) for name in ("train", "dev", "holdout")] == [2, 1, 1]
    assert len(set.union(*groups.values())) == 4
    assert audit_dataset(output, source=train_source)["status"] == "passed"


def test_completion_signal_requires_verified_artifacts(tmp_path):
    signal = tmp_path / "ready.json"
    assert ready(signal) is None
    root = tmp_path / "dataset"
    (root / "audit").mkdir(parents=True)
    write_json(root / "spec.json", {})
    write_json(root / "audit" / "validation.json", dict(status="passed", source_hashes_verified=False))
    write_json(signal, dict(status="complete", datasets=[dict(path=str(root))]))
    with pytest.raises(ValueError, match="full audit"):
        ready(signal)
    write_json(root / "audit" / "validation.json", dict(status="passed", source_hashes_verified=True))
    assert ready(signal)["status"] == "complete"


def test_full_corpus_includes_short_sequences_and_every_take(source, tmp_path):
    # Add an official-format 21-frame segment to a take already present in the
    # test fixture. The full build must include it despite the 200-frame pilot.
    motion = dict(source.motion)
    original = next(value for name, value in motion.items() if name.startswith("take_0___"))
    short = dict(original)
    short.update(start_idx=0, end_idx=60, num_frames=21,
                 aria_traj=original["aria_traj"][:21].clone(),
                 kp3d=original["kp3d"][:21].clone(),
                 smpl_params={key: (value.clone() if key == "betas" else value[:21].clone())
                              for key, value in original["smpl_params"].items()})
    motion["take_0___0___60"] = short
    replacement = source.motion_path.with_suffix(".replacement.pt")
    torch.save(motion, replacement)
    replacement.replace(source.motion_path)
    full_source = EE4DSource(source.root)
    output = tmp_path / "full"
    report = build_dataset(full_source, output, all_sequences=True, num_takes=1)
    assert (report["num_takes"], report["num_episodes"], report["num_variants"]) == (4, 5, 35)
    assert report["source_sequences_covered"] == len(full_source.motion)
    assert report["physical_frames_covered"] == 4*240+21
    assert audit_dataset(output, source=full_source)["source_hashes_verified"]
    dataset = MismatchDataset(output, source=full_source)
    short_records = [r for r in dataset.records if r["base_seq_name"] == "take_0___0___60"]
    assert len(short_records) == 7
    assert {r["bootstrap_frames"] for r in short_records} == {7}
    assert {r["fault_onset"] for r in short_records} == {9}
    assert {r["available_fault_frames"] for r in short_records} == {12}
    drift = next(r for r in short_records if r["variant_name"] == "drift_0p03mps")
    assert drift["operations"][0]["requested_duration_motion_steps"] == 30
    assert drift["operations"][0]["end"] == 21
    assert dataset.arrays(drift["variant_id"])["corruption_mask_gt"][9:, 0].all()
    assert dataset.observations(drift["variant_id"], as_of=20)["aria_traj_obs"].shape == (1, 9)


def test_adaptive_extended_operators_fit_short_sequence():
    clean = clean_arrays(trajectory(21), 0, 0, 7)
    variants = variant_operations(9, 21, 1, "extended", adaptive=True)
    assert len(variants) == 18
    for operations in variants.values():
        arrays, _ = apply_operations(clean, operations, 1, bootstrap_frames=7)
        assert len(arrays["aria_traj_obs"]) == 21
        assert not arrays["corruption_mask_gt"][:7].any()
