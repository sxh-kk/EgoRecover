import json

import joblib
import pytest
import torch

from egorecover.codec import BodyState, planar_reference, transform_from_9d
from egorecover.official import OfficialWindowData
from egorecover.rotations import matrix_to_rotation_6d


class HeadOnlySpy:
    protocol = {"kind": "test_head_only", "mean_beta_zero": True}

    def __init__(self):
        self.calls = []

    def __call__(self, head):
        self.calls.append(head.clone())
        transform = transform_from_9d(head)
        joints = transform[:, None].repeat(1, 22, 1, 1)
        return BodyState(joints, planar_reference(transform), torch.zeros(len(head), 36)), torch.zeros(10)


def source(take, frames, first=3, floor=1.25):
    identity = torch.tensor([1, 0, 0, 0, 1, 0.0])
    head = torch.cat((identity.repeat(frames, 1), torch.zeros(frames, 3)), dim=-1)
    head[:, 6] = torch.arange(frames) * 0.01
    head[:, 8] = floor + 1.7
    joints = torch.zeros(frames, 76, 3)
    joints[:, :, 0] = torch.arange(frames)[:, None] * 0.01
    joints[:, :, 1] = torch.arange(76)[None] * 0.01
    joints[:, :, 2] = floor + 0.03
    last = first + (frames - 1) * 3
    name = f"{take}___{first}___{last}"
    row = {
        "start_idx": first,
        "end_idx": last,
        "num_frames": frames,
        "aria_traj": head,
        "kp3d": joints,
        "floor_height": floor,
        "smpl_params": {
            "global_orient": identity.repeat(frames, 1),
            "body_pose": identity.repeat(frames, 21, 1),
            "betas": torch.full((1, 10), 5.0),
            "transl": joints[:, 0].clone(),
            "left_hand_pose": torch.ones(frames, 12),
            "right_hand_pose": torch.full((frames, 12), 2.0),
        },
    }
    return name, row


@pytest.fixture
def official_root(tmp_path):
    directory = tmp_path / "uniegomotion"
    directory.mkdir()
    (tmp_path / "annotations").mkdir()
    train = dict(source(take, frames) for take, frames in (("train_a", 45), ("train_b", 140), ("train_short", 23)))
    val = dict([source("val_a", 421, floor=0.0)])
    torch.save(train, directory / "ee_train.pt")
    torch.save(val, directory / "ee_val.pt")
    for split, rows in (("train", train), ("val", val)):
        features = {}
        for name, row in rows.items():
            count = row["end_idx"] // 6 + 1
            feature = torch.arange(count)[:, None, None].expand(-1, 5, 1024)
            feature = feature + 1000 * torch.arange(5)[None, :, None]
            features[name.split("___")[0]] = feature.half()
        torch.save({"feats": features, "egocam": {}}, directory / f"egoview_dinov2_{split}.pt")
    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
        "motion_min": torch.zeros(243),
        "motion_max": torch.ones(243),
    }
    torch.save(stats, directory / "v4_beta_ee_train_stats.pt")
    takes, membership = [], {}
    for split, rows in (("train", train), ("val", val)):
        for name in rows:
            take = name.split("___")[0]
            takes.append({"take_name": take, "take_uid": f"uid-{take}"})
            membership[f"uid-{take}"] = split
    (tmp_path / "takes.json").write_text(json.dumps(takes))
    (tmp_path / "annotations/splits.json").write_text(json.dumps({"take_uid_to_split": membership}))
    return tmp_path


def test_features_use_nested_mapping_causal_five_hz_token_zero_and_real_timestamps(official_root):
    data = OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy())
    episode = data.load(0)
    assert episode.images.dtype == torch.float32
    torch.testing.assert_close(episode.images[:6, 0], torch.tensor([0, 1, 1, 2, 2, 3.0]))
    torch.testing.assert_close(episode.timestamps[:3], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
    assert set(data.stats) == {"motion_mean", "motion_std", "traj_mean", "traj_std"}
    assert episode.floor_calibrated is True
    assert data.features["train_a"].dtype == torch.float16
    assert data.features["train_a"].untyped_storage().size() > episode.images.untyped_storage().size()


def test_floor_world_roundtrip_and_global_label_rotations(official_root):
    path = official_root / "uniegomotion/ee_train.pt"
    rows = torch.load(path, weights_only=False)
    name = next(iter(rows))
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    rows[name]["smpl_params"]["global_orient"][:] = matrix_to_rotation_6d(rotation)
    rows[name]["smpl_params"]["body_pose"][:, 0] = matrix_to_rotation_6d(rotation)
    torch.save(rows, path)
    data = OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy())
    episode = data.load(0)
    torch.testing.assert_close(episode.restore_world(episode.truth.joints[..., :3, 3]), rows[name]["kp3d"][:, :22])
    torch.testing.assert_close(episode.truth.joints[:, 0, :3, :3], rotation.expand(len(episode), -1, -1))
    torch.testing.assert_close(episode.truth.joints[:, 1, :3, :3], (rotation @ rotation).expand(len(episode), -1, -1))
    assert episode.truth.joints.shape == (45, 22, 4, 4)
    assert episode.truth.auxiliary.shape == (45, 36)
    torch.testing.assert_close(episode.truth.auxiliary[:, -10:], torch.full((45, 10), 5.0))
    assert torch.equal(episode.beta, torch.zeros(10))


def test_official_val_selection_matches_reference_keys_and_reports_short_tail_and_zero_floor(official_root):
    path = official_root / "uniegomotion/ee_val.pt"
    rows = torch.load(path, weights_only=False)
    name = next(iter(rows))
    reference = {
        f"{name}_start_{start}": {"aria_traj_T": torch.eye(4).repeat(min(80, 421 - start), 1, 1)}
        for start in (0, 200, 400)
    }
    joblib.dump(reference, official_root / "uniegomotion/ee_val_gt_for_evaluation.pkl")
    data = OfficialWindowData(official_root, "dev", bootstrap=HeadOnlySpy())
    assert [row["episode_id"] for row in data.records] == list(reference)[:2]
    assert data.inventory["official_reference_verified"] is True
    assert data.inventory["selected_clip_candidates"] == 3
    assert data.inventory["skipped_short_clips"] == 1
    assert data.inventory["zero_floor_sources"] == 1
    assert data.inventory["uncalibrated_floor_clips"] == 2
    assert data.protocol["floor"] == "provided_official_constant/no_GT_fallback"
    assert torch.equal(data.load(0).original_world_from_floor_world, torch.eye(4))
    assert data.load(0).floor_calibrated is False
    assert OfficialWindowData(official_root, "val", bootstrap=HeadOnlySpy()).digest == data.digest
    assert OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy()).digest == data.digest


def test_test_split_cross_split_take_and_invalid_source_bounds_are_rejected(official_root):
    with pytest.raises(ValueError, match="not be relabeled"):
        OfficialWindowData(official_root, "test", bootstrap=HeadOnlySpy())
    path = official_root / "annotations/splits.json"
    splits = json.loads(path.read_text())
    splits["take_uid_to_split"]["uid-train_a"] = "val"
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="membership mismatch"):
        OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy())
    splits["take_uid_to_split"]["uid-train_a"] = "train"
    path.write_text(json.dumps(splits))
    motion = official_root / "uniegomotion/ee_train.pt"
    rows = torch.load(motion, weights_only=False)
    rows[next(iter(rows))]["start_idx"] += 3
    torch.save(rows, motion)
    with pytest.raises(ValueError, match="stored bounds disagree"):
        OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy())


def test_batch_uses_independent_full_source_queries_with_equal_context_lengths(official_root):
    data = OfficialWindowData(official_root, "train", bootstrap=HeadOnlySpy())
    generator = torch.Generator().manual_seed(62)
    expected_rng = torch.Generator().manual_seed(62)
    expected = []
    names = list(data.source_metadata)[:2]
    for _ in range(16):
        name = names[int(torch.randint(2, (), generator=expected_rng))]
        torch.randint(1, (), generator=expected_rng)
        query = int(torch.randint(24, data.source_metadata[name]["frames"] - 1, (), generator=expected_rng))
        expected.append((name, query - 24))
    batch, frame = data.sample_batch(generator, 16, 4)
    assert frame == 24 and all(len(episode) == 26 for episode in batch.episodes)
    assert [episode.identifier for episode in batch.episodes] == [f"{name}_start_{start}" for name, start in expected]
    assert len({start for _, start in expected}) > 1
    assert any("train_b" in name and start + 24 > 43 for name, start in expected)
    assert data.sampling_inventory[4]["skipped_short_sources"] == 1
    assert data.sampling_inventory[4]["skipped_short_only_takes"] == 1
    batch.cache(20).validate()
    assert batch.observations(frame).head.shape == (16, 21, 9)
    with pytest.raises(ValueError, match="No official sources"):
        data.sample_batch(generator, 1, 1000)
    assert data.sampling_inventory[1000]["skipped_short_sources"] == 3


def test_startup_receives_only_one_head_pose_and_is_independent_of_ground_truth(official_root):
    startup = HeadOnlySpy()
    data = OfficialWindowData(official_root, "train", bootstrap=startup)
    before = data.load(0)
    row = data.motion[data.records[0]["base_seq_name"]]
    row["kp3d"] = row["kp3d"].clone() + 1234
    row["smpl_params"]["betas"] = torch.full((1, 10), 999.0)
    data.cache.clear()
    after = data.load(0)
    assert all(head.shape == (1, 9) for head in startup.calls)
    torch.testing.assert_close(before.teacher.joints, after.teacher.joints)
    torch.testing.assert_close(before.teacher.reference, after.teacher.reference)
    assert torch.equal(before.beta, after.beta)
    assert not torch.equal(before.truth.joints, after.truth.joints)
    assert not torch.equal(before.truth.auxiliary[:, -10:], after.truth.auxiliary[:, -10:])


def test_validation_reference_order_is_verified_and_random_val_sampling_is_forbidden(official_root):
    reference_path = official_root / "uniegomotion/ee_val_gt_for_evaluation.pkl"
    joblib.dump({"wrong_key": {"aria_traj_T": torch.eye(4)[None]}}, reference_path)
    with pytest.raises(ValueError, match="reference keys"):
        OfficialWindowData(official_root, "dev", bootstrap=HeadOnlySpy())
    reference_path.unlink()
    data = OfficialWindowData(official_root, "dev", bootstrap=HeadOnlySpy())
    with pytest.raises(ValueError, match="training split"):
        data.sample_batch(torch.Generator(), 1, 0)
