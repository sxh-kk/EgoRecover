import copy

import pytest
import torch

from motionstreamer_flow.data import Episodes, Normalizer, online_inputs, save_json
from motionstreamer_flow.demo import make_dataset, tiny_config
from motionstreamer_flow.export_egorecover import convert_packet
from motionstreamer_flow.geometry import axis_angle_to_matrix, fk, rotation_6d_to_matrix, to_world
from motionstreamer_flow.model import CausalTAE, MotionFlow
from motionstreamer_flow.streaming import rollout


@pytest.fixture
def source_packet(tmp_path):
    cfg = tiny_config()
    data = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    p = data.load(data.split("train")[0])
    joints, rotations = fk(p["pose"], p["root"], p["offsets"])
    transforms = torch.eye(4).expand(len(joints), 22, 4, 4).clone()
    transforms[..., :3, :3], transforms[..., :3, 3] = rotations, joints
    body = {
        "joints": transforms,
        "reference": torch.eye(4).expand(len(joints), 4, 4).clone(),
        "auxiliary": torch.cat((p["pose"][:, 126:], torch.zeros(len(joints), 10)), -1),
    }
    packet = {
        "truth": {**body, "reference": torch.full_like(body["reference"], float("nan"))},
        "teacher": body,
        "beta_boot": p["beta"],
        "images": torch.nan_to_num(p["images"]),
        "head": p["root"],
        "record": {"floor_height": 2.75},
    }
    return cfg, p, packet


def test_conversion_restores_height_once_with_rotated_translated_anchor(source_packet):
    cfg, p, packet = source_packet
    anchor = torch.eye(4)
    anchor[:3, :3] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 0.7]))
    anchor[:2, 3] = torch.tensor([1.3, -0.8])
    packet["teacher"]["reference"][:] = anchor
    converted = convert_packet(packet, cfg, p["offsets"])
    joints = packet["truth"]["joints"][..., :3, 3]
    torch.testing.assert_close(to_world(converted["joints"], converted["anchor"]), joints)
    expected = joints + torch.tensor([0.0, 0.0, 2.75])
    torch.testing.assert_close(to_world(converted["joints"], converted["original_world_from_anchor"]), expected)
    world_rotation = converted["original_world_from_anchor"][:3, :3] @ rotation_6d_to_matrix(converted["root"][:, :6])
    torch.testing.assert_close(world_rotation, packet["truth"]["joints"][:, 0, :3, :3])
    torch.testing.assert_close(converted["original_world_from_anchor"][:3, :3], anchor[:3, :3])


def test_restoration_changes_outputs_without_changing_generation(source_packet):
    cfg, p, packet = source_packet
    torch.set_num_threads(2)
    torch.manual_seed(62)
    flow, tae, norm = MotionFlow(cfg).eval(), CausalTAE(cfg).eval(), Normalizer(cfg)
    converted = convert_packet(packet, cfg, p["offsets"])
    other = copy.deepcopy(converted)
    other["original_world_from_anchor"][2, 3] += 1.5
    first = rollout(flow, tae, norm, online_inputs(converted), seed=62)
    second = rollout(flow, tae, norm, online_inputs(other), seed=62)
    for name in ("state", "pose", "root", "joints", "root_floor_world", "joints_floor_world"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    torch.testing.assert_close(
        second["joints_world"] - first["joints_world"], torch.tensor([0.0, 0.0, 1.5]).expand_as(first["joints_world"])
    )
    torch.testing.assert_close(
        second["root_world"][:, 6:] - first["root_world"][:, 6:],
        torch.tensor([0.0, 0.0, 1.5]).expand_as(first["root_world"][:, 6:]),
    )


@pytest.mark.parametrize("floor", [None, float("nan"), float("inf")])
def test_conversion_rejects_unknown_world_restoration(source_packet, floor):
    cfg, p, packet = source_packet
    packet["record"]["floor_height"] = floor
    with pytest.raises(ValueError, match="floor_height"):
        convert_packet(packet, cfg, p["offsets"])


@pytest.mark.parametrize("bad_transform", ["reflection", "scale", "bottom_row"])
def test_invalid_restoration_transform_rejected_before_generation(tmp_path, bad_transform):
    cfg = tiny_config()
    data = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    p = data.load(data.split("train")[0])
    transform = p["original_world_from_anchor"]
    if bad_transform == "reflection":
        transform[0, 0] = -1
    elif bad_transform == "scale":
        transform[0, 0] = 2
    else:
        transform[3, 0] = 1
    with pytest.raises(ValueError, match="rigid transform"):
        rollout(MotionFlow(cfg), CausalTAE(cfg), Normalizer(cfg), online_inputs(p))


def test_legacy_dataset_needs_explicit_restoration_migration(tmp_path):
    cfg = tiny_config()
    path = make_dataset(tmp_path / "data", cfg)
    import json

    manifest = json.loads(path.read_text())
    manifest["version"] = 1
    save_json(path, manifest)
    with pytest.raises(ValueError, match="re-export"):
        Episodes(path, cfg)
