"""SMPL-X evaluator contract tests with an explicit synthetic layer.

These verify plumbing and GT isolation, not accuracy of a licensed model.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.smpl_evaluation import decode_committed_rollout, evaluate_saved_case, prepare_ground_truth
from utils.rotation_conversions import matrix_to_rotation_6d


class SyntheticLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("parents", torch.tensor([-1] + [0] * 54))
        self.register_buffer("left_hand_components", torch.eye(45)[:12])
        self.register_buffer("right_hand_components", torch.eye(45)[:12])
        self.register_buffer("shapedirs", torch.zeros(1))
        template = torch.stack([torch.tensor([0.02 * i, 0.01 * (i % 5), 0.03 * (i % 9)]) for i in range(55)])
        self.register_buffer("template", template)

    def forward(self, *, betas, transl=None, return_verts=True, **kwargs):
        count = len(betas)
        root = betas[:, :3]
        if transl is not None:
            root = root + transl
        joints = self.template[None] + root[:, None]
        vertices = joints[:, :4].clone() if return_verts else None
        return SimpleNamespace(joints=joints, vertices=vertices)


def fixture_case():
    smpl = SyntheticLayer()
    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
    }
    codec = MotionCodec(stats)
    count = 3
    frames = [20, 21, 22]
    identity = torch.eye(3)
    pose6d = matrix_to_rotation_6d(identity)
    transl = torch.stack([torch.tensor([0.05 * i, 0.0, 0.0]) for i in range(23)])
    beta = torch.zeros(1, 10)
    gt_joints = smpl.template[None] + transl[:, None]
    supervision = {
        "smpl_params": {
            "betas": beta.numpy(),
            "transl": transl.numpy(),
            "global_orient": pose6d.expand(23, -1).numpy(),
            "body_pose": pose6d.expand(23, 21, -1).numpy(),
            "left_hand_pose": np.zeros((23, 12), dtype=np.float32),
            "right_hand_pose": np.zeros((23, 12), dtype=np.float32),
        },
        "kp3d": gt_joints.numpy(),
        "floor_height": np.float32(0),
    }
    joints = torch.eye(4).expand(count, 22, 4, 4).clone()
    joints[..., :3, 3] = gt_joints[frames, :22] + torch.tensor([0.1, 0.0, 0.0])
    reference = torch.eye(4).expand(count, 4, 4).clone()
    state = BodyState(joints, reference, torch.zeros(count, 36))
    committed = codec.encode_current(state, reference)
    saved = {
        "body_history_source": "model_predictions_only",
        "reference_mode": "planar",
        "bootstrap_frames": 20,
        "frame_indices": frames,
        "committed_motion": committed,
        "references": reference,
        "dense_world_joints": joints[..., :3, 3],
        "beta_boot": torch.zeros(10),
        "floor_estimate_m": 0,
    }
    return smpl, codec, saved, supervision


def test_saved_predictions_flow_through_fk_and_paper_geometry_metrics():
    smpl, codec, saved, supervision = fixture_case()
    gt = prepare_ground_truth(smpl, supervision, saved["frame_indices"], batch_size=2)
    output = evaluate_saved_case(smpl, codec, saved, gt, fault_window=(21, 22), batch_size=2)
    assert output["frames"] == 3
    assert output["diagnostics"]["committed_dense_max_abs_m"] < 1e-5
    assert output["diagnostics"]["fk_vs_dense_body_mm"] < 0.01
    assert output["diagnostics"]["fk_vs_dense_root_mm"] < 0.01
    assert gt["asset_audit"]["hands_max_mm"] < 0.01
    assert output["metrics"]["mpjpe_body_m"] == pytest.approx(0.1, abs=1e-6)
    assert output["metrics"]["mpjpe_hands_m"] == pytest.approx(0.1, abs=1e-6)
    assert output["metrics"]["mpjpe_body_pa_m"] < 1e-5
    assert output["phase_metrics"]["pre_fault"] is None
    assert len(output["per_frame_smpl22_mm"]) == 3


def test_incompatible_gt_asset_or_tampered_prediction_is_rejected():
    smpl, codec, saved, supervision = fixture_case()
    supervision["kp3d"][:, 25:55] += 0.1  # A wrong hand PCA basis must fail even when body joints match.
    with pytest.raises(ValueError, match="does not reproduce EE4D GT"):
        prepare_ground_truth(smpl, supervision, saved["frame_indices"])
    saved["dense_world_joints"] = saved["dense_world_joints"] + 0.1
    with pytest.raises(ValueError, match="disagrees with saved world joints"):
        decode_committed_rollout(codec, saved)


def test_nonfinite_recorded_gt_is_rejected_before_asset_audit():
    smpl, _, saved, supervision = fixture_case()
    supervision["kp3d"][20, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite EE4D GT recorded joints"):
        prepare_ground_truth(smpl, supervision, saved["frame_indices"])


def test_nonfinite_raw_smpl_gt_is_rejected_before_projection():
    smpl, _, saved, supervision = fixture_case()
    supervision["smpl_params"]["body_pose"] = supervision["smpl_params"]["body_pose"].copy()
    supervision["smpl_params"]["body_pose"][20, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite EE4D GT SMPL-X field: body_pose"):
        prepare_ground_truth(smpl, supervision, saved["frame_indices"])
