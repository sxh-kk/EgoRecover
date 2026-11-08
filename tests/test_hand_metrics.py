import pytest
import torch

from scripts.hand_metrics import metrics_from_full_prediction


def test_finger_metric_uses_30_fingers_and_world_translation_in_mm():
    truth = torch.zeros(2, 55, 3)
    prediction = truth.clone()
    prediction[:, 25:55, 0] = 0.1
    prediction[:, [20, 21], 0] = 100
    prediction[:, 23, 1] = 0.02
    vertices = torch.zeros(2, 5, 3)
    vertices[0, :, 2] = 0.01
    vertices[1, :, 2] = -0.03
    result = metrics_from_full_prediction(prediction, truth, vertices, floor_calibrated=True)
    assert result["mpjpe_hand_mm"] == pytest.approx(100)
    assert result["head_translation_leye_mm"] == pytest.approx(20)
    assert result["foot_contact_mesh_mm"] == pytest.approx(20)


def test_unknown_floor_excludes_contact_without_excluding_hand_error():
    joints, vertices = torch.zeros(2, 55, 3), torch.zeros(2, 5, 3)
    result = metrics_from_full_prediction(joints, joints, vertices, floor_calibrated=False)
    assert result["foot_contact_mesh_mm"] is None
    assert result["mpjpe_hand_mm"] == 0
