"""Supplementary UniEgoMotion-style SMPL-X metrics; no model or loss changes."""

import torch

from egorecover.rotations import pca_to_matrix, rotation_6d_to_matrix
from egorecover.state import state_cat, state_map


class EmissionRecorder:
    """Record first emissions while delegating the unchanged online system."""

    def __init__(self, system):
        self.system = system
        self.states = []

    def __getattr__(self, name):
        return getattr(self.system, name)

    def online(self, *args, **kwargs):
        result = self.system.online(*args, **kwargs)
        self.states.append(state_map(result.current, lambda value: value.detach().clone()))
        return result


@torch.no_grad()
def decode_emissions(smpl, states, beta):
    """Decode original emitted rotations/PCA with the existing fixed shape."""
    state = state_cat(states, dim=0)
    rotation = state.joints[..., :3, :3]
    parents = [int(parent) for parent in smpl.parents[:22]]
    local = torch.stack(
        [rotation[:, j] if p < 0 else rotation[:, p].transpose(-1, -2) @ rotation[:, j] for j, p in enumerate(parents)],
        dim=1,
    )
    betas = beta.reshape(1, 10).expand(len(rotation), -1)
    offset = smpl(betas=betas[:1], return_verts=False).joints[:, 0]
    return smpl(
        global_orient=local[:, 0],
        body_pose=local[:, 1:22],
        betas=betas,
        transl=state.joints[:, 0, :3, 3] - offset,
        left_hand_pose=pca_to_matrix(state.auxiliary[:, :12], smpl.left_hand_components),
        right_hand_pose=pca_to_matrix(state.auxiliary[:, 12:24], smpl.right_hand_components),
        return_verts=True,
    )


@torch.no_grad()
def decode_official_truth(smpl, data, record):
    """Labels alone use original GT shape/translation, after reconstruction."""
    row = data.motion[record["source"]]
    params = row["smpl_params"]
    interval = slice(record["start"] + 20, record["start"] + 80)
    device = smpl.v_template.device
    root = rotation_6d_to_matrix(params["global_orient"][interval].to(device).float())
    body = rotation_6d_to_matrix(params["body_pose"][interval].to(device).float())
    translation = params["transl"][interval].to(device).float().clone()
    translation[:, 2] -= data.source_metadata[record["source"]]["floor"]
    beta = params["betas"].to(device).float().reshape(1, 10).expand(len(root), -1)
    return smpl(
        global_orient=root,
        body_pose=body,
        betas=beta,
        transl=translation,
        left_hand_pose=pca_to_matrix(params["left_hand_pose"][interval].to(device).float(), smpl.left_hand_components),
        right_hand_pose=pca_to_matrix(
            params["right_hand_pose"][interval].to(device).float(), smpl.right_hand_components
        ),
        return_verts=False,
    )


def metrics_from_full_prediction(predicted, truth, vertices, *, floor_calibrated):
    """World-frame errors: fingers25:55, left eye23; all lengths in mm."""
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[1] < 55:
        raise ValueError("Matching [T,J>=55,3] SMPL-X joints are required.")
    if not all(bool(torch.isfinite(value).all()) for value in (predicted, truth, vertices)):
        raise ValueError("Nonfinite SMPL-X output.")
    error = (predicted - truth).norm(dim=-1)
    return {
        "mpjpe_hand_mm": float(error[:, 25:55].mean() * 1000),
        "head_translation_leye_mm": float(error[:, 23].mean() * 1000),
        "foot_contact_mesh_mm": (
            float(vertices[..., 2].min(dim=-1).values.abs().mean() * 1000) if floor_calibrated else None
        ),
    }


@torch.no_grad()
def supplementary_metrics(context, record, episode, recorder, body_predictions):
    prediction = decode_emissions(context.smpl, recorder.states, episode.beta)
    truth = decode_official_truth(context.smpl, context.data, record)
    # Extra evaluation must decode the same fixed-shape body that was scored.
    torch.testing.assert_close(prediction.joints[:, :22].cpu(), body_predictions, atol=2e-5, rtol=1e-5)
    result = metrics_from_full_prediction(
        prediction.joints, truth.joints, prediction.vertices, floor_calibrated=episode.floor_calibrated
    )
    return result, prediction.joints[:, 25:55].cpu(), truth.joints[:, 25:55].cpu()
