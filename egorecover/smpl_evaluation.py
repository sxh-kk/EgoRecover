"""Offline UniEgoMotion-style geometry evaluation of saved causal rollouts.

Predictions are reconstructed from committed model output, never from GT body
state. Supervision and the licensed SMPL-X layer enter only after inference.
This covers the paper's geometric reconstruction metrics, not TMR/FID or its
official 80-frame sampling protocol.
"""

import math

import torch

from egorecover.codec import BodyState, transform_from_9d
from egorecover.evaluation_protocol import phase_indices
from egorecover.fk import FixedShapeFK
from eval.metrics import (
    compute_foot_sliding_for_smpl,
    get_air_time,
    get_contact_validity,
    get_foot_penetration,
    reconstruction_error,
)
from utils.pca_conversions import pca_to_matrix
from utils.rotation_conversions import rotation_6d_to_matrix


def _tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def decode_committed_rollout(codec, saved):
    """Restore world 22-joint transforms from saved physical commits.

    The recorded planar reference is authoritative for each frame, so the
    first post-bootstrap frame needs no hidden/bootstrap GT reference.
    """
    if saved.get("body_history_source") != "model_predictions_only":
        raise ValueError("SMPL evaluation requires a prediction-only rollout.")
    if saved.get("reference_mode") != codec.reference_mode:
        raise ValueError("Rollout and codec reference conventions differ.")
    device = codec.motion_mean.device
    motion = _tensor(saved["committed_motion"], device)
    reference = _tensor(saved["references"], device)
    dense = _tensor(saved["dense_world_joints"], device)
    beta_boot = _tensor(saved["beta_boot"], device)
    frames = list(saved["frame_indices"])
    count = len(frames)
    if (
        count < 2
        or motion.shape != (count, 243)
        or reference.shape != (count, 4, 4)
        or dense.shape != (count, 22, 3)
        or beta_boot.shape != (10,)
        or frames != list(range(saved["bootstrap_frames"], saved["bootstrap_frames"] + count))
    ):
        raise ValueError("Saved rollout fields have inconsistent shape or time indices.")
    floor = float(saved["floor_estimate_m"])
    if not math.isfinite(floor) or not all(
        bool(torch.isfinite(x).all()) for x in (motion, reference, dense, beta_boot)
    ):
        raise ValueError("Saved rollout contains nonfinite values.")
    if not torch.allclose(codec.canonical_reference(reference), reference, atol=1e-5, rtol=0):
        raise ValueError("Saved references violate the codec convention.")
    raw = codec.denormalize(motion)
    local_joints = transform_from_9d(raw[:, :198].reshape(count, 22, 9))
    world_joints = reference[:, None] @ local_joints
    world_joints = world_joints.clone()
    world_joints[..., 2, 3] += floor
    mismatch = float((world_joints[..., :3, 3] - dense).abs().max())
    if mismatch > 1e-3:
        raise ValueError(f"Committed motion disagrees with saved world joints by {mismatch:.6g} m.")
    return BodyState(world_joints, reference, raw[:, 207:]), beta_boot, mismatch


@torch.no_grad()
def prepare_ground_truth(smpl, supervision, frame_indices, *, batch_size=32, audit_tolerance_mm=5.0):
    """Regenerate GT with this exact SMPL-X asset and audit source kp3d.

    The audit stops an incompatible model version, joint order or coordinate
    convention from silently producing plausible-looking prediction metrics.
    """
    if batch_size < 1 or audit_tolerance_mm <= 0:
        raise ValueError("Batch size and GT audit tolerance must be positive.")
    device = smpl.shapedirs.device
    params = supervision["smpl_params"]
    total = len(params["global_orient"])
    frames = torch.as_tensor(frame_indices, dtype=torch.long)
    if frames.ndim != 1 or len(frames) < 2 or int(frames.min()) < 0 or int(frames.max()) >= total:
        raise ValueError("GT frame indices are outside the source sequence.")
    select = frames.tolist()
    for name in ("betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"):
        if not bool(torch.isfinite(_tensor(params[name], "cpu")).all()):
            raise ValueError(f"Nonfinite EE4D GT SMPL-X field: {name}.")
    if not math.isfinite(float(supervision["floor_height"])):
        raise ValueError("Nonfinite EE4D GT floor height.")
    betas = _tensor(params["betas"], device).reshape(-1, 10)
    if len(betas) != 1:
        raise ValueError("This EE4D protocol expects one source shape per take.")
    kwargs = {
        "global_orient": rotation_6d_to_matrix(_tensor(params["global_orient"][select], device)),
        "body_pose": rotation_6d_to_matrix(_tensor(params["body_pose"][select], device)),
        "left_hand_pose": pca_to_matrix(_tensor(params["left_hand_pose"][select], device), smpl.left_hand_components),
        "right_hand_pose": pca_to_matrix(
            _tensor(params["right_hand_pose"][select], device), smpl.right_hand_components
        ),
        "betas": betas.expand(len(frames), -1),
        "transl": _tensor(params["transl"][select], device),
    }
    if kwargs["global_orient"].shape != (len(frames), 3, 3) or kwargs["body_pose"].shape != (len(frames), 21, 3, 3):
        raise ValueError("GT SMPL-X pose fields do not match the EE4D body layout.")
    local = torch.cat((kwargs["global_orient"][:, None], kwargs["body_pose"]), dim=1)
    global_rotations = []
    for joint, parent in enumerate(smpl.parents[:22]):
        parent = int(parent)
        global_rotations.append(local[:, joint] if parent < 0 else global_rotations[parent] @ local[:, joint])
    joints, vertices = [], []
    for start in range(0, len(frames), batch_size):
        end = min(start + batch_size, len(frames))
        output = smpl(**{key: value[start:end] for key, value in kwargs.items()}, return_verts=True)
        if output.joints.shape[1] < 55 or output.vertices is None:
            raise ValueError("SMPL-X must expose 55 body/hand joints and vertices.")
        if not bool(torch.isfinite(output.joints).all()) or not bool(torch.isfinite(output.vertices).all()):
            raise ValueError("SMPL-X generated nonfinite GT geometry.")
        joints.append(output.joints[:, :55].cpu())
        vertices.append(output.vertices.cpu())
    joints = torch.cat(joints)
    vertices = torch.cat(vertices)
    recorded = _tensor(supervision["kp3d"][select, :55], "cpu")
    if recorded.shape != joints.shape:
        raise ValueError("EE4D GT must contain the first 55 SMPL-X body/hand joints.")
    if not bool(torch.isfinite(recorded).all()):
        raise ValueError("Nonfinite EE4D GT recorded joints.")
    difference = (joints - recorded).norm(dim=-1)
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("Nonfinite SMPL-X asset audit differences.")
    audit = {
        "mean_mm": float(difference.mean() * 1000),
        "max_mm": float(difference.max() * 1000),
        "body_mean_mm": float(difference[:, :22].mean() * 1000),
        "body_max_mm": float(difference[:, :22].max() * 1000),
        "hands_mean_mm": float(difference[:, 25:55].mean() * 1000),
        "hands_max_mm": float(difference[:, 25:55].max() * 1000),
    }
    if audit["mean_mm"] > audit_tolerance_mm or audit["max_mm"] > 4 * audit_tolerance_mm:
        raise ValueError(
            "This SMPL-X asset/coordinate convention does not reproduce EE4D GT body/hand joints: "
            f"mean={audit['mean_mm']:.3f} mm, max={audit['max_mm']:.3f} mm."
        )
    if not all(math.isfinite(value) for value in audit.values()):
        raise ValueError("Nonfinite SMPL-X asset audit statistics.")
    return {
        "frame_indices": select,
        "joints": joints,
        "vertices": vertices,
        "head_rotation": global_rotations[15].cpu(),
        "recorded_body": recorded[:, :22],
        "floor_height_m": float(supervision["floor_height"]),
        "asset_audit": audit,
    }


def _paper_geometry_metrics(
    pred_joints, gt_joints, pred_vertices, gt_vertices, pred_head_rotation, gt_head_rotation, floor
):
    """Match the geometric definitions in UniEgoMotion's eval.metrics.py."""
    if len(pred_joints) < 2:
        raise ValueError("Foot sliding requires at least two consecutive frames.")
    pred_body, gt_body = pred_joints[:, :22], gt_joints[:, :22]
    rotation_delta = pred_head_rotation.transpose(-1, -2) @ gt_head_rotation - torch.eye(3)
    result = {
        "mpjpe_all55_m": float((pred_joints - gt_joints).norm(dim=-1).mean()),
        "mpjpe_all55_pa_m": float(reconstruction_error(pred_joints.numpy(), gt_joints.numpy())),
        "mpjpe_body_m": float((pred_body - gt_body).norm(dim=-1).mean()),
        "mpjpe_body_root_relative_m": float(
            ((pred_body - pred_body[:, :1]) - (gt_body - gt_body[:, :1])).norm(dim=-1).mean()
        ),
        "mpjpe_body_pa_m": float(reconstruction_error(pred_body.numpy(), gt_body.numpy())),
        "mpjpe_hands_m": float((pred_joints[:, 25:55] - gt_joints[:, 25:55]).norm(dim=-1).mean()),
        "mpjpe_hands_pa_m": float(reconstruction_error(pred_joints[:, 25:55].numpy(), gt_joints[:, 25:55].numpy())),
        "head_rotation_fro": float(torch.linalg.matrix_norm(rotation_delta).mean()),
        "head_translation_m": float((pred_joints[:, 23] - gt_joints[:, 23]).norm(dim=-1).mean()),
        "root_translation_m": float((pred_joints[:, 0] - gt_joints[:, 0]).norm(dim=-1).mean()),
        "foot_slide_mm": float(compute_foot_sliding_for_smpl(pred_body.numpy(), floor)),
        "foot_slide_gt_mm": float(compute_foot_sliding_for_smpl(gt_body.numpy(), floor)),
        "foot_penetration_m": float(get_foot_penetration(pred_vertices, floor)),
        "foot_penetration_gt_m": float(get_foot_penetration(gt_vertices, floor)),
        "air_time_fraction": float(get_air_time(pred_vertices, floor)),
        "air_time_gt_fraction": float(get_air_time(gt_vertices, floor)),
        "foot_contact_m": float(get_contact_validity(pred_vertices, floor)),
        "foot_contact_gt_m": float(get_contact_validity(gt_vertices, floor)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite SMPL-X evaluation metric.")
    return result


@torch.no_grad()
def evaluate_saved_case(smpl, codec, saved, ground_truth, *, fault_window, batch_size=32):
    """Evaluate an already completed causal run; no online model is invoked."""
    state, beta_boot, dense_roundtrip = decode_committed_rollout(codec, saved)
    frames = list(saved["frame_indices"])
    if frames != ground_truth["frame_indices"]:
        raise ValueError("Prediction and audited GT frame indices differ.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    fk = FixedShapeFK(smpl, beta_boot)
    joints, vertices = [], []
    for start in range(0, len(frames), batch_size):
        end = min(start + batch_size, len(frames))
        part = BodyState(state.joints[start:end], state.reference[start:end], state.auxiliary[start:end])
        output_joints, output_vertices = fk.project(part, joint_count=55, return_verts=True)
        joints.append(output_joints.cpu())
        vertices.append(output_vertices.cpu())
    joints = torch.cat(joints)
    vertices = torch.cat(vertices)
    pred_head = state.joints[:, 15, :3, :3].cpu()
    gt_joints = ground_truth["joints"]
    gt_vertices = ground_truth["vertices"]
    gt_head = ground_truth["head_rotation"]
    floor = ground_truth["floor_height_m"]
    overall = _paper_geometry_metrics(joints, gt_joints, vertices, gt_vertices, pred_head, gt_head, floor)
    sections = phase_indices(frames, fault_window)
    phases = {
        name: (
            _paper_geometry_metrics(
                joints[items],
                gt_joints[items],
                vertices[items],
                gt_vertices[items],
                pred_head[items],
                gt_head[items],
                floor,
            )
            if len(items) >= 2
            else None
        )
        for name, items in sections.items()
    }
    dense_error = (state.joints[..., :3, 3].cpu() - ground_truth["recorded_body"]).norm(dim=-1).mean() * 1000
    per_joint_error = (joints[:, :22] - gt_joints[:, :22]).norm(dim=-1) * 1000
    direct_error = (state.joints[..., :3, 3].cpu() - ground_truth["recorded_body"]).norm(dim=-1) * 1000
    if not bool(torch.isfinite(dense_error)) or not bool(torch.isfinite(per_joint_error).all()):
        raise ValueError("Nonfinite SMPL-X per-frame or direct-body errors.")
    return {
        "frames": len(frames),
        "metrics": overall,
        "phase_metrics": phases,
        "per_frame_smpl22_mm": per_joint_error.mean(dim=-1).tolist(),
        "per_frame_dense22_mm": direct_error.mean(dim=-1).tolist(),
        "per_joint_smpl22_mean_mm": per_joint_error.mean(dim=0).tolist(),
        "diagnostics": {
            "dense22_mm_recomputed": float(dense_error),
            "committed_dense_max_abs_m": dense_roundtrip,
            "fk_vs_dense_body_mm": float((joints[:, :22] - state.joints[..., :3, 3].cpu()).norm(dim=-1).mean() * 1000),
            "fk_vs_dense_root_mm": float((joints[:, 0] - state.joints[:, 0, :3, 3].cpu()).norm(dim=-1).mean() * 1000),
            "floor_estimate_minus_annotation_m": float(saved["floor_estimate_m"]) - floor,
        },
    }
