"""Reproduce offline shape/rotation interventions on one saved causal trace.

GT shape and rotations are diagnostic substitutions only. They never enter
rollout, training cache, action labels, or formal fixed-shape FK metrics.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from dataset.smpl_utils import get_smpl
from egorecover.codec import BodyState, MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.evaluation_protocol import event_window, file_sha256, phase_indices
from egorecover.fk import FixedShapeFK
from egorecover.smpl_evaluation import decode_committed_rollout, prepare_ground_truth
from eval.metrics import reconstruction_error
from utils.rotation_conversions import rotation_6d_to_matrix


def gt_global_rotations(supervision, frames, parents, device):
    params = supervision["smpl_params"]
    root = rotation_6d_to_matrix(torch.as_tensor(params["global_orient"][frames], dtype=torch.float32, device=device))
    body = rotation_6d_to_matrix(torch.as_tensor(params["body_pose"][frames], dtype=torch.float32, device=device))
    local = torch.cat((root[:, None], body), dim=1)
    global_rotations = []
    for joint, parent in enumerate(parents[:22]):
        global_rotations.append(local[:, joint] if int(parent) < 0 else global_rotations[int(parent)] @ local[:, joint])
    return torch.stack(global_rotations, dim=1)


@torch.no_grad()
def project(smpl, state, betas, batch_size):
    result = []
    for start in range(0, len(state.joints), batch_size):
        end = min(start + batch_size, len(state.joints))
        current = BodyState(state.joints[start:end], state.reference[start:end], state.auxiliary[start:end])
        shape = betas[start:end] if betas.ndim == 2 else betas
        result.append(FixedShapeFK(smpl, shape).project(current)[0].cpu())
    return torch.cat(result)


def metrics(joints, gt, dense, parents, frames, fault_window):
    difference = (joints - gt).norm(dim=-1) * 1000
    relative = ((joints - joints[:, :1]) - (gt - gt[:, :1])).norm(dim=-1) * 1000
    disagreement = (joints - dense).norm(dim=-1) * 1000
    bone = (
        torch.stack(
            [
                (joints[:, joint] - joints[:, int(parents[joint])]).norm(dim=-1)
                - (gt[:, joint] - gt[:, int(parents[joint])]).norm(dim=-1)
                for joint in range(1, 22)
            ],
            dim=1,
        ).abs()
        * 1000
    )
    phases = phase_indices(frames, fault_window)
    return {
        "world_mpjpe_mm": float(difference.mean()),
        "root_mpjpe_mm": float(difference[:, 0].mean()),
        "root_relative_mpjpe_mm": float(relative.mean()),
        "pa_mpjpe_mm": float(reconstruction_error(joints.numpy(), gt.numpy()) * 1000),
        "fk_vs_dense_mm": float(disagreement.mean()),
        "bone_length_abs_error_mm": float(bone.mean()),
        "per_joint_world_mpjpe_mm": difference.mean(0).tolist(),
        "per_joint_fk_vs_dense_mm": disagreement.mean(0).tolist(),
        "phase_world_mpjpe_mm": {
            name: float(difference[indices].mean()) if indices else None for name, indices in phases.items()
        },
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("gaussian", "history"), required=True)
    parser.add_argument("--variant", choices=("clean", "freeze_3s", "drift_0p03mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--smplx-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Choose a new output file and positive batch size.")
    torch.set_num_threads(4)
    if args.smplx_dir:
        os.environ["SMPLX_MODEL_PATH"] = str(args.smplx_dir.resolve())
    model_file = (
        Path(os.environ.get("SMPLX_MODEL_PATH", Path(__file__).resolve().parents[1] / "body_models/smplx"))
        / "SMPLX_NEUTRAL.npz"
    )
    report = json.loads((args.rollout / "report.json").read_text())
    trace_path = args.rollout / f"{args.source_mode}_{args.variant}.pt"
    saved = torch.load(trace_path, map_location="cpu", weights_only=True)
    dataset, _ = open_dataset(signal=args.signal)
    record = next(
        r for r in dataset.records if r["base_take_name"] == report["take"] and r["variant_name"] == args.variant
    )
    clean = next(r for r in dataset.records if r["base_take_name"] == report["take"] and r["variant_name"] == "clean")
    supervision = dataset.supervision(clean["variant_id"])
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    device = torch.device(args.device)
    codec = MotionCodec(
        torch.load(stats_path, map_location="cpu", weights_only=False), reference_mode=report["reference_mode"]
    ).to(device)
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    frames = saved["frame_indices"]
    gt = prepare_ground_truth(smpl, supervision, frames, batch_size=args.batch_size)
    state, fixed_beta, _ = decode_committed_rollout(codec, saved)
    gt_rotations = gt_global_rotations(supervision, frames, smpl.parents, device)
    predicted_rotations = state.joints[..., :3, :3]
    local_deltas = []
    for joint, parent in enumerate(smpl.parents[:22]):
        predicted_local = (
            predicted_rotations[:, joint]
            if int(parent) < 0
            else predicted_rotations[:, int(parent)].transpose(-1, -2) @ predicted_rotations[:, joint]
        )
        gt_local = (
            gt_rotations[:, joint]
            if int(parent) < 0
            else gt_rotations[:, int(parent)].transpose(-1, -2) @ gt_rotations[:, joint]
        )
        trace = (predicted_local.transpose(-1, -2) @ gt_local).diagonal(dim1=-2, dim2=-1).sum(-1)
        local_deltas.append(torch.acos(((trace - 1) / 2).clamp(-1, 1)) * (180 / torch.pi))
    gt_state_joints = state.joints.clone()
    gt_state_joints[..., :3, :3] = gt_rotations
    gt_rotation_state = BodyState(gt_state_joints, state.reference, state.auxiliary)
    gt_beta = torch.as_tensor(supervision["smpl_params"]["betas"], dtype=torch.float32, device=device).reshape(-1, 10)[
        0
    ]
    interventions = {
        "predicted_rotation_fixed_bootstrap_beta": (state, fixed_beta),
        "predicted_rotation_per_frame_beta": (state, state.auxiliary[:, -10:]),
        "predicted_rotation_gt_beta_diagnostic_only": (state, gt_beta),
        "gt_rotation_fixed_bootstrap_beta_diagnostic_only": (gt_rotation_state, fixed_beta),
        "gt_rotation_gt_beta_diagnostic_only": (gt_rotation_state, gt_beta),
    }
    gt_body = gt["joints"][:, :22]
    dense = state.joints[..., :3, 3].cpu()
    window = event_window(record)
    result = {
        "scope": "offline_intervention_diagnostic_only",
        "source_rollout_scope": report.get("scope"),
        "bootstrap_is_trained": report.get("bootstrap_is_trained"),
        "take": report["take"],
        "source_mode": args.source_mode,
        "variant": args.variant,
        "frame_indices": frames,
        "fault_window": list(window) if window else None,
        "trace_sha256": file_sha256(trace_path),
        "rollout_report_sha256": file_sha256(args.rollout / "report.json"),
        "stats_sha256": file_sha256(stats_path),
        "smplx_asset_sha256": file_sha256(model_file),
        "gt_asset_audit": gt["asset_audit"],
        "fixed_bootstrap_beta": fixed_beta.cpu().tolist(),
        "gt_beta_diagnostic_only": gt_beta.cpu().tolist(),
        "mean_per_frame_predicted_beta": state.auxiliary[:, -10:].mean(0).cpu().tolist(),
        "mean_local_rotation_error_deg": torch.stack(local_deltas, dim=1).mean().item(),
        "per_joint_local_rotation_error_deg": torch.stack(local_deltas, dim=1).mean(0).tolist(),
        "floor_estimate_minus_gt_m": float(saved["floor_estimate_m"]) - gt["floor_height_m"],
        "direct_dense22_world_mpjpe_mm": float((dense - gt_body).norm(dim=-1).mean() * 1000),
        "interventions": {},
    }
    for name, (body, beta) in interventions.items():
        joints = project(smpl, body, beta, args.batch_size)
        result["interventions"][name] = metrics(joints, gt_body, dense, smpl.parents, frames, window)
        print(name, result["interventions"][name]["world_mpjpe_mm"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
