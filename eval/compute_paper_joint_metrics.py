#!/usr/bin/env python3
"""Compute the joint metrics and time windows used by the UniEgoMotion paper."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

from dataset.smpl_utils import evaluate_smpl, get_smpl
from eval.metrics import reconstruction_error
from utils.torch_utils import to_device


PAPER_WINDOWS = {
    "recon": (0, None),       # complete reconstruction clip
    "gen": (0, 20),          # first 2 s at 10 fps
    "fore": (20, 40),        # first 2 s after the 2 s context
}


def _mean_joint_error(pred, gt):
    return torch.linalg.vector_norm(pred - gt, dim=-1).mean().item()


def _pa_error(pred, gt):
    return reconstruction_error(pred.numpy(), gt.numpy()).item()


def compute_task_metrics(smpl, gt, preds, task):
    start, fixed_end = PAPER_WINDOWS[task]
    all_metrics = {
        "mpjpe_body": [],
        "mpjpe_body_pa": [],
        "mpjpe_hand": [],
        "mpjpe_hand_pa": [],
        "root_trans_error": [],
        "head_trans_error": [],
    }

    for key in tqdm(list(preds), desc=task):
        gt_params = gt[key]["smpl_params"]
        pred_params = preds[key]["smpl_params"]
        num_frames = len(gt_params["global_orient"])
        end = num_frames if fixed_end is None else min(fixed_end, num_frames)
        if end <= start:
            continue

        gt_slice = {name: value[start:end] for name, value in gt_params.items()}
        pred_slice = {name: value[start:end] for name, value in pred_params.items()}
        if any(len(value) != end - start for value in pred_slice.values()):
            raise ValueError(f"Prediction {key} is shorter than paper window [{start}, {end})")

        gt_joints = evaluate_smpl(smpl, gt_slice, return_joints_only=True).cpu()
        pred_joints = evaluate_smpl(smpl, pred_slice, return_joints_only=True).cpu()
        gt_body, pred_body = gt_joints[:, :22], pred_joints[:, :22]
        gt_hand, pred_hand = gt_joints[:, 25:55], pred_joints[:, 25:55]

        all_metrics["mpjpe_body"].append(_mean_joint_error(pred_body, gt_body))
        all_metrics["mpjpe_body_pa"].append(_pa_error(pred_body, gt_body))
        all_metrics["mpjpe_hand"].append(_mean_joint_error(pred_hand, gt_hand))
        all_metrics["mpjpe_hand_pa"].append(_pa_error(pred_hand, gt_hand))
        all_metrics["root_trans_error"].append(_mean_joint_error(pred_joints[:, :1], gt_joints[:, :1]))
        all_metrics["head_trans_error"].append(_mean_joint_error(pred_joints[:, 23:24], gt_joints[:, 23:24]))

    return {name: np.asarray(values) for name, values in all_metrics.items()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_path = Path(args.exp_path)
    suffix = args.eval_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    output = exp_path / f"metrics_paper_protocol{suffix}.pkl"
    if output.exists() and not args.overwrite:
        print(f"Metrics already computed at {output}")
        return

    smpl = get_smpl().to(device)
    gt = joblib.load(Path(args.data_dir) / "uniegomotion/ee_val_gt_for_evaluation.pkl")
    gt = to_device(gt, device)
    results = {}
    for task in PAPER_WINDOWS:
        pred_path = exp_path / f"preds_ee4d_{task}{suffix}.pkl"
        preds = to_device(joblib.load(pred_path), device)
        results[task] = compute_task_metrics(smpl, gt, preds, task)
        del preds
    joblib.dump(results, output)
    print(f"Saved paper-protocol joint metrics to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", required=True)
    parser.add_argument("--eval-suffix", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
