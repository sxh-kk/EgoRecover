"""Measure coordinate-gauge burden without changing any physical body target."""

import argparse
from pathlib import Path

import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.fk import FixedShapeFK, load_smpl
from egorecover.io import file_sha256, save_json
from egorecover.models import HistoryPrior
from egorecover.prior_data import predictor_forward, select_examples


def reencode_reference(codec, state, reference, previous_reference):
    return codec.encode_current(BodyState(state.joints, reference, state.auxiliary), previous_reference)


def losses(codes, target):
    error = (codes - target).double().square()
    return {
        "code_loss": (error.sum(-1) + 7 * error[:, 198:207].sum(-1)) / 306,
        "reference9_contribution": 8 * error[:, 198:207].sum(-1) / 306,
        "body6d_MSE": error[:, :198].reshape(-1, 22, 9)[:, :, :6].mean((-1, -2)),
        "dense_xyz_MSE": error[:, :198].reshape(-1, 22, 9)[:, :, 6:].mean((-1, -2)),
    }


@torch.no_grad()
def diagnose(args):
    torch.set_num_threads(2)
    root = args.root
    bank = torch.load(root / "banks/D128/bank.pt", map_location="cpu", mmap=True, weights_only=True)
    saved = torch.load(root / "P01_W3G10k_s62/last.pt", map_location="cpu", weights_only=True)
    model = HistoryPrior(**saved["model_config"]).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(saved["state_dict"])
    codec = MotionCodec(bank["stats"]).to(args.device)
    smpl = load_smpl(args.device)
    fk = FixedShapeFK(smpl, torch.zeros(10, device=args.device), shape_mode="predicted")
    values = {}
    equivalence = []
    for start in range(0, len(bank["rows"]), 32):
        e = select_examples(bank["tensors"], list(range(start, min(start + 32, len(bank["rows"])))), args.device)
        pred = predictor_forward(model, e)[:, 0]
        base = e["base"][:, 0]
        target = e["target"][:, 0]
        cv = codec.decode_current(base, e["reference"])
        p = codec.decode_current(pred, e["reference"])
        gt = codec.decode_current(target, e["reference"])
        common_target = reencode_reference(codec, gt, cv.reference, e["reference"])
        common_cv = reencode_reference(codec, cv, cv.reference, e["reference"])
        common_p = reencode_reference(codec, p, cv.reference, e["reference"])
        projected_p = reencode_reference(codec, p, p.reference, e["reference"])
        projected_cv = reencode_reference(codec, cv, cv.reference, e["reference"])
        same_body_P_reference = reencode_reference(codec, cv, p.reference, e["reference"])
        same_body_GT_reference = reencode_reference(codec, cv, gt.reference, e["reference"])
        all_losses = {
            "original_CV": losses(base, target),
            "original_P": losses(pred, target),
            "projected_own_reference_CV": losses(projected_cv, target),
            "projected_own_reference_P": losses(projected_p, target),
            "CV_body_P_reference": losses(same_body_P_reference, target),
            "CV_body_GT_reference_oracle": losses(same_body_GT_reference, target),
            "common_CV_reference_CV": losses(common_cv, common_target),
            "common_CV_reference_P": losses(common_p, common_target),
        }
        for name, codes in (("CV", base), ("P", pred)):
            raw = codec.denormalize(codes)[:, :198].reshape(-1, 22, 9)[..., :6]
            first, second = raw[..., :3], raw[..., 3:]
            for metric, value in {
                "first6d_basis_norm": first.norm(dim=-1).mean(-1),
                "second6d_basis_norm": second.norm(dim=-1).mean(-1),
                "basis_abs_dot": (first * second).sum(-1).abs().mean(-1),
            }.items():
                values.setdefault(name + "/" + metric, []).append(value.cpu())
        for name, metrics in all_losses.items():
            for metric, value in metrics.items():
                values.setdefault(name + "/" + metric, []).append(value.cpu())
        for name, state in [("CV", cv), ("P", p)]:
            difference = (state.joints[..., :3, :3] - gt.joints[..., :3, :3]).double()
            values.setdefault(name + "/global_rotation_Frobenius", []).append(
                difference.square().sum((-1, -2)).sqrt().mean(-1).cpu()
            )
        relative = cv.reference[..., :3, :3].transpose(-1, -2) @ gt.reference[..., :3, :3]
        values.setdefault("reference_yaw_abs_deg", []).append(
            torch.atan2(relative[:, 1, 0], relative[:, 0, 0]).abs().double().cpu() * 180 / torch.pi
        )
        gt_common = codec.decode_current(common_target, e["reference"])
        cv_changed_ref = codec.decode_current(same_body_GT_reference, e["reference"])
        equivalence.extend(
            [(fk(gt) - fk(gt_common)).abs().max().item(), (fk(cv) - fk(cv_changed_ref)).abs().max().item()]
        )
        equivalence.append((fk(p) - fk(codec.decode_current(projected_p, e["reference"]))).abs().max().item())
        equivalence.append((fk(p) - fk(codec.decode_current(common_p, e["reference"]))).abs().max().item())
        equivalence.append(
            (fk(cv) - fk(codec.decode_current(same_body_P_reference, e["reference"]))).abs().max().item()
        )
    values = {k: torch.cat(v) for k, v in values.items()}
    rows = bank["rows"]
    summary = {}
    for phase in ("all", "mature", "startup"):
        indices = [i for i, row in enumerate(rows) if phase == "all" or (row["frame"] >= 40) == (phase == "mature")]
        groups = {}
        for i in indices:
            groups.setdefault(rows[i]["episode"], []).append(i)
        result = {k: sum(float(v[js].mean()) for js in groups.values()) / len(groups) for k, v in values.items()}
        result["clips"] = len(groups)
        result["reference_yaw_p95_deg"] = float(values["reference_yaw_abs_deg"][indices].quantile(0.95))
        result["reference_yaw_gt30deg_fraction"] = float(
            (values["reference_yaw_abs_deg"][indices] > 30).double().mean()
        )
        summary[phase] = result
    report = {
        "scope": "Frozen EMA2000/D128 coordinate-gauge diagnostic only; no learning, feedback or changes to targets used in P01.",
        "bank_sha256": file_sha256(root / "banks/D128/bank.pt"),
        "checkpoint_sha256": file_sha256(root / "P01_W3G10k_s62/last.pt"),
        "weights_denominator": 306,
        "summary": summary,
        "physical_equivalence_max_abs_error_m": max(equivalence),
        "GT_reference_oracle": "Uses GT only after inference to reparameterize the SAME CV physical body; cannot be used by deployed P.",
        "common_CV_reference": "Uses causal CV reference for both physical predictions and truth encoding; GT is still label only.",
        "interpretation": "Gauge changes can change normalized code loss without changing body FK; common-frame code is a diagnostic alternate objective, not a demonstrated training gain.",
    }
    if max(equivalence) > 1e-5:
        raise ValueError("Reencoding changed physical FK beyond the diagnostic precision tolerance.")
    save_json(args.output, report)
    print(
        __import__("json").dumps(
            {"summary": summary, "physical_equivalence_max_abs_error_m": max(equivalence)}, ensure_ascii=False
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("exp/formal-p-w3-g10k-20261113"))
    p.add_argument("--output", type=Path, default=Path("exp/formal-p-w3-g10k-20261113/diagnosis/gauge.json"))
    p.add_argument("--device", default="cuda")
    diagnose(p.parse_args())


if __name__ == "__main__":
    main()
