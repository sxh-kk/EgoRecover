"""Offline dense and differentiable SMPL-X objectives in physical world space."""

import torch

from egorecover.fk import FixedShapeFK


def weighted_representation_mse(prediction, target):
    weights = prediction.new_ones(243, dtype=torch.float32)
    weights[198:207] = 8
    return ((prediction.float() - target.float()).square() * weights).sum(-1).mean() / weights.sum()


def dense_position_errors(codec, prediction, previous_reference, target_joints):
    if prediction.ndim != 3 or prediction.shape[1:] != (1, 243):
        raise ValueError("Expected one [B,1,243] current prediction.")
    if previous_reference.shape != (len(prediction), 4, 4) or target_joints.shape != (len(prediction), 22, 3):
        raise ValueError("Expected matching previous references and world target joints.")
    # Decode outside autocast: 6D Gram-Schmidt and geometry need float32.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        body = codec.decode_current(prediction[:, 0].float(), previous_reference.float())
        return body.joints[..., :3, 3] - target_joints.float()


def fk_position_errors(codec, prediction, batch, smpl):
    """Differentiable world SMPL22 error with per-sample predicted startup shape."""
    if prediction.ndim != 3 or prediction.shape[1:] != (1, 243):
        raise ValueError("Expected one [B,1,243] prediction for FK loss.")
    betas = batch.get("beta_boot")
    if betas is None or betas.shape != (len(prediction), 10):
        raise ValueError("FK loss requires one bootstrap beta vector per training sample.")
    if (
        "beta_boot_is_model" not in batch
        or batch["beta_boot_is_model"].dtype != torch.bool
        or batch["beta_boot_is_model"].shape != (len(prediction),)
        or not bool(batch["beta_boot_is_model"].all())
    ):
        raise ValueError("Formal FK loss requires a model-generated bootstrap shape for every sample.")
    if batch["target_joints"].shape != (len(prediction), 22, 3):
        raise ValueError("FK loss requires matching world GT joints.")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        state = codec.decode_current(prediction[:, 0].float(), batch["previous_reference"].float())
        body = FixedShapeFK(smpl, betas.float()).project(state)[0]
        error = body - batch["target_joints"].float()
    if not bool(torch.isfinite(error).all()):
        raise ValueError("Nonfinite differentiable FK error.")
    return error


def physical_objective(codec, prediction, batch, *, geometry_weight=1.0, fk_weight=0.0, smpl=None, scale_m=0.1,
                       return_components=False):
    if geometry_weight < 0 or fk_weight < 0 or scale_m <= 0:
        raise ValueError("Invalid physical-loss weight or scale.")
    representation = weighted_representation_mse(prediction, batch["target"])
    geometry = representation.new_zeros(())
    if geometry_weight:
        error = dense_position_errors(codec, prediction, batch["previous_reference"], batch["target_joints"])
        geometry = error.square().sum(-1).mean() / (scale_m**2)
    fk_geometry = representation.new_zeros(())
    if fk_weight:
        if smpl is None:
            raise ValueError("FK loss requires an SMPL-X layer.")
        fk_error = fk_position_errors(codec, prediction, batch, smpl)
        fk_geometry = fk_error.square().sum(-1).mean() / (scale_m**2)
    total = representation + geometry_weight * geometry + fk_weight * fk_geometry
    if return_components:
        return {"loss": total, "representation": representation,
                "weighted_dense": geometry_weight * geometry, "weighted_fk": fk_weight * fk_geometry}
    return total


@torch.no_grad()
def fk_position_mm(codec, prediction, batch, smpl):
    return float(fk_position_errors(codec, prediction, batch, smpl).norm(dim=-1).mean() * 1000)


@torch.no_grad()
def dense_position_mm(codec, prediction, batch):
    return float(
        dense_position_errors(codec, prediction, batch["previous_reference"], batch["target_joints"])
        .norm(dim=-1)
        .mean()
        * 1000
    )
