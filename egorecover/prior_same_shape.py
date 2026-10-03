"""Training-only pose FK supervision using the same model shape on both sides."""
import torch
from egorecover.fk import FixedShapeFK
from egorecover.losses import physical_objective as original_objective


def same_shape_position_errors(codec, prediction, batch, smpl):
    if prediction.ndim != 3 or prediction.shape[1:] != (1, 243):
        raise ValueError('Expected [B,1,243] prediction.')
    betas = batch.get('beta_boot')
    valid = batch.get('beta_boot_is_model')
    if (betas is None or betas.shape != (len(prediction), 10) or valid is None
            or valid.dtype != torch.bool or valid.shape != (len(prediction),) or not bool(valid.all())):
        raise ValueError('Same-shape FK requires model-generated bootstrap shapes.')
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        fk = FixedShapeFK(smpl, betas.float())
        pred = codec.decode_current(prediction[:, 0].float(), batch['previous_reference'].float())
        predicted_joints = fk.project(pred)[0]
        # Both paths use the SAME bootstrap beta, never the target's shape channels.
        with torch.no_grad():
            target = codec.decode_current(batch['target'][:, 0].float(), batch['previous_reference'].float())
            target_joints = fk.project(target)[0]
        error = predicted_joints - target_joints
    if not bool(torch.isfinite(error).all()):
        raise ValueError('Nonfinite same-shape FK error.')
    return error


def physical_objective(codec, prediction, batch, *, same_shape_weight=0., geometry_weight=1.,
                       fk_weight=0., smpl=None, scale_m=.1, return_components=False):
    if same_shape_weight < 0 or fk_weight != 0:
        raise ValueError('Use nonnegative same-shape weight and disable original FK loss.')
    parts = original_objective(codec, prediction, batch, geometry_weight=geometry_weight,
                               fk_weight=0., smpl=smpl, scale_m=scale_m, return_components=True)
    pose = parts['loss'].new_zeros(())
    if same_shape_weight:
        if smpl is None:
            raise ValueError('Same-shape FK requires a SMPL layer.')
        error = same_shape_position_errors(codec, prediction, batch, smpl)
        pose = same_shape_weight * error.square().sum(-1).mean() / scale_m**2
    parts['weighted_same_shape_fk'] = pose
    parts['loss'] = parts['loss'] + pose
    return parts if return_components else parts['loss']
