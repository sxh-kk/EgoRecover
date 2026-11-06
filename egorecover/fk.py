"""Optional SMPL22 FK with fixed bootstrap shape and no GT fallbacks.

The caller provides a licensed SMPL-X layer using load_smpl().
Importing this module alone does not require assets. Actual asset-backed FK
must be validated before these outputs become formal utility labels.
"""

import os
from pathlib import Path

import torch
from torch import nn

from egorecover.rotations import pca_to_matrix


def smpl_path():
    directory = Path(os.environ.get("SMPLX_MODEL_PATH", Path.cwd() / "body_models/smplx")).expanduser()
    return directory / "SMPLX_NEUTRAL.npz"


def smpl_identity():
    from .io import file_sha256

    return file_sha256(smpl_path())


def load_smpl(device):
    """Licensed asset is required explicitly; never fall back to a dummy body."""
    path = smpl_path()
    if not path.is_file():
        raise FileNotFoundError(f"SMPL-X neutral model not found: {path}. Set SMPLX_MODEL_PATH to its directory.")
    import smplx

    return (
        smplx.SMPLXLayer(str(path.parent), ext="npz", gender="NEUTRAL", use_pca=True, num_pca_comps=12)
        .to(device)
        .eval()
        .requires_grad_(False)
    )


def joint_decoder(episode, synthetic, smpl=None):
    if synthetic:
        from .diagnostics import DenseJoints

        return DenseJoints()
    if smpl is None:
        raise ValueError("Production reconstruction requires the licensed SMPL-X asset.")
    beta = episode.beta
    return FixedShapeFK(smpl, beta[None] if beta.ndim == 1 else beta)


class FixedShapeFK(nn.Module):
    def __init__(self, smpl, beta_boot):
        super().__init__()
        if (
            beta_boot.shape != (10,) and (beta_boot.ndim != 2 or beta_boot.shape[1] != 10 or beta_boot.shape[0] < 1)
        ) or not bool(torch.isfinite(beta_boot).all()):
            raise ValueError("Provide finite [10] or [batch,10] beta coefficients from model bootstrap.")
        self.smpl = smpl.eval().requires_grad_(False)
        self.register_buffer("beta_boot", beta_boot.detach().clone())
        self.parents = [int(value) for value in smpl.parents[:22]]
        if (
            len(self.parents) != 22
            or self.parents[0] != -1
            or any(not 0 <= parent < joint for joint, parent in enumerate(self.parents[1:], start=1))
        ):
            raise ValueError("Expected topologically ordered SMPL22 parents.")
        with torch.no_grad():
            betas = self.beta_boot[None] if self.beta_boot.ndim == 1 else self.beta_boot
            root = self.smpl(betas=betas, return_verts=False).joints[:, 0]
        self.register_buffer("root_offset", root.detach().clone())

    def train(self, mode=True):
        super().train(mode)
        self.smpl.eval()
        return self

    def forward(self, state):
        """Decode only the 22 body joints required by the current plan."""
        if state.joints.shape[-3:] != (22, 4, 4) or state.auxiliary.shape[-1] != 36:
            raise ValueError("Expected dense BodyState with 22 transforms and 36 auxiliary channels.")
        prefix = state.joints.shape[:-3]
        if state.auxiliary.shape[:-1] != prefix:
            raise ValueError("Body transforms and auxiliary channels must have identical batch dimensions.")
        joints = state.joints.reshape(-1, 22, 4, 4)
        auxiliary = state.auxiliary.reshape(-1, 36)
        global_rotation = joints[..., :3, :3]
        local = torch.stack(
            [
                (
                    global_rotation[:, joint]
                    if parent < 0
                    else global_rotation[:, parent].transpose(-1, -2) @ global_rotation[:, joint]
                )
                for joint, parent in enumerate(self.parents)
            ],
            dim=1,
        )
        count = len(joints)
        if self.beta_boot.ndim == 1:
            betas = self.beta_boot.expand(count, -1)
            root_offset = self.root_offset.expand(count, -1)
        else:
            if len(self.beta_boot) != count:
                raise ValueError("Per-sample bootstrap shapes must match the flattened body batch.")
            betas = self.beta_boot
            root_offset = self.root_offset
        output = self.smpl(
            global_orient=local[:, 0],
            body_pose=local[:, 1:22],
            betas=betas,
            transl=joints[:, 0, :3, 3] - root_offset,
            left_hand_pose=pca_to_matrix(auxiliary[:, :12], self.smpl.left_hand_components),
            right_hand_pose=pca_to_matrix(auxiliary[:, 12:24], self.smpl.right_hand_components),
            return_verts=False,
        )
        if output.joints.shape[1] < 22:
            raise ValueError("SMPL layer must expose the first 22 body joints.")
        result = output.joints[:, :22].reshape(*prefix, 22, 3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("SMPL FK produced nonfinite joints.")
        return result
