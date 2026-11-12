"""Shape-aware SMPL22 FK without computing unused skinned vertices.

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


def joint_decoder(episode, synthetic, smpl=None, *, shape_mode="bootstrap"):
    if synthetic:
        from .diagnostics import DenseJoints

        return DenseJoints()
    if smpl is None:
        raise ValueError("Production reconstruction requires the licensed SMPL-X asset.")
    beta = episode.beta
    return FixedShapeFK(smpl, beta[None] if beta.ndim == 1 else beta, shape_mode=shape_mode)


class FixedShapeFK(nn.Module):
    def __init__(self, smpl, beta_boot, *, shape_mode="bootstrap", backend="auto"):
        super().__init__()
        if shape_mode not in ("bootstrap", "predicted"):
            raise ValueError("Shape mode must be bootstrap or predicted.")
        self.shape_mode = shape_mode
        if (
            beta_boot.shape != (10,) and (beta_boot.ndim != 2 or beta_boot.shape[1] != 10 or beta_boot.shape[0] < 1)
        ) or not bool(torch.isfinite(beta_boot).all()):
            raise ValueError("Provide finite [10] or [batch,10] beta coefficients from model bootstrap.")
        self.smpl = smpl.eval().requires_grad_(False)
        self.register_buffer("beta_boot", beta_boot.detach().clone())
        geometry = getattr(smpl, "_body_joint_geometry", None)
        if geometry is None and all(hasattr(smpl, name) for name in ("J_regressor", "v_template", "shapedirs")):
            geometry = BodyJointGeometry(smpl)
            smpl.add_module("_body_joint_geometry", geometry)
        if backend not in ("auto", "body", "full"):
            raise ValueError("FK backend must be auto, body or full.")
        if backend == "body" and geometry is None:
            raise ValueError("Body FK requires SMPL-X joint regressor, template and shape directions.")
        self.backend = "body" if backend != "full" and geometry is not None else "full"
        self.parents = geometry.parents if geometry is not None else smpl.parents[:22].detach().cpu().tolist()
        if (
            len(self.parents) != 22
            or self.parents[0] != -1
            or any(not 0 <= parent < joint for joint, parent in enumerate(self.parents[1:], start=1))
        ):
            raise ValueError("Expected topologically ordered SMPL22 parents.")
        # The full decoder remains an explicit numerical reference and supports
        # layer adapters that do not expose the original SMPL-X geometry.
        with torch.no_grad():
            betas = self.beta_boot[None] if self.beta_boot.ndim == 1 else self.beta_boot
            root = (
                geometry.rest_joints(betas)[:, 0]
                if self.backend == "body"
                else self.smpl(betas=betas, return_verts=False).joints[:, 0]
            )
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
        count = len(joints)
        if self.shape_mode == "predicted":
            betas = auxiliary[:, -10:]
            root_offset = None
        elif self.beta_boot.ndim == 1:
            betas = self.beta_boot.expand(count, -1)
            root_offset = self.root_offset.expand(count, -1)
        else:
            if len(self.beta_boot) != count:
                raise ValueError("Per-sample bootstrap shapes must match the flattened body batch.")
            betas = self.beta_boot
            root_offset = self.root_offset
        if self.backend == "body":
            positions = self.smpl._body_joint_geometry(global_rotation, joints[:, 0, :3, 3], betas)
            result = positions.reshape(*prefix, 22, 3)
            if not bool(torch.isfinite(result).all()):
                raise ValueError("SMPL FK produced nonfinite joints.")
            return result
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
        output = self.smpl(
            global_orient=local[:, 0],
            body_pose=local[:, 1:22],
            betas=betas,
            transl=None if root_offset is None else joints[:, 0, :3, 3] - root_offset,
            left_hand_pose=pca_to_matrix(auxiliary[:, :12], self.smpl.left_hand_components),
            right_hand_pose=pca_to_matrix(auxiliary[:, 12:24], self.smpl.right_hand_components),
            return_verts=False,
        )
        if output.joints.shape[1] < 22:
            raise ValueError("SMPL layer must expose the first 22 body joints.")
        positions = output.joints[:, :22]
        if self.shape_mode == "predicted":
            # Recenter with this prediction's own shape-dependent pelvis. The
            # gradient through shape is retained; no GT shape enters decoding.
            positions = positions + (joints[:, 0, :3, 3] - positions[:, 0])[:, None]
        result = positions.reshape(*prefix, 22, 3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("SMPL FK produced nonfinite joints.")
        return result


class BodyJointGeometry(nn.Module):
    """The first 22 LBS joints depend on shaped rest joints and body rotations.

    Regressing the template and ten shape bases once avoids the 10k-vertex
    blend-shape, pose-corrective and skinning work on every loss call. Hand PCA
    and facial rotations have no ancestors among these body joints. Local-to-
    global rotation composition is retained, including its gradients, rather
    than assuming finite-precision input rotations are perfectly orthogonal.
    """

    def __init__(self, smpl):
        super().__init__()
        self.parents = smpl.parents[:22].detach().cpu().tolist()
        with torch.no_grad():
            self.register_buffer("rest", torch.einsum("jv,vc->jc", smpl.J_regressor[:22], smpl.v_template))
            self.register_buffer("shape", torch.einsum("jv,vcq->jcq", smpl.J_regressor[:22], smpl.shapedirs[..., :10]))

    def rest_joints(self, betas):
        return self.rest + torch.einsum("bq,jcq->bjc", betas, self.shape)

    def forward(self, rotations, root, betas):
        rest = self.rest_joints(betas)
        positions, composed = [root], [rotations[:, 0]]
        for joint, parent in enumerate(self.parents[1:], 1):
            offset = rest[:, joint] - rest[:, parent]
            positions.append(positions[parent] + (composed[parent] @ offset[..., None])[..., 0])
            local = rotations[:, parent].transpose(-1, -2) @ rotations[:, joint]
            composed.append(composed[parent] @ local)
        return torch.stack(positions, 1)
