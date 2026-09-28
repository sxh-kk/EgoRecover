import os
from pathlib import Path

from loguru import logger
import smplx
import torch


def get_smpl(smpl_type="smplx"):
    logger.warning(f"Loading SMPL model: {smpl_type}")
    assert smpl_type in ["smplx"]

    default_model_path = Path(__file__).resolve().parents[1] / "body_models" / smpl_type
    smpl_path = Path(os.environ.get("SMPLX_MODEL_PATH", default_model_path)).expanduser()
    model_file = smpl_path / "SMPLX_NEUTRAL.npz"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"SMPL-X neutral model not found at {model_file}. "
            "Download the licensed SMPL-X neutral model and extract it there, or set SMPLX_MODEL_PATH "
            "to the directory containing SMPLX_NEUTRAL.npz."
        )

    # Load SMPL
    # joint_mapper = JointMapper(smpl_to_openpose(smpl_type, use_hands=args.use_hands, use_face=False))
    # Don't use joint mapper because we need root joint to be at pelvis. We can sample the needed joints later.
    # This will give all 145 joints as shown here: https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx/joint_names.py#L19
    # First 55 SMPLX kintree joints are 22 body + 3 face + 15 left hand + 15 right hand. 76 joints cover body and hands. Rest are faces.
    # Rest of the joints are other helper joints (e.g. for openpose) and face contours.
    smpl_layer_cls = {"smpl": smplx.SMPLLayer, "smplx": smplx.SMPLXLayer}[smpl_type]
    smpl = smpl_layer_cls(
        str(smpl_path),
        ext="npz",
        gender="NEUTRAL",
        use_pca=True,
        # joint_mapper=joint_mapper,
        num_pca_comps=12,
    )
    return smpl


def evaluate_smpl(smpl, smpl_params, max_parallel_smpl_evals=1000, return_joints_only=False):
    kp3d = []
    verts = []
    full_pose = []
    bs = max_parallel_smpl_evals
    nf = len(smpl_params["body_pose"])
    for i in range(0, nf, bs):
        smpl_params_batch = {
            "body_pose": smpl_params["body_pose"][i : i + bs],
            "global_orient": smpl_params["global_orient"][i : i + bs],
            "betas": smpl_params["betas"][i : i + bs],
            "transl": smpl_params["transl"][i : i + bs],
        }
        if "left_hand_pose" in smpl_params:
            smpl_params_batch["left_hand_pose"] = smpl_params["left_hand_pose"][i : i + bs]
            smpl_params_batch["right_hand_pose"] = smpl_params["right_hand_pose"][i : i + bs]

        with torch.no_grad():
            smpl_output = smpl(
                **smpl_params_batch,
                return_verts=not return_joints_only,
                return_full_pose=not return_joints_only,
            )

        kp3d.append(smpl_output.joints)
        if not return_joints_only:
            verts.append(smpl_output.vertices)
            full_pose.append(smpl_output.full_pose)

    kp3d = torch.cat(kp3d, dim=0)
    if return_joints_only:
        return kp3d

    verts = torch.cat(verts, dim=0)
    full_pose = torch.cat(full_pose, dim=0)

    return kp3d, verts, full_pose
