"""Shared single-frame interventions for diagnostics and response supervision."""

from .codec import BodyState


def intervention_state(predicted, truth, joints, mode):
    if mode == "full_state":
        return BodyState(*(getattr(truth, key).detach() for key in ("joints", "reference", "auxiliary")))
    if mode != "body_pose":
        raise ValueError("Unknown intervention mode.")
    pose = BodyState(truth.joints.detach(), predicted.reference.detach(), predicted.auxiliary.detach())
    transforms = pose.joints.clone()
    transforms[..., :3, 3] = joints(pose)
    return BodyState(transforms, pose.reference, pose.auxiliary)
