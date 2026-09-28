"""Per-policy physical history: candidates are decoded without committing."""

from collections import deque

import torch

from egorecover.actions import action_masks
from egorecover.codec import BodyState, planar_reference
from egorecover.conditioning import build_conditioning


class HistoryBuffer:
    def __init__(self, codec, initial_reference, *, length=20):
        if initial_reference.shape != (4, 4) or length < 1:
            raise ValueError("Expected one initial [4,4] reference and positive history length.")
        self.codec = codec
        self.initial_reference = codec.canonical_reference(initial_reference).detach().clone()
        self.length = length
        self.states = deque(maxlen=length)
        self.frame_indices = deque(maxlen=length)
        self.beta_boot = None

    @property
    def previous_reference(self):
        return self.states[-1].reference if self.states else self.initial_reference

    def decode_candidate(self, motion):
        if motion.shape != (243,):
            raise ValueError("One candidate must have shape [243].")
        return self.codec.decode_current(motion, self.previous_reference)

    @torch.no_grad()
    def commit(self, motion, frame_index):
        """Append one selected prediction; never accept a GT state dictionary."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("frame_index must be an integer.")
        if self.frame_indices and frame_index != self.frame_indices[-1] + 1:
            raise ValueError("Commit exactly once per consecutive physical frame.")
        try:
            state = self.decode_candidate(motion)
        except ValueError as error:
            raise ValueError(f"Cannot commit physical frame {frame_index}: {error}") from error
        state = BodyState(*(value.detach().clone() for value in (state.joints, state.reference, state.auxiliary)))
        self.states.append(state)
        self.frame_indices.append(frame_index)
        return state

    @classmethod
    def from_bootstrap(cls, codec, predictions, initial_reference, *, start_index=0, length=20):
        """predictions MUST be model outputs for the shared clean startup."""
        if predictions.shape != (length, 243):
            raise ValueError(f"Bootstrap requires [{length},243] normalized model predictions.")
        buffer = cls(codec, initial_reference, length=length)
        for index, prediction in enumerate(predictions):
            buffer.commit(prediction, start_index + index)
        buffer.beta_boot = torch.stack([state.auxiliary[-10:] for state in buffer.states]).mean(0).detach().clone()
        return buffer

    def encoded(self):
        if len(self.states) != self.length:
            raise ValueError("Complete the common model bootstrap before current-frame inference.")
        states = BodyState(
            *(
                torch.stack([getattr(state, name) for state in self.states])[None]
                for name in ("joints", "reference", "auxiliary")
            )
        )
        # This reference depends only on committed predictions, never the
        # current head observation or the candidate being evaluated.
        outer = planar_reference(self.states[0].reference)[None]
        return self.codec.encode_history(states, outer)

    def continuation_prior(self):
        return self.codec.continuation_prior(self.states[-1]).reshape(1, 1, 243)

    def conditions(self, observation, *, action="a11", prior_mu=None, floor_height=0.0):
        """One online packet, and an explicit startup/calibrated floor only.

        observation is a MismatchDataset.observations(...,history_frames=1)
        result, already on the buffer's device. GT/fault fields are rejected.
        floor_height must not be looked up from supervision inside a rollout.
        """
        allowed = {"img_feats", "aria_traj_obs", "img_available", "traj_available", "frame_id_30fps"}
        if set(observation) != allowed or observation["aria_traj_obs"].shape != (1, 9):
            raise ValueError("Expected a current-only online observation packet with the documented fields.")
        img_available = observation["img_available"].reshape(1, 1)
        traj_available = observation["traj_available"].reshape(1, 1)
        _, traj_mask = action_masks(action, img_available=img_available, traj_available=traj_available)
        trajectory = observation["aria_traj_obs"].clone()
        trajectory[:, 8] -= floor_height
        trajectory = self.codec.encode_observation(trajectory, self.previous_reference[None], ~traj_mask[:, 0])
        history = self.encoded()
        return build_conditioning(
            history_motion=history,
            history_valid=torch.ones(history.shape[:2], device=history.device, dtype=torch.bool),
            prior_mu=self.continuation_prior() if prior_mu is None else prior_mu,
            traj=trajectory[None],
            img_embs=observation["img_feats"][None],
            img_available=img_available,
            traj_available=traj_available,
            action=action,
        )
