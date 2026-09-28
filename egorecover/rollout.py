"""Causal predicted-history rollout. This module never reads supervision."""

from collections import deque
import time

import torch

from egorecover.actions import action_masks
from egorecover.calibration import estimate_bootstrap_floor
from egorecover.codec import planar_reference, transform_from_9d
from egorecover.history import HistoryBuffer
from egorecover.utility import choose_action


@torch.no_grad()
def initialize_history(initializer, initializer_flow, codec, prefix, *, seed=62, length=20):
    """One shared clean model initialization; no GT body, beta or floor input."""
    if initializer.training:
        raise ValueError("The bootstrap model must be frozen/eval.")
    if prefix["aria_traj_obs"].shape != (length, 9):
        raise ValueError("Bootstrap packet must contain exactly the clean prefix.")
    if not bool(prefix["img_available"].all() & prefix["traj_available"].all()):
        raise ValueError("This protocol requires available clean startup observations.")
    floor, encoded, references = encode_startup_observations(codec, prefix, length=length)
    trajectory = prefix["aria_traj_obs"].clone()
    trajectory[:, 8] -= floor
    valid = torch.ones(1, length, device=trajectory.device, dtype=torch.bool)
    y = dict(
        traj=encoded[None],
        img_embs=prefix["img_feats"][None],
        valid_frames=valid,
        valid_img_embs=valid,
        traj_mask=~prefix["traj_available"][None],
        img_mask=~prefix["img_available"][None],
    )
    noise = torch.randn(
        1, length, 243, device=trajectory.device, generator=torch.Generator(device=trajectory.device).manual_seed(seed)
    )
    predictions = initializer_flow.sample_loop(initializer, noise.shape, {"y": y}, noise=noise, repaint_enabled=False)[
        0
    ]
    buffer = HistoryBuffer.from_bootstrap(codec, predictions, references[0], length=length)
    buffer.bootstrap_motion = predictions.detach().clone()
    return buffer, floor, encoded


def encode_startup_observations(codec, prefix, *, length=20):
    """Causal coordinate/floor calibration for both fresh and cached startup."""
    if prefix["aria_traj_obs"].shape != (length, 9):
        raise ValueError("Bootstrap packet must contain exactly the clean prefix.")
    if not bool(prefix["img_available"].all() & prefix["traj_available"].all()):
        raise ValueError("This protocol requires available clean startup observations.")
    floor = estimate_bootstrap_floor(prefix, codec, length)
    trajectory = prefix["aria_traj_obs"].clone()
    trajectory[:, 8] -= floor
    references = planar_reference(transform_from_9d(trajectory))
    previous = torch.cat((references[:1], references[:-1]), dim=0)
    encoded = codec.encode_observation(trajectory, previous, prefix["traj_available"])
    return floor, encoded, references


@torch.no_grad()
def run_episode(
    model,
    flow,
    initializer,
    initializer_flow,
    codec,
    observations,
    *,
    num_frames,
    prior=None,
    utility=None,
    action="a11",
    seed=62,
    history_length=20,
    observation_length=20,
    startup=None,
    prior_only=False,
):
    """observations(as_of=..., history_frames=...) returns online fields only.

    Each strategy owns its buffer. P/Q run once and only one G action is sampled
    per physical frame. All decoding/commits occur without current sensors.
    """
    if model.training or (prior is not None and prior.training) or (utility is not None and utility.training):
        raise ValueError("All online networks must be eval/frozen.")
    if num_frames <= history_length:
        raise ValueError("Episode must extend beyond startup.")
    if prior_only and prior is None:
        raise ValueError("Prior-only rollout requires a trained history prior.")
    device = next(model.parameters()).device

    def packet(as_of, history_frames):
        return {
            name: value.to(device) for name, value in observations(as_of=as_of, history_frames=history_frames).items()
        }

    prefix = packet(history_length - 1, history_length)
    if startup is None:
        buffer, floor, startup_traj = initialize_history(
            initializer, initializer_flow, codec, prefix, seed=seed, length=history_length
        )
    else:
        floor, startup_traj, references = encode_startup_observations(codec, prefix, length=history_length)
        if abs(floor - float(startup["floor_estimate_m"])) > 1e-5:
            raise ValueError("Cached startup and current clean-prefix floor differ.")
        buffer = HistoryBuffer.from_bootstrap(
            codec,
            startup["normalized_motion"].to(device),
            startup["initial_reference"].to(device),
            length=history_length,
        )
        if not torch.allclose(buffer.initial_reference, references[0], atol=1e-5, rtol=0):
            raise ValueError("Cached startup reference differs from the legal clean prefix.")
        if not torch.allclose(buffer.beta_boot.cpu(), startup["beta_boot"], atol=1e-5, rtol=0):
            raise ValueError("Cached startup beta differs from its stored body motion.")
        if not torch.allclose(
            torch.stack([state.reference for state in buffer.states]).cpu(),
            startup["references"],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Cached startup references differ from committed model motion.")
    bootstrap_references = torch.stack([state.reference for state in buffer.states]).cpu()
    bootstrap_positions = torch.stack([state.joints[..., :3, 3] for state in buffer.states]).cpu()
    bootstrap_positions[..., 2] += floor
    if startup is not None and not torch.allclose(bootstrap_positions, startup["world_joints"], atol=1e-4, rtol=0):
        raise ValueError("Cached startup world joints differ from committed model motion.")
    observed_history = deque(maxlen=observation_length)
    for index in range(history_length):
        observed_history.append(
            (
                prefix["img_feats"][index],
                startup_traj[index],
                prefix["img_available"][index],
                prefix["traj_available"][index],
            )
        )
    positions, references, motions, committed_motions, actions, latency = [], [], [], [], [], []
    for index in range(history_length, num_frames):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        current = packet(index, 1)
        y = buffer.conditions(current, action="a11" if utility is not None else action, floor_height=floor)
        if prior is not None:
            y["prior_mu"] = prior(y["history_motion"], y["history_valid"], y["prior_mu"])
        observed_history.append(
            (current["img_feats"][0], y["traj"][0, 0], current["img_available"][0], current["traj_available"][0])
        )
        selected = action
        if utility is not None:
            image, trajectory, iv, tv = (
                torch.stack([item[column] for item in observed_history])[None] for column in range(4)
            )
            gains = utility(
                history_motion=y["history_motion"],
                history_valid=y["history_valid"],
                prior_mu=y["prior_mu"],
                img_embs=image,
                traj=trajectory,
                img_available=iv,
                traj_available=tv,
                observation_valid=torch.ones_like(iv),
            )
            selected = choose_action(
                gains,
                img_available=current["img_available"].reshape(1, 1),
                traj_available=current["traj_available"].reshape(1, 1),
            )
        y["img_mask"], y["traj_mask"] = action_masks(
            selected,
            img_available=current["img_available"].reshape(1, 1),
            traj_available=current["traj_available"].reshape(1, 1),
        )
        epsilon = torch.randn(
            1, 1, 243, device=device, generator=torch.Generator(device=device).manual_seed(seed + index + 1)
        )
        prediction = y["prior_mu"][0, 0] if prior_only else flow.sample(model, y, epsilon=epsilon)[0, 0]
        previous_reference = buffer.previous_reference
        state = buffer.commit(prediction, index)
        committed_motions.append(codec.encode_current(state, previous_reference).cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency.append((time.perf_counter() - start) * 1000)
        position = state.joints[..., :3, 3].clone()
        position[:, 2] += floor  # Report back in the official physical world.
        positions.append(position.cpu())
        references.append(state.reference.cpu())
        motions.append(prediction.cpu())
        actions.append(int(selected.item()) if isinstance(selected, torch.Tensor) else selected)
    return {
        "dense_world_joints": torch.stack(positions),
        "bootstrap_world_joints": bootstrap_positions,
        "bootstrap_references": bootstrap_references,
        "bootstrap_motion": (
            buffer.bootstrap_motion.cpu() if startup is None else startup["normalized_motion"].cpu().clone()
        ),
        "bootstrap_initial_reference": buffer.initial_reference.cpu(),
        "bootstrap_sampling_seed": seed if startup is None else startup["sampling_seed"],
        "references": torch.stack(references),
        "normalized_motion": torch.stack(motions),
        "normalized_motion_semantics": "raw G prediction; decode using the recorded reference_mode",
        "committed_motion": torch.stack(committed_motions),
        "reference_mode": codec.reference_mode,
        "actions": actions,
        "latency_ms": latency,
        "frame_indices": list(range(history_length, num_frames)),
        "beta_boot": buffer.beta_boot.cpu(),
        "floor_estimate_m": floor,
        "bootstrap_frames": history_length,
        "body_history_source": "model_predictions_only",
        "metric_space": "dense positions, not SMPL FK",
    }
