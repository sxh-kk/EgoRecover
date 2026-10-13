"""Causal predicted-history rollout. This module never reads supervision."""

from collections import deque
from numbers import Integral
import time

import torch

from egorecover.actions import binary_mask
from egorecover.calibration import estimate_bootstrap_floor
from egorecover.codec import BodyState, planar_reference, transform_from_9d
from egorecover.history import HistoryBuffer
from egorecover.utility import choose_action, validate_min_gain


def _memory_conditions(codec, buffer, packets, prior, length, floor, action, scope):
    """Rebuild R0/R1 memory from arrived raw sensors and this policy's bodies."""
    from egorecover.observation_reference import conditions

    head, images, iv, tv = (
        torch.cat([packet[key] for packet in packets])
        for key in ("aria_traj_obs", "img_feats", "img_available", "traj_available")
    )
    head = head.clone()
    head[:, 8] -= floor
    padding = length - len(head)
    valid = torch.arange(length, device=head.device) >= padding

    def pad(value):
        return torch.cat((value.new_zeros(padding, *value.shape[1:]), value))[None]

    history = BodyState(
        *(
            torch.stack([getattr(state, key) for state in buffer.states])[None]
            for key in ("joints", "reference", "auxiliary")
        )
    )
    return conditions(
        codec,
        history,
        pad(head),
        pad(images),
        valid[None],
        prior,
        False,
        img_available=pad(iv),
        traj_available=pad(tv),
        action=action,
        action_scope=scope,
    )[0]


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
    min_gain=0.0,
    episode_id=None,
):
    """observations(as_of=..., history_frames=...) returns online fields only.

    Each strategy owns its buffer. P/Q run once and only one G action is sampled
    per physical frame. All decoding/commits occur without current sensors.
    min_gain is the dev-calibrated utility threshold, in the Q label units.
    R0/R1 use their configured observation memory and the calibrated P baseline.
    episode_id enables episode/frame/seed-keyed paired noise; omission preserves
    the legacy seed schedule. This returns dense state, not an FK accuracy score.
    """
    validate_min_gain(min_gain)
    if model.training or (prior is not None and prior.training) or (utility is not None and utility.training):
        raise ValueError("All online networks must be eval/frozen.")
    if prior_only and prior is None:
        raise ValueError("Prior-only rollout requires a trained history prior.")
    if prior_only and utility is not None:
        raise ValueError("Prior-only rollout does not select G actions.")
    for name, value in (
        ("num_frames", num_frames),
        ("history_length", history_length),
        ("observation_length", observation_length),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if num_frames <= history_length:
        raise ValueError("Episode must extend beyond startup.")
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise ValueError("episode_id must be a nonempty string.")
    memory_length = getattr(model, "observation_length", None)
    if memory_length is not None:
        from egorecover.observation_reference import ObservationFlow

        if (
            not isinstance(flow, ObservationFlow)
            or memory_length not in (21, 80)
            or history_length != 20
            or prior is None
        ):
            raise ValueError("R0/R1 rollout requires ObservationFlow, calibrated P and 20 body-history frames.")
        if getattr(model, "anchored", False) or codec.reference_mode != "planar":
            raise ValueError("This online rollout requires a prediction-defined reference (R0/R1).")
    elif getattr(flow, "conditioning_protocol", None) != "history-current-v1":
        raise ValueError("The current-only model requires HistoryFlow conditioning.")
    base_mode = getattr(prior, "base_mode", "hold")
    if base_mode not in ("hold", "constant_velocity") or (base_mode == "constant_velocity" and history_length < 2):
        raise ValueError("Invalid P physical baseline or insufficient history for constant velocity.")
    device = next(model.parameters()).device

    def packet(as_of, history_frames):
        values = observations(as_of=as_of, history_frames=history_frames)
        allowed = {"img_feats", "aria_traj_obs", "img_available", "traj_available", "frame_id_30fps"}
        if set(values) != allowed:
            raise ValueError("Online observation packets must contain only the documented fields.")
        result = {name: value.to(device) for name, value in values.items()}
        for name, shape in (
            ("aria_traj_obs", (history_frames, 9)),
            ("img_feats", (history_frames, 1024)),
            ("frame_id_30fps", (history_frames,)),
        ):
            if result[name].shape != shape:
                raise ValueError(f"Invalid online {name} shape; expected {shape}.")
        for name in ("img_available", "traj_available"):
            result[name] = binary_mask(result[name], (history_frames,), device, name)
        indices = result["frame_id_30fps"]
        if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise ValueError("Source frame indices must be integers on the 30 Hz grid.")
        if bool((indices < 0).any()) or (len(indices) > 1 and not bool((indices[1:] - indices[:-1] == 3).all())):
            raise ValueError("Source frame indices must advance by three on the 10 Hz motion grid.")
        return result

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
    raw_memory = deque(maxlen=memory_length) if memory_length is not None else None
    for index in range(history_length):
        if raw_memory is not None:
            raw_memory.append({key: value[index : index + 1].clone() for key, value in prefix.items()})
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
        expected_frame = prefix["frame_id_30fps"][-1] + 3 * (index - history_length + 1)
        if current["frame_id_30fps"][0] != expected_frame:
            raise ValueError("Current packet does not match the requested causal physical frame.")
        initial_action = "a11" if utility is not None else action
        if raw_memory is not None:
            raw_memory.append({key: value.clone() for key, value in current.items()})
            y = _memory_conditions(
                codec, buffer, raw_memory, prior, memory_length, floor, initial_action, flow.action_scope
            )
        else:
            y = buffer.conditions(current, action=initial_action, floor_height=floor)
            if prior is not None:
                base = y["prior_mu"]
                if base_mode == "constant_velocity":
                    base = codec.constant_velocity_prior(buffer.states[-2], buffer.states[-1])[None, None]
                y["prior_mu"] = prior(y["history_motion"], y["history_valid"], base)
            observed_history.append(
                (current["img_feats"][0], y["traj"][0, 0], current["img_available"][0], current["traj_available"][0])
            )
        selected = action
        if utility is not None:
            if raw_memory is not None:
                image, trajectory = y["memory_img"], y["memory_traj"]
                iv, tv = ~y["memory_img_mask"], ~y["memory_traj_mask"]
                ov = y["memory_valid"]
            else:
                image, trajectory, iv, tv = (
                    torch.stack([item[column] for item in observed_history])[None] for column in range(4)
                )
                ov = torch.ones_like(iv)
            gains = utility(
                history_motion=y["history_motion"],
                history_valid=y["history_valid"],
                prior_mu=y["prior_mu"],
                img_embs=image,
                traj=trajectory,
                img_available=iv,
                traj_available=tv,
                observation_valid=ov,
            )
            img_available, traj_available = flow.action_availability(y)
            selected = choose_action(
                gains,
                img_available=img_available,
                traj_available=traj_available,
                min_gain=min_gain,
            )
        y = flow.apply_action(y, selected)
        sampling_seed = seed + index + 1
        if episode_id is not None:
            from egorecover.prior_real_history import noise_seed

            sampling_seed = noise_seed(episode_id, seed, index)
        epsilon = torch.randn(
            1, 1, 243, device=device, generator=torch.Generator(device=device).manual_seed(sampling_seed)
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
        "protocol": {
            "conditioning": flow.conditioning_protocol,
            "action_scope": getattr(flow, "action_scope", "current"),
            "observation_length": memory_length if memory_length is not None else observation_length,
            "prior_base_mode": base_mode,
            "min_gain": min_gain,
            "source_mode": flow.source_mode,
            "sigma": flow.sigma,
            "nfe_per_frame": 0 if prior_only else flow.flow.num_steps,
            "seed": seed,
            "episode_id": episode_id,
            "noise_schedule": "episode_frame_seed" if episode_id is not None else "legacy_seed_plus_frame",
            "noise_device": str(device),
            "floor_source": "clean_prefix_head_height_estimate",
            "reference_mode": codec.reference_mode,
        },
    }
