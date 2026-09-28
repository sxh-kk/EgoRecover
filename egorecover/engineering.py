"""Explicitly teacher-forced development batches from the engineering pilot.

This module reads supervision and clean counterparts. NEVER use it as an
online policy input builder. The online path is HistoryBuffer.conditions().
"""

import random

import torch

from egorecover.annotations import body_states_from_supervision
from egorecover.calibration import estimate_bootstrap_floor
from egorecover.codec import BodyState, planar_reference
from egorecover.conditioning import build_conditioning


def split_takes(dataset, seed=62):
    takes = sorted({record["base_take_name"] for record in dataset.records})
    if len(takes) < 3:
        raise ValueError("Use the pilot with at least three takes for separate engineering splits.")
    random.Random(seed).shuffle(takes)
    heldout = max(1, len(takes) // 6)
    return {"train": takes[: -2 * heldout], "dev": takes[-2 * heldout : -heldout], "holdout": takes[-heldout:]}


class TeacherForcedFrames:
    def __init__(
        self,
        dataset,
        codec,
        takes,
        *,
        clean_only=False,
        stride=1,
        history_length=20,
        contact_floor_source="annotation",
        bootstrap_shapes=None,
    ):
        if stride < 1:
            raise ValueError("stride must be positive.")
        if contact_floor_source not in ("annotation", "startup_estimate"):
            raise ValueError("Unknown offline contact-label floor source.")
        records = [
            record
            for record in dataset.records
            if record["base_take_name"] in set(takes) and (not clean_only or record["variant_name"] == "clean")
        ]
        if not records:
            raise ValueError("No records selected.")
        fields = {
            name: []
            for name in (
                "history_motion",
                "base_mu",
                "velocity_mu",
                "target",
                "traj",
                "img_embs",
                "img_available",
                "traj_available",
                "previous_reference",
                "target_joints",
                "beta_boot",
                "beta_boot_is_model",
                "floor_estimate_m",
            )
        }
        self.record_ids, self.time_indices, self.floor_diagnostics = [], [], {}
        cache = {}
        for record in records:
            count = record["num_frames"]
            observed = dataset.observations(record["variant_id"], as_of=count - 1, history_frames=count)
            episode_id = record["episode_id"]
            if episode_id not in cache:
                clean = dataset.observations(record["clean_variant_id"], as_of=count - 1, history_frames=count)
                labels = dataset.supervision(record["variant_id"])
                floor = estimate_bootstrap_floor(clean, codec, record["bootstrap_frames"])
                contact_floor = float(labels["floor_height"]) if contact_floor_source == "annotation" else floor
                states = body_states_from_supervision(
                    labels, clean["aria_traj_obs"], floor_height=floor, contact_floor_height=contact_floor
                )
                time = torch.arange(history_length, count, stride)
                hist_idx = time[:, None] + torch.arange(-history_length, 0)[None]
                hist = BodyState(*(getattr(states, key)[hist_idx] for key in ("joints", "reference", "auxiliary")))
                current = BodyState(*(getattr(states, key)[time] for key in ("joints", "reference", "auxiliary")))
                last = BodyState(*(getattr(states, key)[time - 1] for key in ("joints", "reference", "auxiliary")))
                previous = BodyState(*(getattr(states, key)[time - 2] for key in ("joints", "reference", "auxiliary")))
                cache[episode_id] = {
                    "history_motion": codec.encode_history(hist, planar_reference(states.reference[hist_idx[:, 0]])),
                    "base_mu": codec.continuation_prior(last)[:, None],
                    "velocity_mu": codec.constant_velocity_prior(previous, last)[:, None],
                    "target": codec.encode_current(current, last.reference)[:, None],
                    "previous_reference": last.reference,
                    "target_joints": current.joints[..., :3, 3],
                    "time": time,
                    "floor": floor,
                    "beta_boot": (
                        bootstrap_shapes.for_record(record)
                        if bootstrap_shapes is not None
                        else torch.zeros(10, dtype=states.auxiliary.dtype, device=states.auxiliary.device)
                    )
                    .to(states.auxiliary)
                    .expand(len(time), -1),
                    "beta_boot_is_model": torch.full((len(time),), bootstrap_shapes is not None, dtype=torch.bool),
                    "floor_estimate_m": torch.full((len(time),), floor, dtype=states.auxiliary.dtype),
                }
                self.floor_diagnostics[record["base_take_name"]] = {
                    "estimate_m": floor,
                    "annotation_m_for_offline_comparison_only": float(labels["floor_height"]),
                    "error_m": floor - float(labels["floor_height"]),
                }
            cached = cache[episode_id]
            time, floor = cached["time"], cached["floor"]
            for name in (
                "history_motion",
                "base_mu",
                "velocity_mu",
                "target",
                "previous_reference",
                "target_joints",
                "beta_boot",
                "beta_boot_is_model",
                "floor_estimate_m",
            ):
                fields[name].append(cached[name])
            trajectory = observed["aria_traj_obs"][time].clone()
            trajectory[:, 8] -= floor
            fields["traj"].append(
                codec.encode_observation(trajectory, cached["previous_reference"], observed["traj_available"][time])[
                    :, None
                ]
            )
            fields["img_embs"].append(observed["img_feats"][time, None])
            for name in ("img_available", "traj_available"):
                fields[name].append(observed[name][time, None])
            self.record_ids.extend([record["variant_id"]] * len(time))
            self.time_indices.extend(time.tolist())
        self.tensors = {name: torch.cat(values) for name, values in fields.items()}
        self.history_length = history_length

    def __len__(self):
        return len(self.record_ids)

    def batch(self, indices, device):
        return {name: value[indices].to(device) for name, value in self.tensors.items()}


def conditions_from_batch(batch, *, prior=None, action="a11"):
    history = batch["history_motion"]
    valid = torch.ones(history.shape[:2], device=history.device, dtype=torch.bool)
    mu = batch["base_mu"] if prior is None else prior(history, valid, batch["base_mu"])
    return build_conditioning(
        history_motion=history,
        history_valid=valid,
        prior_mu=mu,
        traj=batch["traj"],
        img_embs=batch["img_embs"],
        img_available=batch["img_available"],
        traj_available=batch["traj_available"],
        action=action,
    )
