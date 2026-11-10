"""Separate sampled physical target/context from the history roll-in policy."""

import hashlib
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryTraining:
    mode: str
    rollin_frames: int
    sampling_frames: int
    startup_fraction: float
    reference_batch: int
    microbatch: int
    seed: int

    @classmethod
    def from_options(cls, args, reference_world_size):
        sampling = args.rollin_frames if args.sampling_rollin_frames is None else args.sampling_rollin_frames
        if args.history_mode not in ("fixed", "mixed"):
            raise ValueError("Unknown history training mode.")
        for name, value in (("rollin", args.rollin_frames), ("sampling", sampling)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} history length must be a nonnegative integer.")
        if sampling < args.rollin_frames:
            raise ValueError("Sampling context must cover the maximum history roll-in.")
        fraction = args.history_startup_fraction
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("History startup fraction must be between zero and one.")
        reference_batch = args.batch_size // reference_world_size
        if args.history_mode == "mixed":
            if args.sampling_rollin_frames is None or args.rollin_frames <= 4:
                raise ValueError("Mixed history requires an explicit shared sampling context and roll-in >4.")
            chunks = reference_batch / args.microbatch_size
            short = chunks * fraction
            if not chunks.is_integer() or not short.is_integer():
                raise ValueError("Startup fraction must select complete microbatches in each reference batch.")
        return cls(
            args.history_mode,
            args.rollin_frames,
            sampling,
            fraction,
            reference_batch,
            args.microbatch_size,
            args.seed,
        )

    def frames(self, step, global_offset):
        if self.mode == "fixed":
            return self.rollin_frames
        reference_rank, offset = divmod(global_offset, self.reference_batch)
        if offset % self.microbatch:
            raise ValueError("History schedule must be queried at microbatch boundaries.")
        chunks = self.reference_batch // self.microbatch

        def key(slot):
            return hashlib.sha256(f"history/{self.seed}/{step}/{reference_rank}/{slot}".encode()).digest()

        startup_slots = sorted(range(chunks), key=key)[: round(chunks * self.startup_fraction)]
        slot = offset // self.microbatch
        return (0, 1, 4)[int.from_bytes(key(slot)[:4], "big") % 3] if slot in startup_slots else self.rollin_frames

    def protocol(self):
        return {
            "mode": self.mode,
            "maximum_rollin_frames": self.rollin_frames,
            "shared_sampling_rollin_frames": self.sampling_frames,
            "target_local_frame": 20 + self.sampling_frames,
            "startup_fraction": self.startup_fraction if self.mode == "mixed" else 0,
            "startup_rollin_choices": [0, 1, 4] if self.mode == "mixed" else [],
            "schedule": "stateless per-step/reference-rank microbatch allocation; no sample/dropout RNG draws",
            "history_gradients": "detached self-rollout; current training objective only",
        }


def sampling_signature(rows):
    """Exclude history policy so four cells can verify shared queries/time/noise."""
    ordered = sorted(rows, key=lambda row: row["global_offset"])
    common = [{k: v for k, v in row.items() if k != "rollin_frames"} for row in ordered]
    return hashlib.sha256(json.dumps(common, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
