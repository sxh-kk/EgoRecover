"""Read official EE4D files lazily, preserving their split and source timebase.

Only requested motion/feature slices move to the training device. The legacy
Episode.teacher slot holds a head-only startup prior here, never E7 output.
"""

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import torch

from .codec import BodyState, MotionCodec, planar_reference, rigid_transform, transform_from_9d
from .data import Episode, TrainingBatch
from .io import file_sha256
from .rotations import rotation_6d_to_matrix
from .state import FIELDS, state_map
from .timing import validate_timing

SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
STATS_NAMES = ("motion_mean", "motion_std", "traj_mean", "traj_std")


def _load(path):
    # These are the user-supplied official trusted Torch files, not arbitrary
    # model checkpoints. Mapping avoids copying the complete DINO tensor.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


class OfficialWindowData:
    """Official train/val data with independent take-balanced training queries."""

    def __init__(self, root, split, *, device="cpu", bootstrap=None, clip_frames=80, train_stride=20):
        if split == "test":
            raise ValueError("The official package has train/val only; val must not be relabeled as test.")
        if split not in ("train", "dev", "val"):
            raise ValueError("Official EE4D split must be train, dev, or val.")
        self.root, self.device = Path(root).resolve(), torch.device(device)
        self.split = "train" if split == "train" else "dev"
        self.official_split = "train" if split == "train" else "val"
        self.clip_frames = _integer(clip_frames, "clip_frames", 22)
        self.train_stride = _integer(train_stride, "train_stride", 1)
        self.bootstrap = bootstrap
        self.directory = self.root / "uniegomotion"
        loaded = _load(self.directory / "v4_beta_ee_train_stats.pt")
        self.stats = {name: torch.as_tensor(loaded[name]).float().clone() for name in STATS_NAMES}
        MotionCodec(self.stats)
        self.motion = _load(self.directory / f"ee_{self.official_split}.pt")
        features = _load(self.directory / f"egoview_dinov2_{self.official_split}.pt")
        if not isinstance(features, dict) or not isinstance(features.get("feats"), dict):
            raise ValueError("Official DINO file requires its feats mapping.")
        self.features = features["feats"]
        self._validate_sources()
        self.protocol = {
            "format": "official_ee4d_head_only_v1",
            "split_policy": "official train and val; dev aliases val; test forbidden",
            "motion_fps": 10,
            "source_fps": 30,
            "source_stride": 3,
            "feature_fps": 5,
            "feature_index": "floor((sequence_first30 + 3*motion_index)/6), token 0, float32",
            "floor": "provided_official_constant/no_GT_fallback",
            "zero_floor_assumption": "zero is unknown sentinel; retained without GT fallback; floor metrics excluded",
            "startup_history_frames": 20,
            "startup_body": "head-only neutral SMPL prior, not legacy teacher predictions",
            "startup_shape": "mean_beta_zero; no GT beta supplied",
            "bootstrap": getattr(bootstrap, "protocol", {"kind": "HeadAlignedBootstrap", "mean_beta_zero": True}),
            "clip_frames": self.clip_frames,
            "train_stride": self.train_stride,
            "validation_selection": "all source sequences in insertion order; range(0,T-20,20); global [::10]",
            "validation_short_clips": "skip selected clips shorter than 22; report count",
            "evaluation_boundary": "score first emissions after 20 startup frames, not the full official 80-frame score",
            "training_queries": "take-balanced source; independent uniform q in [20+rollin,T-2]; slice q-20-rollin:q+2",
            "fine_hands": "not reconstructed or scored; official PCA channels retained only in labels",
        }
        self.digest = self._identity()
        self.cache = {}
        self.sampling_inventory = {}
        self._sampling_groups = {}
        self._make_records()

    def _validate_sources(self):
        take_path, split_path = self.root / "takes.json", self.root / "annotations/splits.json"
        if not take_path.is_file() or not split_path.is_file():
            raise FileNotFoundError("Official take/split annotations are required to verify source split membership.")
        takes = json.loads(take_path.read_text())
        membership = json.loads(split_path.read_text())["take_uid_to_split"]
        names = {}
        for take in takes:
            name, uid = take["take_name"], take["take_uid"]
            if name in names and names[name] != uid:
                raise ValueError("Official take names have conflicting UIDs.")
            names[name] = uid
        if not isinstance(self.motion, dict) or not self.motion:
            raise ValueError("Official motion file must contain source sequences.")
        self.source_metadata = {}
        for name, row in self.motion.items():
            try:
                take, first, last = name.rsplit("___", 2)
                first, last = int(first), int(last)
            except (AttributeError, TypeError, ValueError):
                raise ValueError("Official source name must end in ___first30___last30.") from None
            if take not in names or membership.get(names[take]) != self.official_split:
                raise ValueError(f"Official split membership mismatch for {take} in {self.official_split}.")
            if first < 0 or last < first or (last - first) % 3:
                raise ValueError("Official source bounds must follow the original 30Hz/stride3 grid.")
            count = _integer(row["num_frames"], "num_frames", 1)
            if count != (last - first) // 3 + 1 or row["start_idx"] != first or row["end_idx"] != last:
                raise ValueError("Official source name, frame count and stored bounds disagree.")
            floor = row["floor_height"]
            if isinstance(floor, torch.Tensor) and floor.numel() == 1:
                floor = float(floor)
            if isinstance(floor, bool) or not isinstance(floor, Real) or not math.isfinite(floor):
                raise ValueError("Official floor_height must be a finite scalar.")
            if row["aria_traj"].shape != (count, 9) or row["kp3d"].shape != (count, 76, 3):
                raise ValueError("Official head/joint tensor dimensions disagree with source count.")
            params = row["smpl_params"]
            shapes = {"global_orient": (count, 6), "body_pose": (count, 21, 6), "betas": (1, 10), "transl": (count, 3)}
            shapes.update({f"{side}_hand_pose": (count, 12) for side in ("left", "right")})
            if any(params[key].shape != shape for key, shape in shapes.items()):
                raise ValueError("Official SMPL parameter dimensions disagree with source count.")
            feature = self.features.get(take)
            if not isinstance(feature, torch.Tensor) or feature.ndim != 3 or feature.shape[1:] != (5, 1024):
                raise ValueError(f"Official DINO feats missing or invalid for {take}.")
            if last // 6 >= len(feature):
                raise ValueError(f"Official DINO feature timeline does not cover source {name}.")
            self.source_metadata[name] = {"take": take, "first30": first, "frames": count, "floor": float(floor)}

    def _identity(self):
        filenames = [
            *(
                f"uniegomotion/{name}"
                for name in (
                    "ee_train.pt",
                    "ee_val.pt",
                    "egoview_dinov2_train.pt",
                    "egoview_dinov2_val.pt",
                    "ee_val_gt_for_evaluation.pkl",
                )
            ),
        ]
        files = {}
        for name in filenames:
            path = self.root / name
            info = path.stat() if path.is_file() else None
            files[name] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns} if info else None
        small = ("takes.json", "annotations/splits.json", "uniegomotion/v4_beta_ee_train_stats.pt")
        identity = {
            "large_file_identity": "size and mtime_ns metadata, not full-payload SHA256",
            "files": files,
            "small_file_sha256": {name: file_sha256(self.root / name) for name in small},
            "stats": {name: self.stats[name].tolist() for name in STATS_NAMES},
            "protocol": self.protocol,
        }
        self.identity = identity
        return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _make_records(self):
        stride = self.train_stride if self.split == "train" else 20
        candidates = [
            (name, start)
            for name, meta in self.source_metadata.items()
            for start in range(0, meta["frames"] - 20, stride)
        ]
        if self.split == "dev":
            candidates = candidates[::10]
        self.inventory = {
            "official_split": self.official_split,
            "source_sequences": len(self.source_metadata),
            "source_takes": len({meta["take"] for meta in self.source_metadata.values()}),
            "zero_floor_sources": sum(meta["floor"] == 0 for meta in self.source_metadata.values()),
            "selected_clip_candidates": len(candidates),
            "skipped_short_clips": 0,
            "official_reference_verified": False,
        }
        reference_path = self.directory / "ee_val_gt_for_evaluation.pkl"
        if self.split == "dev" and reference_path.is_file():
            import joblib

            reference = joblib.load(reference_path)
            expected = [f"{name}_start_{start}" for name, start in candidates]
            if list(reference) != expected:
                raise ValueError("Official validation candidates do not match the supplied evaluation reference keys.")
            if self.clip_frames == 80 and any(
                len(reference[key]["aria_traj_T"]) != min(80, self.source_metadata[name]["frames"] - start)
                for key, (name, start) in zip(expected, candidates)
            ):
                raise ValueError("Official evaluation reference clip lengths disagree with source slices.")
            self.inventory["official_reference_verified"] = True
            del reference
        self.records = []
        for name, start in candidates:
            meta = self.source_metadata[name]
            count = min(self.clip_frames, meta["frames"] - start)
            if count < 22:
                self.inventory["skipped_short_clips"] += 1
                continue
            self.records.append(
                {
                    "episode_id": f"{name}_start_{start}",
                    "base_seq_name": name,
                    "base_take_name": meta["take"],
                    "group": self.split,
                    "episode_start_motion_idx": start,
                    "num_frames": count,
                    "floor_height": meta["floor"],
                }
            )
        self.inventory["usable_clips"] = len(self.records)
        self.inventory["uncalibrated_floor_clips"] = sum(row["floor_height"] == 0 for row in self.records)
        self.inventory["scored_after_startup_frames"] = sum(row["num_frames"] - 20 for row in self.records)
        self.takes = sorted({meta["take"] for meta in self.source_metadata.values() if meta["frames"] >= 22})
        if not self.records or not self.takes:
            raise ValueError("Official split has no clips with twenty startup and two following frames.")

    def _startup(self, head):
        if self.bootstrap is None:
            from .bootstrap import HeadAlignedBootstrap
            from .fk import load_smpl

            self.bootstrap = HeadAlignedBootstrap.from_smpl(load_smpl("cpu"))
        # Invoke on each sensor pose separately: even a custom bootstrap cannot
        # inspect a later head observation when constructing an earlier state.
        states, beta = [], None
        for index in range(len(head)):
            state, current_beta = self.bootstrap(head[index : index + 1].clone())
            current_beta = torch.as_tensor(current_beta).detach().float().cpu()
            if current_beta.shape != (10,) or not bool(torch.equal(current_beta, torch.zeros(10))):
                raise ValueError("Official head-only bootstrap must return the fixed zero-beta mean shape.")
            if beta is not None and not torch.equal(beta, current_beta):
                raise ValueError("Bootstrap changed fixed startup shape across frames.")
            beta = current_beta
            if any(
                getattr(state, key).shape != shape
                for key, shape in (("joints", (1, 22, 4, 4)), ("reference", (1, 4, 4)), ("auxiliary", (1, 36)))
            ):
                raise ValueError("Bootstrap must return one unbatched BodyState per head observation.")
            states.append(state_map(state, lambda value: value.detach().float().cpu()))
        return BodyState(*(torch.cat([getattr(state, key) for state in states]) for key in FIELDS)), beta

    def _episode(self, name, start, count):
        meta, row = self.source_metadata[name], self.motion[name]
        if start < 0 or count < 22 or start + count > meta["frames"]:
            raise ValueError("Requested official clip escapes its source sequence or is too short.")
        interval = slice(start, start + count)
        head = row["aria_traj"][interval].float().clone()
        head[:, 8] -= meta["floor"]
        points = row["kp3d"][interval, :22].float().clone()
        points[:, :, 2] -= meta["floor"]
        params = row["smpl_params"]
        local = rotation_6d_to_matrix(
            torch.cat((params["global_orient"][interval, None], params["body_pose"][interval]), dim=1).float()
        )
        rotations = []
        for joint, parent in enumerate(SMPL22_PARENTS):
            rotations.append(local[:, joint] if parent < 0 else rotations[parent] @ local[:, joint])
        joints = rigid_transform(torch.stack(rotations, dim=1), points)
        reference = planar_reference(transform_from_9d(head))
        feet = points[:, [10, 11]]
        moving = (feet[1:] - feet[:-1]).norm(dim=-1) < 0.02
        slow = torch.cat((moving[:1], moving), dim=0)
        contacts = (slow & (feet[..., 2] < 0.10)).float()
        auxiliary = torch.cat(
            (
                params["left_hand_pose"][interval].float(),
                params["right_hand_pose"][interval].float(),
                contacts,
                params["betas"].float().expand(count, -1),
            ),
            dim=-1,
        )
        truth = BodyState(joints, reference, auxiliary)
        startup, beta = self._startup(head)
        frame_ids = meta["first30"] + 3 * torch.arange(start, start + count, dtype=torch.int64)
        timestamps = frame_ids.double() / 30
        validate_timing(timestamps, 10, count)
        feature_ids = torch.div(frame_ids, 6, rounding_mode="floor")
        # Advanced indexing copies only this small clip's token zero, never the
        # entire memory-mapped take or feature file.
        images = self.features[meta["take"]][feature_ids, 0].float()
        if any(not bool(torch.isfinite(getattr(truth, key)).all()) for key in FIELDS):
            raise ValueError("Official clip contains nonfinite supervised body state.")
        world = torch.eye(4, device=self.device)
        world[2, 3] = meta["floor"]
        episode = Episode(
            f"{name}_start_{start}",
            meta["take"],
            state_map(truth, lambda value: value.to(self.device)),
            state_map(startup, lambda value: value.to(self.device)),
            beta.to(self.device),
            head.to(self.device),
            images.to(self.device),
            torch.ones(count, dtype=torch.bool, device=self.device),
            torch.ones(count, dtype=torch.bool, device=self.device),
            timestamps,
            self.stats,
            world,
        )
        episode.floor_calibrated = meta["floor"] != 0
        episode.cache().validate()
        episode.observations(20).validate(1, 20)
        return episode

    def load(self, index):
        if index in self.cache:
            return self.cache[index]
        record = self.records[index]
        episode = self._episode(record["base_seq_name"], record["episode_start_motion_idx"], record["num_frames"])
        if len(self.cache) >= 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = episode
        return episode

    def sample_batch(self, generator, count, rollin_frames):
        if self.split != "train":
            raise ValueError("Random official queries are restricted to the training split.")
        count = _integer(count, "batch count", 1)
        rollin_frames = _integer(rollin_frames, "rollin_frames")
        frame, minimum = 20 + rollin_frames, 22 + rollin_frames
        if rollin_frames not in self._sampling_groups:
            groups = {}
            short = 0
            for name, meta in self.source_metadata.items():
                if meta["frames"] < minimum:
                    short += 1
                else:
                    groups.setdefault(meta["take"], []).append(name)
            self.sampling_inventory[rollin_frames] = {
                "required_source_frames": minimum,
                "eligible_takes": len(groups),
                "eligible_sources": sum(map(len, groups.values())),
                "skipped_short_sources": short,
                "skipped_short_only_takes": self.inventory["source_takes"] - len(groups),
            }
            self._sampling_groups[rollin_frames] = list(groups.values())
        groups = self._sampling_groups[rollin_frames]
        if not groups:
            raise ValueError(f"No official sources have the {minimum} frames required for this rollin budget.")
        episodes = []
        for _ in range(count):
            names = groups[int(torch.randint(len(groups), (), generator=generator))]
            name = names[int(torch.randint(len(names), (), generator=generator))]
            query = int(torch.randint(frame, self.source_metadata[name]["frames"] - 1, (), generator=generator))
            episodes.append(self._episode(name, query - frame, minimum))
        return TrainingBatch(tuple(episodes)), frame

    def sample(self, generator):
        return self.sample_batch(generator, 1, 0)[0].episodes[0]
