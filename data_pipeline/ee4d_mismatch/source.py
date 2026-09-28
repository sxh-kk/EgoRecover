"""Read the official processed files without initializing E7 or SMPL-X."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .corruptions import clean_arrays, stable_seed


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def annotation_digest(reference):
    """Hash the physical labels, including shape; no model encoding is involved."""
    digest = hashlib.sha256()

    def visit(name, value):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(name + "/" + key, value[key])
        else:
            array = np.ascontiguousarray(as_numpy(value))
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"Invalid supervision in {name}")
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes())

    visit("labels", reference)
    return digest.hexdigest()


class EE4DSource:
    def __init__(self, root, split="val"):
        if split not in {"train", "val"}:
            raise ValueError("Source split must be train or val")
        self.root = Path(root).resolve()
        self.split = split
        self.motion_path = self.root / "uniegomotion" / f"ee_{split}.pt"
        self.feature_path = self.root / "uniegomotion" / f"egoview_dinov2_{split}.pt"
        self.metadata_path = self.root / "takes.json"
        self.split_path = self.root / "annotations" / "splits.json"
        for path in (self.motion_path, self.feature_path, self.metadata_path, self.split_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required processed asset is not ready: {path}")
        # Official torch files contain NumPy scalars as well as tensor storages.
        # Like the upstream loader, this expects trusted official processed files.
        self.motion = torch.load(self.motion_path, map_location="cpu", weights_only=False, mmap=True)
        feature_file = torch.load(self.feature_path, map_location="cpu", weights_only=False, mmap=True)
        self.features = feature_file["feats"]
        self.metadata = {item["take_name"]: item for item in json.loads(self.metadata_path.read_text())}
        if not isinstance(self.motion, dict) or not isinstance(self.features, dict):
            raise ValueError("Expected official sequence and feature dictionaries")

    def fingerprints(self):
        paths = [self.motion_path, self.feature_path, self.metadata_path, self.split_path]
        stats = self.root / "uniegomotion" / "v4_beta_ee_train_stats.pt"
        if stats.is_file():
            paths.append(stats)
        return {
            str(path.relative_to(self.root)): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        }

    def inventory(self, frames=200):
        candidates, rejected = {}, {}
        for name, item in self.motion.items():
            try:
                take, start, end = name.rsplit("___", 2)
                start, end = int(start), int(end)
                count = int(item["num_frames"])
                if count != (end - start) // 3 + 1 or (end - start) % 3:
                    raise ValueError("sequence_time_mismatch")
                if tuple(item["aria_traj"].shape) != (count, 9):
                    raise ValueError("trajectory_shape")
                if count < frames:
                    raise ValueError("too_short")
                if take not in self.features:
                    raise ValueError("missing_visual_take")
                feature = self.features[take]
                if feature.ndim not in (2, 3) or feature.shape[-1] != 1024:
                    raise ValueError("feature_shape")
                if feature.ndim == 3 and feature.shape[1] < 1:
                    raise ValueError("missing_feature_token")
                if end // 3 // 2 >= len(feature):
                    raise ValueError("feature_time_bounds")
                if take not in self.metadata:
                    raise ValueError("missing_take_metadata")
                candidates.setdefault(take, []).append(name)
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error)
                rejected[reason] = rejected.get(reason, 0) + 1
        return candidates, {
            "source_sequences": len(self.motion),
            "source_takes": len({name.rsplit("___", 2)[0] for name in self.motion}),
            "feature_takes": len(self.features),
            "eligible_takes": len(candidates),
            "eligible_sequences": sum(map(len, candidates.values())),
            "required_frames": frames,
            "rejected_sequences": rejected,
        }

    def select_episodes(self, num_takes, frames, seed, bootstrap=20, recovery=80):
        if num_takes < 1 or bootstrap < 1 or recovery < 0:
            raise ValueError("Invalid episode selection limits")
        minimum_onset = bootstrap + 10
        maximum_onset = frames - recovery - 30
        if maximum_onset < minimum_onset:
            raise ValueError("Episode is too short for bootstrap, event and recovery observation")
        candidates, inventory = self.inventory(frames)
        if num_takes > len(candidates):
            raise ValueError(f"Requested {num_takes} takes; only {len(candidates)} are eligible")
        # Round-robin task strata, with deterministic randomization within strata.
        groups = {}
        for take in candidates:
            metadata = self.metadata[take]
            task = metadata.get("parent_task_name") or metadata.get("task_name") or "unknown"
            groups.setdefault(task, []).append(take)
        rng = np.random.default_rng(seed)
        tasks = sorted(groups)
        rng.shuffle(tasks)
        for names in groups.values():
            names.sort()
            rng.shuffle(names)
        chosen = []
        while len(chosen) < num_takes:
            for task in tasks:
                if groups[task] and len(chosen) < num_takes:
                    chosen.append((groups[task].pop(), task))
        episodes = []
        for take, task in chosen:
            local = np.random.default_rng(stable_seed(seed, take, "episode"))
            names = sorted(candidates[take])
            name = names[int(local.integers(len(names)))]
            length = int(self.motion[name]["num_frames"])
            start = int(local.integers(length - frames + 1))
            onset = int(local.integers(minimum_onset, maximum_onset + 1))
            identity = f"{name}:{start}:{frames}"
            episode_id = "ep_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            episodes.append(dict(
                episode_id=episode_id, base_take_name=take, base_seq_name=name,
                base_split=self.split, episode_start_motion_idx=start, num_frames=frames,
                bootstrap_frames=bootstrap, fault_onset=onset, task_name=task,
                seed=stable_seed(seed, episode_id),
            ))
        return episodes, inventory

    def select_all_sequences(self, seed, bootstrap=20, recovery=80):
        """Use each source sequence exactly once, including sequences below 200 frames.

        Short sequences have a shorter clean bootstrap and a clipped fault so
        that each sequence still has a meaningful, post-bootstrap corruption.
        The requested durations and attainable minimum recovery are recorded.
        """
        if bootstrap < 2 or recovery < 0:
            raise ValueError("Full validation requires bootstrap >= 2 and recovery >= 0")
        candidates, inventory = self.inventory(1)
        if inventory["eligible_sequences"] != inventory["source_sequences"]:
            raise ValueError(f"Cannot cover every source sequence: {inventory['rejected_sequences']}")
        episodes = []
        for name in sorted(self.motion):
            take = name.rsplit("___", 2)[0]
            length = int(self.motion[name]["num_frames"])
            if length < 6:
                raise ValueError(f"Sequence is too short for an observable mismatch: {name}")
            boot = min(bootstrap, max(2, length // 3))
            precontext = min(10, max(2, length // 8))
            if boot + precontext + 2 > length:
                raise ValueError(f"No room for at least two faulty observations: {name}")
            available_fault = min(30, length - boot - precontext)
            minimum_recovery = min(recovery, length - boot - precontext - available_fault)
            first_onset = boot + precontext
            last_onset = length - minimum_recovery - available_fault
            local = np.random.default_rng(stable_seed(seed, name, "full_sequence_onset"))
            onset = int(local.integers(first_onset, last_onset + 1))
            identity = f"{name}:0:{length}"
            episode_id = "ep_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            metadata = self.metadata[take]
            task = metadata.get("parent_task_name") or metadata.get("task_name") or "unknown"
            episodes.append(dict(
                episode_id=episode_id, base_take_name=take, base_seq_name=name,
                base_split=self.split, episode_start_motion_idx=0, num_frames=length,
                bootstrap_frames=boot, fault_onset=onset, task_name=task,
                minimum_recovery_frames=minimum_recovery,
                available_fault_frames=available_fault,
                seed=stable_seed(seed, episode_id),
            ))
        if len({episode["episode_id"] for episode in episodes}) != len(episodes):
            raise ValueError("Episode identifier collision")
        return episodes, inventory

    def clean(self, episode):
        name = episode["base_seq_name"]
        item = self.motion[name]
        start = episode["episode_start_motion_idx"]
        end = start + episode["num_frames"]
        if start < 0 or end > int(item["num_frames"]):
            raise ValueError("Episode exceeds source sequence")
        start30 = int(name.rsplit("___", 2)[1])
        return clean_arrays(as_numpy(item["aria_traj"])[start:end], start30, start, episode["bootstrap_frames"])

    def supervision(self, episode):
        """Copy physical annotations separately from the observation API."""
        item = self.motion[episode["base_seq_name"]]
        start = episode["episode_start_motion_idx"]
        end = start + episode["num_frames"]
        parameters = {}
        for name, value in item["smpl_params"].items():
            array = as_numpy(value)
            parameters[name] = array.copy() if name == "betas" else array[start:end].copy()
        return {
            "smpl_params": parameters,
            "kp3d": as_numpy(item["kp3d"])[start:end].copy(),
            "body_root_offset": as_numpy(item["body_root_offset"]).copy(),
            "floor_height": as_numpy(item["floor_height"]).copy(),
        }

    def visual_payload(self, take, indices, available):
        """Missing -1 indices never reach tensor indexing; inputs are returned as copies."""
        indices = np.asarray(indices, dtype=np.int64)
        available = np.asarray(available, dtype=bool)
        if indices.shape != available.shape or indices.ndim != 1:
            raise ValueError("Visual indices and availability must have shape [T]")
        feature = self.features[take]
        valid_indices = indices[available]
        if np.any(valid_indices < 0) or np.any(valid_indices >= len(feature)):
            raise ValueError("Visual index is out of bounds")
        output = torch.zeros((len(indices), 1024), dtype=torch.float32)
        if len(valid_indices):
            selected = feature[torch.from_numpy(valid_indices.copy())]
            if selected.ndim == 3:
                selected = selected[:, 0]
            selected = torch.as_tensor(selected).float()
            if not torch.isfinite(selected).all():
                raise ValueError("Non-finite DINO payload")
            output[torch.from_numpy(available)] = selected
        return output
