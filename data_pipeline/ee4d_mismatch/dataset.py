"""A reader that keeps online observations separate from annotations and audit data."""

import json
from pathlib import Path

import numpy as np
import torch

from .corruptions import validate_arrays
from .source import EE4DSource, file_sha256


class MismatchDataset:
    def __init__(self, root, data_root=None, source=None):
        self.root = Path(root).resolve()
        self.spec = json.loads((self.root / "spec.json").read_text())
        if self.spec["schema_version"] != "ee4d-mismatch-v1":
            raise ValueError("Unsupported mismatch schema")
        self.records = []
        for manifest in self.spec["manifests"]:
            path = self._inside(manifest)
            self.records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        ids = [item["variant_id"] for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate variant IDs")
        self.by_id = {item["variant_id"]: item for item in self.records}
        self.source = source or EE4DSource(data_root or self.spec["source_root"], self.spec["base_split"])
        if self.source.split != self.spec["base_split"]:
            raise ValueError("Source split does not match manifest")

    def _inside(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Dataset path escapes its root")
        return path

    def __len__(self):
        return len(self.records)

    def record(self, index):
        return self.by_id[index] if isinstance(index, str) else self.records[index]

    def arrays(self, index, verify_hash=False):
        """Offline data access. This includes audit fields, unlike observations()."""
        record = self.record(index)
        path = self._inside(record["array_file"])
        if verify_hash and file_sha256(path) != record["array_sha256"]:
            raise ValueError(f"Array checksum mismatch: {record['variant_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays, len(self.source.features[record["base_take_name"]]))
        return arrays

    def observations(self, index, *, as_of, history_frames=1):
        """Return only actual payloads up to as_of; no fault labels, source indices or GT.

        The returned trajectory is absolute 9D in the source world frame. An E7
        adapter must compute its 18D conditions separately using a legal reference.
        """
        record = self.record(index)
        if isinstance(as_of, bool) or not isinstance(as_of, int) or not 0 <= as_of < record["num_frames"]:
            raise ValueError("as_of must be a valid episode frame")
        if isinstance(history_frames, bool) or not isinstance(history_frames, int) or history_frames < 1:
            raise ValueError("history_frames must be positive")
        arrays = self.arrays(index)
        selected = slice(max(0, as_of-history_frames+1), as_of+1)
        image = self.source.visual_payload(record["base_take_name"], arrays["img_source_idx"][selected],
                                           arrays["img_available"][selected])
        return {
            "img_feats": image,
            "aria_traj_obs": torch.from_numpy(arrays["aria_traj_obs"][selected].copy()),
            "img_available": torch.from_numpy(arrays["img_available"][selected].copy()),
            "traj_available": torch.from_numpy(arrays["traj_available"][selected].copy()),
            "frame_id_30fps": torch.from_numpy(arrays["frame_id_30fps"][selected].copy()),
        }

    def supervision(self, index):
        return self.source.supervision(self.record(index))
