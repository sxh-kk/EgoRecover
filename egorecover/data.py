"""Read the published dense paired caches without converting to latent poses."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.timing import source_timing

from .io import file_sha256
from .state import FIELDS, ObservationWindow, WindowCache, state_map


def keyed_noise(identifier, seed, frame, width, device, *, draw=0):
    digest = hashlib.sha256(f"{identifier}/{seed}/{frame}/{draw}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    # Generate all ten slots before slicing, so different W share current noise.
    return torch.randn(1, 10, 243, generator=generator)[:, -width:].to(device)


@dataclass
class Episode:
    identifier: str
    take: str
    truth: BodyState
    teacher: BodyState
    beta: torch.Tensor
    head: torch.Tensor
    images: torch.Tensor
    head_valid: torch.Tensor
    image_valid: torch.Tensor
    timestamps: torch.Tensor
    stats: dict
    original_world_from_floor_world: torch.Tensor
    synthetic: bool = False

    def __len__(self):
        return self.head.shape[0]

    def restore_world(self, joints):
        transform = self.original_world_from_floor_world.to(joints)
        return joints @ transform[:3, :3].T + transform[:3, 3]

    def cache(self, frame=20):
        if not 20 <= frame < len(self):
            raise ValueError("Cache requires twenty prior predicted frames.")
        history = state_map(self.teacher, lambda x: x[frame - 20 : frame][None].detach())
        return WindowCache(history, self.beta[None].detach(), self.teacher.reference[:1].detach(), frame - 1)

    def observations(self, frame):
        if not 20 <= frame < len(self):
            raise ValueError("Observation request outside episode.")
        interval = slice(frame - 20, frame + 1)
        return ObservationWindow(
            self.head[None, interval],
            self.images[None, interval],
            self.head_valid[None, interval],
            self.image_valid[None, interval],
            frame,
        )

    def target(self, frame, width=1):
        if not 0 <= frame - width + 1 <= frame < len(self):
            raise ValueError("Target request outside episode.")
        return state_map(self.truth, lambda x: x[frame - width + 1 : frame + 1][None])


@dataclass(frozen=True)
class TrainingBatch:
    """Independent episodes at a common physical index, vectorized on device."""

    episodes: tuple[Episode, ...]

    def __post_init__(self):
        if not self.episodes:
            raise ValueError("Training batches must contain at least one episode.")

    def __len__(self):
        return len(self.episodes)

    @property
    def beta(self):
        return torch.stack([e.beta for e in self.episodes])

    def cache(self, frame):
        caches = [e.cache(frame) for e in self.episodes]
        body = BodyState(*(torch.cat([getattr(c.body, key) for c in caches]) for key in FIELDS))
        return WindowCache(body, torch.cat([c.beta for c in caches]), torch.cat([c.anchor for c in caches]), frame - 1)

    def observations(self, frame):
        observations = [e.observations(frame) for e in self.episodes]
        return ObservationWindow(
            *(
                torch.cat([getattr(o, key) for o in observations])
                for key in ("head", "images", "head_valid", "image_valid")
            ),
            frame,
        )

    def target(self, frame, width=1):
        targets = [e.target(frame, width) for e in self.episodes]
        return BodyState(*(torch.cat([getattr(t, key) for t in targets]) for key in FIELDS))

    def noise(self, seed, frame, width, *, tag, offset=0, draw=0):
        return torch.cat(
            [
                keyed_noise(f"{e.identifier}/{tag}/{offset + i}", seed, frame, width, e.beta.device, draw=draw)
                for i, e in enumerate(self.episodes)
            ]
        )


def manifest_inventory(root):
    """Validate the complete split before any episode payload is opened."""
    path = Path(root).resolve() / "data/manifest.json"
    manifest = json.loads(path.read_text())
    membership, ids = {}, set()
    for record in manifest["records"]:
        take, group, identifier = record["base_take_name"], record["group"], record["episode_id"]
        if group not in ("train", "dev", "test") or identifier in ids:
            raise ValueError("Invalid split or duplicate episode identifier.")
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier:
            raise ValueError("Invalid episode identifier.")
        ids.add(identifier)
        if membership.setdefault(take, group) != group:
            raise ValueError("Take leakage between splits.")
    splits = {}
    for group in ("train", "dev", "test"):
        records = [r for r in manifest["records"] if r["group"] == group]
        splits[group] = {
            "episodes": len(records),
            "takes": len({r["base_take_name"] for r in records}),
            "frames": sum(r["num_frames"] for r in records),
        }
    return manifest, {
        "sha256": file_sha256(path),
        "splits": splits,
        "frame_lengths": dict(Counter(r["num_frames"] for r in manifest["records"])),
    }


class PreparedWindowData:
    def __init__(self, root, split, *, device="cpu", episode_ids=None):
        self.root, self.device = Path(root).resolve(), device
        self.path = self.root / "data/manifest.json"
        manifest, inventory = manifest_inventory(self.root)
        self.digest = inventory["sha256"]
        if split not in ("train", "dev", "test"):
            raise ValueError("Unknown dataset split.")
        self.records = [r for r in manifest["records"] if r["group"] == split]
        if episode_ids is not None:
            allowed = {r["episode_id"] for r in self.records}
            if not set(episode_ids) <= allowed:
                raise ValueError("Audit episode selection is outside the requested split.")
            self.records = [r for r in self.records if r["episode_id"] in episode_ids]
        if not self.records:
            raise ValueError(f"Empty {split} split.")
        self.split, self.cache = split, {}
        self.takes = sorted({r["base_take_name"] for r in self.records})
        self.take_rows = [[i for i, r in enumerate(self.records) if r["base_take_name"] == t] for t in self.takes]
        self.stats = self.load(0).stats

    def load(self, index):
        if index in self.cache:
            return self.cache[index]
        record = self.records[index]
        path = (self.root / "data/episodes" / f"{record['episode_id']}.pt").resolve()
        if not path.is_relative_to(self.root / "data/episodes"):
            raise ValueError("Episode path escapes dataset directory.")
        marker = json.loads(path.with_suffix(".json").read_text())
        if marker.get("completed") is not True or marker["manifest_sha256"] != self.digest:
            raise ValueError("Unfinished episode or changed manifest.")
        if file_sha256(path) != marker["sha256"]:
            raise ValueError("Episode checksum mismatch.")
        packet = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if packet["record"] != record:
            raise ValueError("Episode identity mismatch.")
        _, timestamps, _ = source_timing(packet, SimpleNamespace(fps=10.0))
        floor = record.get("floor_height")
        if isinstance(floor, bool) or not isinstance(floor, (float, int)) or not math.isfinite(floor):
            raise ValueError("Source record must declare its finite removed floor_height.")
        original_world_from_floor_world = torch.eye(4, device=self.device)
        original_world_from_floor_world[2, 3] = floor
        n = len(timestamps)
        if n < 22:
            raise ValueError("Episode is too short for feedback training.")
        stats = {k: torch.as_tensor(v).float() for k, v in packet["stats"].items()}
        MotionCodec(stats)  # Validate all four training-stat buffers.
        if hasattr(self, "stats") and any(not torch.equal(stats[k], self.stats[k]) for k in self.stats):
            raise ValueError("Training statistics changed across episodes/splits.")
        device = self.device
        truth, teacher = [
            BodyState(*(packet[key][name].float().to(device) for name in FIELDS)) for key in ("truth", "teacher")
        ]
        head, images, beta = (packet[k].float().to(device) for k in ("head", "images", "beta_boot"))
        head_valid = packet.get("head_valid", torch.ones(n, dtype=torch.bool)).to(device)
        image_valid = packet.get("image_valid", torch.ones(n, dtype=torch.bool)).to(device)
        episode = Episode(
            record["episode_id"],
            record["base_take_name"],
            truth,
            teacher,
            beta,
            head,
            images,
            head_valid,
            image_valid,
            timestamps,
            stats,
            original_world_from_floor_world,
        )
        episode.cache().validate()
        for state in (truth, teacher):
            if (
                state.joints.shape != (n, 22, 4, 4)
                or state.reference.shape != (n, 4, 4)
                or state.auxiliary.shape != (n, 36)
            ):
                raise ValueError("Invalid dense episode state dimensions.")
            if any(not bool(torch.isfinite(getattr(state, key)).all()) for key in FIELDS):
                raise ValueError("Nonfinite dense episode state.")
        episode.observations(20).validate(1, 20)
        if head.shape != (n, 9) or images.shape != (n, 1024):
            raise ValueError("Invalid complete observation timeline.")
        if head_valid.shape != (n,) or image_valid.shape != (n,):
            raise ValueError("Invalid observation availability timeline.")
        for values, valid in ((head, head_valid), (images, image_valid)):
            if valid.dtype != torch.bool or not bool(torch.isfinite(values[valid]).all()):
                raise ValueError("Invalid available observation timeline.")
        if len(self.cache) >= 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = episode
        return episode

    def sample(self, generator):
        if self.split != "train":
            raise ValueError("Random sampling is restricted to training takes.")
        rows = self.take_rows[int(torch.randint(len(self.takes), (), generator=generator))]
        return self.load(rows[int(torch.randint(len(rows), (), generator=generator))])
