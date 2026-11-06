"""Read the published dense paired caches without converting to latent poses."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from egorecover.codec import BodyState, MotionCodec, transform_to_9d
from motionstreamer_flow.time_protocol import source_timing, validate_timing
from .state import FIELDS, ObservationWindow, WindowCache, state_map


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class PreparedWindowData:
    def __init__(self, root, split, *, device="cpu"):
        self.root, self.device = Path(root).resolve(), device
        self.path = self.root / "data/manifest.json"
        self.digest = file_sha256(self.path)
        manifest = json.loads(self.path.read_text())
        if split not in ("train", "dev", "test"):
            raise ValueError("Unknown dataset split.")
        membership, ids = {}, set()
        for record in manifest["records"]:
            take, group = record["base_take_name"], record["group"]
            if group not in ("train", "dev", "test") or record["episode_id"] in ids:
                raise ValueError("Invalid split or duplicate episode identifier.")
            ids.add(record["episode_id"])
            if membership.setdefault(take, group) != group:
                raise ValueError("Take leakage between splits.")
        self.records = [r for r in manifest["records"] if r["group"] == split]
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
        if not marker.get("completed") or marker["manifest_sha256"] != self.digest:
            raise ValueError("Unfinished episode or changed manifest.")
        if file_sha256(path) != marker["sha256"]:
            raise ValueError("Episode checksum mismatch.")
        packet = torch.load(path, map_location="cpu", weights_only=True)
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


def synthetic_episode(*, device="cpu", length=36):
    """Explicit integration fixture, never a substitute for measured accuracy."""
    if length < 32:
        raise ValueError("Synthetic episode must support the ten-step diagnostic.")
    t = torch.arange(length, device=device).float()
    reference = torch.eye(4, device=device).repeat(length, 1, 1)
    reference[:, 0, 3] = t * 0.02
    joints = reference[:, None].repeat(1, 22, 1, 1)
    joints[:, :, 1, 3] += torch.linspace(-0.3, 0.3, 22, device=device)
    joints[:, :, 2, 3] += torch.linspace(0.2, 1.7, 22, device=device)
    auxiliary = torch.zeros(length, 36, device=device)
    truth = BodyState(joints, reference, auxiliary)
    teacher_joints = joints.clone()
    teacher_joints[..., 0, 3] += 0.01 * torch.sin(t)[:, None]
    teacher = BodyState(teacher_joints, reference.clone(), auxiliary.clone())
    head_transform = reference.clone()
    head_transform[:, 2, 3] = 1.7
    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
    }
    identity = torch.tensor([1, 0, 0, 0, 1, 0.0])
    for joint in range(23):
        stats["motion_mean"][joint * 9 : joint * 9 + 6] = identity
    episode = Episode(
        "synthetic",
        "synthetic",
        truth,
        teacher,
        torch.zeros(10, device=device),
        transform_to_9d(head_transform),
        torch.zeros(length, 1024, device=device),
        torch.ones(length, dtype=torch.bool, device=device),
        torch.ones(length, dtype=torch.bool, device=device),
        torch.arange(length).double() / 10,
        stats,
        torch.eye(4, device=device),
        synthetic=True,
    )
    validate_timing(episode.timestamps, 10, length)
    return episode
