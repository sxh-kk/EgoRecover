import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .geometry import validate_rigid_transform
from .time_protocol import validate_timing


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Normalizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        for name, dimension in (("pose", cfg.pose_dim), ("root", 9), ("head", 18), ("latent", cfg.latent_dim)):
            self.register_buffer(name + "_mean", torch.zeros(dimension))
            self.register_buffer(name + "_std", torch.ones(dimension))

    def normalize(self, name, x):
        return (x.float() - getattr(self, name + "_mean")) / getattr(self, name + "_std")

    def restore(self, name, x):
        return x.float() * getattr(self, name + "_std") + getattr(self, name + "_mean")

    @torch.no_grad()
    def fit(self, name, values):
        # Streaming double-precision population moments, train split only.
        count, total, square = 0, None, None
        for x in values:
            x = x.detach().reshape(-1, x.shape[-1]).double().cpu()
            if len(x) == 0:
                continue
            if not torch.isfinite(x).all():
                raise ValueError(f"Non-finite {name} in normalization data.")
            count += len(x)
            total = x.sum(0) if total is None else total + x.sum(0)
            square = x.square().sum(0) if square is None else square + x.square().sum(0)
        if count < 2:
            raise ValueError(f"Insufficient train samples for {name} statistics.")
        mean = total / count
        std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.05)
        getattr(self, name + "_mean").copy_(mean.float())
        getattr(self, name + "_std").copy_(std.float())

    def project_state(self, state, latent_dim):
        from .geometry import matrix_to_rotation_6d, rotation_6d_to_matrix

        root = self.restore("root", state[..., latent_dim:])
        root = torch.cat((matrix_to_rotation_6d(rotation_6d_to_matrix(root[..., :6])), root[..., 6:]), -1)
        return torch.cat((state[..., :latent_dim].float(), self.normalize("root", root)), -1)


class Episodes:
    """Audited standalone episode manifest. Split membership is assigned by take."""

    def __init__(self, manifest, cfg):
        self.path = Path(manifest).resolve()
        self.cfg = cfg
        self.digest = sha256(self.path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("version") != 3:
            raise ValueError(
                "Dataset schema v3 required; re-export old data with coordinate restoration and verified timing."
            )
        if self.manifest.get("fps") != cfg.fps:
            raise ValueError("Manifest FPS mismatch.")
        self.records = self.manifest["records"]
        takes, ids = {}, set()
        for record in self.records:
            if record["split"] not in ("train", "dev", "test") or record["id"] in ids:
                raise ValueError("Invalid split or duplicate episode id.")
            ids.add(record["id"])
            old = takes.setdefault(record["take"], record["split"])
            if old != record["split"]:
                raise ValueError("Take leakage between dataset splits.")
            self.resolve(record)
        self.cache = {}

    def resolve(self, record):
        path = (self.path.parent / record["path"]).resolve()
        if not path.is_relative_to(self.path.parent) or not path.is_file():
            raise ValueError("Episode path must reference a file inside the dataset directory.")
        return path

    def split(self, name):
        records = [r for r in self.records if r["split"] == name]
        if not records:
            raise ValueError(f"Empty {name} split.")
        return records

    def load(self, record):
        if record["id"] in self.cache:
            return self.cache[record["id"]]
        path = self.resolve(record)
        if sha256(path) != record["sha256"]:
            raise ValueError("Episode checksum mismatch.")
        packet = torch.load(path, map_location="cpu", weights_only=True)
        validate_episode(packet, self.cfg)
        # Bound host memory: production manifests may contain thousands of episodes.
        if len(self.cache) >= 16:
            self.cache.pop(next(iter(self.cache)))
        self.cache[record["id"]] = packet
        return packet

    def sample(self, generator):
        train = self.split("train")
        samples = []
        length = self.cfg.bootstrap + self.cfg.target_frames
        for _ in range(self.cfg.batch_size):
            record = train[int(torch.randint(len(train), (), generator=generator))]
            packet = self.load(record)
            t = len(packet["pose"])
            start = int(torch.randint(max(t - length + 1, 1), (), generator=generator))
            sample = {
                key: value[start : start + length]
                for key, value in packet.items()
                if key in ("pose", "root", "images", "head", "image_valid", "head_valid", "valid", "joints")
            }
            actual = len(sample["pose"])
            for key, value in sample.items():
                if actual < length:
                    sample[key] = torch.cat((value, value.new_zeros(length - actual, *value.shape[1:])))
            sample["beta"] = packet["beta"]
            sample["offsets"] = packet["offsets"]
            sample["startup_pose"] = packet["startup_pose"] if start == 0 else sample["pose"][: self.cfg.bootstrap]
            sample["startup_root"] = packet["startup_root"] if start == 0 else sample["root"][: self.cfg.bootstrap]
            samples.append(sample)
        return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def validate_episode(p, cfg):
    t = len(p["pose"])
    shapes = {
        "pose": (t, 152),
        "root": (t, 9),
        "images": (t, cfg.image_dim),
        "head": (t, 18),
        "image_valid": (t,),
        "head_valid": (t,),
        "valid": (t,),
        "joints": (t, 22, 3),
        "beta": (10,),
        "offsets": (22, 3),
        "anchor": (4, 4),
        "original_world_from_anchor": (4, 4),
        "timestamps_s": (t,),
        "startup_pose": (cfg.bootstrap, 152),
        "startup_root": (cfg.bootstrap, 9),
    }
    if t <= cfg.bootstrap:
        raise ValueError("Episode must extend beyond the predicted startup prefix.")
    for key, shape in shapes.items():
        if key not in p or tuple(p[key].shape) != shape:
            raise ValueError(f"Invalid episode shape for {key}: expected {shape}.")
        mask = p["image_valid"] if key == "images" else p["head_valid"] if key == "head" else None
        value = p[key] if mask is None else p[key][mask]
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite available episode field: {key}.")
    if any(p[key].dtype != torch.bool for key in ("valid", "image_valid", "head_valid")):
        raise ValueError("Validity masks must be boolean tensors.")
    if not p["valid"][: cfg.bootstrap].all():
        raise ValueError("Startup body prefix must be valid.")
    if p.get("startup_source") not in ("predicted", "synthetic_predicted"):
        raise ValueError("Startup must be supplied by a predictor, not ground truth.")
    for key in ("anchor", "original_world_from_anchor"):
        validate_rigid_transform(p[key], key)
    validate_timing(p["timestamps_s"], cfg.fps, t)
    if "source_frame_id_30fps" in p:
        indices = p["source_frame_id_30fps"]
        if (
            not isinstance(indices, torch.Tensor)
            or indices.shape != (t,)
            or indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
            or not torch.allclose(indices.double() / 30, p["timestamps_s"].double(), atol=1e-8, rtol=0)
        ):
            raise ValueError("Source frame indices and timestamps_s disagree.")


def online_inputs(packet):
    """Whitelist only legal online inputs. Never return pose/root/joints truth."""
    keys = (
        "images",
        "head",
        "image_valid",
        "head_valid",
        "startup_pose",
        "startup_root",
        "beta",
        "offsets",
        "anchor",
        "original_world_from_anchor",
    )
    return {key: packet[key] for key in keys}
