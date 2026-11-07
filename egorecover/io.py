"""Atomic artifacts and reproducibility identity shared by train/evaluate/restore."""

import hashlib
import io
import json
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    """Hash tensor values independently of their device and shared storage."""
    stream = io.BytesIO()
    torch.save({name: value.detach().cpu().clone() for name, value in sorted(state.items())}, stream)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def code_identity():
    package = Path(__file__).resolve().parent
    return {str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))}
