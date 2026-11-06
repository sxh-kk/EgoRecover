"""Pinned asset restoration with checked transport parts and atomic writes."""

import gzip
import json
import shutil
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

from .io import file_sha256


def asset_registry():
    return json.loads(gzip.decompress(files("egorecover").joinpath("resources/assets.json.gz").read_bytes()))


def destination(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe asset destination: {relative}")
    return path


def matches(path, item):
    return path.is_file() and path.stat().st_size == item["bytes"] and file_sha256(path) == item["sha256"]


def select_assets(registry, preset):
    """Metadata and a small train/dev check do not require the full dataset."""
    if preset == "metadata":
        return [f for f in registry["files"] if f["path"] == "datasets/paired/data/manifest.json"]
    if preset == "preflight":
        wanted = {"datasets/paired/data/manifest.json", "pretrained/prior.pt"}
        sizes = {a["path"]: a["bytes"] for a in registry["archives"]}
        for group in ("train", "dev"):
            candidates = [
                f
                for f in registry["files"]
                if f["path"].endswith(".pt") and f.get("archive", {}).get("path", "").startswith(f"bundles/g-{group}/")
            ]
            if not candidates:
                raise ValueError(f"No {group} episode available for preflight.")
            chosen = min(candidates, key=lambda f: (sizes[f["archive"]["path"]], f["path"]))
            wanted.update((chosen["path"], str(Path(chosen["path"]).with_suffix(".json"))))
        return [f for f in registry["files"] if f["path"] in wanted]
    return [item for item in registry["files"] if preset in item["presets"]]


def asset_plan(registry, preset):
    selected = select_assets(registry, preset)
    archive_names = {f["archive"]["path"] for f in selected if "archive" in f}
    archives = [a for a in registry.get("archives", []) if a["path"] in archive_names]
    payload = sum(f["bytes"] for f in selected if "archive" not in f) + sum(a["bytes"] for a in archives)
    logical = sum(f["bytes"] for f in selected)
    return {
        "preset": preset,
        "files": len(selected),
        "restored_bytes": logical,
        "download_bytes": payload,
        "estimated_retained_bytes_with_cache": logical + payload + sum(a["bytes"] for a in archives),
    }


class AssetStore:
    def __init__(self, root, registry, downloader=None):
        self.root = Path(root).resolve()
        self.registry = registry
        self.downloader = downloader
        self.verified = set()

    def download(self, item, source=None):
        downloader = self.downloader
        if downloader is None:
            from huggingface_hub import hf_hub_download

            downloader = hf_hub_download
        kind = item.get("repo_type", "dataset")
        revision = self.registry["repositories"][kind]["revision"]
        if not revision:
            raise ValueError("Asset revision must be pinned.")
        return Path(
            downloader(
                repo_id=item["repo_id"],
                repo_type=kind,
                revision=revision,
                filename=source or item.get("source_path", item["path"]),
                cache_dir=str(self.root / ".downloads"),
            )
        )

    def assemble(self, item, destination_path):
        with destination_path.open("wb") as out:
            for part in item["parts"]:
                cached = self.download(item, part["path"])
                if not matches(cached, part):
                    raise ValueError(f"Asset transport part checksum mismatch: {part['path']}")
                with cached.open("rb") as source:
                    shutil.copyfileobj(source, out)

    def archive(self, name):
        item = next(a for a in self.registry["archives"] if a["path"] == name)
        path = destination(self.root / ".archives", item["path"])
        if not path.exists():
            self.write(item, path, use_archive=False)
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if key not in self.verified:
            if not matches(path, item):
                raise ValueError("Cached archive checksum mismatch.")
            self.verified.add(key)
        return path

    def write(self, item, path, *, use_archive=True):
        if path.exists():
            raise ValueError(f"Refusing to overwrite an existing asset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".restore-", delete=False) as out:
                temp = Path(out.name)
                if "archive" in item and use_archive:
                    with zipfile.ZipFile(self.archive(item["archive"]["path"])) as archive:
                        info = archive.getinfo(item["archive"]["member"])
                        if info.file_size != item["bytes"]:
                            raise ValueError("Archive member size mismatch.")
                        with archive.open(info) as source:
                            shutil.copyfileobj(source, out)
                elif "parts" in item:
                    out.close()
                    self.assemble(item, temp)
                else:
                    cached = self.download(item)
                    with cached.open("rb") as source:
                        shutil.copyfileobj(source, out)
            if not matches(temp, item):
                raise ValueError(f"Restored asset checksum mismatch: {item['path']}")
            # Exclusive creation also protects a destination that appeared while
            # downloading. The completed bytes are never partially published.
            import os

            os.link(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def restore(root, preset="models", *, dry_run=False, verify_only=False, registry=None, downloader=None):
    registry = asset_registry() if registry is None else registry
    selected = select_assets(registry, preset)
    if not selected:
        raise ValueError(f"Empty asset preset: {preset}")
    store = AssetStore(root, registry, downloader)
    for item in selected:
        destination(store.root, item["path"])
    logical_bytes = sum(item["bytes"] for item in selected)
    archives = {item["archive"]["path"] for item in selected if "archive" in item}
    download_bytes = sum(item["bytes"] for item in selected if "archive" not in item)
    download_bytes += sum(item["bytes"] for item in registry.get("archives", []) if item["path"] in archives)
    print(
        f"{preset}: {len(selected)} files, {logical_bytes / 1024**3:.3f} GiB restored, "
        f"up to {download_bytes / 1024**3:.3f} GiB download payload",
        flush=True,
    )
    if dry_run:
        return
    for index, item in enumerate(selected, 1):
        path = destination(store.root, item["path"])
        if not matches(path, item):
            if verify_only:
                raise ValueError(f"Missing or changed asset: {item['path']}")
            store.write(item, path)
        if index % 100 == 0 or index == len(selected):
            print(f"Verified {index}/{len(selected)} asset files.", flush=True)
    print("All selected assets match their recorded sizes and SHA256 hashes.")
