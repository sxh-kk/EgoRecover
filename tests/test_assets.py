import hashlib
import io
import zipfile

import pytest

from egorecover.assets import asset_plan, asset_registry, destination, restore, select_assets


def entry(path, content, **kwargs):
    return {
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "repo_id": "test/repository",
        "repo_type": "dataset",
        "presets": ["data"],
        **kwargs,
    }


def registry(files, archives=()):
    return {"repositories": {"dataset": {"revision": "pinned-revision"}}, "files": files, "archives": archives}


def downloader(tmp_path, payloads):
    def fetch(**kwargs):
        assert kwargs["revision"] == "pinned-revision"
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[kwargs["filename"]])
        return str(path)

    return fetch


def test_registry_contains_only_current_assets_with_pinned_revisions():
    data = asset_registry()
    assert all(repo["revision"] for repo in data["repositories"].values())
    paths = [item["path"] for item in data["files"]]
    assert len(paths) == len(set(paths))
    assert {p for p in paths if p.startswith("pretrained/")} == {"pretrained/e7.ckpt", "pretrained/prior.pt"}
    assert all(p.startswith(("pretrained/", "datasets/paired/data/")) for p in paths)


def test_preflight_downloads_train_and_dev_without_test_payloads():
    data = asset_registry()
    selected = select_assets(data, "preflight")
    episodes = [item for item in selected if "archive" in item]
    assert len(episodes) == 2
    assert any("g-train" in item["archive"]["path"] for item in episodes)
    assert any("g-dev" in item["archive"]["path"] for item in episodes)
    assert not any("g-test" in item["archive"]["path"] for item in episodes)
    assert len(select_assets(data, "metadata")) == 1
    assert asset_plan(data, "preflight")["download_bytes"] < 300 * 1024**2


def test_train_dev_selects_all_training_development_pairs_and_prior_without_e7_or_test():
    data = asset_registry()
    selected = select_assets(data, "train-dev")
    paths = {item["path"] for item in selected}
    assert len(selected) == 9188
    assert {path for path in paths if path.startswith("pretrained/")} == {"pretrained/prior.pt"}
    assert "datasets/paired/data/manifest.json" in paths
    episodes = [item for item in selected if "archive" in item]
    assert len(episodes) == 4369 + 224
    for item in episodes:
        assert item["archive"]["path"].startswith(("bundles/g-train/", "bundles/g-dev/"))
        assert item["path"].endswith(".pt")
        assert item["path"].removesuffix(".pt") + ".json" in paths
    test_paths = {
        item["path"] for item in data["files"] if item.get("archive", {}).get("path", "").startswith("bundles/g-test/")
    }
    assert paths.isdisjoint(test_paths)
    assert paths.isdisjoint({path.removesuffix(".pt") + ".json" for path in test_paths})
    assert asset_plan(data, "train-dev")["download_bytes"] < asset_plan(data, "training")["download_bytes"]
    for preset in ("models", "training", "data", "smoke"):
        assert select_assets(data, preset) == [item for item in data["files"] if preset in item["presets"]]


def test_dry_run_does_not_create_directories_or_download(tmp_path):
    root = tmp_path / "absent"
    restore(
        root,
        "data",
        dry_run=True,
        registry=registry([entry("data.pt", b"abc")]),
        downloader=lambda **_: pytest.fail("download during dry-run"),
    )
    assert not root.exists()


def test_direct_restore_uses_original_source_name_and_verifies_existing_file(tmp_path):
    item = entry("new/data.pt", b"abc", source_path="old/data.pt")
    root = tmp_path / "destination"
    restore(
        root, "data", registry=registry([item]), downloader=downloader(tmp_path / "remote", {"old/data.pt": b"abc"})
    )
    assert (root / "new/data.pt").read_bytes() == b"abc"
    restore(
        root,
        "data",
        verify_only=True,
        registry=registry([item]),
        downloader=lambda **_: pytest.fail("download during verification"),
    )


def test_parts_are_assembled_in_order_and_wrong_bytes_never_published(tmp_path):
    one, two = entry("parts/one", b"ab"), entry("parts/two", b"cd")
    item = entry("model.pt", b"abcd", parts=[one, two])
    restore(
        tmp_path / "ok",
        "data",
        registry=registry([item]),
        downloader=downloader(tmp_path / "remote", {"parts/one": b"ab", "parts/two": b"cd"}),
    )
    assert (tmp_path / "ok/model.pt").read_bytes() == b"abcd"
    with pytest.raises(ValueError, match="checksum"):
        restore(
            tmp_path / "bad",
            "data",
            registry=registry([item]),
            downloader=downloader(tmp_path / "remote", {"parts/one": b"ab", "parts/two": b"xx"}),
        )
    assert not (tmp_path / "bad/model.pt").exists()
    assert not list((tmp_path / "bad").glob("*.restore-*"))


def test_archive_restores_only_named_member_and_checks_original_bytes(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("original/episode.pt", b"episode")
        archive.writestr("../../unrequested", b"unused")
    packed = stream.getvalue()
    member = entry("data/episode.pt", b"episode", archive={"path": "bundle.zip", "member": "original/episode.pt"})
    bundle = entry("bundle.zip", packed)
    restore(
        tmp_path / "result",
        "data",
        registry=registry([member], [bundle]),
        downloader=downloader(tmp_path / "remote", {"bundle.zip": packed}),
    )
    assert (tmp_path / "result/data/episode.pt").read_bytes() == b"episode"
    assert not (tmp_path / "unrequested").exists()


@pytest.mark.parametrize("path", ["../outside", "/absolute", "folder/../../outside"])
def test_unsafe_destinations_rejected_before_download(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        restore(
            tmp_path,
            "data",
            registry=registry([entry(path, b"x")]),
            downloader=lambda **_: pytest.fail("unsafe download"),
        )


def test_symlink_escape_and_changed_destination_are_preserved(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        destination(root, "alias/escaped.pt")
    (root / "data.pt").write_bytes(b"local work")
    with pytest.raises(ValueError, match="overwrite"):
        restore(root, "data", registry=registry([entry("data.pt", b"remote")]))
    assert (root / "data.pt").read_bytes() == b"local work"
