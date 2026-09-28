import json

import pytest
import torch

from egorecover.evaluation_protocol import file_sha256
from run.merge_predicted_histories import merge


def shard(root, index, take, *, scope="training_only_predicted_histories"):
    root.mkdir()
    identity = {"shard_count": 2, "shard_index": index, "selected_takes": [take],
                "group_takes": ["a", "b"], "variants": ["clean"], "scope": scope,
                "group": "train", "p_sha256": "fixed-prior", "g_sha256": {"history": "fixed-generator"}}
    payload = {"identity": identity, "tensors": {"target": torch.zeros(360, 1, 3)},
               "record_ids": [f"{take}_clean"] * 360, "time_indices": list(range(20, 200)) * 2,
               "take_names": [take] * 360,
               "generator_modes": [mode for mode in ("gaussian", "history") for _ in range(180)]}
    torch.save(payload, root / "frames.pt")
    (root / "report.json").write_text(json.dumps({"completed": True, "identity": identity, "frames": 360,
                                                "episodes": [], "cache_sha256": file_sha256(root / "frames.pt")}))
    return root


def test_merge_requires_complete_disjoint_shards_with_same_scope(tmp_path):
    a = shard(tmp_path / "a", 0, "a")
    b = shard(tmp_path / "b", 1, "b")
    report = merge([a, b], tmp_path / "merged")
    assert report["completed"] and report["frames"] == 720
    with pytest.raises(ValueError, match="exactly one shard"):
        merge([a], tmp_path / "missing")
    with pytest.raises(ValueError, match="exactly one shard"):
        merge([a, a], tmp_path / "duplicate")
    wrong = shard(tmp_path / "wrong", 1, "b", scope="development_diagnostic_predicted_histories")
    with pytest.raises(ValueError, match="identities disagree"):
        merge([a, wrong], tmp_path / "mixed")
    overlap = shard(tmp_path / "overlap", 1, "a")
    with pytest.raises(ValueError, match="overlap"):
        merge([a, overlap], tmp_path / "overlapping")
    (b / "frames.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="differs from its report"):
        merge([a, b], tmp_path / "corrupted")
