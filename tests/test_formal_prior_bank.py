from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from egorecover import WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.diagnostics import synthetic_episode
from egorecover.formal_prior_bank import (
    FORMAT,
    ProducerRuntime,
    build_shard,
    generate_context_batch,
    merge_bank,
    select_contexts,
    training_anchors,
    validate_producer,
)
from egorecover.io import file_sha256
from egorecover.prior_data import TENSOR_FIELDS, prior_example, reconstructed_history_count
from egorecover.state import state_map


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FakeData:
    def __init__(self, split="train", lengths=(22, 25, 41, 65, 180)):
        self.official_split, self.digest = split, f"{split}-manifest"
        self.source_metadata = {
            f"source_{index}": {"take": f"take_{index}", "frames": length, "floor": 0.0 if index == 0 else 1.0}
            for index, length in enumerate(lengths)
        }
        self.stats = synthetic_episode().stats
        self.records = []
        self.label_beta = 0.0
        self.fail_source = None

    def _episode(self, source, start, count):
        if source == self.fail_source:
            raise RuntimeError("planned interruption")
        item = synthetic_episode(length=max(36, start + count))
        fields = {"truth": state_map(item.truth, lambda value: value[start : start + count].clone())}
        fields["truth"].auxiliary[:, -10:] = self.label_beta
        fields["teacher"] = state_map(item.teacher, lambda value: value[start : start + count].clone())
        for key in ("head", "images", "head_valid", "image_valid", "timestamps"):
            fields[key] = getattr(item, key)[start : start + count].clone()
        return replace(
            item,
            identifier=f"{source}_start_{start}",
            take=self.source_metadata[source]["take"],
            bootstrap_only=True,
            floor_calibrated=self.source_metadata[source]["floor"] != 0,
            **fields,
        )


class StableG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value, time, conditions):
        self.calls.append(len(value))
        output = conditions["center"].clone()
        output[..., 6:9] += 0.001
        output[..., -10:] += 0.001
        return output


def runtime(split="train", lengths=(22, 25, 41, 65, 180)):
    data = FakeData(split, lengths)
    config = WindowConfig(window=3, source="gaussian", shape_mode="predicted", geometry_target="official")
    system = WindowReconstructor(MotionCodec(data.stats), None, config)

    return ProducerRuntime(
        data,
        system,
        StableG(),
        lambda state: state.joints[..., :3, 3] + state.auxiliary[:, None, -10:-7] * 0.01,
        data.stats,
        {
            "checkpoint_sha256": "fixed-G",
            "step": 10000,
            "window": 3,
            "shape_mode": "predicted",
            "source": "gaussian",
            "NFE": 1,
            "training_manifest": "train-manifest",
        },
        {"fixed_source": "test"},
    )


def frozen_selection(data, counts=(80, 80), starts=(0, 20)):
    records = []
    for index, (count, start) in enumerate(zip(counts, starts)):
        source = f"source_{index}"
        records.append(
            {
                "episode_id": f"{source}_start_{start}",
                "source": source,
                "take": f"take_{index}",
                "start": start,
                "frames": count,
            }
        )
    return {"sha256": "frozen-list", "records": records}


def test_train_contexts_are_source_keyed_bounded_unique_and_have_exact_phase_anchors():
    data = FakeData(lengths=(21, 22, 24, 25, 40, 41, 43, 160, 400))
    before = torch.get_rng_state().clone()
    selected = select_contexts(data, "train")
    assert torch.equal(before, torch.get_rng_state())
    assert selected == select_contexts(data, "train")
    assert selected["excluded_short_sources"] == 1
    assert len(selected["contexts"]) == 8
    for context in selected["contexts"]:
        length = min(160, data.source_metadata[context["source"]]["frames"])
        assert context["context_length"] == length
        assert 0 <= context["context_start"] <= data.source_metadata[context["source"]]["frames"] - length
        anchors = context["anchors"]
        assert anchors == sorted(set(anchors)) and all(20 <= value < length for value in anchors)
        early = [value - 20 for value in anchors if value < 40]
        mature = [value - 20 for value in anchors if value >= 40]
        assert len(early) <= 3 and all(value in (0, 1, 4, 8, 12, 19) for value in early)
        if mature:
            assert mature[0] == 20 and mature[-1] == length - 21
            assert len(mature) == min(5, length - 40)
    assert training_anchors(22, "a") == [20, 21]


def test_frozen_validation_is_dense_and_preflight_selection_is_distinct():
    data = FakeData("val", lengths=(140, 140))
    selection = frozen_selection(data)
    full = select_contexts(data, "D128", selection=selection)
    assert full["sha256"] == "frozen-list" and full["expected_rows"] == 120
    assert full["contexts"][1]["anchors"] == list(range(20, 80))
    partial = select_contexts(data, "D128", selection=selection, limit_contexts=1)
    assert partial["sha256"] != full["sha256"]
    assert partial["parent_selection_sha256"] == full["sha256"] and partial["limited_preflight"]
    bad = deepcopy(selection)
    bad["records"][0]["take"] = "wrong take"
    with pytest.raises(ValueError, match="physical source"):
        select_contexts(data, "D400", selection=bad)


def test_continuous_p_samples_are_before_G_call_and_flags_use_local_L_after_nonzero_start():
    r = runtime("val", lengths=(150, 150))
    selection = select_contexts(r.data, "D128", selection=frozen_selection(r.data))
    rows, tensors, generation = generate_context_batch(r, selection["contexts"])
    assert generation["rows"] == 120 and generation["G_calls"] == 118
    assert generation["batched_G_calls"] == 59 and r.model.calls == [2] * 59
    for index, row in enumerate(rows):
        count = reconstructed_history_count(row["L"], 3, True)
        assert row["L"] == row["frame"] - 20
        assert row["source_query"] == row["context_start"] + row["frame"]
        torch.testing.assert_close(tensors["body_valid"][index], torch.arange(20) >= 20 - count)
    first = next(i for i, row in enumerate(rows) if row["context_start"] == 20 and row["L"] == 0)
    context = selection["contexts"][1]
    episode = r.data._episode(context["source"], 20, 80)
    truth = state_map(episode.target(20), lambda value: value[:, -1])
    expected = prior_example(r.system.codec, episode.cache().body, truth, generated_frames=0)
    for key in TENSOR_FIELDS:
        torch.testing.assert_close(tensors[key][first : first + 1], expected[key])


def test_labels_do_not_enter_predictor_inputs_or_CV_own_beta_geometry():
    a = runtime(lengths=(50,))
    selection = select_contexts(a.data, "train")
    rows_a, tensors_a, _ = generate_context_batch(a, selection["contexts"])
    b = runtime(lengths=(50,))
    b.data.label_beta = 100.0
    rows_b, tensors_b, _ = generate_context_batch(b, selection["contexts"])
    for field in ("history", "base", "reference", "body_valid", "truth_xyz"):
        torch.testing.assert_close(tensors_a[field], tensors_b[field], rtol=0, atol=0)
    assert not torch.equal(tensors_a["target"], tensors_b["target"])
    assert [row["cv_body_mm"] for row in rows_a] == [row["cv_body_mm"] for row in rows_b]


def test_chunks_resume_without_replaying_completed_contexts_and_merge_without_concatenating_all(tmp_path, monkeypatch):
    r = runtime(lengths=(22, 25, 41))
    selected = select_contexts(r.data, "train")
    r.data.fail_source = "source_1"
    with pytest.raises(RuntimeError, match="planned interruption"):
        build_shard(r, selected, tmp_path, batch_contexts=1, minimum_free_bytes=0)
    first = tmp_path / "shard0000/chunks/chunk000000.pt"
    saved_sha = file_sha256(first)
    resumed = runtime(lengths=(22, 25, 41))
    complete = build_shard(resumed, selected, tmp_path, batch_contexts=1, resume=True, minimum_free_bytes=0)
    assert complete["completed_contexts"] == 3 and file_sha256(first) == saved_sha
    assert len(resumed.model.calls) == 4 + 20  # Context0 was already committed.
    monkeypatch.setattr(
        "egorecover.formal_prior_bank.concatenate_examples",
        lambda values: (_ for _ in ()).throw(AssertionError("Full-bank cat is forbidden during merge")),
    )
    merged = merge_bank(tmp_path, minimum_free_bytes=0)
    bank = torch.load(tmp_path / "bank.pt", weights_only=True, map_location="cpu", mmap=True)
    assert bank["format"] == FORMAT and bank["producer_window"] == 3
    assert bank["stats"].keys() == r.stats.keys()
    assert bank["manifest_sha256"] == bank["producer"]["training_manifest"] == "train-manifest"
    assert len(bank["rows"]) == selected["expected_rows"] == merged["rows"]
    assert merged["coverage"]["contexts"] == 3
    assert not list(tmp_path.glob(".merge-tensors-*"))
    assert merge_bank(tmp_path, minimum_free_bytes=0) == merged


def test_multiple_shards_merge_and_val_manifest_is_separate_from_producer_train_manifest(tmp_path):
    r = runtime("val", lengths=(150, 150))
    selected = select_contexts(r.data, "D400", selection=frozen_selection(r.data))
    for shard in range(3):
        build_shard(r, selected, tmp_path, shard=shard, shards=3, batch_contexts=2, minimum_free_bytes=0)
    result = merge_bank(tmp_path, minimum_free_bytes=0)
    bank = torch.load(tmp_path / "bank.pt", weights_only=True, map_location="cpu", mmap=True)
    assert result["contexts"] == 2 and len(bank["rows"]) == 120
    assert bank["manifest_sha256"] == "val-manifest"
    assert bank["producer"]["training_manifest"] == "train-manifest"


def test_resume_rejects_corrupted_finite_chunk_tensors(tmp_path):
    r = runtime(lengths=(22,))
    selected = select_contexts(r.data, "train")
    build_shard(r, selected, tmp_path, minimum_free_bytes=0)
    path = tmp_path / "shard0000/chunks/chunk000000.pt"
    packet = torch.load(path, weights_only=True)
    packet["tensors"]["history"][0, 0, 0] += 1
    torch.save(packet, path)
    with pytest.raises(ValueError, match="checksum"):
        build_shard(r, selected, tmp_path, resume=True, minimum_free_bytes=0)


def test_producer_identity_rejects_wrong_shape_code_checkpoint_or_manifest():
    saved = {
        "step": 10000,
        "protocol": {
            "history": 20,
            "window": 3,
            "source": "gaussian",
            "sigma": 1.0,
            "steps": 1,
            "revise_recent": True,
            "body_feedback": True,
            "shape_mode": "predicted",
            "geometry_target": "official",
        },
        "identity": {
            "synthetic": False,
            "data_format": "official",
            "prior": "constant_velocity",
            "manifest": "native-train",
            "code": {"g.py": "fixed"},
        },
    }
    expected = {"sha256": "G", "step": 10000, "native_checkpoint_manifest": "native-train"}
    result = validate_producer(saved, expected, checkpoint_sha256="G", actual_code={"g.py": "fixed"})
    assert result["training_manifest"] == "native-train"
    for key, value in (("window", 1), ("shape_mode", "bootstrap"), ("revise_recent", False), ("steps", 2)):
        changed = deepcopy(saved)
        changed["protocol"][key] = value
        with pytest.raises(ValueError, match="producer|Producer"):
            validate_producer(changed, expected, checkpoint_sha256="G", actual_code={"g.py": "fixed"})
    with pytest.raises(ValueError, match="code"):
        validate_producer(saved, expected, checkpoint_sha256="G", actual_code={"g.py": "changed"})
    with pytest.raises(ValueError, match="SHA"):
        validate_producer(saved, expected, checkpoint_sha256="different", actual_code={"g.py": "fixed"})
    with pytest.raises(ValueError, match="manifest"):
        validate_producer(
            saved,
            {**expected, "native_checkpoint_manifest": "planning-digest"},
            checkpoint_sha256="G",
            actual_code={"g.py": "fixed"},
        )


def test_packed_contexts_with_different_lengths_match_continuous_serial_histories():
    packed = runtime(lengths=(22, 41, 65))
    selected = select_contexts(packed.data, "train")
    rows, tensors, generation = generate_context_batch(packed, selected["contexts"])
    lookup = {(row["context_index"], row["frame"]): index for index, row in enumerate(rows)}
    assert generation["G_calls"] == sum(max(row["anchors"]) - 20 for row in selected["contexts"])
    assert generation["batched_G_calls"] == max(max(row["anchors"]) - 20 for row in selected["contexts"])
    serial = runtime(lengths=(22, 41, 65))
    for context in selected["contexts"]:
        expected_rows, expected_tensors, _ = generate_context_batch(serial, [context])
        for index, row in enumerate(expected_rows):
            actual = lookup[(row["context_index"], row["frame"])]
            assert row == rows[actual]
            for field in TENSOR_FIELDS:
                torch.testing.assert_close(tensors[field][actual], expected_tensors[field][index], rtol=0, atol=0)


def test_long_context_selection_preserves_original_intervals_and_all_dense_queries():
    data = FakeData("val", lengths=(180, 180))
    selection = frozen_selection(data, counts=(100, 120), starts=(0, 10))
    selected = select_contexts(data, "long50", selection=selection)
    assert selected["expected_rows"] == 180
    assert selected["contexts"][1]["context_start"] == 10
    assert selected["contexts"][1]["anchors"] == list(range(20, 120))


def test_build_driver_restarts_inside_frozen_G_package_and_archives_new_builder_separately(tmp_path, monkeypatch):
    import json

    from scripts import build_formal_prior_bank as driver

    snapshot = tmp_path / "producer-source"
    (snapshot / "egorecover").mkdir(parents=True)
    (snapshot / "egorecover/__init__.py").write_text("# immutable producer fixture\n")
    plan = {"producer": {"source_snapshot": str(snapshot)}, "proposed_output_root": str(tmp_path / "exp/P01")}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path,
        output=tmp_path / "bank",
        cohort="D128",
        shard=0,
        shards=1,
        batch_contexts=2,
        device="cpu",
        limit_contexts=2,
        resume=False,
        compile_eval=False,
        smpl_directory=None,
        minimum_free_gib=0,
    )
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        driver.os, "execve", lambda executable, argv, env: captured.update(executable=executable, argv=argv, env=env)
    )
    driver.freeze_and_restart(args)
    assert captured["env"]["PYTHONPATH"] == str(snapshot)
    assert captured["env"]["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert "--_frozen-worker" in captured["argv"]
    assert (args.output / "builder-source/formal_prior_bank.py").is_file()
    assert (snapshot / "egorecover/__init__.py").read_text() == "# immutable producer fixture\n"
