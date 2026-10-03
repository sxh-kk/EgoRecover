import json
import pytest
import torch

from egorecover.evaluation_protocol import file_sha256
from egorecover.prior_adaptation import (FixedHistoryFrames, SCOPES, SHAPES, TEACHER_KEYS,
                                         evaluate, summarize_errors, validate_teachers)
from run.adapt_prior_on_predictions import dependency_status
from run.complete_stages import shared_gpu_indices


def test_start_requires_both_egorecover_queues_to_finish(tmp_path):
    paths = [tmp_path / name for name in ("stages.json", "ablation.json")]
    assert not dependency_status(paths)[0]
    done = {"status": "complete", "tasks": {"job": {"status": "complete"}}}
    for path in paths:
        path.write_text(json.dumps(done))
    assert dependency_status(paths)[0]
    for state in ("running", "failed", "waiting_for_gpu_memory"):
        paths[0].write_text(json.dumps({**done, "status": state}))
        assert not dependency_status(paths)[0]
    paths[0].write_text(json.dumps({**done, "tasks": {"job": {"status": "running"}}}))
    assert not dependency_status(paths)[0]
    paths[0].write_text("partial json")
    assert not dependency_status(paths)[0]


def test_shared_gpu_policy_does_not_wait_for_lingbot(monkeypatch):
    def query(argv, **kwargs):
        # Only memory/default compute mode is relevant; occupied cards are allowed.
        assert not any("query-compute-apps" in item or "utilization" in item for item in argv)
        return "0, 98304, 0, Default\n4, 98304, 50000, Default\n5, 98304, 80000, Default\n"
    monkeypatch.setattr("run.complete_stages.subprocess.check_output", query)
    assert shared_gpu_indices([4, 5, 6, 7], {}, 32) == {4}


def make_cache(tmp_path, group):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dataset_spec_sha256": "spec", "splits":
                                {"train": ["a"], "dev": ["b"], "holdout": ["c"]}}))
    stats = tmp_path / "stats.pt"
    stats.write_bytes(b"normalization")
    take = {"train": "a", "dev": "b"}[group]
    records = {name: {"base_take_name": name, "variant_name": "clean", "bootstrap_frames": 20,
                      "num_frames": 22} for name in ("a", "b", "c")}
    identity = {key: "same" for key in TEACHER_KEYS}
    identity.update(scope=SCOPES[group], group=group, bootstrap_is_model=True,
                    selected_takes=[take], group_takes=[take], variants=["clean"],
                    stats_sha256=file_sha256(stats), split_manifest_sha256=file_sha256(split),
                    dataset_spec_sha256="spec")
    payload = {"identity": identity,
               "tensors": {key: torch.zeros(4, *shape) for key, shape in SHAPES.items()},
               "record_ids": [take] * 4, "take_names": [take] * 4,
               "time_indices": [20, 21] * 2, "generator_modes": ["gaussian"] * 2 + ["history"] * 2}
    payload["tensors"]["beta_boot_is_model"] = torch.ones(4, dtype=torch.bool)
    folder = tmp_path / group
    folder.mkdir()
    path = folder / "frames.pt"
    def write():
        torch.save(payload, path)
        (folder / "report.json").write_text(json.dumps({"completed": True, "frames": 4,
            "identity": identity, "cache_sha256": file_sha256(path)}))
    write()
    return path, dict(group=group, split_path=split, stats_path=stats, records=records), payload, write


def test_cache_rejects_split_leaks_scope_swaps_and_duplicate_targets(tmp_path):
    path, args, payload, write = make_cache(tmp_path, "train")
    frames = FixedHistoryFrames(path, **args)
    assert len(frames) == 4
    with pytest.raises(ValueError, match="scope/group"):
        FixedHistoryFrames(path, **{**args, "group": "dev"})
    payload["take_names"][0] = "b"
    write()
    with pytest.raises(ValueError, match="different take/split"):
        FixedHistoryFrames(path, **args)
    payload["take_names"][0] = "a"
    payload["time_indices"][1] = 20
    write()
    with pytest.raises(ValueError, match="duplicated"):
        FixedHistoryFrames(path, **args)
    payload["time_indices"][1] = 21
    payload["tensors"]["beta_boot_is_model"][0] = False
    write()
    with pytest.raises(ValueError, match="model-generated"):
        FixedHistoryFrames(path, **args)


def test_train_dev_use_identical_frozen_teacher(tmp_path):
    path, args, _, _ = make_cache(tmp_path, "train")
    train = FixedHistoryFrames(path, **args)
    path, args, _, _ = make_cache(tmp_path, "dev")
    dev = FixedHistoryFrames(path, **args)
    validate_teachers(train, dev)
    dev.identity["p_sha256"] = "adapted-prior-cannot-regenerate-selection-histories"
    with pytest.raises(ValueError, match="same frozen P/G"):
        validate_teachers(train, dev)


def test_fixed_state_evaluation_uses_fk_for_all_three_predictions(tmp_path, monkeypatch):
    path, args, _, _ = make_cache(tmp_path, "dev")
    frames = FixedHistoryFrames(path, **args)
    frames.tensors["base_mu"][:] = 2
    frames.tensors["velocity_mu"][:] = 3
    original = frames.tensors["history_motion"].clone()
    class Prior(torch.nn.Module):
        def forward(self, history, valid, base):
            assert valid.all()
            return base + 3
    observed = []
    def fk(codec, prediction, batch, smpl):
        assert not torch.is_grad_enabled()
        observed.append(batch["history_motion"].clone())
        error = torch.zeros(len(prediction), 22, 3)
        error[..., 0] = prediction[:, 0, 0, None]
        return error
    monkeypatch.setattr("egorecover.prior_adaptation.fk_position_errors", fk)
    scores, errors = evaluate(Prior(), frames, None, None, "cpu", 3, baselines=True)
    assert scores["prior"]["macro_fk22_mm"] == 5000
    assert scores["hold"]["macro_fk22_mm"] == 2000
    assert scores["constant_velocity"]["macro_fk22_mm"] == 3000
    assert all(len(values) == 4 for values in errors.values())
    torch.testing.assert_close(frames.tensors["history_motion"], original)
    assert len(observed) == 6  # Same batch for P, hold and CV, including the final partial batch.


def test_summary_weights_takes_equally_despite_unequal_frame_counts():
    class Frames:
        metadata = {"take_names": ["a", "a", "a", "b"], "generator_modes": ["history"] * 4}
        variants = ["clean"] * 4
        def __len__(self):
            return 4
    assert summarize_errors(torch.tensor([10., 10., 10., 30.]), Frames())["macro_fk22_mm"] == 20
