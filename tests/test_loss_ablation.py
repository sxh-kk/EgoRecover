import json

import pytest
import torch

from egorecover.fixed_prior import load_fixed_prior
from egorecover.losses import physical_objective
from mydiffusion.flow_matching import FlowMatching
from run import complete_stages
from run.loss_ablation import result_summary, tasks_for


def test_fixed_prior_checks_data_bootstrap_selection_and_weights(tmp_path):
    keys = ("stats_sha256", "split_manifest_sha256", "bootstrap_cache_sha256", "reference_mode",
            "e7_checkpoint_sha256", "e7_weight_source", "splits", "selection_split_manifest_sha256",
            "selection_dataset_spec_sha256", "selection_bootstrap_cache_sha256", "prior_selection")
    target = {key: key for key in keys}
    report = {**target, "completed": True, "prior": {"selected_step": 20}, "prior_steps": 2400,
              "geometry_weight": 1, "fk_weight": 0}
    checkpoint = {**target, "kind": "history_prior", "selected_step": 20,
                  "state_dict": {"weight": torch.ones(2)}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    torch.save(checkpoint, tmp_path / "prior.pt")
    _, _, identity = load_fixed_prior(tmp_path, target)
    assert identity["original_geometry_weight"] == 1
    for key in ("bootstrap_cache_sha256", "splits", "e7_checkpoint_sha256", "selection_split_manifest_sha256"):
        with pytest.raises(ValueError, match=key):
            load_fixed_prior(tmp_path, {**target, key: "different"})
    checkpoint["state_dict"]["weight"][0] = float("nan")
    torch.save(checkpoint, tmp_path / "prior.pt")
    with pytest.raises(ValueError, match="nonfinite"):
        load_fixed_prior(tmp_path, target)


def test_zero_extra_losses_match_original_e7_value_and_gradient():
    prediction = torch.randn(3, 1, 243, requires_grad=True)
    target = torch.randn_like(prediction)
    parts = physical_objective(None, prediction, {"target": target}, geometry_weight=0, fk_weight=0,
                               return_components=True)
    original = FlowMatching()._valid_frame_mse(prediction - target, {
        "y": {"valid_frames": torch.ones(3, 1, dtype=torch.bool)}})[0].mean()
    torch.testing.assert_close(parts["loss"], original)
    torch.testing.assert_close(torch.autograd.grad(parts["loss"], prediction, retain_graph=True)[0],
                               torch.autograd.grad(original, prediction)[0])
    assert parts["weighted_dense"] == 0 and parts["weighted_fk"] == 0


def test_loss_ablation_runs_four_matched_generators_on_correct_cards(tmp_path):
    tasks, cases = tasks_for(tmp_path, [4, 5, 6, 7])
    assert len(tasks) == 8
    assert [(case["geometry_weight"], case["fk_weight"], case["mode"]) for case in cases] == [
        (0, 0, "gaussian"), (0, 0, "history"), (1, 1, "gaussian"), (1, 1, "history")]
    assert [task.gpu for task in tasks] == [4, 5, 6, 7, 4, 5, 6, 7]
    for train, evaluation in zip(tasks[:4], tasks[4:]):
        assert train.argv[train.argv.index("--prior-steps") + 1] == "0"
        assert "--fixed-prior-experiment" in train.argv
        assert "--history-cache" not in train.argv
        assert evaluation.expected == 36 and evaluation.deps == [train.name]
        assert complete_stages.requested_modes(train.argv, ()) == complete_stages.requested_modes(evaluation.argv, ())


def test_assigned_gpu_is_not_replaced_by_another_available_card(tmp_path, monkeypatch):
    queue = complete_stages.Queue(tmp_path / "queue", [4, 5], 1, allow_shared=True)
    task = complete_stages.Task("pinned", [], tmp_path / "train", "train", gpu=5)
    snapshots = iter([{4}, {4}, {4, 5}, {4, 5}])
    monkeypatch.setattr(complete_stages, "shared_gpu_indices", lambda *args: next(snapshots))
    monkeypatch.setattr(complete_stages.time, "sleep", lambda _: None)
    launched = []
    def launch(task, gpu):
        launched.append(gpu)
        queue.state["tasks"][task.name]["status"] = "complete"
    monkeypatch.setattr(queue, "launch", launch)
    queue.run([task])
    assert launched == [5]
    queue.lock.close()


def test_summary_compares_clean_and_faults_without_mixing_generators(tmp_path):
    _, cases = tasks_for(tmp_path, [4, 5, 6, 7])
    identity = {key: "same" for key in ("group", "sampling_seed", "dataset_spec_sha256",
                "evaluation_split_manifest_sha256", "bootstrap_cache_sha256", "e7_checkpoint_sha256", "stats_sha256")}
    variants = ["clean", "freeze_3s", "drift_0p03mps"]
    def write(path, label, modes, offset):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"experiment": label, "mode": mode, "take": take, "variant": variant,
                 "smpl22_mpjpe_mm": 100 + offset + (50 if mode == "history" else 0) + (30 if variant != "clean" else 0),
                 "paired_fault_delta": {key: 0 for key in ("fault_signed_mean_mm", "fault_positive_auc_mm_s", "fault_peak_increase_mm")}}
                for mode in modes for take in ("a", "b") for variant in variants]
        path.write_text(json.dumps({**identity, "variants": variants, "completed": True,
                                   "experiments": {label: label}, "results": rows}))
    baseline = tmp_path / "baseline.json"
    write(baseline, "72take_2400", ["gaussian", "history"], 0)
    for case in cases:
        write(tmp_path / "eval" / case["label"] / "report.json", case["label"], [case["mode"]],
              -10 if case["fk_weight"] else 5)
    result = result_summary(cases, baseline_eval=baseline)["comparisons"]
    assert result["g1f1_minus_g0f0_gaussian/clean"]["mean_difference_mm"] == -15
    assert result["g1f1_minus_g0f0_history/clean"]["mean_difference_mm"] == -15
    assert result["g0f0_gaussian_minus_g1f0/clean"]["current_mean_mm"] == 105
    assert result["g0f0_gaussian_minus_g1f0/all_variants"]["current_mean_mm"] == 125
