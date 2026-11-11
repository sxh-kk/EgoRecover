import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from egorecover import WindowConfig
from egorecover.io import state_sha256
from egorecover.training import compatible_initialization


def _run_process(command):
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            output, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=10)
            pytest.fail(f"CPU training subprocess exceeded 120 seconds:\n{output[-10000:]}")
        assert process.returncode == 0, f"Training/evaluation subprocess failed:\n{output[-10000:]}"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)


def _assert_identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_identical(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_identical(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("route", ("single", "two_step", "two_step_detached", "two_step_velocity", "eight_slow"))
def test_two_rank_training_resumes_exactly_with_rank_rng_and_light_best_full_dev(tmp_path, route):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("The real two-rank training test requires the Gloo CPU backend")
    unroll_steps = 1 if route == "single" else (8 if route == "eight_slow" else 2)
    config = WindowConfig(window=3, feedback="none", current_weight=0.03 if unroll_steps > 1 else 0)
    config_path = tmp_path / "architecture-W3.json"
    config_path.write_text(json.dumps(config.to_dict()))
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--module",
        "egorecover",
        "train",
        "--synthetic",
        "--compact",
        "--from-scratch",
        "--device",
        "cpu",
        "--config",
        str(config_path),
        "--batch-size",
        "4",
        "--microbatch-size",
        "1",
        "--rollin-frames",
        "1",
        "--train-steps",
        "3",
        "--eval-every",
        "2",
        "--dev-limit",
        "1",
        "--full-dev-final",
        "--lr-decay-step",
        "2",
        "--threads",
        "1",
        "--unroll-steps",
        str(unroll_steps),
    ]
    if unroll_steps == 2:
        common += ["--isolate-forward-rng"]
    if route == "two_step_detached":
        common += ["--detach-between-steps"]
    if route == "two_step_velocity":
        common += ["--velocity-weight", "0.02"]
    if route == "eight_slow":
        common += [
            "--detach-between-steps",
            "--slow-weight",
            "0.03",
            "--velocity-weight",
            "0.02",
            "--isolate-forward-rng",
        ]
    full_root, resumed_root = tmp_path / "full", tmp_path / "resumed"
    _run_process(common + ["--output", str(full_root)])
    _run_process(common + ["--output", str(resumed_root), "--stop-after", "2"])
    stopped = json.loads((resumed_root / "status.json").read_text())
    assert stopped["status"] == "stopped_after_checkpoint" and stopped["step"] == 2
    assert not (resumed_root / "full-dev.json").exists()
    _run_process(common + ["--output", str(resumed_root), "--resume"])

    full = torch.load(full_root / "resume.pt", weights_only=True)
    resumed = torch.load(resumed_root / "resume.pt", weights_only=True)
    assert full["step"] == resumed["step"] == 3
    for name in ("model", "ema", "feedback_model", "optimizer", "scheduler", "rank_rng_states"):
        _assert_identical(resumed[name], full[name])
    assert full["scheduler"]["last_epoch"] == 3
    assert full["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-6)

    for root, checkpoint in ((full_root, full), (resumed_root, resumed)):
        rank_states = checkpoint["rank_rng_states"]
        assert len(rank_states) == 2
        assert not torch.equal(rank_states[0]["sample_rng"], rank_states[1]["sample_rng"])
        assert not torch.equal(rank_states[0]["cpu_rng"], rank_states[1]["cpu_rng"])
        assert all(state["cuda_rng"] is None for state in rank_states)
        assert {state["model_sha256"] for state in rank_states} == {state_sha256(checkpoint["model"])}
        assert {state["ema_sha256"] for state in rank_states} == {state_sha256(checkpoint["ema"])}
        progress = json.loads((root / "progress.json").read_text())
        assert progress["target_queries_seen"] == 12
        assert progress["effective_batch_size"] == 4 and progress["local_batch_size"] == 2
        assert progress["world_size"] == 2 and progress["microbatch_size"] == 1
        assert progress["learning_rate"] == pytest.approx(3e-6)
        status = json.loads((root / "status.json").read_text())
        assert status == {"status": "complete", "step": 3}
        identity = checkpoint["identity"]
        assert identity["supervision"]["detach_between_steps"] == (route in ("two_step_detached", "eight_slow"))
        assert identity["supervision"]["velocity_weight"] == (
            0.02 if route in ("two_step_velocity", "eight_slow") else 0.0
        )
        assert identity["distributed"]["world_size"] == 2
        assert identity["distributed"]["local_batch_size"] == 2
        assert identity["distributed"]["global_batch_size"] == 4
        assert identity["full_dev_final"] is True
        # There is one synthetic validation episode, so rank 1 has no local
        # rows. The complete report exercises empty-rank gathering as well.
        full_dev = json.loads((root / "full-dev.json").read_text())
        rows = full_dev["results"]["1"]["rows"]["rollout"]
        assert len(rows) == 1 and rows[0]["episode"] == "synthetic"
        assert full_dev["training_step"] == 3
        assert full_dev["results"]["1"]["summary"]["rollout"]["comparison"]["mpjpe_world_mm"]["mean"] > 0
        with (root / "metrics.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 2 and all(row["mpjpe_world_mm"] for row in csv_rows)

        best = torch.load(root / "best.pt", weights_only=True)
        assert best["purpose"] == "evaluation_or_initialization_only"
        assert not {"model", "optimizer", "scheduler", "rank_rng_states", "feedback_model"} & best.keys()
        assert {"format", "model_config", "identity", "protocol", "stats", "ema", "step", "best"} <= best.keys()
        compatible_initialization(best, config, best["identity"])

    # The public evaluator must accept the lightweight checkpoint directly,
    # without requiring resume-only optimizer, raw-model or rank RNG payloads.
    evaluation_root = tmp_path / "light-best-evaluation"
    _run_process(
        [
            sys.executable,
            "-m",
            "egorecover",
            "evaluate",
            "--checkpoint",
            str(full_root / "best.pt"),
            "--synthetic",
            "--device",
            "cpu",
            "--mode",
            "rollout",
            "--steps",
            "1",
            "--skip-paired",
            "--threads",
            "1",
            "--output",
            str(evaluation_root),
        ]
    )
    report = json.loads((evaluation_root / "report.json").read_text())
    assert report["results"]["1"]["paired_horizons"] == []
    assert report["results"]["1"]["summary"]["rollout"]["comparison"]["mpjpe_world_mm"]["mean"] > 0
    with (evaluation_root / "metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
