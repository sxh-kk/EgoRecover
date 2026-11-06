import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

from egorecover.cli import main


def test_installed_module_runs_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "egorecover", "--help"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert all(command in result.stdout for command in ("train", "evaluate", "experiments", "restore"))


def test_all_experiments_generate_configs_without_assets_or_training(tmp_path):
    main(["experiments", "--output", str(tmp_path)])
    matrix = json.loads((tmp_path / "matrix.json").read_text())
    assert matrix
    for row in matrix:
        assert json.loads((tmp_path / (row["name"] + ".json")).read_text()) == row["protocol"]
    assert not list(tmp_path.rglob("*.pt"))


def test_end_to_end_public_cli_train_and_evaluate(tmp_path):
    from egorecover import WindowConfig

    config = tmp_path / "config.json"
    config.write_text(json.dumps(WindowConfig(feedback="paired").to_dict()))
    output = tmp_path / "training"
    main(
        [
            "train",
            "--config",
            str(config),
            "--synthetic",
            "--compact",
            "--output",
            str(output),
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--eval-every",
            "1",
            "--rollin-frames",
            "1",
        ]
    )
    report = tmp_path / "evaluation"
    main(
        [
            "evaluate",
            "--checkpoint",
            str(output / "best.pt"),
            "--synthetic",
            "--output",
            str(report),
            "--steps",
            "1",
            "2",
            "4",
            "10",
            "--save-predictions",
        ]
    )
    saved = json.loads((report / "report.json").read_text())
    assert set(saved["results"]) == {"1", "2", "4", "10"}
    for group in saved["results"].values():
        assert set(group["summary"]) == {"rollout", "matched"}
        diagnostics = group["paired_horizons"][0]
        assert diagnostics["cache_source"] == "own_continuous_rollout"
        assert set(diagnostics["anchors"]["20"]) == {"1", "2", "4", "10"}
        for mode in ("rollout", "matched"):
            metrics = group["summary"][mode]["comparison"]
            assert metrics["mpjpe_world_mm"]["mean"] > 0
            assert metrics["wrist_pe_mm"]["mean"] > 0
            assert "pa_mpjpe_mm" in metrics and "foot_slide_uniegomotion_mm" in metrics
    protocol = saved["identity"]["comparison_protocol"]
    assert "MPJPE-H" in protocol["paper_boundaries"]["excluded_by_scope"]
    assert protocol["units"]["mpjpe_world_mm"] == "mm"
    with (report / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16  # four solver budgets x two history modes x two aggregations
    assert {r["aggregation"] for r in rows} == {"take_mean", "episode_mean"}
    assert {r["mode"] for r in rows} == {"rollout", "matched"}
    assert all(r["synthetic"] == "True" and r["mpjpe_world_mm"] for r in rows)
    assert len(list(report.glob("*.pt"))) == 8


def test_runtime_imports_do_not_depend_on_removed_implementations():
    import egorecover

    obsolete = {
        "config",
        "model",
        "module",
        "run",
        "dataset",
        "eval",
        "utils",
        "motionstreamer_flow",
        "data_pipeline",
        "mydiffusion",
        "pytorch_lightning",
        "yacs",
        "loguru",
    }
    for path in Path(egorecover.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in obsolete for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".")[0] not in obsolete, path
