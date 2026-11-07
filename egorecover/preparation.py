"""Prepare the current plan without launching experiments or selecting on test."""

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .assets import asset_plan, asset_registry, matches
from .codec import MotionCodec
from .data import PreparedWindowData, manifest_inventory, resolve_data_format
from .experiments import experiment_matrix
from .fk import FixedShapeFK, load_smpl, smpl_path
from .io import code_identity, file_sha256, save_json
from .state import state_map


def environment_report():
    hardware = subprocess.run(
        [
            "bash",
            "-c",
            "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cuda_devices": torch.cuda.device_count() if available else 0,
        "hardware": hardware.stdout.strip().splitlines() if hardware.returncode == 0 else [],
    }


@torch.no_grad()
def audit_available_episodes(root, *, count=1, smpl=None):
    manifest, inventory = manifest_inventory(root)
    reports = {}
    training_stats = None
    for group in ("train", "dev"):
        records = [r for r in manifest["records"] if r["group"] == group]
        available = [
            r
            for r in records
            if (Path(root) / "data/episodes" / (r["episode_id"] + ".pt")).is_file()
            and (Path(root) / "data/episodes" / (r["episode_id"] + ".json")).is_file()
        ]
        rows = []
        if available and count:
            ids = [r["episode_id"] for r in available[:count]]
            data = PreparedWindowData(root, group, episode_ids=ids)
            for index in range(len(data.records)):
                episode = data.load(index)
                if training_stats is None:
                    training_stats = episode.stats
                elif any(not torch.equal(training_stats[k], episode.stats[k]) for k in training_stats):
                    raise ValueError("Train/dev episodes do not share the fixed training statistics.")
                codec = MotionCodec(episode.stats)
                reference = episode.cache().body.reference[:, -1:]
                errors = {}
                for width in (1, 3, 5, 10):
                    target = episode.target(20, width)
                    restored = codec.decode_current(codec.encode_current(target, reference), reference)
                    error = float((restored.joints - target.joints).abs().max())
                    if error > 5e-5:
                        raise ValueError(f"Dense coordinate roundtrip failed: {episode.identifier}, W={width}.")
                    errors[str(width)] = error
                row = {
                    "episode": episode.identifier,
                    "take": episode.take,
                    "frames": len(episode),
                    "fps": 1 / float(episode.timestamps[1] - episode.timestamps[0]),
                    "dense_roundtrip_max_abs_m": errors,
                    "smpl_asset_audit": "not_run_missing_asset",
                }
                if smpl is not None:
                    frames = sorted({0, 19, len(episode) // 2, len(episode) - 1})
                    truth = state_map(episode.truth, lambda x: x[frames])
                    positions = FixedShapeFK(smpl, truth.auxiliary[:, -10:])(truth)
                    error = (positions - truth.joints[..., :3, 3]).norm(dim=-1) * 1000
                    row["smpl_asset_audit"] = {"mean_mm": float(error.mean()), "max_mm": float(error.max())}
                    if float(error.max()) > 5:
                        raise ValueError("SMPL-X asset does not reconstruct the cached GT within 5mm.")
                rows.append(row)
        reports[group] = {"available_episodes": len(available), "total_episodes": len(records), "audited": rows}
    return inventory, reports


def prepare(args):
    if args.check_episodes < 0:
        raise ValueError("check-episodes cannot be negative.")
    if resolve_data_format(args.data_root, getattr(args, "data_format", "auto")) == "official":
        from .official_preparation import prepare_official

        return prepare_official(args)
    initialization = getattr(args, "initialization", "scratch")
    if initialization not in ("scratch", "e7"):
        raise ValueError("Choose scratch or e7 G initialization.")
    registry = asset_registry()
    matrix = experiment_matrix(args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    for row in matrix:
        save_json(args.output / "configs" / (row["name"] + ".json"), row["protocol"])
    save_json(args.output / "matrix.json", matrix)
    jobs = []
    for row in matrix:
        for seed in row["seeds"]:
            output = (args.output / "runs" / row["name"] / f"seed{seed}").resolve()
            command = [
                "conda",
                "run",
                "-n",
                "egorecover",
                "--no-capture-output",
                "python",
                "-m",
                "egorecover",
                "train",
                "--config",
                str((args.output / "configs" / (row["name"] + ".json")).resolve()),
                "--data-root",
                str(args.data_root.resolve()),
                "--prior-checkpoint",
                str((args.assets_root / "pretrained/prior.pt").resolve()),
                "--output",
                str(output),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--batch-size",
                str(args.batch_size),
                "--microbatch-size",
                str(args.microbatch_size),
                "--train-steps",
                str(args.train_steps),
            ]
            dependency = row["initialize_from"] or row["reuse"]
            if row["initialize_from"]:
                checkpoint = args.output / "runs" / row["initialize_from"] / f"seed{seed}" / "best.pt"
                command.extend(("--initialize", str(checkpoint.resolve())))
            elif initialization == "e7":
                command.extend(("--e7-checkpoint", str((args.assets_root / "pretrained/e7.ckpt").resolve())))
            else:
                command.append("--from-scratch")
            jobs.append(
                {
                    "name": row["name"],
                    "seed": seed,
                    "working_directory": str(Path.cwd()),
                    "depends_on": dependency,
                    "reuse": row["reuse"],
                    "argv": None if row["reuse"] else command,
                    "started": False,
                }
            )
    save_json(args.output / "jobs.json", jobs)
    plans = {
        key: asset_plan(registry, key) for key in ("metadata", "preflight", "train-dev", "data", "models", "training")
    }
    report = {
        "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": environment_report(),
        "downloads": plans,
        "issues": [],
        "code": code_identity(),
        "experiments_started": False,
        "g_initialization": initialization,
    }
    manifest_path = args.data_root / "data/manifest.json"
    smpl = None
    if not smpl_path().is_file():
        report["issues"].append(
            {
                "kind": "missing_smplx",
                "path": str(smpl_path()),
                "action": "Download the licensed SMPL-X v1.1 neutral NPZ from the official website.",
            }
        )
    else:
        report["smpl_sha256"] = file_sha256(smpl_path())
        smpl = load_smpl("cpu")
    if manifest_path.is_file():
        try:
            inventory, audits = audit_available_episodes(args.data_root, count=args.check_episodes, smpl=smpl)
            report["dataset"] = inventory
            report["episode_audits"] = audits
            for group, audit in audits.items():
                if audit["available_episodes"] < audit["total_episodes"]:
                    report["issues"].append(
                        {
                            "kind": "incomplete_data",
                            "split": group,
                            "available": audit["available_episodes"],
                            "total": audit["total_episodes"],
                        }
                    )
        except (ValueError, OSError, RuntimeError) as error:
            report["issues"].append({"kind": "data_audit_failed", "message": str(error)})
    else:
        report["issues"].append({"kind": "missing_manifest", "path": str(manifest_path)})
    report["models"] = {}
    required_models = {"pretrained/prior.pt"}
    if initialization == "e7":
        required_models.add("pretrained/e7.ckpt")
    for item in registry["files"]:
        if item["path"] in required_models:
            path = args.assets_root / item["path"]
            valid = matches(path, item)
            report["models"][item["path"]] = {"path": str(path), "verified": valid}
            if not valid:
                report["issues"].append({"kind": "missing_or_changed_model", "path": str(path)})
    if args.device == "cuda" and not report["environment"]["cuda_available"]:
        report["issues"].append(
            {"kind": "cuda_unavailable", "action": "Use the CUDA-enabled egorecover Conda environment."}
        )
    protocol = {
        "history_frames": 20,
        "observation_frames": 21,
        "body_dimensions": 243,
        "windows": [1, 3, 5, 10],
        "source_modes": ["gaussian", "prediction"],
        "training_solver_steps": 1,
        "evaluation_solver_steps": [1, 2, 4, 10],
        "paired_horizons_frames": [1, 2, 4, 10],
        "paired_anchor_frames": [20, 60, 100, 140, 180],
        "paired_cache_source": "own_continuous_rollout",
        "reference_label_writes": 1,
        "fixed_startup_shape": True,
        "g_initialization": initialization,
        "predictor_initialization": "published_calibrated_prior",
        "bootstrap_history_source": "published_teacher_predictions",
        "bootstrap_shape_source": "published_beta_boot",
        "entire_system_from_scratch": False,
        "test_used_for_model_selection": False,
        "selected_window_for_later_experiments": args.window,
        "window_selection_finalized": False,
        "training_budget": {
            "effective_batch": args.batch_size,
            "microbatch": args.microbatch_size,
            "optimizer_steps": args.train_steps,
            "target_queries": args.batch_size * args.train_steps,
        },
    }
    if "dataset" in report:
        protocol["manifest_sha256"] = report["dataset"]["sha256"]
    save_json(args.output / "protocol.json", protocol)
    report["status"] = "ready_for_training" if not report["issues"] else "needs_assets_or_environment"
    save_json(args.output / "preparation.json", report)
    print(
        f"Preparation: {report['status']}; {len(report['issues'])} remaining requirements. Report: {args.output / 'preparation.json'}"
    )
    return report
