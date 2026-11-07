"""Small official-data readiness checks, without restoring legacy artifacts."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .bootstrap import HeadAlignedBootstrap
from .checkpoint import load_predictor, resolve_prior_mode
from .codec import MotionCodec
from .experiments import experiment_matrix
from .fk import FixedShapeFK, load_smpl, smpl_identity, smpl_path
from .io import code_identity, save_json
from .official import OfficialWindowData


@torch.no_grad()
def prepare_official(args):
    from .preparation import environment_report

    if getattr(args, "initialization", "scratch") != "scratch":
        raise ValueError("Official-data preparation uses random G initialization, with no E7 weights.")
    prior_checkpoint = getattr(args, "prior_checkpoint", None)
    prior_mode = resolve_prior_mode(getattr(args, "prior_mode", "auto"), prior_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_format": "official",
        "g_initialization": "scratch",
        "prior_mode": prior_mode,
        "environment": environment_report(),
        "models": {},
        "issues": [],
        "experiments_started": False,
        "automatic_download": False,
        "code": code_identity(),
    }
    datasets = {}
    smpl = None
    if not smpl_path().is_file():
        report["issues"].append({"kind": "missing_smplx", "path": str(smpl_path())})
    else:
        report["smpl_sha256"] = smpl_identity()
        smpl = load_smpl("cpu")
        bootstrap = HeadAlignedBootstrap.from_smpl(smpl)
        report["bootstrap"] = bootstrap.protocol
        try:
            for split in ("train", "dev"):
                data = OfficialWindowData(args.data_root, split, bootstrap=bootstrap)
                datasets[split] = data
                rows = []
                for index in range(min(args.check_episodes, len(data.records))):
                    episode = data.load(index)
                    codec = MotionCodec(episode.stats)
                    reference = episode.cache().body.reference[:, -1:]
                    roundtrip = {}
                    for width in (1, 3, 5, 10):
                        truth = episode.target(20, width)
                        restored = codec.decode_current(codec.encode_current(truth, reference), reference)
                        error = float((truth.joints - restored.joints).abs().max())
                        if error > 5e-5:
                            raise ValueError(f"Official coordinate roundtrip failed for W={width}.")
                        roundtrip[str(width)] = error
                    history = episode.cache().body
                    expected = history.joints[..., :3, 3]
                    projected = FixedShapeFK(smpl, episode.beta)(history)
                    bootstrap_error = float((projected - expected).abs().max())
                    if bootstrap_error > 5e-5:
                        raise ValueError("Head-only bootstrap is inconsistent with neutral SMPL geometry.")
                    rows.append(
                        {
                            "episode": episode.identifier,
                            "frames": len(episode),
                            "dense_roundtrip_max_abs_m": roundtrip,
                            "bootstrap_fk_max_abs_m": bootstrap_error,
                        }
                    )
                report.setdefault("dataset", {})[split] = {
                    "inventory": data.inventory,
                    "sample_checks": rows,
                    "digest": data.digest,
                }
            if datasets["train"].digest != datasets["dev"].digest:
                raise ValueError("Official train/val dataset identities disagree.")
            if any(
                not torch.equal(datasets["train"].stats[k], datasets["dev"].stats[k]) for k in datasets["train"].stats
            ):
                raise ValueError("Official train/val normalization statistics disagree.")
            report["data_protocol"] = datasets["train"].protocol
            if prior_mode == "learned":
                load_predictor(prior_checkpoint, "cpu", datasets["train"].digest)
                report["models"]["predictor"] = {"verified": True, "path": str(prior_checkpoint)}
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            report["issues"].append({"kind": "official_data_check_failed", "message": str(error)})
    if args.device == "cuda" and not report["environment"]["cuda_available"]:
        report["issues"].append({"kind": "cuda_unavailable"})

    matrix = experiment_matrix(args.window)
    jobs = []
    for row in matrix:
        config_path = args.output / "configs" / (row["name"] + ".json")
        save_json(config_path, row["protocol"])
        for seed in row["seeds"]:
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
                str(config_path.resolve()),
                "--data-root",
                str(args.data_root.resolve()),
                "--data-format",
                "official",
                "--prior-mode",
                prior_mode,
                "--output",
                str((args.output / "runs" / row["name"] / f"seed{seed}").resolve()),
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
            if prior_checkpoint is not None:
                command += ["--prior-checkpoint", str(prior_checkpoint.resolve())]
            if row["initialize_from"]:
                checkpoint = args.output / "runs" / row["initialize_from"] / f"seed{seed}" / "best.pt"
                command += ["--initialize", str(checkpoint.resolve())]
            else:
                command += ["--from-scratch"]
            jobs.append(
                {
                    "name": row["name"],
                    "seed": seed,
                    "working_directory": str(Path.cwd()),
                    "depends_on": row["initialize_from"] or row["reuse"],
                    "reuse": row["reuse"],
                    "argv": None if row["reuse"] else command,
                    "started": False,
                }
            )
    save_json(args.output / "matrix.json", matrix)
    save_json(args.output / "jobs.json", jobs)
    save_json(
        args.output / "protocol.json",
        {
            "data_format": "official",
            "g_initialization": "scratch",
            "prior_mode": prior_mode,
            "startup": report.get("bootstrap"),
            "data_protocol": report.get("data_protocol"),
            "history_frames": 20,
            "observation_frames": 21,
            "windows": [1, 3, 5, 10],
            "window_selection_finalized": False,
            "selected_window_for_later_experiments": args.window,
            "training_budget": {
                "effective_batch": args.batch_size,
                "microbatch": args.microbatch_size,
                "optimizer_steps": args.train_steps,
                "target_queries": args.batch_size * args.train_steps,
            },
            "official_val_used_for_development": True,
            "unseen_test_claim": False,
        },
    )
    report["status"] = "ready_for_training" if not report["issues"] else "needs_assets_or_environment"
    save_json(args.output / "preparation.json", report)
    print(
        f"Official preparation: {report['status']}; {len(report['issues'])} remaining requirements. Report: {args.output / 'preparation.json'}"
    )
    return report
