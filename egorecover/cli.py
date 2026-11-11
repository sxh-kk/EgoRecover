"""The single public command line for the current research plan."""

import argparse
import json
from pathlib import Path


def build_parser():
    from .config import TrainOptions
    from .data import OFFICIAL_DATA_ROOT

    parser = argparse.ArgumentParser(prog="egorecover", description="Causal rolling-window body reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a window model with optional paired feedback")
    for name in ("config", "data-root", "prior-checkpoint", "e7-checkpoint", "initialize"):
        train.add_argument(f"--{name}", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default=TrainOptions.device)
    for name in ("seed", "train-steps", "batch-size", "microbatch-size", "rollin-frames", "eval-every"):
        train.add_argument(f"--{name}", type=int, default=getattr(TrainOptions, name.replace("-", "_")))
    train.add_argument("--dev-limit", type=int)
    train.add_argument("--sample-reference-world-size", type=int)
    train.add_argument("--sampling-rollin-frames", type=int)
    train.add_argument("--history-mode", choices=("fixed", "mixed"), default=TrainOptions.history_mode)
    train.add_argument("--history-startup-fraction", type=float, default=TrainOptions.history_startup_fraction)
    train.add_argument("--unroll-steps", type=int, choices=(1, 2, 8), default=TrainOptions.unroll_steps)
    train.add_argument("--detach-between-steps", action="store_true")
    train.add_argument("--velocity-weight", type=float, default=TrainOptions.velocity_weight)
    train.add_argument("--velocity-scale-m-s", type=float, default=TrainOptions.velocity_scale_m_s)
    train.add_argument("--slow-weight", type=float, default=TrainOptions.slow_weight)
    train.add_argument("--feedback-intervention", choices=("full_state", "body_pose"), default="full_state")
    train.add_argument("--isolate-forward-rng", action="store_true")
    train.add_argument("--allow-initialize-rewrite-change", action="store_true")
    train.add_argument("--dev-seed", type=int, default=TrainOptions.dev_seed)
    train.add_argument("--full-dev-final", action="store_true")
    train.add_argument("--stop-after", type=int)
    for flag in ("resume", "synthetic", "compact", "from-scratch"):
        train.add_argument(f"--{flag}", action="store_true")
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--data-format", choices=("auto", "official", "paired"), default="auto")
    train.add_argument("--prior-mode", choices=("auto", "constant_velocity", "learned"), default="auto")

    evaluation = commands.add_parser("evaluate", help="Matched-cache, continuous and paired-horizon evaluation")
    for name in ("checkpoint", "output"):
        evaluation.add_argument(f"--{name}", type=Path, required=True)
    for name in ("data-root", "prior-checkpoint"):
        evaluation.add_argument(f"--{name}", type=Path)
    evaluation.add_argument("--split", choices=("dev", "val", "test"), default="dev")
    evaluation.add_argument("--steps", type=int, nargs="+", choices=(1, 2, 4, 10), default=[1, 2, 4, 10])
    evaluation.add_argument("--mode", choices=("both", "rollout", "matched"), default="both")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--draw", type=int, default=1062)
    evaluation.add_argument("--anchors", type=int, nargs="+", default=[20, 60, 100, 140, 180])
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--synthetic", action="store_true")
    evaluation.add_argument("--save-predictions", action="store_true")
    evaluation.add_argument(
        "--skip-paired", action="store_true", help="Skip paired diagnostics for reconstruction-only runs"
    )
    evaluation.add_argument("--threads", type=int, default=2)
    evaluation.add_argument("--data-format", choices=("auto", "official", "paired"), default="auto")
    evaluation.add_argument("--prior-mode", choices=("auto", "constant_velocity", "learned"), default="auto")

    experiments = commands.add_parser("experiments", help="Generate the four controlled experiment families")
    experiments.add_argument("--window", type=int, choices=(1, 3, 5, 10), default=3)
    experiments.add_argument("--output", type=Path)

    restore = commands.add_parser("restore", help="Restore only assets used by the current pipeline")
    restore.add_argument("--root", type=Path, default=Path("assets"))
    restore.add_argument(
        "--preset",
        choices=("metadata", "preflight", "train-dev", "models", "data", "training", "smoke"),
        default="models",
    )
    restore.add_argument("--dry-run", action="store_true")
    restore.add_argument("--verify-only", action="store_true")

    prepare = commands.add_parser("prepare", help="Audit data/environment and materialize the current experiment plan")
    prepare.add_argument("--output", type=Path, default=Path("exp/preparation"))
    prepare.add_argument("--assets-root", type=Path, default=Path("assets"))
    prepare.add_argument("--data-root", type=Path, default=OFFICIAL_DATA_ROOT)
    prepare.add_argument("--data-format", choices=("auto", "official", "paired"), default="auto")
    prepare.add_argument("--prior-mode", choices=("auto", "constant_velocity", "learned"), default="auto")
    prepare.add_argument("--prior-checkpoint", type=Path)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--window", type=int, choices=(1, 3, 5, 10), default=3)
    prepare.add_argument("--check-episodes", type=int, default=1)
    prepare.add_argument("--batch-size", type=int, default=64)
    prepare.add_argument("--microbatch-size", type=int, default=8)
    prepare.add_argument("--train-steps", type=int, default=2400)
    prepare.add_argument("--initialization", choices=("scratch", "e7"), default="scratch")
    for command in (train, prepare):
        for name in ("learning-rate", "weight-decay", "grad-clip-norm", "lr-decay-gamma"):
            command.add_argument(f"--{name}", type=float, default=getattr(TrainOptions, name.replace("-", "_")))
        command.add_argument("--lr-schedule", choices=("uniegomotion", "constant"), default=TrainOptions.lr_schedule)
        command.add_argument("--lr-decay-step", type=int)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("train", "evaluate"):
        if args.threads < 1:
            parser.error("--threads must be positive")
        import torch

        torch.set_num_threads(args.threads)
    if args.command == "train":
        from .config import TrainOptions
        from .training import train

        values = vars(args).copy()
        values.pop("command")
        values.pop("threads")
        train(TrainOptions(**values))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(args)
    elif args.command == "experiments":
        from .experiments import experiment_matrix
        from .io import save_json

        matrix = experiment_matrix(args.window)
        if args.output:
            for row in matrix:
                save_json(args.output / (row["name"] + ".json"), row["protocol"])
            save_json(args.output / "matrix.json", matrix)
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
    elif args.command == "restore":
        from .assets import restore

        restore(args.root, args.preset, dry_run=args.dry_run, verify_only=args.verify_only)
    elif args.command == "prepare":
        from .preparation import prepare

        if args.batch_size < 1 or args.microbatch_size < 1 or args.train_steps < 1:
            parser.error("Preparation budgets must be positive")
        prepare(args)
