"""The single public command line for the current research plan."""

import argparse
import json
from pathlib import Path


def build_parser():
    from .config import TrainOptions

    parser = argparse.ArgumentParser(prog="egorecover", description="Causal rolling-window body reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a window model with optional paired feedback")
    for name in ("config", "data-root", "prior-checkpoint", "e7-checkpoint", "initialize"):
        train.add_argument(f"--{name}", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default=TrainOptions.device)
    for name in ("seed", "train-steps", "batch-size", "rollin-frames", "eval-every"):
        train.add_argument(f"--{name}", type=int, default=getattr(TrainOptions, name.replace("-", "_")))
    train.add_argument("--learning-rate", type=float, default=TrainOptions.learning_rate)
    train.add_argument("--dev-limit", type=int)
    train.add_argument("--stop-after", type=int)
    for flag in ("resume", "synthetic", "compact"):
        train.add_argument(f"--{flag}", action="store_true")
    train.add_argument("--threads", type=int, default=2)

    evaluation = commands.add_parser("evaluate", help="Matched-cache, continuous and paired-horizon evaluation")
    for name in ("checkpoint", "output"):
        evaluation.add_argument(f"--{name}", type=Path, required=True)
    for name in ("data-root", "prior-checkpoint"):
        evaluation.add_argument(f"--{name}", type=Path)
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--steps", type=int, nargs="+", choices=(1, 2, 4, 10), default=[1, 2, 4, 10])
    evaluation.add_argument("--mode", choices=("both", "rollout", "matched"), default="both")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--draw", type=int, default=1062)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--synthetic", action="store_true")
    evaluation.add_argument("--save-predictions", action="store_true")
    evaluation.add_argument("--threads", type=int, default=2)

    experiments = commands.add_parser("experiments", help="Generate the four controlled experiment families")
    experiments.add_argument("--window", type=int, choices=(1, 3, 5, 10), default=3)
    experiments.add_argument("--output", type=Path)

    restore = commands.add_parser("restore", help="Restore only assets used by the current pipeline")
    restore.add_argument("--root", type=Path, default=Path("assets"))
    restore.add_argument("--preset", choices=("models", "data", "training", "smoke"), default="models")
    restore.add_argument("--dry-run", action="store_true")
    restore.add_argument("--verify-only", action="store_true")
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
