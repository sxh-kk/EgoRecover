"""Build/merge P01 v2 banks; build restarts under the frozen producer package.

Example (one GPU per shard):
  python scripts/build_formal_prior_bank.py build --plan PLAN.json --cohort D128 \
    --output exp/P01/banks/D128 --shard 0 --shards 4 --batch-contexts 8 --resume
  python scripts/build_formal_prior_bank.py merge --output exp/P01/banks/D128

Use --limit-contexts with a separate output root for small-bank preflights.
This command does not start P training or silently launch additional shards.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def module_path():
    sibling = Path(__file__).resolve().with_name("formal_prior_bank.py")
    return sibling if sibling.is_file() else Path(__file__).resolve().parents[1] / "egorecover/formal_prior_bank.py"


def load_module(path):
    spec = importlib.util.spec_from_file_location("formal_prior_bank_worker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freeze_and_restart(args):
    """Archive the builder separately; never modify the producer snapshot."""
    plan_file = args.plan.resolve()
    plan = json.loads(plan_file.read_text())
    snapshot = Path(plan["producer"]["source_snapshot"]).resolve()
    if not (snapshot / "egorecover/__init__.py").is_file():
        raise FileNotFoundError("Producer source snapshot is missing.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archived = output / "builder-source"
    with (output / "builder-source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if archived.exists():
            if json.loads((archived / "plan.json").read_text()) != plan:
                raise ValueError("Bank output was frozen from a different plan; choose a new output root.")
            manifest = json.loads((archived / "manifest.json").read_text())
            if any(hashlib.sha256((archived / name).read_bytes()).hexdigest() != sha for name, sha in manifest.items()):
                raise ValueError("Archived bank-builder source or plan checksum changed.")
            for current, saved in (
                (Path(__file__), archived / "driver.py"),
                (module_path(), archived / "formal_prior_bank.py"),
            ):
                if not args.resume and current.read_bytes() != saved.read_bytes():
                    raise ValueError("Builder source changed; resume the archived builder or use a new output root.")
        else:
            archived.mkdir()
            shutil.copy2(__file__, archived / "driver.py")
            shutil.copy2(module_path(), archived / "formal_prior_bank.py")
            (archived / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
            (archived / "manifest.json").write_text(
                json.dumps(
                    {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archived.iterdir() if p.is_file()},
                    indent=2,
                )
                + "\n"
            )
    body_models = (
        Path(args.smpl_directory).resolve()
        if args.smpl_directory
        else Path(
            os.environ.get("SMPLX_MODEL_PATH", str(Path(plan["proposed_output_root"]).parents[1] / "body_models/smplx"))
        ).resolve()
    )
    env = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(snapshot),
        SMPLX_MODEL_PATH=str(body_models),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    argv = [
        sys.executable,
        str(archived / "driver.py"),
        "build",
        "--plan",
        str(archived / "plan.json"),
        "--cohort",
        args.cohort,
        "--output",
        str(output),
        "--shard",
        str(args.shard),
        "--shards",
        str(args.shards),
        "--batch-contexts",
        str(args.batch_contexts),
        "--device",
        args.device,
        "--minimum-free-gib",
        str(args.minimum_free_gib),
        "--_frozen-worker",
    ]
    if args.limit_contexts is not None:
        argv += ["--limit-contexts", str(args.limit_contexts)]
    if args.resume:
        argv.append("--resume")
    if args.compile_eval:
        argv.append("--compile-eval")
    os.chdir(snapshot)
    os.execve(sys.executable, argv, env)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="Produce one explicitly requested shard on one device.")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--cohort", choices=("train", "D128", "D400", "long50"), required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--shard", type=int, default=0)
    build.add_argument("--shards", type=int, default=1)
    build.add_argument("--batch-contexts", type=int, default=8)
    build.add_argument("--device", default="cuda")
    build.add_argument("--smpl-directory", type=Path)
    build.add_argument("--limit-contexts", type=int)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--compile-eval", action="store_true")
    build.add_argument("--minimum-free-gib", type=float, default=10)
    build.add_argument("--_frozen-worker", action="store_true", help=argparse.SUPPRESS)
    merge = sub.add_parser("merge", help="CPU streaming merge after every shard completed.")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--minimum-free-gib", type=float, default=10)
    return result


def main():
    args = parser().parse_args()
    if args.minimum_free_gib < 0:
        raise ValueError("minimum-free-gib must be nonnegative.")
    if args.action == "build" and not args._frozen_worker:
        freeze_and_restart(args)
        return
    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if args.action == "merge":
        archived = args.output / "builder-source/formal_prior_bank.py"
        module = load_module(archived if archived.is_file() else module_path())
        report = module.merge_bank(args.output, minimum_free_bytes=int(args.minimum_free_gib * 1024**3))
    else:
        args.driver_file = Path(__file__).resolve()
        report = load_module(module_path()).build_from_args(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
