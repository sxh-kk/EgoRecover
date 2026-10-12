"""Single-GPU training. Legacy modules and output directories are never imported."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from .config import Config
from .data import Episodes, Normalizer, save_json, save_torch, sha256
from .evaluation import evaluate
from .model import CausalTAE, MotionFlow
from .objective import masked_mean, tae_loss, two_forward_loss


def resolve_device(name, precision):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use a CUDA PyTorch build/GPU, or explicitly --device cpu for smoke tests."
            )
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bfloat16; set precision=fp16 or fp32 in a separate config.")
        torch.set_float32_matmul_precision("high")
    elif device.type != "cpu":
        raise ValueError("Supported devices: cuda[:index] and cpu.")
    return device


def autocast(device, precision):
    if device.type == "cpu" or precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def fit_base_statistics(norm, episodes):
    for field, mask in (("pose", "valid"), ("root", "valid"), ("head", "head_valid")):
        print(
            json.dumps({"event": "fit_train_statistics", "field": field, "episodes": len(episodes.split("train"))}),
            flush=True,
        )
        norm.fit(
            field, (episodes.load(record)[field][episodes.load(record)[mask]] for record in episodes.split("train"))
        )


@torch.no_grad()
def fit_latent_statistics(norm, tae, episodes, device):
    def values():
        records = episodes.split("train")
        for index, record in enumerate(records):
            p = episodes.load(record)
            latent, _ = tae.encode(
                norm.normalize("pose", p["pose"].to(device))[None],
                p["beta"].to(device)[None],
                p["valid"].to(device)[None],
            )
            yield latent[0, p["valid"].to(device)]
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(
                    json.dumps({"event": "fit_train_latent_statistics", "episodes": index + 1, "total": len(records)}),
                    flush=True,
                )

    norm.fit("latent", values())


@torch.no_grad()
def validate_tae(tae, norm, episodes, device):
    errors, count = 0.0, 0
    for record in episodes.split("dev"):
        p = episodes.load(record)
        target = norm.normalize("pose", p["pose"].to(device))[None]
        mask = p["valid"].to(device)[None]
        mean, _ = tae.encode(target, p["beta"].to(device)[None], mask)
        recon = tae.decode(mean, p["beta"].to(device)[None], mask)
        frames = int(mask.sum())
        errors += float(masked_mean((recon.float() - target).square(), mask)) * frames
        count += frames
    return {"normalized_reconstruction_mse": errors / count, "frames": count}


def load_bundle(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**{**checkpoint["config"], "tae_dilations": tuple(checkpoint["config"]["tae_dilations"])})
    norm, tae = Normalizer(cfg).to(device), CausalTAE(cfg).to(device)
    norm.load_state_dict(checkpoint["normalizer"])
    tae.load_state_dict(checkpoint["tae"])
    tae.eval().requires_grad_(False)
    flow = None
    if checkpoint["stage"] == "flow":
        flow = MotionFlow(cfg).to(device)
        flow.load_state_dict(checkpoint["flow"])
        flow.eval()
    return cfg, norm, tae, flow, checkpoint


def run_training(
    *, stage, config, manifest, output, device_name="cuda", tae_path=None, resume=None, stop_after=None, dev_limit=None
):
    cfg = Config.load(config)
    device = resolve_device(device_name, cfg.precision)
    # CPU runs are explicit; bound CPU thread count for smoke tests rather than oversubscribing tiny convolutions.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    output = Path(output).resolve()
    episodes = Episodes(manifest, cfg)
    episodes.split("train")
    episodes.split("dev")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Output is nonempty; use a fresh independent directory or --resume.")
    if resume and output.exists() and any(output.iterdir()) and not (output / "provenance.json").is_file():
        raise ValueError("Resume output must be an existing standalone run or an empty new directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    norm, tae = Normalizer(cfg).to(device), CausalTAE(cfg).to(device)
    identity = {"stage": stage, "manifest_sha256": episodes.digest, "tae_sha256": None, "device": str(device)}
    flow = None
    if stage == "flow":
        if tae_path is None:
            raise ValueError("Flow training requires --tae from a trained TAE stage.")
        identity["tae_sha256"] = sha256(tae_path)
        tc, norm, tae, _, taecp = load_bundle(tae_path, device)
        if taecp["stage"] != "tae" or taecp["manifest_sha256"] != episodes.digest:
            raise ValueError("TAE must originate from this dataset and stage.")
        architecture_fields = ("pose_dim", "latent_dim", "tae_width", "tae_dilations", "image_dim", "fps", "bootstrap")
        if any(getattr(tc, field) != getattr(cfg, field) for field in architecture_fields):
            raise ValueError("Incompatible TAE architecture/data config.")
        # The TAE decoder uses the Flow config's same receptive field; all architecture fields were checked.
        tae.cfg = cfg
        flow = MotionFlow(cfg).to(device)
        model = flow
        if resume is None:
            fit_latent_statistics(norm, tae, episodes, device)
    elif stage == "tae":
        model = tae
        if resume is None:
            fit_base_statistics(norm, episodes)
    else:
        raise ValueError("Stage must be tae or flow.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.precision == "fp16")
    step, best = 0, float("inf")
    if resume:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        comparison = {**identity, "config": cfg.to_dict(), "dev_limit": dev_limit, "device_type": device.type}
        for key, value in comparison.items():
            # JSON config lists and dataclass tuples represent the same dilation sequence.
            if json.dumps(previous[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(f"Resume identity mismatch: {key}.")
        norm.load_state_dict(previous["normalizer"])
        tae.load_state_dict(previous["tae"])
        if flow is not None:
            flow.load_state_dict(previous["flow"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        step, best = previous["step"], previous["best"]
        generator.set_state(previous["data_rng"])
        torch.set_rng_state(previous["cpu_rng"])
        if device.type == "cuda":
            if previous["cuda_rng"] is None:
                raise ValueError("Cannot resume CPU smoke training as an equivalent CUDA run.")
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    cfg.save(output / "config.json")
    save_json(
        output / "provenance.json",
        {
            **identity,
            "dataset_backend": episodes.manifest.get("backend"),
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "supervised_frames_per_update": cfg.batch_size
            * cfg.accumulation
            * (cfg.target_frames + (cfg.bootstrap if stage == "tae" else 0)),
            "note": "Flow is conditioned on mixed histories during training and predicted histories at evaluation.",
        },
    )
    last_step = min(cfg.total_steps, stop_after if stop_after is not None else cfg.total_steps)
    if last_step <= step:
        raise ValueError("stop-after must be greater than the resumed step.")
    start_time = time.monotonic()
    overflow_retries = 0
    log = (output / "metrics.jsonl").open("a" if resume else "w")
    try:
        while step < last_step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for _ in range(cfg.accumulation):
                batch = {
                    key: (value.pin_memory() if device.type == "cuda" else value).to(device, non_blocking=True)
                    for key, value in episodes.sample(generator).items()
                }
                with autocast(device, cfg.precision):
                    if stage == "tae":
                        loss, stats = tae_loss(tae, norm, batch, cfg)
                    else:
                        loss, stats = two_forward_loss(flow, tae, norm, batch, cfg, step + 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss; no checkpoint has been committed for this update."
                    )
                scaler.scale(loss / cfg.accumulation).backward()
                for key, value in {"loss": loss.detach(), **stats}.items():
                    accumulated[key] = accumulated.get(key, 0) + float(value) / cfg.accumulation
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, error_if_nonfinite=not scaler.is_enabled()
            )
            if not torch.isfinite(grad_norm):
                # GradScaler's recorded overflow skips this optimizer update and lowers the scale.
                scaler.step(optimizer)
                scaler.update()
                overflow_retries += 1
                print(
                    json.dumps({"event": "fp16_overflow", "next_update": step + 1, "scale": scaler.get_scale()}),
                    flush=True,
                )
                if overflow_retries >= 20:
                    raise FloatingPointError("20 consecutive fp16 overflows; inspect data/model or use bf16/fp32.")
                continue
            overflow_retries = 0
            scaler.step(optimizer)
            scaler.update()
            step += 1
            entry = {
                "step": step,
                "stage": stage,
                **accumulated,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - start_time,
            }
            do_validation = step % cfg.eval_every == 0 or step == cfg.total_steps
            improved = False
            if do_validation:
                model.eval()
                # Isolate validation RNG from the training stream so stopping/resuming does not change updates.
                devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    if stage == "tae":
                        entry["dev"] = validate_tae(tae, norm, episodes, device)
                        score = entry["dev"]["normalized_reconstruction_mse"]
                    else:
                        entry["dev"] = evaluate(flow, tae, norm, episodes, "dev", seed=cfg.seed, limit=dev_limit)
                        score = entry["dev"]["metrics"]["body_mm"]
                improved = score < best
                best = min(best, score)
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or do_validation or step % 50 == 0:
                print(
                    json.dumps({key: value for key, value in entry.items() if key != "dev"}, allow_nan=False),
                    flush=True,
                )
            if do_validation or step == last_step:
                state = {
                    **identity,
                    "config": cfg.to_dict(),
                    "step": step,
                    "best": best,
                    "normalizer": norm.state_dict(),
                    "tae": tae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "data_rng": generator.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "validation": entry.get("dev"),
                    "dev_limit": dev_limit,
                    "device_type": device.type,
                }
                if flow is not None:
                    state["flow"] = flow.state_dict()
                save_torch(output / "latest.pt", state)
                if improved:
                    save_torch(output / "best.pt", state)
    finally:
        log.close()
    return output / "latest.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tae", "flow"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tae")
    parser.add_argument("--resume")
    parser.add_argument(
        "--stop-after", type=int, help="Absolute update index, useful for interrupt/resume smoke checks."
    )
    parser.add_argument("--dev-limit", type=int, help="Debug only: limit dev episodes; formal runs should omit this.")
    args = parser.parse_args()
    run_training(
        stage=args.stage,
        config=args.config,
        manifest=args.manifest,
        output=args.output,
        device_name=args.device,
        tae_path=args.tae,
        resume=args.resume,
        stop_after=args.stop_after,
        dev_limit=args.dev_limit,
    )


if __name__ == "__main__":
    main()
