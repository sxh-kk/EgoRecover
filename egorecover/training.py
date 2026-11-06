"""Train the rolling-window model, with full-response paired EMA supervision."""

import json
import math
import platform

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.checkpoint import FORMAT, load_e7_weights, load_predictor
from egorecover.codec import MotionCodec
from egorecover.data import PreparedWindowData, keyed_noise
from egorecover.diagnostics import synthetic_episode
from egorecover.evaluation import evaluate_episode, summarize
from egorecover.fk import joint_decoder, load_smpl, smpl_identity
from egorecover.io import code_identity, file_sha256, save_checkpoint, save_json
from egorecover.losses import frozen_ema, training_objective, update_ema
from egorecover.models import MotionModel
from egorecover.state import state_map


def compatible_initialization(saved, config, identity):
    if saved.get("format") != FORMAT:
        raise ValueError("Expected a rolling-window checkpoint.")
    for key in (
        "history",
        "window",
        "source",
        "sigma",
        "steps",
        "body_feedback",
        "revise_recent",
        "endpoint_probability",
    ):
        if saved["protocol"][key] != config.to_dict()[key]:
            raise ValueError(f"Initialization changes controlled setting: {key}.")
    for key in ("manifest", "prior", "synthetic", "compact", "smpl_asset", "model_config"):
        if saved["identity"][key] != identity[key]:
            raise ValueError(f"Initialization identity changed: {key}.")


def train(args):
    device = torch.device(args.device)
    config = WindowConfig.from_json(args.config)
    if args.resume and args.initialize:
        raise ValueError("Resume and initialization are mutually exclusive.")
    if args.dev_limit is not None and args.dev_limit < 1:
        raise ValueError("dev-limit must be positive.")
    if args.train_steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.rollin_frames < 0:
        raise ValueError("Invalid training budget.")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if args.synthetic:
        if not args.compact or args.data_root or args.prior_checkpoint or args.e7_checkpoint:
            raise ValueError("Synthetic checks require --compact and no production assets.")
        fixture = synthetic_episode(device=device)
        train_data = dev_data = None
        stats, manifest = fixture.stats, "synthetic_diagnostic"
    else:
        if args.data_root is None or args.prior_checkpoint is None:
            raise ValueError("Provide --data-root and the calibrated --prior-checkpoint.")
        if args.compact:
            raise ValueError("Compact mode is restricted to explicit synthetic checks.")
        if args.e7_checkpoint is None and args.initialize is None and not args.resume:
            raise ValueError("Start with --e7-checkpoint or --initialize.")
        train_data = PreparedWindowData(args.data_root, "train", device=device)
        dev_data = PreparedWindowData(args.data_root, "dev", device=device)
        stats, manifest = train_data.stats, train_data.digest
        if any(not torch.equal(stats[k], dev_data.stats[k]) for k in stats):
            raise ValueError("Train/dev normalization statistics disagree.")
    torch.manual_seed(args.seed)
    codec = MotionCodec(stats).to(device)
    prior = load_predictor(args.prior_checkpoint, device, None if args.synthetic else manifest)
    system = WindowReconstructor(codec, prior, config)
    model_config = ModelConfig.compact() if args.compact else ModelConfig()
    model = MotionModel(config, model_config).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01
    )
    identity = {
        "protocol": config.to_dict(),
        "seed": args.seed,
        "manifest": manifest,
        "prior": file_sha256(args.prior_checkpoint) if args.prior_checkpoint else "constant_velocity",
        "e7": file_sha256(args.e7_checkpoint) if args.e7_checkpoint else None,
        "initialize": file_sha256(args.initialize) if args.initialize else None,
        "synthetic": args.synthetic,
        "compact": args.compact,
        "model_config": model_config.to_dict(),
        "smpl_asset": "dense_diagnostic" if args.synthetic else smpl_identity(),
        "feedback_teacher": "fixed_initial_ema",
        "code": code_identity(),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "rollin_frames": args.rollin_frames,
        "learning_rate": args.learning_rate,
        "dev_limit": args.dev_limit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run identity changed; use a new output directory.")
    checkpoint_path = args.output / "resume.pt"
    if checkpoint_path.exists() and not args.resume:
        raise ValueError("Output already has a checkpoint; pass --resume or choose a new directory.")
    saved = None
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved["identity"] != identity or saved.get("format") != FORMAT:
            raise ValueError("Resume identity changed.")
        model.load_state_dict(saved["model"], strict=True)
    elif args.initialize:
        initial = torch.load(args.initialize, map_location=device, weights_only=True)
        compatible_initialization(initial, config, identity)
        model.load_state_dict(initial["ema"], strict=True)
    elif args.e7_checkpoint:
        load_e7_weights(model, args.e7_checkpoint, weight_source="ema")
    ema = frozen_ema(model)
    # Keep the response map identical across paired/direct comparisons for the
    # entire continuation. A separate moving EMA is used for model selection.
    feedback_model = frozen_ema(model)
    smpl = None if args.synthetic else load_smpl(device)
    generator = torch.Generator().manual_seed(args.seed + 1)
    start, best, curve = 0, float("inf"), []
    if saved:
        ema.load_state_dict(saved["ema"], strict=True)
        feedback_model.load_state_dict(saved["feedback_model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sample_rng"].cpu())
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        start, best, curve = saved["step"], saved["best"], saved["curve"]
        del saved
    save_json(identity_path, identity)
    final = min(args.train_steps, start + args.stop_after) if args.stop_after else args.train_steps
    for step in range(start + 1, final + 1):
        optimizer.zero_grad(set_to_none=True)
        components = {k: 0.0 for k in ("loss", "flow", "current", "feedback")}
        for sample in range(args.batch_size):
            episode = fixture if args.synthetic else train_data.sample(generator)
            frame = int(torch.randint(20, len(episode) - 1, (), generator=generator))
            first = max(20, frame - args.rollin_frames)
            cache = episode.cache(first)
            model.eval()
            for t in range(first, frame):
                noise = keyed_noise(episode.identifier, args.seed + step, t, config.window, device)
                cache = system.online(model, cache, episode.observations(t), noise).cache.detach()
            model.train()
            # Stateless streams keep episode/time/endpoint draws identical when
            # W or feedback draw count changes. Terminal slots match across W.
            epsilon = keyed_noise(
                f"{episode.identifier}/train/{sample}", args.seed + step, frame, config.window, device
            )
            future_noise = torch.stack(
                [
                    keyed_noise(
                        f"{episode.identifier}/feedback/{sample}",
                        args.seed + step,
                        frame + 1,
                        config.window,
                        device,
                        draw=draw,
                    )
                    for draw in range(config.feedback_draws)
                ]
            )
            tau = 0.001 + 0.999 * torch.rand(1, generator=generator)
            if torch.rand((), generator=generator) < config.endpoint_probability:
                tau.fill_(1.0)
            result = training_objective(
                system,
                model,
                feedback_model,
                cache,
                episode.observations(frame),
                episode.target(frame, config.window),
                epsilon,
                tau.to(device),
                joint_decoder(episode, args.synthetic, smpl),
                next_observations=episode.observations(frame + 1),
                next_epsilon=future_noise,
                truth_next=state_map(episode.target(frame + 1), lambda x: x[:, -1]),
            )
            if not bool(torch.isfinite(result["loss"])):
                raise ValueError("Nonfinite training objective.")
            (result["loss"] / args.batch_size).backward()
            for key in components:
                components[key] += float(result[key].detach()) / args.batch_size
            del result
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        progress = {"step": step, **components, "gradient_norm": float(norm), "synthetic": args.synthetic}
        print(json.dumps(progress, allow_nan=False), flush=True)
        save_json(args.output / "progress.json", progress)
        if step % args.eval_every == 0 or step == final:
            rows = []
            count = 1 if args.synthetic else len(dev_data.records)
            if args.dev_limit is not None:
                count = min(count, args.dev_limit)
            for index in range(count):
                episode = fixture if args.synthetic else dev_data.load(index)
                row = evaluate_episode(system, ema, episode, joint_decoder(episode, args.synthetic, smpl))
                row.pop("predictions")
                rows.append(row)
            score = summarize(rows)
            curve.append({"step": step, "scores": score})
            improved = score["body_mm"]["mean"] < best
            best = min(best, score["body_mm"]["mean"])
            checkpoint = {
                "format": FORMAT,
                "model_config": model_config.to_dict(),
                "identity": identity,
                "protocol": config.to_dict(),
                "stats": stats,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "feedback_model": feedback_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best": best,
                "curve": curve,
                "sample_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            save_checkpoint(checkpoint_path, checkpoint)
            if improved:
                save_checkpoint(args.output / "best.pt", checkpoint)
            save_json(
                args.output / "selection.json",
                {"curve": curve, "best_body_mm": best, "synthetic": args.synthetic, "selection_split": "dev"},
            )
            save_json(args.output / "dev.json", rows)
