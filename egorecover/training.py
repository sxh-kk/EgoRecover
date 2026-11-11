"""Train the rolling-window model, with full-response paired EMA supervision."""

import hashlib
import json
import math
import platform
from collections import Counter
from contextlib import nullcontext

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.checkpoint import FORMAT, load_e7_weights, load_predictor, resolve_prior_mode
from egorecover.codec import MotionCodec
from egorecover.data import TrainingBatch, open_window_data, resolve_data_format
from egorecover.diagnostics import synthetic_episode
from egorecover.distributed import DistributedContext
from egorecover.evaluation import evaluate_episode, summarize
from egorecover.fk import joint_decoder, load_smpl, smpl_identity
from egorecover.history_training import HistoryTraining, sampling_signature
from egorecover.io import code_identity, file_sha256, save_checkpoint, save_json, state_sha256
from egorecover.losses import frozen_ema, multi_step_objective, training_objective, two_step_objective, update_ema
from egorecover.models import MotionModel
from egorecover.optimization import make_optimizer, make_scheduler, optimization_protocol
from egorecover.sampling import reference_microbatch_plan
from egorecover.selection import select_dev_records
from egorecover.state import state_map


def compatible_initialization(saved, config, identity, *, allow_rewrite_change=False):
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
        if key == "revise_recent" and allow_rewrite_change:
            continue
        if saved["protocol"][key] != config.to_dict()[key]:
            raise ValueError(f"Initialization changes controlled setting: {key}.")
    for key in (
        "manifest",
        "prior",
        "synthetic",
        "compact",
        "smpl_asset",
        "model_config",
        "data_format",
        "data_protocol",
    ):
        if saved["identity"][key] != identity[key]:
            raise ValueError(f"Initialization identity changed: {key}.")


def train(args):
    context = DistributedContext.from_environment(args.device)
    try:
        return _train(args, context)
    finally:
        context.close()


def _rng_state(generator, device):
    return {
        "sample_rng": generator.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _evaluate_indices(system, model, data, fixture, indices, smpl, args, context, *, comparison=False):
    rows = []
    for index in indices[context.rank :: context.world_size]:
        episode = fixture if args.synthetic else data.load(index)
        row = evaluate_episode(
            system,
            model,
            episode,
            joint_decoder(episode, args.synthetic, smpl, shape_mode=system.config.shape_mode),
            comparison=comparison,
        )
        row.pop("predictions")
        row.pop("body_rotations", None)
        baseline = (episode.teacher.joints[20:, :, :3, 3] - episode.truth.joints[20:, :, :3, 3]).norm(dim=-1)
        row["startup_prior_body_mm"] = float(baseline.mean() * 1000)
        rows.append(row)
    return sorted(context.gather_rows(rows), key=lambda row: row["episode"])


def _train(args, context):
    device = context.device
    config = WindowConfig.from_json(args.config)
    if isinstance(args.unroll_steps, bool) or args.unroll_steps not in (1, 2, 8):
        raise ValueError("Training unroll steps must be 1, 2 or 8.")
    if args.feedback_intervention not in ("full_state", "body_pose"):
        raise ValueError("Unknown feedback intervention.")
    if args.unroll_steps > 1 and (config.feedback != "none" or not config.current_weight):
        raise ValueError("Two-step baseline requires current geometry and no paired feedback.")
    if (args.detach_between_steps or args.velocity_weight) and args.unroll_steps == 1:
        raise ValueError("Cache detachment and emitted velocity supervision require multiple supervised calls.")
    if not math.isfinite(args.slow_weight) or args.slow_weight < 0:
        raise ValueError("Persistent-error weight must be finite and nonnegative.")
    if args.slow_weight and args.unroll_steps != 8:
        raise ValueError("Persistent-error supervision requires eight supervised calls.")
    if args.unroll_steps == 8 and not args.detach_between_steps:
        raise ValueError("The eight-call control requires explicit cache detachment between calls.")
    if not math.isfinite(args.velocity_weight) or args.velocity_weight < 0:
        raise ValueError("Velocity weight must be finite and nonnegative.")
    if not math.isfinite(args.velocity_scale_m_s) or args.velocity_scale_m_s <= 0:
        raise ValueError("Velocity normalization must be finite and positive in m/s.")
    if args.resume and args.initialize:
        raise ValueError("Resume and initialization are mutually exclusive.")
    if args.allow_initialize_rewrite_change and not (args.initialize or args.resume):
        raise ValueError("Rewrite initialization override requires --initialize or its native --resume.")
    if args.from_scratch and (args.e7_checkpoint or args.initialize):
        raise ValueError("Random G initialization cannot be combined with --e7-checkpoint or --initialize.")
    if args.dev_limit is not None and args.dev_limit < 1:
        raise ValueError("dev-limit must be positive.")
    if (
        args.train_steps < 1
        or args.batch_size < 1
        or args.microbatch_size < 1
        or args.eval_every < 1
        or args.rollin_frames < 0
    ):
        raise ValueError("Invalid training budget.")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be positive.")
    if args.batch_size % context.world_size:
        raise ValueError("Global batch-size must be divisible by the number of DDP ranks.")
    local_batch_size = args.batch_size // context.world_size
    reference_world_size = (
        context.world_size if args.sample_reference_world_size is None else args.sample_reference_world_size
    )
    if (
        isinstance(args.sample_reference_world_size, bool)
        or not isinstance(reference_world_size, int)
        or reference_world_size < 1
        or context.world_size % reference_world_size
        or args.batch_size % reference_world_size
        or (
            args.sample_reference_world_size is not None
            and (
                local_batch_size % args.microbatch_size
                or (args.batch_size // reference_world_size) % args.microbatch_size
            )
        )
    ):
        raise ValueError("Reference sampling ranks and microbatches must partition the physical/global batch.")
    sampling_rank = context.rank // (context.world_size // reference_world_size)
    history_training = HistoryTraining.from_options(args, reference_world_size)
    controlled_history = args.sampling_rollin_frames is not None
    optimization_protocol(args)
    prior_mode = resolve_prior_mode(args.prior_mode, args.prior_checkpoint)
    smpl = None
    dataset_format, dataset_protocol = "synthetic", {"scope": "synthetic_diagnostic"}
    if args.synthetic:
        if not args.compact or args.data_root or args.prior_checkpoint or args.e7_checkpoint:
            raise ValueError("Synthetic checks require --compact and no production assets.")
        fixture = synthetic_episode(device=device)
        train_data = dev_data = None
        stats, manifest = fixture.stats, "synthetic_diagnostic"
    else:
        if args.data_root is None:
            raise ValueError("Provide --data-root for production training.")
        if args.compact:
            raise ValueError("Compact mode is restricted to explicit synthetic checks.")
        if args.e7_checkpoint is None and args.initialize is None and not args.resume and not args.from_scratch:
            raise ValueError("Choose --from-scratch, --e7-checkpoint or --initialize explicitly.")
        dataset_format = resolve_data_format(args.data_root, args.data_format)
        if dataset_format == "official" and args.e7_checkpoint is not None:
            raise ValueError("Official-data runs use random G initialization; omit --e7-checkpoint.")
        smpl = load_smpl(device)
        bootstrap = None
        if dataset_format == "official":
            from .bootstrap import HeadAlignedBootstrap

            bootstrap = HeadAlignedBootstrap.from_smpl(smpl)
        train_data = open_window_data(
            args.data_root, "train", device=device, data_format=dataset_format, bootstrap=bootstrap
        )
        dev_data = open_window_data(
            args.data_root, "dev", device=device, data_format=dataset_format, bootstrap=bootstrap
        )
        stats, manifest = train_data.stats, train_data.digest
        dataset_protocol = getattr(train_data, "protocol", {"scope": "legacy_paired_cache"})
        if any(not torch.equal(stats[k], dev_data.stats[k]) for k in stats):
            raise ValueError("Train/dev normalization statistics disagree.")
    records = (
        [{"episode_id": fixture.identifier, "base_take_name": fixture.take}] if args.synthetic else dev_data.records
    )
    dev_indices, dev_selection = select_dev_records(records, args.dev_limit, args.dev_seed)
    if args.sample_reference_world_size is not None and dataset_format != "official":
        raise ValueError("Reference-rank sampling currently requires official data.")
    if controlled_history and (dataset_format != "official" or args.sample_reference_world_size is None):
        raise ValueError("Shared physical-target history ablations require official reference-rank sampling.")
    if args.unroll_steps == 8 and not args.synthetic and not controlled_history:
        raise ValueError("Production eight-call training requires the controlled official query sampler.")
    torch.manual_seed(args.seed)
    codec = MotionCodec(stats).to(device)
    prior = load_predictor(args.prior_checkpoint, device, None if args.synthetic else manifest)
    system = WindowReconstructor(codec, prior, config)
    model_config = ModelConfig.compact() if args.compact else ModelConfig()
    model = MotionModel(config, model_config).to(device)
    training_model = context.wrap(model)
    initial_sha = context.all_gather_object(state_sha256(model.state_dict()) if context.is_primary else None)[0]
    reference_clips = train_data.inventory["selected_clip_candidates"] if dataset_format == "official" else None
    optimization = optimization_protocol(args, reference_clips)
    optimizer = make_optimizer((p for p in model.parameters() if p.requires_grad), optimization)
    scheduler = make_scheduler(optimizer, optimization)
    identity = {
        "protocol": config.to_dict(),
        "seed": args.seed,
        "manifest": manifest,
        "prior": file_sha256(args.prior_checkpoint) if args.prior_checkpoint else "constant_velocity",
        "e7": file_sha256(args.e7_checkpoint) if args.e7_checkpoint else None,
        "initialize": file_sha256(args.initialize) if args.initialize else None,
        "synthetic": args.synthetic,
        "compact": args.compact,
        "from_scratch": args.from_scratch,
        "data_format": dataset_format,
        "data_protocol": dataset_protocol,
        "prior_mode": prior_mode,
        "model_config": model_config.to_dict(),
        "random_initial_model_sha256": initial_sha,
        "smpl_asset": "dense_diagnostic" if args.synthetic else smpl_identity(),
        "feedback_teacher": "fixed_initial_ema",
        "supervision": {
            "unroll_steps": args.unroll_steps,
            "detach_between_steps": args.detach_between_steps,
            "velocity_weight": args.velocity_weight,
            "velocity_scale_m_s": args.velocity_scale_m_s,
            "velocity_target": (
                "consecutive emitted world FK velocities against official GT joints"
                if config.geometry_target == "official"
                else "consecutive emitted world FK velocities; GT uses same fixed bootstrap shape"
            ),
            "velocity_timebase": "each training episode's measured consecutive timestamps",
            "feedback_intervention": args.feedback_intervention,
            "isolate_forward_rng": args.isolate_forward_rng,
            "loss_reduction": "mean across supervised online calls",
            "future_tau": "same sampled tau; each frame preserves the endpoint mixture marginal",
            "cache_gradients": (
                ("stopped between both calls" if args.detach_between_steps else "between both calls")
                if args.unroll_steps > 1
                else "current through frozen response only"
            ),
        },
        "code": code_identity(),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "eval_every": args.eval_every,
        "rollin_frames": args.rollin_frames,
        "learning_rate": args.learning_rate,
        "optimization": optimization,
        "dev_limit": args.dev_limit,
        "dev_selection": dev_selection,
        "full_dev_final": args.full_dev_final,
        "distributed": {
            "world_size": context.world_size,
            "global_batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "sample_seed": "seed + 1 + sampling_rank",
            "sample_reference_world_size": args.sample_reference_world_size,
            "logical_sampling_world_size": reference_world_size,
            "logical_batch_size": args.batch_size // reference_world_size,
            "sampling_partition": "contiguous physical-rank microbatches within reference-rank batches",
            "dropout_seed": "seed + 100000 * rank",
        },
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    if controlled_history:
        identity["history_training"] = history_training.protocol()
    if args.unroll_steps == 8:
        identity["supervision"][
            "cache_gradients"
        ] = "stopped between all eight calls; each emission retains its own parameter gradient"
        identity["supervision"]["persistent_error"] = {
            "weight": args.slow_weight,
            "pooling_scales": [2, 4, 8],
            "coordinate_frame": "shared floor-calibrated world axes",
            "reduction": "mean squared signed pooled errors; all overlapping blocks; equal per-scale weight",
            "cache_gradients": "stopped between calls; all eight emitted output graphs retained",
            "supervised_query_horizon": 8,
        }
    if args.allow_initialize_rewrite_change:
        identity["allow_initialize_rewrite_change"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "identity.json"
    # Continuations retain their original initialization provenance on resume;
    # the native resume checkpoint supplies the live model/optimizer states.
    if args.resume and identity_path.is_file():
        original_identity = json.loads(identity_path.read_text())
        identity["initialize"] = original_identity.get("initialize")
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
        compatible_initialization(initial, config, identity, allow_rewrite_change=args.allow_initialize_rewrite_change)
        model.load_state_dict(initial["ema"], strict=True)
        loaded_sha = context.all_gather_object(state_sha256(model.state_dict()) if context.is_primary else None)[0]
        if context.is_primary:
            save_json(
                args.output / "initialization.json",
                {
                    "checkpoint_sha256": identity["initialize"],
                    "checkpoint_step": initial["step"],
                    "weight_source": "ema",
                    "loaded_model_sha256": loaded_sha,
                    "optimizer": "fresh",
                    "allowed_rewrite_change": args.allow_initialize_rewrite_change,
                },
            )
        del initial
    elif args.e7_checkpoint:
        load_e7_weights(model, args.e7_checkpoint, weight_source="ema")
    ema = frozen_ema(model)
    # Keep the response map identical across paired/direct comparisons for the
    # entire continuation. A separate moving EMA is used for model selection.
    feedback_model = frozen_ema(model)
    torch.manual_seed(args.seed + 100000 * context.rank)
    generator = torch.Generator().manual_seed(args.seed + 1 + sampling_rank)
    start, best, curve = 0, float("inf"), []
    if saved:
        ema.load_state_dict(saved["ema"], strict=True)
        feedback_model.load_state_dict(saved["feedback_model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        rank_rng = saved["rank_rng_states"][context.rank]
        generator.set_state(rank_rng["sample_rng"].cpu())
        torch.set_rng_state(rank_rng["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(rank_rng["cuda_rng"].cpu(), device)
        start, best, curve = saved["step"], saved["best"], saved["curve"]
        del saved
    if context.is_primary:
        save_json(identity_path, identity)
        save_json(
            args.output / "status.json", {"status": "training", "start_step": start, "world_size": context.world_size}
        )
    context.barrier()
    final = min(args.train_steps, start + args.stop_after) if args.stop_after else args.train_steps
    for step in range(start + 1, final + 1):
        optimizer.zero_grad(set_to_none=True)
        components = {k: 0.0 for k in ("loss", "flow", "hand_pca", "current", "feedback")}
        if args.unroll_steps > 1:
            components["velocity"] = 0.0
        if args.unroll_steps == 8:
            components["slow"] = 0.0
        sampling_rows = []
        reference_plan = (
            reference_microbatch_plan(
                train_data,
                generator,
                rank=context.rank,
                world_size=context.world_size,
                reference_world_size=reference_world_size,
                batch_size=args.batch_size,
                microbatch_size=args.microbatch_size,
                rollin_frames=history_training.sampling_frames,
                endpoint_probability=config.endpoint_probability,
                horizon=max(2, args.unroll_steps),
            )
            if args.sample_reference_world_size is not None
            else None
        )
        for offset in range(0, local_batch_size, args.microbatch_size):
            count = min(args.microbatch_size, local_batch_size - offset)
            global_offset = context.rank * local_batch_size + offset
            planned = reference_plan[offset // args.microbatch_size] if reference_plan is not None else None
            if planned is not None:
                if (planned["offset"], planned["global_offset"], planned["count"]) != (offset, global_offset, count):
                    raise ValueError("Reference sampler disagrees with physical microbatch partition.")
                batch, frame = train_data.load_training_queries(planned["queries"]), planned["frame"]
            elif not args.synthetic and hasattr(train_data, "sample_batch"):
                batch, frame = train_data.sample_batch(generator, count, history_training.sampling_frames)
            else:
                episodes = tuple(fixture if args.synthetic else train_data.sample(generator) for _ in range(count))
                batch = TrainingBatch(episodes)
                frame = int(
                    torch.randint(
                        20, min(len(e) for e in episodes) - max(2, args.unroll_steps) + 1, (), generator=generator
                    )
                )
            rollin_frames = history_training.frames(step, global_offset)
            first = max(20, frame - rollin_frames)
            cache = batch.cache(first)
            model.eval()
            for t in range(first, frame):
                noise = batch.noise(args.seed + step, t, config.window, tag="rollin", offset=global_offset)
                cache = system.online(model, cache, batch.observations(t), noise).cache.detach()
            training_model.train()
            # Stateless streams keep episode/time/endpoint draws identical when
            # W or feedback draw count changes. Terminal slots match across W.
            epsilon = batch.noise(args.seed + step, frame, config.window, tag="train", offset=global_offset)
            future_noise = torch.stack(
                [
                    batch.noise(
                        args.seed + step, frame + 1, config.window, tag="feedback", offset=global_offset, draw=draw
                    )
                    for draw in range(config.feedback_draws)
                ]
            )
            if planned is None:
                tau = 0.001 + 0.999 * torch.rand(count, generator=generator)
                tau[torch.rand(count, generator=generator) < config.endpoint_probability] = 1.0
            else:
                tau = planned["tau"]
            if controlled_history:
                sampling_rows.append(
                    {
                        "global_offset": global_offset,
                        "queries": planned["queries"],
                        "target_local_frame": frame,
                        "tau": tau.tolist(),
                        "epsilon_sha256": hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest(),
                        **(
                            {
                                "next_epsilon_sha256": hashlib.sha256(
                                    future_noise[0].detach().cpu().numpy().tobytes()
                                ).hexdigest()
                            }
                            if args.isolate_forward_rng
                            else {}
                        ),
                        "rollin_frames": frame - first,
                    }
                )
            synchronization = (
                training_model.no_sync()
                if context.world_size > 1 and offset + count < local_batch_size
                else nullcontext()
            )
            with synchronization:
                forward_seed = (
                    int.from_bytes(
                        hashlib.sha256(f"objective/{args.seed}/{step}/{global_offset}".encode()).digest()[:8], "big"
                    )
                    % (2**62)
                    if args.isolate_forward_rng
                    else None
                )
                arguments = dict(
                    next_observations=batch.observations(frame + 1),
                    next_epsilon=future_noise[0] if args.unroll_steps == 2 else future_noise,
                    forward_seed=forward_seed,
                )
                if args.unroll_steps == 2:
                    arguments.update(
                        next_truth_window=batch.target(frame + 1, config.window),
                        detach_between=args.detach_between_steps,
                        velocity_weight=args.velocity_weight,
                        velocity_scale_m_s=args.velocity_scale_m_s,
                        dt=torch.stack([e.timestamps[frame + 1] - e.timestamps[frame] for e in batch.episodes]),
                    )
                else:
                    arguments.update(
                        truth_next=state_map(batch.target(frame + 1), lambda x: x[:, -1]),
                        feedback_intervention=args.feedback_intervention,
                    )
                if args.unroll_steps == 8:
                    noises = torch.stack(
                        [epsilon]
                        + [
                            batch.noise(
                                args.seed + step, frame + i, config.window, tag="feedback", offset=global_offset
                            )
                            for i in range(1, 8)
                        ]
                    )
                    if controlled_history:
                        sampling_rows[-1]["all_emission_noise_sha256"] = hashlib.sha256(
                            noises.detach().cpu().numpy().tobytes()
                        ).hexdigest()
                    objective = multi_step_objective
                    current_observations = [batch.observations(frame + i) for i in range(8)]
                    current_targets = [batch.target(frame + i, config.window) for i in range(8)]
                    current_noise = noises
                    arguments = {
                        "dt": torch.stack(
                            [
                                e.timestamps[frame + 1 : frame + 8] - e.timestamps[frame : frame + 7]
                                for e in batch.episodes
                            ]
                        ),
                        "velocity_weight": args.velocity_weight,
                        "velocity_scale_m_s": args.velocity_scale_m_s,
                        "slow_weight": args.slow_weight,
                        "forward_seed": forward_seed,
                    }
                else:
                    objective = two_step_objective if args.unroll_steps == 2 else training_objective
                    current_observations = batch.observations(frame)
                    current_targets = batch.target(frame, config.window)
                    current_noise = epsilon
                result = objective(
                    system,
                    training_model,
                    feedback_model,
                    cache,
                    current_observations,
                    current_targets,
                    current_noise,
                    tau.to(device),
                    joint_decoder(batch, args.synthetic, smpl, shape_mode=config.shape_mode),
                    **arguments,
                )
                if controlled_history and args.isolate_forward_rng and step == 1:
                    sampling_rows[-1]["first_current_codes_sha256"] = hashlib.sha256(
                        result["result"].codes.detach().cpu().numpy().tobytes()
                    ).hexdigest()
                if not bool(torch.isfinite(result["loss"])):
                    raise ValueError("Nonfinite training objective.")
                fraction = count / local_batch_size
                (result["loss"] * fraction).backward()
            for key in components:
                components[key] += float(result[key].detach()) * fraction
            del result
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        update_ema(ema, model, config.ema_decay)
        components = context.sync_scalar_dict(components)
        progress = {
            "step": step,
            **components,
            "gradient_norm": float(norm),
            "synthetic": args.synthetic,
            "target_queries_seen": step * args.batch_size,
            "supervised_current_frames_seen": step * args.batch_size * args.unroll_steps,
            "unroll_steps": args.unroll_steps,
            "effective_batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "learning_rate": used_lr,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "world_size": context.world_size,
            "local_batch_size": local_batch_size,
        }
        if controlled_history:
            shared = [row for rank_rows in context.all_gather_object(sampling_rows) for row in rank_rows]
            expected = list(range(0, args.batch_size, args.microbatch_size))
            if sorted(row["global_offset"] for row in shared) != expected:
                raise RuntimeError("Shared history queries do not partition the global batch.")
            counts = Counter()
            for row in shared:
                counts[str(row["rollin_frames"])] += args.microbatch_size
            progress["sampling_sha256"] = sampling_signature(shared)
            progress["rollin_frames_counts"] = dict(counts)
            progress["target_local_frame"] = 20 + history_training.sampling_frames
            if context.is_primary and step == 1:
                save_json(
                    args.output / "sampling-first-step.json", sorted(shared, key=lambda row: row["global_offset"])
                )
        if context.is_primary:
            print(json.dumps(progress, allow_nan=False), flush=True)
            save_json(args.output / "progress.json", progress)
            with (args.output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
        if step % args.eval_every == 0 or step == final:
            if context.is_primary:
                save_json(
                    args.output / "status.json", {"status": "validation", "step": step, "episodes": len(dev_indices)}
                )
            rows = _evaluate_indices(
                system, ema, dev_data, fixture if args.synthetic else None, dev_indices, smpl, args, context
            )
            score = summarize(rows)
            curve.append({"step": step, "scores": score})
            improved = score["body_mm"]["mean"] < best
            best = min(best, score["body_mm"]["mean"])
            rank_state = {
                **_rng_state(generator, device),
                "model_sha256": state_sha256(model.state_dict()),
                "ema_sha256": state_sha256(ema.state_dict()),
            }
            rank_rng_states = context.all_gather_object(rank_state)
            if (
                len({state["model_sha256"] for state in rank_rng_states}) != 1
                or len({state["ema_sha256"] for state in rank_rng_states}) != 1
            ):
                raise ValueError("DDP workers disagree on synchronized model or EMA parameters.")
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
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step": step,
                "best": best,
                "curve": curve,
                **rank_rng_states[0],
                "rank_rng_states": rank_rng_states,
            }
            if context.is_primary:
                save_checkpoint(checkpoint_path, checkpoint)
                if improved:
                    best_checkpoint = {
                        key: checkpoint[key]
                        for key in ("format", "model_config", "identity", "protocol", "stats", "ema", "step", "best")
                    }
                    best_checkpoint["purpose"] = "evaluation_or_initialization_only"
                    save_checkpoint(args.output / "best.pt", best_checkpoint)
                save_json(
                    args.output / "selection.json",
                    {"curve": curve, "best_body_mm": best, "synthetic": args.synthetic, "selection_split": "dev"},
                )
                save_json(args.output / "dev.json", rows)
                save_json(args.output / "status.json", {"status": "training", "step": step})
            context.barrier()
    if args.full_dev_final and final == args.train_steps:
        if context.is_primary:
            save_json(args.output / "status.json", {"status": "full_validation", "training_step": final})
        chosen = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True, mmap=True)
        ema.load_state_dict(chosen["ema"], strict=True)
        chosen_step = chosen["step"]
        del chosen
        rows = _evaluate_indices(
            system,
            ema,
            dev_data,
            fixture if args.synthetic else None,
            list(range(len(records))),
            smpl,
            args,
            context,
            comparison=True,
        )
        if context.is_primary:
            report = {
                "identity": {
                    "synthetic": args.synthetic,
                    "training": identity,
                    "latency_note": "concurrent multi-GPU validation; not an isolated latency benchmark",
                },
                "window": config.window,
                "checkpoint_step": chosen_step,
                "training_step": final,
                "results": {str(config.steps): {"rows": {"rollout": rows}, "summary": {"rollout": summarize(rows)}}},
            }
            from .evaluate import save_metrics_csv

            save_json(args.output / "full-dev.json", report)
            save_metrics_csv(args.output / "metrics.csv", report)
    if context.is_primary:
        save_json(
            args.output / "status.json",
            {"status": "complete" if final == args.train_steps else "stopped_after_checkpoint", "step": final},
        )
    context.barrier()
