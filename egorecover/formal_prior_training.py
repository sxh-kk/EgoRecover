"""Frozen-producer P01: pure G histories, exact phase quotas and resumable LR."""

import copy
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from .codec import MotionCodec
from .distributed import DistributedContext
from .fk import FixedShapeFK, load_smpl, smpl_identity
from .formal_prior_evaluation import evaluate_formal_prior, selection_take_excluded_report
from .io import code_identity, file_sha256, save_checkpoint, save_json, state_sha256
from .losses import update_ema
from .models import HistoryPrior
from .prior_data import TENSOR_FIELDS, predictor_forward, reconstructed_history_count, select_examples
from .prior_objectives import prior_objective_losses


class FormalPriorSampler:
    """Take/source/anchor balanced draws, with exactly 6:2 per micro8."""

    def __init__(self, rows):
        self.rows = rows
        groups = {True: defaultdict(lambda: defaultdict(list)), False: defaultdict(lambda: defaultdict(list))}
        for index, row in enumerate(rows):
            if row["kind"] != "predicted" or row["window"] != 3 or row["frame"] < 20:
                raise ValueError("Formal P samples must be actual W3 G histories.")
            source = row.get("source", row["episode"])
            groups[row["frame"] >= 40][row["take"]][source].append(index)
        if any(not group for group in groups.values()):
            raise ValueError("Formal P sampling needs both early and mature histories.")
        self.groups = {
            phase: [
                (take, [(source, sorted(indices)) for source, indices in sorted(sources.items())])
                for take, sources in sorted(takes.items())
            ]
            for phase, takes in groups.items()
        }
        self.visited = set()
        self.counts = Counter()

    def sample(self, generator, count=8):
        if count < 4 or count % 4:
            raise ValueError("Exact75:25 P sampling requires microbatch divisible by4.")
        indices = []
        for phase, amount in ((True, 3 * count // 4), (False, count // 4)):
            takes = self.groups[phase]
            for _ in range(amount):
                _, sources = takes[int(torch.randint(len(takes), (), generator=generator))]
                _, anchors = sources[int(torch.randint(len(sources), (), generator=generator))]
                index = anchors[int(torch.randint(len(anchors), (), generator=generator))]
                indices.append(index)
                self.visited.add(index)
                self.counts["mature" if phase else "early"] += 1
                self.counts[f"L{self.rows[index]['frame'] - 20}"] += 1
        return indices

    def state_dict(self):
        return {"visited_indices": sorted(self.visited), "sampling_counts": dict(self.counts)}

    def load_state_dict(self, state):
        self.visited = set(state["visited_indices"])
        self.counts = Counter(state["sampling_counts"])


def validate_formal_bank(bank, split, protocol, *, check_tensors=True, cohort=None, allow_limited=False):
    if bank.get("format") != "egorecover_prior_bank_v2" or bank.get("split") != split:
        raise ValueError("Formal P bank format/split mismatch.")
    expected = protocol["producer"]
    actual = bank.get("producer", {})
    keys = {
        "checkpoint_sha256": "sha256",
        "step": "step",
        "window": "window",
        "shape_mode": "shape_mode",
        "source": "source",
        "NFE": "NFE",
        "training_manifest": "native_checkpoint_manifest",
    }
    for bank_key, plan_key in keys.items():
        if actual.get(bank_key) != expected[plan_key]:
            raise ValueError(f"Formal P bank producer mismatch: {bank_key}.")
    if bank.get("producer_window") != 3 or not bank.get("manifest_sha256") or not bank.get("selection_sha256"):
        raise ValueError("Formal P bank misses producer/data/selection identity.")
    if not bank.get("rows") or set(bank.get("tensors", {})) != set(TENSOR_FIELDS):
        raise ValueError("Formal P bank has no rows or has incompatible tensor fields.")
    if cohort is not None:
        if bank.get("cohort") != cohort:
            raise ValueError("Formal P bank cohort mismatch.")
        if bank.get("limited_preflight") is not False and not allow_limited:
            raise ValueError("A limited preflight bank cannot be used for formal training/evaluation.")
        if cohort != "train" and not allow_limited:
            selection = protocol["evaluation"][cohort]
            if bank["selection_sha256"] != selection["selection_id"]:
                raise ValueError("Formal P bank selection mismatch.")
            if len({row["episode"] for row in bank["rows"]}) != selection["clips"]:
                raise ValueError("Formal P bank clip coverage mismatch.")
            if "total_prediction_rows" in selection:
                expected_rows = selection["total_prediction_rows"]
            else:
                frozen = json.loads(Path(selection["file"]).read_text())
                expected_rows = sum(row.get("num_frames", row.get("frames")) - 20 for row in frozen["records"])
            if len(bank["rows"]) != expected_rows:
                raise ValueError("Formal P dense evaluation frame coverage mismatch.")
    if any(len(tensor) != len(bank["rows"]) for tensor in bank["tensors"].values()):
        raise ValueError("Formal P bank tensors and metadata disagree.")
    MotionCodec(bank["stats"])
    for row in bank["rows"]:
        if (
            row["kind"] != "predicted"
            or row["window"] != 3
            or row["frame"] < 20
            or row.get("L", row["frame"] - 20) != row["frame"] - 20
        ):
            raise ValueError("Invalid local query or history provenance in formal P bank.")
    if check_tensors:
        for tensor in bank["tensors"].values():
            for start in range(0, len(tensor), 2048):
                if not bool(torch.isfinite(tensor[start : start + 2048]).all()):
                    raise ValueError("Nonfinite formal P bank tensor.")
        flags = bank["tensors"]["body_valid"]
        if flags.dtype != torch.bool or flags.shape != (len(bank["rows"]), 20):
            raise ValueError("Formal P history source flags have wrong shape/type.")
        for start in range(0, len(flags), 2048):
            counts = torch.tensor(
                [reconstructed_history_count(row["frame"] - 20, 3, True) for row in bank["rows"][start : start + 2048]]
            )[:, None]
            expected_flags = torch.arange(20)[None] >= 20 - counts
            if not torch.equal(flags[start : start + 2048], expected_flags):
                raise ValueError("Formal P source flags do not reflect W3 local roll-in.")


def load_formal_bank(path, split, protocol, *, check_tensors=True, cohort=None, allow_limited=False):
    bank = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_formal_bank(bank, split, protocol, check_tensors=check_tensors, cohort=cohort, allow_limited=allow_limited)
    return bank


def formal_settings(args, protocol, context):
    opt = protocol["optimization"]
    sampling = protocol["sampling"]
    objective = protocol["loss"].get("objective", "raw_code")
    if objective not in ("raw_code", "common_cv_valid_state"):
        raise ValueError("Unknown formal P objective.")
    fixed_weight = protocol["loss"].get("fixed_geometry_weight")
    if fixed_weight is not None and (
        isinstance(fixed_weight, bool) or not math.isfinite(fixed_weight) or fixed_weight <= 0
    ):
        raise ValueError("A fixed P geometry weight must be finite and positive.")
    if (
        args.batch_size != sampling["global_batch"]
        or args.microbatch_size != sampling["microbatch"]
        or args.seed != protocol["model"]["seed"]
        or args.learning_rate != opt["learning_rate"]
        or args.ema_decay != opt["ema_decay"]
    ):
        raise ValueError("Formal P hyperparameters disagree with the frozen protocol.")
    decay = opt["lr_step_size"] if args.lr_decay_step is None else args.lr_decay_step
    if args.formal_preflight:
        if not 1 <= args.train_steps <= 4 or args.eval_every != 1 or "preflight" not in str(args.output):
            raise ValueError("Shortened formal preflight must be explicitly isolated and at most4 updates.")
    elif (
        args.train_steps != opt["max_updates"]
        or args.eval_every != opt["eval_every"]
        or decay != opt["lr_step_size"]
        or args.lr_gamma != opt["lr_gamma"]
        or context.world_size != sampling["GPUs"]
    ):
        raise ValueError("Formal P budget, LR schedule or GPU count changed.")
    if (
        decay < 1
        or not 0 < args.lr_gamma <= 1
        or args.batch_size % context.world_size
        or (args.batch_size // context.world_size) % args.microbatch_size
        or (args.stop_after is not None and args.stop_after < 1)
    ):
        raise ValueError("Invalid formal P schedule, batch partition or stopping budget.")
    return {
        "train_steps": args.train_steps,
        "eval_every": args.eval_every,
        "lr_step_size": decay,
        "lr_gamma": args.lr_gamma,
        "preflight": args.formal_preflight,
    }


def gradient_audit(model, example_tensors, indices, codec, fk, device, *, microbatch=8, objective="raw_code"):
    """Norms of mean gradients over a fixed train-only diagnostic batch."""
    model.eval()
    parameters = tuple(model.parameters())
    accum = {key: [torch.zeros_like(p) for p in parameters] for key in ("code", "geometry")}
    for start in range(0, len(indices), microbatch):
        selected = indices[start : start + microbatch]
        losses = prior_objective_losses(
            model, select_examples(example_tensors, selected, device), codec, fk, objective=objective
        )
        for key in accum:
            grads = torch.autograd.grad(losses[key], parameters, retain_graph=key == "code", allow_unused=True)
            for target, gradient in zip(accum[key], grads):
                if gradient is not None:
                    target.add_(gradient, alpha=len(selected) / len(indices))
    squared = {
        key: torch.stack([value.square().sum() for value in gradients]).sum() for key, gradients in accum.items()
    }
    norms = {key: float(value.sqrt()) for key, value in squared.items()}
    dot = float(torch.stack([(a * b).sum() for a, b in zip(accum["code"], accum["geometry"])]).sum())
    if any(not math.isfinite(n) for n in norms.values()):
        raise ValueError("Nonfinite P diagnostic gradient.")
    cosine = dot / max(norms["code"] * norms["geometry"], 1e-30)
    return {"gradient_norms": norms, "gradient_cosine": cosine, "finite": math.isfinite(cosine)}


def calibrate_formal_geometry(model, bank, codec, fk, device, protocol):
    calibration = protocol["loss"]["geometry_calibration"]
    sampler = FormalPriorSampler(bank["rows"])
    generator = torch.Generator().manual_seed(calibration["seed"])
    indices = []
    for _ in range(calibration["samples"] // 8):
        indices.extend(sampler.sample(generator, 8))
    objective = protocol["loss"].get("objective", "raw_code")
    audit = gradient_audit(model, bank["tensors"], indices, codec, fk, device, objective=objective)
    norms = audit["gradient_norms"]
    if min(norms.values()) <= 1e-12:
        raise ValueError("Degenerate P geometry calibration gradients.")
    fixed_weight = protocol["loss"].get("fixed_geometry_weight")
    weight = (
        float(fixed_weight)
        if fixed_weight is not None
        else min(calibration["weight_cap"], calibration["gradient_ratio_target"] * norms["code"] / norms["geometry"])
    )
    return {
        **audit,
        "weight": weight,
        "indices": indices,
        "calibration_seed": calibration["seed"],
        "samples": len(indices),
        "target_weighted_geometry_ratio": calibration["gradient_ratio_target"] if fixed_weight is None else None,
        "initial_weighted_geometry_to_code_ratio": weight * norms["geometry"] / norms["code"],
        "weight_mode": "P01_fixed_not_recalibrated" if fixed_weight is not None else "train_gradient_calibration",
        "objective": objective,
    }


def initial_health(model, bank, codec, fk, device):
    example = select_examples(bank["tensors"], list(range(min(8, len(bank["rows"])))), device)
    model.eval()
    with torch.no_grad():
        result = predictor_forward(model, example)
        identical = torch.equal(result, example["base"])
        perturbed = {key: value.clone() for key, value in example.items()}
        perturbed["target"].add_(123)
        perturbed["truth_xyz"].add_(10)
        causal = torch.equal(result, predictor_forward(model, perturbed))
    prediction = example["base"][:, 0].detach().requires_grad_(True)
    state = codec.decode_current(prediction, example["reference"])
    gradient = torch.autograd.grad(fk(state).square().sum(), prediction)[0][..., -10:]
    return {
        "initial_P_equals_CV": identical,
        "causal_inputs_test": causal,
        "beta_FK_gradient": bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0),
    }


def _formal_report(model, bank, codec, fk, context, *, foot_diagnostics=False):
    return evaluate_formal_prior(
        model,
        bank,
        codec,
        fk,
        device=context.device,
        context=context,
        include_foot_diagnostics=foot_diagnostics,
    )


def train_formal_prior(args):
    with DistributedContext.from_environment(args.device) as context:
        return _train_formal_prior(args, context)


def _checkpoint_export(ema, model_config, identity, stats, step, summary):
    # The best snapshot must not share mutable EMA storage with later updates.
    weights = {key: value.detach().cpu().clone() for key, value in ema.items()}
    return {
        "format": "egorecover_prior_v1",
        "manifest_sha256": identity["manifest_sha256"],
        "model_config": copy.deepcopy(model_config),
        "state_dict": weights,
        "state_sha256": state_sha256(weights),
        "step": step,
        "validation_summary": copy.deepcopy(summary),
        "identity": identity,
        "stats": stats,
    }


def _publish_prior_export(path, export):
    """Preserve an already valid file, including its externally bound byte SHA."""
    path = Path(path)
    if path.exists():
        try:
            existing = torch.load(path, weights_only=True, map_location="cpu")
            metadata = (
                "format",
                "manifest_sha256",
                "model_config",
                "step",
                "validation_summary",
                "identity",
                "state_sha256",
            )
            if (
                isinstance(existing, dict)
                and all(existing.get(key) == export[key] for key in metadata)
                and existing["stats"].keys() == export["stats"].keys()
                and all(torch.equal(existing["stats"][key], export["stats"][key]) for key in export["stats"])
                and state_sha256(existing["state_dict"]) == export["state_sha256"]
            ):
                return
        except (
            OSError,
            RuntimeError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
            KeyError,
            TypeError,
            AttributeError,
        ):
            pass  # A damaged derived export is repairable from the committed resume.
    save_checkpoint(path, export)


def _recover_uncommitted_logs(directory, step):
    """Keep committed traces, preserving rolled-back tails as audit artifacts."""
    directory = Path(directory)
    log = directory / "train.jsonl"
    kept, discarded = [], []
    if step and not log.exists():
        raise ValueError("Committed training trace is missing; recover the original log before continuing.")
    if log.exists():
        lines = log.read_bytes().splitlines(keepends=True)
        previous = 0
        for ordinal, line in enumerate(lines):
            try:
                row = json.loads(line)
                row_step = int(row["step"])
            except (ValueError, KeyError, TypeError):
                if ordinal != len(lines) - 1:
                    raise ValueError("Invalid committed training trace; refuse to rewrite history.") from None
                discarded.append(line)
                continue
            if row_step > step:
                discarded.append(line)
            else:
                if discarded or row_step != previous + 1:
                    raise ValueError("Committed training trace is duplicated or out of order.")
                kept.append(line)
                previous = row_step
        if step and previous != step:
            raise ValueError("Committed training trace does not reach the recovery checkpoint.")
    future = []
    for pattern in ("validation-step*.json", "gradient-step*.json"):
        for path in directory.glob(pattern):
            number = path.stem.rsplit("step", 1)[1]
            if number.isdigit() and int(number) > step:
                future.append(path)
    if not discarded and not future:
        return None
    archive = directory / "recovery" / str(time.time_ns())
    archive.mkdir(parents=True)
    if discarded:
        (archive / "uncommitted-train.jsonl").write_bytes(b"".join(discarded))
        temporary = log.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(kept))
        temporary.replace(log)
    for path in future:
        path.replace(archive / path.name)
    save_json(archive / "rollback.json", {"committed_step": step, "discarded_log_lines": len(discarded)})
    return str(archive)


def recover_formal_checkpoint(directory, saved=None):
    """Republish derived artifacts from one committed, self-contained resume.

    The controller can call this after workers exit even if a pilot/cap step is
    already complete. Never call it while a worker is writing this directory.
    Older snapshots remain readable; only v3 owns the new recovery contract.
    """
    directory = Path(directory)
    if saved is None:
        path = directory / "resume.pt"
        if not path.exists():
            return {"status": "no_checkpoint"}
        saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved.get("format") != "egorecover_formal_prior_resume_v3":
        return {"status": "legacy_checkpoint", "step": saved.get("step")}
    step, identity = saved["step"], saved["identity"]
    current, best = saved["export"], saved["best_export"]
    record = min(saved["curve"], key=lambda row: row["selection_score"])
    if not math.isclose(float(record["selection_score"]), float(saved["best"]), rel_tol=0, abs_tol=1e-9):
        raise ValueError("Committed best score differs from its selection curve.")
    for export, expected_step in ((current, step), (best, record["step"])):
        if (
            export["identity"] != identity
            or export["step"] != expected_step
            or export["model_config"] != identity["model_config"]
            or export["state_sha256"] != state_sha256(export["state_dict"])
        ):
            raise ValueError("Committed P export identity, step or weights are inconsistent.")
    if current["state_sha256"] != state_sha256(saved["ema"]):
        raise ValueError("Committed current export does not match the recoverable EMA.")
    saved_identity = directory / "identity.json"
    if saved_identity.exists() and json.loads(saved_identity.read_text()) != identity:
        raise ValueError("Cannot repair exports under a different run identity.")
    archive = _recover_uncommitted_logs(directory, step)
    _publish_prior_export(directory / "last.pt", current)
    _publish_prior_export(directory / "best.pt", best)
    if step in (2000, 8000):
        _publish_prior_export(directory / f"ema-step{step}.pt", current)
    if step == identity["settings"]["train_steps"]:
        _publish_prior_export(directory / "final.pt", current)
    for name, value in saved["artifacts"].items():
        save_json(directory / name, value)
    return {"status": "recovered", "step": step, "best_step": best["step"], "rollback_archive": archive}


def _train_formal_prior(args, context):
    torch.set_num_threads(2)
    protocol = json.loads(args.formal_protocol.read_text())
    settings = formal_settings(args, protocol, context)
    device = context.device
    # Read-only mappings are shared through the host page cache, not four copies of DINO/motion.
    train_bank = load_formal_bank(
        args.train_bank,
        "train",
        protocol,
        check_tensors=context.is_primary,
        cohort="train",
        allow_limited=args.formal_preflight,
    )
    val_bank = load_formal_bank(
        args.val_bank,
        "val",
        protocol,
        check_tensors=context.is_primary,
        cohort="D128",
        allow_limited=args.formal_preflight,
    )
    train_takes = {row["take"] for row in train_bank["rows"]}
    if train_takes & {row["take"] for row in val_bank["rows"]}:
        raise ValueError("Formal P train/val histories share takes.")
    if any(not torch.equal(train_bank["stats"][k], val_bank["stats"][k]) for k in train_bank["stats"]):
        raise ValueError("Formal P train/val normalization differs.")
    if not args.formal_preflight:
        inventory = protocol["data"]["train_inventory"]
        if (
            len(train_takes) != inventory["eligible_takes"]
            or len({r["source"] for r in train_bank["rows"]}) != inventory["eligible_sources"]
        ):
            raise ValueError("Formal train bank does not cover the declared eligible takes/sources.")
        for phase in ("early", "mature"):
            selected = [row for row in train_bank["rows"] if (row["frame"] >= 40) == (phase == "mature")]
            if (
                len({row["take"] for row in selected}) != inventory[f"{phase}_pool_takes"]
                or len({row["source"] for row in selected}) != inventory[f"{phase}_pool_sources"]
            ):
                raise ValueError(f"Formal {phase} sampling pool does not match its declared coverage.")
        if len(train_bank["rows"]) != protocol["data"]["bank_size"]["expected_rows_from_length_metadata"]:
            raise ValueError("Formal train anchor coverage differs from the frozen plan.")
    codec = MotionCodec(train_bank["stats"]).to(device)
    smpl = load_smpl(device)
    asset_identity = smpl_identity()
    if any(bank.get("provenance", {}).get("producer_smpl_asset") != asset_identity for bank in (train_bank, val_bank)):
        raise ValueError("Formal P FK asset differs from the frozen history producer asset.")
    fk = FixedShapeFK(smpl, torch.zeros(10, device=device), shape_mode="predicted")
    torch.manual_seed(args.seed)
    model_config = {key: protocol["model"][key] for key in ("width", "layers", "heads", "dropout", "source_flags")}
    if "hold_beta" in protocol["model"]:
        model_config["hold_beta"] = protocol["model"]["hold_beta"]
    model = HistoryPrior(**model_config).to(device)
    # This copy never receives DDP reducer hooks; train-only autograd diagnostics
    # must not start a rank0-only backward on the live distributed parameters.
    audit_model = copy.deepcopy(model) if context.is_primary else None
    old_identity = json.loads((args.output / "identity.json").read_text()) if args.resume else None
    initial_sha = state_sha256(model.state_dict())
    if context.is_primary:
        audit = (
            old_identity["geometry_audit"]
            if old_identity
            else calibrate_formal_geometry(audit_model, train_bank, codec, fk, device, protocol)
        )
        health = initial_health(model, val_bank, codec, fk, device)
        bank_hashes = {"train": file_sha256(args.train_bank), "val": file_sha256(args.val_bank)}
    else:
        audit = health = bank_hashes = None
    audit = context.all_gather_object(audit)[0]
    health = context.all_gather_object(health)[0]
    bank_hashes = context.all_gather_object(bank_hashes)[0]
    health.update(
        producer_identity=True,
        train_val_take_disjoint=True,
        finite_objective_and_gradients=True,
        exact_phase_quota=True,
        gradient_diagnostics_finite=audit["finite"],
    )
    if not all(health.values()):
        raise ValueError(f"Initial formal P health check failed: {health}")
    identity = {
        "protocol_sha256": file_sha256(args.formal_protocol),
        "model_config": model_config,
        "seed": args.seed,
        "from_scratch": True,
        "random_initial_model_sha256": initial_sha,
        "manifest_sha256": train_bank["manifest_sha256"],
        "val_manifest_sha256": val_bank["manifest_sha256"],
        "smpl_asset_sha256": asset_identity,
        "bank_sha256": bank_hashes,
        "producer": train_bank["producer"],
        "settings": settings,
        "world_size": context.world_size,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "learning_rate": args.learning_rate,
        "ema_decay": args.ema_decay,
        "geometry_audit": audit,
        "objective": protocol["loss"].get("objective", "raw_code"),
        "hold_beta": model_config.get("hold_beta", False),
        "code": code_identity(),
        "shape_mode": "predicted",
        "geometry_target": "official",
        "phase_quotas": {"mature": 48, "early": 16},
        "GT_history_fraction": 0,
        "TwoForward": False,
        "current_observations_to_P": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "identity.json").exists() and json.loads((args.output / "identity.json").read_text()) != identity:
        raise ValueError("Formal P run identity changed; use a new run.")
    if (args.output / "resume.pt").exists() and not args.resume:
        raise ValueError("Formal P resume exists; require explicit --resume.")
    training_model = context.wrap(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings["lr_step_size"], gamma=settings["lr_gamma"])
    generator = torch.Generator().manual_seed(args.seed + 1 + context.rank)
    sampler = FormalPriorSampler(train_bank["rows"])
    torch.manual_seed(args.seed + 100000 * context.rank)
    start, best, curve, best_export = 0, float("inf"), [], None
    if args.resume:
        saved = torch.load(args.output / "resume.pt", weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError("Formal P resume identity changed.")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["rank_rng_states"][context.rank]
        sampler.load_state_dict(state)
        generator.set_state(state["sample_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start, best, curve = saved["step"], saved["best"], saved["curve"]
        if context.is_primary:
            recover_formal_checkpoint(args.output, saved)
            best_export = saved.get("best_export")
            if best_export is None:
                # Legacy v2 checkpoints cannot recover a missing historical
                # best, so fail closed rather than silently substituting last.
                candidate = args.output / "best.pt"
                if not candidate.exists():
                    raise ValueError("Legacy P resume has no best export; recover its original checkpoint first.")
                best_export = torch.load(candidate, weights_only=True, map_location="cpu")
                best_record = min(curve, key=lambda row: row["selection_score"])
                if best_export["identity"] != identity or best_export["step"] != best_record["step"]:
                    raise ValueError("Legacy P best export differs from the committed selection curve.")
                best_export["state_sha256"] = state_sha256(best_export["state_dict"])
        if "artifacts" in saved:
            health = dict(saved["artifacts"]["health_checks.json"]["checks"])
        del saved
    if context.is_primary:
        if not start:
            _recover_uncommitted_logs(args.output, 0)
        save_json(args.output / "identity.json", identity)
        save_json(args.output / "geometry-calibration.json", audit)
        save_json(
            args.output / "status.json", {"phase": "initial_validation" if not start else "training", "step": start}
        )
        save_json(args.output / "health_checks.json", {"step": start, "passed": all(health.values()), "checks": health})
    context.barrier()
    if not start:
        report = _formal_report(
            ema,
            val_bank,
            codec,
            fk,
            context,
            foot_diagnostics=protocol["evaluation"].get("foot_diagnostics", False),
        )
        if (
            abs(float(report["selection_score"]) - float(report["summary"]["mature"]["metrics"]["mpjpe_mm"]["CV"]))
            > 1e-6
        ):
            raise ValueError("Initial P and matched CV validation are not equivalent.")
        if context.is_primary:
            save_json(args.output / "validation-step0.json", {"step": 0, **report})
    final = min(args.train_steps, start + args.stop_after) if args.stop_after else args.train_steps
    local_batch = args.batch_size // context.world_size
    for step in range(start + 1, final + 1):
        t0 = time.monotonic()
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        lr_used = optimizer.param_groups[0]["lr"]
        metrics = {"code": 0.0, "geometry": 0.0, "hand_pca": 0.0, "loss": 0.0}
        traces = []
        phase_counts = Counter()
        for offset in range(0, local_batch, args.microbatch_size):
            indices = sampler.sample(generator, args.microbatch_size)
            traces.extend(indices)
            phase_counts.update("mature" if train_bank["rows"][i]["frame"] >= 40 else "early" for i in indices)
            example = select_examples(train_bank["tensors"], indices, device)
            sync = (
                training_model.no_sync()
                if context.world_size > 1 and offset + args.microbatch_size < local_batch
                else nullcontext()
            )
            with sync:
                losses = prior_objective_losses(training_model, example, codec, fk, objective=identity["objective"])
                loss = losses["code"] + audit["weight"] * losses["geometry"]
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite formal P objective.")
                (loss * args.microbatch_size / local_batch).backward()
            for key in losses:
                metrics[key] += float(losses[key].detach()) * args.microbatch_size / local_batch
            metrics["loss"] += float(loss.detach()) * args.microbatch_size / local_batch
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, args.ema_decay)
        metrics = context.sync_scalar_dict(metrics)
        sampled = context.all_gather_object({"indices": traces, "counts": dict(phase_counts)})
        global_counts = Counter()
        for partition in sampled:
            global_counts.update(partition["counts"])
        if global_counts != {"mature": 3 * args.batch_size // 4, "early": args.batch_size // 4}:
            raise RuntimeError("Formal P global phase quota drifted.")
        trace_hash = hashlib.sha256(json.dumps([p["indices"] for p in sampled]).encode()).hexdigest()
        progress = {
            "step": step,
            **metrics,
            "lr": lr_used,
            "lr_next": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(norm),
            "geometry_weight": audit["weight"],
            "phase": "frozen_W3_G_history",
            "mature_count": global_counts["mature"],
            "early_count": global_counts["early"],
            "trace_hash": trace_hash,
            "elapsed_seconds": time.monotonic() - t0,
        }
        if context.is_primary:
            save_json(args.output / "progress.json", progress)
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
            if step <= 5 or step % 50 == 0:
                print(json.dumps(progress), flush=True)
        if step in protocol["loss"]["geometry_calibration"]["diagnostic_steps"] or args.formal_preflight:
            if context.is_primary:
                audit_model.load_state_dict(model.state_dict(), strict=True)
                diagnostic = gradient_audit(
                    audit_model,
                    train_bank["tensors"],
                    audit["indices"],
                    codec,
                    fk,
                    device,
                    objective=identity["objective"],
                )
            else:
                diagnostic = None
            diagnostic = context.all_gather_object(diagnostic)[0]
            health["gradient_diagnostics_finite"] = health["gradient_diagnostics_finite"] and diagnostic["finite"]
            if context.is_primary:
                diagnostic["weighted_geometry_to_code_ratio"] = (
                    audit["weight"]
                    * diagnostic["gradient_norms"]["geometry"]
                    / max(diagnostic["gradient_norms"]["code"], 1e-30)
                )
                save_json(args.output / f"gradient-step{step}.json", diagnostic)
        if step % args.eval_every == 0 or step == final:
            if context.is_primary:
                save_json(args.output / "status.json", {"phase": "validation", "step": step})
            report = _formal_report(
                ema,
                val_bank,
                codec,
                fk,
                context,
                foot_diagnostics=protocol["evaluation"].get("foot_diagnostics", False),
            )
            score = float(report["selection_score"])
            if not math.isfinite(score):
                raise ValueError("Nonfinite formal P validation.")
            improved = score < best
            best = min(best, score)
            curve.append({"step": step, "selection_score": score, "summary": report["summary"]})
            rank_state = {
                "sample_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "model_sha256": state_sha256(model.state_dict()),
                **sampler.state_dict(),
            }
            rank_states = context.all_gather_object(rank_state)
            if len({s["model_sha256"] for s in rank_states}) != 1:
                raise RuntimeError("Formal P DDP rank model weights disagree.")
            if context.is_primary:
                seen = set().union(*(set(s["visited_indices"]) for s in rank_states))
                artifacts = {
                    f"validation-step{step}.json": {"step": step, **report},
                    "validation.json": {"step": step, **report},
                    "selection.json": {
                        "best_selection_mm": best,
                        "selection_key": report["selection_key"],
                        "curve": curve,
                    },
                    "health_checks.json": {"step": step, "passed": all(health.values()), "checks": dict(health)},
                    "coverage.json": {
                        "step": step,
                        "unique_anchors": len(seen),
                        "bank_anchors": len(train_bank["rows"]),
                        "unique_sources": len(
                            {train_bank["rows"][i].get("source", train_bank["rows"][i]["episode"]) for i in seen}
                        ),
                        "unique_takes": len({train_bank["rows"][i]["take"] for i in seen}),
                        "sampling_counts_per_rank": [s["sampling_counts"] for s in rank_states],
                    },
                    "progress.json": progress,
                }
                export = _checkpoint_export(
                    ema.state_dict(), model_config, identity, train_bank["stats"], step, report["summary"]
                )
                if improved:
                    best_export = export
                committed = {
                    "format": "egorecover_formal_prior_resume_v3",
                    "identity": identity,
                    "model": model.state_dict(),
                    "ema": export["state_dict"],
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rank_rng_states": rank_states,
                    "step": step,
                    "best": best,
                    "curve": curve,
                    "export": export,
                    "best_export": best_export,
                    "artifacts": artifacts,
                }
                # One atomic commit owns weights, the immutable historical
                # best and selection/health records. All other files can be
                # republished after any interruption, including at2k/10k.
                save_checkpoint(args.output / "resume.pt", committed)
                recover_formal_checkpoint(args.output, committed)
                save_json(args.output / "status.json", {"phase": "training", "step": step})
            context.barrier()
    if context.is_primary:
        save_json(
            args.output / "status.json",
            {"phase": "complete" if final == args.train_steps else "stopped_after_checkpoint", "step": final},
        )


def evaluate_formal_checkpoint(args):
    with DistributedContext.from_environment(args.device) as context:
        torch.set_num_threads(2)
        protocol = json.loads(args.formal_protocol.read_text())
        saved = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
        bank = load_formal_bank(args.val_bank, "val", protocol, check_tensors=context.is_primary)
        if bank.get("cohort") not in ("D128", "D400", "long50"):
            raise ValueError("Formal checkpoint evaluation needs a declared fixed evaluation cohort.")
        validate_formal_bank(bank, "val", protocol, check_tensors=False, cohort=bank["cohort"])
        if saved["identity"]["producer"] != bank["producer"]:
            raise ValueError("P checkpoint and evaluation bank producers differ.")
        if any(not torch.equal(saved["stats"][k], bank["stats"][k]) for k in saved["stats"]):
            raise ValueError("P checkpoint and evaluation statistics differ.")
        codec = MotionCodec(saved["stats"]).to(context.device)
        smpl = load_smpl(context.device)
        asset_identity = smpl_identity()
        if (
            bank.get("provenance", {}).get("producer_smpl_asset") != asset_identity
            or saved["identity"]["smpl_asset_sha256"] != asset_identity
        ):
            raise ValueError("Formal P evaluation FK asset changed.")
        fk = FixedShapeFK(smpl, torch.zeros(10, device=context.device), shape_mode="predicted")
        model = HistoryPrior(**saved["model_config"]).to(context.device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        report = evaluate_formal_prior(
            model,
            bank,
            codec,
            fk,
            device=context.device,
            context=context,
            include_pa=True,
            include_parts=True,
            include_beta=True,
            include_shared_beta=True,
            include_foot_diagnostics=protocol["evaluation"].get("foot_diagnostics", False),
        )
        if context.is_primary:
            supplement = {}
            if bank["cohort"] in ("D400", "long50"):
                selection = protocol["evaluation"]["D128"]
                selection_path = Path(selection["file"])
                frozen = json.loads(selection_path.read_text())
                if frozen["sha256"] != selection["selection_id"]:
                    raise ValueError("D128 exclusion selection differs from the frozen protocol.")
                expected_hash = selection.get("selection_file_sha256")
                if expected_hash is not None and file_sha256(selection_path) != expected_hash:
                    raise ValueError("D128 exclusion file contents changed after protocol freeze.")
                excluded = {row["take"] for row in frozen["records"]}
                if (
                    len(frozen["records"]) != selection["clips"]
                    or len(frozen["records"]) != frozen["clips"]
                    or len(excluded) != frozen["takes"]
                ):
                    raise ValueError("D128 exclusion selection has inconsistent clip/take counts.")
                supplementary = selection_take_excluded_report(
                    report,
                    excluded,
                    selection_id=selection["selection_id"],
                    bootstrap_samples=protocol["evaluation"]["paired_CI"]["replicates"],
                    bootstrap_seed=protocol["evaluation"]["paired_CI"]["seed"],
                )
                expected = protocol["evaluation"]["selection_and_report_overlap"][bank["cohort"]]
                if (
                    supplementary["overlap_clips"] != expected["overlap_clips"]
                    or supplementary["clips"] != expected["remaining_clips_excluding_D128_takes"]
                    or supplementary["takes"] != expected["remaining_takes_excluding_D128"]
                ):
                    raise ValueError("D128-take-excluded coverage differs from the declared protocol.")
                supplementary.update(selection_file_sha256=file_sha256(selection_path))
                supplement["selection_take_excluded"] = supplementary
            save_json(
                args.output / "report.json",
                {
                    "checkpoint_step": saved["step"],
                    "checkpoint_sha256": file_sha256(args.evaluate_checkpoint),
                    "bank_sha256": file_sha256(args.val_bank),
                    "bank_cohort": bank["cohort"],
                    "bank_selection_sha256": bank["selection_sha256"],
                    "bank_rows": len(bank["rows"]),
                    "bank_clips": len({row["episode"] for row in bank["rows"]}),
                    **report,
                    **supplement,
                },
            )
