"""Version-two frozen-G history banks with resumable, bounded-size chunks.

The production CLI loads this module by filename inside the producer's frozen
package environment. All G, codec, data, bootstrap and FK imports then resolve
to that snapshot rather than the editable P-training implementation.
"""

import fcntl
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.bootstrap import HeadAlignedBootstrap
from egorecover.codec import MotionCodec
from egorecover.data import TrainingBatch, keyed_noise
from egorecover.fk import FixedShapeFK, load_smpl, smpl_identity
from egorecover.io import code_identity, file_sha256, save_checkpoint, save_json, state_sha256
from egorecover.models import MotionModel
from egorecover.official import OfficialWindowData
from egorecover.prior_data import TENSOR_FIELDS, concatenate_examples, prior_example, reconstructed_history_count
from egorecover.state import WindowCache, state_map

FORMAT = "egorecover_prior_bank_v2"
EARLY_L = (0, 1, 4, 8, 12, 19)
STATS_FIELDS = ("motion_mean", "motion_std", "traj_mean", "traj_std")
TENSOR_SHAPES = {
    "history": (20, 243),
    "base": (1, 243),
    "target": (1, 243),
    "reference": (4, 4),
    "truth_xyz": (22, 3),
    "body_valid": (20,),
}


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _rng(seed, role, source):
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}/{role}/{source}".encode()).digest(), "big"))


def training_anchors(context_length, source, *, seed=2062):
    """Local queries: up to three early anchors and five evenly spaced mature."""
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 22:
        raise ValueError("A training context needs at least22 frames.")
    maximum = context_length - 21
    candidates = [value for value in EARLY_L if value <= maximum]
    early = _rng(seed, "early", source).sample(candidates, min(3, len(candidates)))
    mature_count = min(5, max(0, maximum - 19))
    if mature_count == 1:
        mature = [20]
    elif mature_count:
        mature = [20 + round(index * (maximum - 20) / (mature_count - 1)) for index in range(mature_count)]
    else:
        mature = []
    return [20 + value for value in sorted(early + mature)]


def select_contexts(data, cohort, *, selection=None, limit_contexts=None, seed=2062):
    """Plan contexts from metadata only, before materializing any episode."""
    if cohort not in ("train", "D128", "D400", "long50"):
        raise ValueError("Choose train, D128, D400 or long50.")
    split = "train" if cohort == "train" else "val"
    if data.official_split != split:
        raise ValueError("Context selection and actual official data split disagree.")
    if limit_contexts is not None and (isinstance(limit_contexts, bool) or limit_contexts < 1):
        raise ValueError("limit_contexts must be positive.")
    contexts, excluded = [], 0
    if cohort == "train":
        if selection is not None:
            raise ValueError("Training contexts are selected per source, not from an old clip bank.")
        for source, metadata in sorted(data.source_metadata.items()):
            length = metadata["frames"]
            if length < 22:
                excluded += 1
                continue
            count = min(160, length)
            start = _rng(seed, "context", source).randrange(length - count + 1)
            contexts.append(
                {
                    "source": source,
                    "take": metadata["take"],
                    "context_start": start,
                    "context_length": count,
                    "anchors": training_anchors(count, source, seed=seed),
                }
            )
        parent_sha = None
    else:
        if not selection or not selection.get("sha256") or not selection.get("records"):
            raise ValueError("Validation banks require an explicit frozen physical selection.")
        parent_sha = selection["sha256"]
        for record in selection["records"]:
            source, start, count = record["source"], record["start"], record["frames"]
            if source not in data.source_metadata:
                raise ValueError("Frozen validation source is absent from the actual split.")
            metadata = data.source_metadata[source]
            if cohort in ("D128", "D400") and count != 80:
                raise ValueError("D128/D400 require the frozen complete80-frame clips.")
            if metadata["take"] != record["take"] or start < 0 or count < 22 or start + count > metadata["frames"]:
                raise ValueError("Frozen validation context no longer matches its physical source.")
            episode = f"{source}_start_{start}"
            if record["episode_id"] != episode:
                raise ValueError("Frozen validation episode identifier disagrees with its physical context.")
            if "record_index" in record:
                actual = data.records[record["record_index"]]
                if (
                    actual["episode_id"] != episode
                    or actual["base_seq_name"] != source
                    or actual["episode_start_motion_idx"] != start
                    or actual["num_frames"] != count
                ):
                    raise ValueError("Frozen validation record ordering or context changed.")
            contexts.append(
                {
                    "source": source,
                    "take": record["take"],
                    "context_start": start,
                    "context_length": count,
                    "anchors": list(range(20, count)),
                }
            )
    original_count = len(contexts)
    if limit_contexts is not None:
        contexts = contexts[:limit_contexts]
    if not contexts:
        raise ValueError("No eligible contexts for this bank.")
    seen = set()
    for index, context in enumerate(contexts):
        context.update(index=index, episode=f"{context['source']}_start_{context['context_start']}")
        key = (context["source"], context["context_start"], context["context_length"])
        if key in seen:
            raise ValueError("A bank cannot contain duplicated physical contexts.")
        seen.add(key)
    policy = {
        "version": "source160_early3_mature5_v1" if split == "train" else "frozen_dense_queries_v1",
        "seed": seed,
        "startup_frames": 20,
        "early_L_candidates": list(EARLY_L),
        "max_early": 3,
        "max_mature": 5,
        "local_L": "frame-20 actual previous G calls in this context, before current G call",
        "noise_seed": 1062,
        "noise_key": "native episode identifier source_start_context_start; local query; generate10 slots then sliceW3",
    }
    content = {"split": split, "cohort": cohort, "manifest_sha256": data.digest, "contexts": contexts, "policy": policy}
    full_selection = len(contexts) == original_count
    selection_sha = parent_sha if parent_sha and full_selection else json_sha256(content)
    return {
        **content,
        "sha256": selection_sha,
        "content_sha256": json_sha256(content),
        "parent_selection_sha256": parent_sha,
        "limited_preflight": not full_selection,
        "original_context_count": original_count,
        "excluded_short_sources": excluded,
        "expected_rows": sum(len(value["anchors"]) for value in contexts),
        "G_call_upper_bound": sum(value["context_length"] - 21 for value in contexts),
    }


def validate_producer(saved, expected, *, checkpoint_sha256, actual_code):
    """Reject a substituted producer even when it has a compatible network."""
    if checkpoint_sha256 != expected["sha256"] or saved.get("step") != expected["step"]:
        raise ValueError("Frozen producer checkpoint SHA or step changed.")
    required = {
        "history": 20,
        "window": 3,
        "source": "gaussian",
        "sigma": 1.0,
        "steps": 1,
        "revise_recent": True,
        "body_feedback": True,
        "shape_mode": "predicted",
        "geometry_target": "official",
    }
    if any(saved["protocol"].get(key) != value for key, value in required.items()):
        raise ValueError("Producer must be the frozen W3 predicted-beta Gaussian/CV NFE1 rewrite control.")
    identity = saved["identity"]
    if (
        identity.get("synthetic")
        or identity.get("data_format") != "official"
        or identity.get("prior") != "constant_velocity"
    ):
        raise ValueError("Producer must use official data and the constant-velocity prior.")
    if identity["code"] != actual_code:
        raise ValueError("Imported producer source differs from its native frozen checkpoint code.")
    if expected.get("native_checkpoint_manifest", identity["manifest"]) != identity["manifest"]:
        raise ValueError("Producer native training manifest changed.")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "step": saved["step"],
        "window": 3,
        "shape_mode": "predicted",
        "source": "gaussian",
        "NFE": 1,
        "training_manifest": identity["manifest"],
        "revise_recent": True,
        "prior": "constant_velocity",
    }


@dataclass
class ProducerRuntime:
    data: object
    system: object
    model: object
    fk: object
    stats: dict
    producer: dict
    provenance: dict


def load_runtime(plan, cohort, *, device="cuda", compile_eval=False):
    """Production-only loading; the CLI first restarts inside frozen G source."""
    expected = plan["producer"]
    package = Path(expected["source_snapshot"]).resolve() / "egorecover"
    import egorecover

    if Path(egorecover.__file__).resolve().parent != package:
        raise ValueError("Invoke scripts/build_formal_prior_bank.py so G uses its frozen source environment.")
    checkpoint = Path(expected["checkpoint"])
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    producer = validate_producer(
        saved, expected, checkpoint_sha256=file_sha256(checkpoint), actual_code=code_identity()
    )
    device = torch.device(device)
    smpl = load_smpl(device)
    if smpl_identity() != saved["identity"]["smpl_asset"]:
        raise ValueError("Licensed SMPL-X asset changed from the frozen producer.")
    split = "train" if cohort == "train" else "val"
    data = OfficialWindowData(
        plan["data"]["root"], split, device=device, bootstrap=HeadAlignedBootstrap.from_smpl(smpl)
    )
    if split == "train" and data.digest != producer["training_manifest"]:
        raise ValueError("Actual train data no longer matches the producer native training manifest.")
    if any(not torch.equal(data.stats[key].cpu(), saved["stats"][key].cpu()) for key in STATS_FIELDS):
        raise ValueError("Actual split statistics differ from frozen producer statistics.")
    config = WindowConfig(**saved["protocol"])
    codec = MotionCodec(saved["stats"]).to(device)
    model = MotionModel(config, ModelConfig(**saved["model_config"])).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["ema"], strict=True)
    if compile_eval:
        if device.type != "cuda":
            raise ValueError("The optional compiled producer path requires CUDA.")
        model.enable_eval_compile()
    stats = {key: saved["stats"][key].detach().cpu().clone() for key in STATS_FIELDS}
    return ProducerRuntime(
        data,
        WindowReconstructor(codec, None, config),
        model,
        FixedShapeFK(smpl, torch.zeros(10, device=device), shape_mode="predicted"),
        stats,
        producer,
        {
            "producer_source_snapshot": str(package.parent),
            "producer_code": saved["identity"]["code"],
            "producer_smpl_asset": saved["identity"]["smpl_asset"],
            "producer_data_protocol": saved["identity"]["data_protocol"],
            "data_identity": data.identity,
            "data_protocol": data.protocol,
            "stats_sha256": state_sha256(stats),
            "builder_module_sha256": file_sha256(__file__),
            "compile_eval": bool(compile_eval),
            "device_type": device.type,
        },
    )


def _slice_cache(cache, indices):
    return WindowCache(
        state_map(cache.body, lambda value: value[indices]),
        cache.beta[indices],
        cache.anchor[indices],
        cache.frame,
        cache.generated_frames,
    )


@torch.no_grad()
def generate_context_batch(runtime, contexts):
    """Continuously advance contexts together; extract P rows BEFORE G(t)."""
    started = time.perf_counter()
    active = []
    for descriptor in contexts:
        episode = runtime.data._episode(descriptor["source"], descriptor["context_start"], descriptor["context_length"])
        if episode.identifier != descriptor["episode"] or episode.take != descriptor["take"]:
            raise ValueError("Materialized episode disagrees with selected physical context.")
        if len(episode) != descriptor["context_length"] or not episode.bootstrap_only:
            raise ValueError("Bank contexts require the complete selected head-only startup episode.")
        active.append((descriptor, episode))
    if not active:
        raise ValueError("Cannot generate an empty context batch.")
    cache = TrainingBatch(tuple(episode for _, episode in active)).cache(20)
    rows, examples, g_calls, batched_calls = [], [], 0, 0
    runtime.model.eval()
    maximum = max(max(descriptor["anchors"]) for descriptor, _ in active)
    for query in range(20, maximum + 1):
        selected = [index for index, (descriptor, _) in enumerate(active) if query in descriptor["anchors"]]
        if selected:
            local_L = query - 20
            if cache.generated_frames != local_L:
                raise ValueError("Context history length no longer matches actual previous G calls.")
            history = state_map(cache.body, lambda value: value[selected])
            truth = state_map(
                TrainingBatch(tuple(active[index][1] for index in selected)).target(query), lambda value: value[:, -1]
            )
            count = reconstructed_history_count(local_L, 3, True)
            example = prior_example(runtime.system.codec, history, truth, generated_frames=count)
            cv = runtime.system.codec.decode_current(example["base"][:, 0], example["reference"])
            errors = (runtime.fk(cv) - example["truth_xyz"]).norm(dim=-1).mean(-1).cpu().tolist()
            examples.append({key: value.detach().cpu().contiguous().clone() for key, value in example.items()})
            for index, error in zip(selected, errors):
                descriptor, episode = active[index]
                rows.append(
                    {
                        "episode": episode.identifier,
                        "take": episode.take,
                        "frame": query,
                        "kind": "predicted",
                        "window": 3,
                        "timestamp_s": float(episode.timestamps[query]),
                        "floor_calibrated": bool(episode.floor_calibrated),
                        "cv_body_mm": float(error) * 1000,
                        "source": descriptor["source"],
                        "context_index": descriptor["index"],
                        "context_start": descriptor["context_start"],
                        "context_length": descriptor["context_length"],
                        "source_query": descriptor["context_start"] + query,
                        "L": local_L,
                    }
                )
        # A context whose final P label was just extracted needs no current G
        # call. Other contexts continue without resetting at sparse anchors.
        keep = [index for index, (descriptor, _) in enumerate(active) if max(descriptor["anchors"]) > query]
        if not keep:
            break
        cache = _slice_cache(cache, keep)
        active = [active[index] for index in keep]
        batch = TrainingBatch(tuple(episode for _, episode in active))
        noise = torch.cat([keyed_noise(episode.identifier, 1062, query, 3, cache.beta.device) for _, episode in active])
        cache = runtime.system.online(runtime.model, cache, batch.observations(query), noise).cache.detach()
        g_calls += len(active)
        batched_calls += 1
    tensors = concatenate_examples(examples)
    return (
        rows,
        tensors,
        {
            "contexts": len(contexts),
            "rows": len(rows),
            "G_calls": g_calls,
            "batched_G_calls": batched_calls,
            "generation_seconds": time.perf_counter() - started,
        },
    )


def bank_metadata(runtime, selection, *, shards, batch_contexts):
    if selection["split"] == "train" and selection["manifest_sha256"] != runtime.producer["training_manifest"]:
        raise ValueError("Train split manifest must match the actual producer training manifest.")
    metadata = {
        "format": FORMAT,
        "split": selection["split"],
        "cohort": selection["cohort"],
        "manifest_sha256": selection["manifest_sha256"],
        "producer": runtime.producer,
        "producer_window": 3,
        "producer_checkpoint_sha256": runtime.producer["checkpoint_sha256"],
        "selection_sha256": selection["sha256"],
        "selection_content_sha256": selection["content_sha256"],
        "parent_selection_sha256": selection["parent_selection_sha256"],
        "limited_preflight": selection["limited_preflight"],
        "source_flags": "last min(20,L+2 if L>0 else0) history slots reconstructed; L is context-local previous G calls",
        "source_flag_semantics": {"history": 20, "window": 3, "revise_recent": True, "L": "frame-20"},
        "CV_geometry": "own predicted-beta body22 FK against official current joints; no GT beta substitution",
        "policy": selection["policy"],
        "shards": shards,
        "batch_contexts": batch_contexts,
        "expected_contexts": len(selection["contexts"]),
        "expected_rows": selection["expected_rows"],
        "provenance": runtime.provenance,
    }
    return {**metadata, "bank_identity_sha256": json_sha256(metadata)}


def _tensor_digest(tensors):
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _content_digest(packet):
    return json_sha256(
        {
            "rows": json_sha256(packet["rows"]),
            "tensors": _tensor_digest(packet["tensors"]),
            "stats": _tensor_digest(packet["stats"]),
        }
    )


def validate_chunk(packet, metadata, contexts):
    if any(packet.get(key) != value for key, value in metadata.items()):
        raise ValueError("Bank chunk identity changed.")
    rows, tensors = packet["rows"], packet["tensors"]
    if set(tensors) != set(TENSOR_FIELDS):
        raise ValueError("Bank chunk tensor fields changed.")
    if packet.get("content_sha256") != _content_digest(packet):
        raise ValueError("Bank chunk content checksum changed.")
    expected_stats = metadata["provenance"].get("stats_sha256")
    if expected_stats and state_sha256(packet["stats"]) != expected_stats:
        raise ValueError("Bank chunk statistics differ from the producer codec identity.")
    for key, suffix in TENSOR_SHAPES.items():
        value = tensors[key]
        dtype = torch.bool if key == "body_valid" else torch.float32
        if value.shape != (len(rows), *suffix) or value.dtype != dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid bank chunk tensor: {key}.")
    seen = set()
    for index, row in enumerate(rows):
        descriptor = contexts[row["context_index"]]
        key = (row["context_index"], row["frame"])
        if (
            key in seen
            or row["kind"] != "predicted"
            or row["window"] != 3
            or row["source"] != descriptor["source"]
            or row["take"] != descriptor["take"]
            or row["episode"] != descriptor["episode"]
            or row["context_start"] != descriptor["context_start"]
            or row["context_length"] != descriptor["context_length"]
            or row["frame"] not in descriptor["anchors"]
            or row["source_query"] != row["context_start"] + row["frame"]
            or row["L"] != row["frame"] - 20
            or not math.isfinite(row["cv_body_mm"])
            or not math.isfinite(row["timestamp_s"])
        ):
            raise ValueError("Bank chunk row is duplicated or disagrees with its local context.")
        count = reconstructed_history_count(row["L"], 3, True)
        expected = torch.arange(20) >= 20 - count
        if not torch.equal(tensors["body_valid"][index], expected):
            raise ValueError("Bank flags must use context-local G-call count, including W3 rewrites.")
        seen.add(key)
    expected = {(index, query) for index in packet["context_indices"] for query in contexts[index]["anchors"]}
    if seen != expected:
        raise ValueError("Bank chunk omitted or added selected context anchors.")


def _space(path, minimum_free_bytes):
    info = os.statvfs(path)
    if info.f_bavail * info.f_frsize < minimum_free_bytes:
        raise RuntimeError("Insufficient project-disk headroom; existing chunks are preserved.")


def _shared_files(output, metadata, selection):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "identity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = output / "identity.json"
        if identity.exists():
            if (
                json.loads(identity.read_text()) != metadata
                or json.loads((output / "selection.json").read_text()) != selection
            ):
                raise ValueError("Output already belongs to another bank selection or producer.")
        else:
            save_json(output / "selection.json", selection)
            save_json(identity, metadata)


def _coverage(rows):
    return {
        "rows": len(rows),
        "contexts": len({row["context_index"] for row in rows}),
        "sources": len({row["source"] for row in rows}),
        "takes": len({row["take"] for row in rows}),
        "early_rows": sum(row["L"] < 20 for row in rows),
        "mature_rows": sum(row["L"] >= 20 for row in rows),
        "local_L_counts": dict(sorted(Counter(str(row["L"]) for row in rows).items(), key=lambda item: int(item[0]))),
    }


def build_shard(
    runtime, selection, output, *, shard=0, shards=1, batch_contexts=8, resume=False, minimum_free_bytes=10 * 1024**3
):
    """Save one bounded context batch per atomic chunk; resume completed batches."""
    if shards < 1 or not 0 <= shard < shards or batch_contexts < 1:
        raise ValueError("Invalid shard or context batch size.")
    output = Path(output)
    metadata = bank_metadata(runtime, selection, shards=shards, batch_contexts=batch_contexts)
    _shared_files(output, metadata, selection)
    directory = output / f"shard{shard:04d}"
    directory.mkdir(exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chunks = directory / "chunks"
        chunks.mkdir(exist_ok=True)
        existing = list(chunks.glob("chunk*.pt"))
        if existing and not resume:
            raise FileExistsError("Shard contains saved chunks; use --resume or a new output directory.")
        assigned = selection["contexts"][shard::shards]
        batches = [assigned[start : start + batch_contexts] for start in range(0, len(assigned), batch_contexts)]
        records, all_rows = [], []
        total_calls, batched_calls, generation_seconds = 0, 0, 0.0
        try:
            for chunk_index, descriptors in enumerate(batches):
                _space(directory, minimum_free_bytes)
                path = chunks / f"chunk{chunk_index:06d}.pt"
                indices = [context["index"] for context in descriptors]
                if path.exists():
                    packet = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                    if packet["context_indices"] != indices or packet["chunk_index"] != chunk_index:
                        raise ValueError("Saved chunk no longer matches this deterministic shard batch.")
                    validate_chunk(packet, metadata, selection["contexts"])
                else:
                    save_json(
                        directory / "progress.json",
                        {
                            "phase": "building",
                            "completed_contexts": sum(len(record["context_indices"]) for record in records),
                            "target_contexts": len(assigned),
                            "in_flight_context_indices": indices,
                            "rows": len(all_rows),
                        },
                    )
                    rows, tensors, generation = generate_context_batch(runtime, descriptors)
                    packet = {
                        **metadata,
                        "stats": runtime.stats,
                        "rows": rows,
                        "tensors": tensors,
                        "context_indices": indices,
                        "chunk_index": chunk_index,
                        "generation": generation,
                    }
                    packet["content_sha256"] = _content_digest(packet)
                    validate_chunk(packet, metadata, selection["contexts"])
                    save_checkpoint(path, packet)
                    del rows, tensors
                generation = packet["generation"]
                all_rows.extend(packet["rows"])
                total_calls += generation["G_calls"]
                batched_calls += generation["batched_G_calls"]
                generation_seconds += generation["generation_seconds"]
                records.append(
                    {
                        "file": str(path.relative_to(directory)),
                        "sha256": file_sha256(path),
                        "context_indices": indices,
                        "rows": len(packet["rows"]),
                    }
                )
                del packet
                save_json(
                    directory / "progress.json",
                    {
                        "phase": "building",
                        "completed_contexts": sum(len(record["context_indices"]) for record in records),
                        "target_contexts": len(assigned),
                        "rows": len(all_rows),
                        "G_calls": total_calls,
                        "batched_G_calls": batched_calls,
                        "committed_generation_seconds": generation_seconds,
                        "G_frames_per_second": total_calls / max(generation_seconds, 1e-12),
                        "rows_per_second": len(all_rows) / max(generation_seconds, 1e-12),
                    },
                )
            if len(existing) > len(batches):
                raise ValueError("Shard contains chunks outside the selected context partition.")
            complete = {
                "phase": "complete",
                "bank_identity_sha256": metadata["bank_identity_sha256"],
                "shard": shard,
                "shards": shards,
                "completed_contexts": len(assigned),
                "target_contexts": len(assigned),
                "chunks": records,
                "coverage": _coverage(all_rows),
                "rows": len(all_rows),
                "G_calls": total_calls,
                "batched_G_calls": batched_calls,
                "committed_generation_seconds": generation_seconds,
                "G_frames_per_second": total_calls / max(generation_seconds, 1e-12),
                "rows_per_second": len(all_rows) / max(generation_seconds, 1e-12),
                "timing_scope": "materialize contexts, continuous producer rollout and example extraction; excludes model/data startup and chunk serialization; committed batches only",
            }
            save_json(directory / "complete.json", complete)
            save_json(directory / "progress.json", complete)
            return complete
        except Exception as error:
            save_json(
                directory / "progress.json",
                {
                    "phase": "failed",
                    "error": str(error),
                    "completed_chunks": len(records),
                    "target_contexts": len(assigned),
                    "saved_chunks_preserved": True,
                },
            )
            raise


def merge_bank(output, *, minimum_free_bytes=10 * 1024**3):
    """Fill disk-mapped final tensors one chunk at a time, without a full cat."""
    output = Path(output)
    with (output / "merge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = json.loads((output / "identity.json").read_text())
        selection = json.loads((output / "selection.json").read_text())
        target = output / "bank.pt"
        if (output / "complete.json").exists():
            complete = json.loads((output / "complete.json").read_text())
            if (
                complete["bank_identity_sha256"] != metadata["bank_identity_sha256"]
                or file_sha256(target) != complete["bank_sha256"]
            ):
                raise ValueError("Completed merged bank identity or checksum changed.")
            return complete
        chunk_files, covered, total_rows, total_g_calls, generation_seconds = [], [], 0, 0, 0.0
        for shard in range(metadata["shards"]):
            directory = output / f"shard{shard:04d}"
            complete = json.loads((directory / "complete.json").read_text())
            if complete["bank_identity_sha256"] != metadata["bank_identity_sha256"] or complete["phase"] != "complete":
                raise ValueError("All shards must finish with the same frozen bank identity.")
            expected = [context["index"] for context in selection["contexts"][shard :: metadata["shards"]]]
            actual = [index for chunk in complete["chunks"] for index in chunk["context_indices"]]
            if actual != expected:
                raise ValueError("Shard completion does not cover its selected context partition.")
            covered.extend(actual)
            for chunk in complete["chunks"]:
                path = directory / chunk["file"]
                if file_sha256(path) != chunk["sha256"]:
                    raise ValueError("Saved bank chunk checksum changed.")
                chunk_files.append(path)
                total_rows += chunk["rows"]
            total_g_calls += complete["G_calls"]
            generation_seconds += complete["committed_generation_seconds"]
        if sorted(covered) != list(range(len(selection["contexts"]))) or total_rows != metadata["expected_rows"]:
            raise ValueError("Merged contexts or anchor count do not match the frozen selection.")
        _space(output, minimum_free_bytes)
        rows, offset, stats = [], 0, None
        with tempfile.TemporaryDirectory(prefix=".merge-tensors-", dir=output) as temporary:
            tensors = {
                key: torch.from_file(
                    str(Path(temporary) / f"{key}.raw"),
                    shared=True,
                    size=total_rows * math.prod(shape),
                    dtype=torch.bool if key == "body_valid" else torch.float32,
                ).reshape(total_rows, *shape)
                for key, shape in TENSOR_SHAPES.items()
            }
            for index, path in enumerate(chunk_files):
                _space(output, minimum_free_bytes)
                packet = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                validate_chunk(packet, metadata, selection["contexts"])
                if stats is None:
                    stats = {key: value.clone() for key, value in packet["stats"].items()}
                elif any(not torch.equal(value, packet["stats"][key]) for key, value in stats.items()):
                    raise ValueError("Chunk normalization statistics changed.")
                count = len(packet["rows"])
                for key in TENSOR_FIELDS:
                    tensors[key][offset : offset + count].copy_(packet["tensors"][key])
                rows.extend(packet["rows"])
                offset += count
                del packet
                save_json(
                    output / "progress.json",
                    {
                        "phase": "merging",
                        "completed_chunks": index + 1,
                        "target_chunks": len(chunk_files),
                        "rows": offset,
                        "target_rows": total_rows,
                    },
                )
            if offset != total_rows or stats is None:
                raise ValueError("Merged bank has missing tensor rows.")
            final = {**metadata, "stats": stats, "rows": rows, "tensors": tensors, "coverage": _coverage(rows)}
            save_checkpoint(target, final)
            del final
            tensors.clear()
            gc.collect()
        complete = {
            "phase": "complete",
            "bank_identity_sha256": metadata["bank_identity_sha256"],
            "bank": str(target),
            "bank_sha256": file_sha256(target),
            "coverage": _coverage(rows),
            "contexts": len(covered),
            "rows": total_rows,
            "G_calls": total_g_calls,
            "committed_generation_seconds": generation_seconds,
            "G_frames_per_second": total_g_calls / max(generation_seconds, 1e-12),
        }
        save_json(output / "complete.json", complete)
        save_json(output / "progress.json", complete)
        return complete


def build_from_args(args):
    plan = json.loads(Path(args.plan).read_text())
    started = time.perf_counter()
    runtime = load_runtime(plan, args.cohort, device=args.device, compile_eval=args.compile_eval)
    selection_file = None if args.cohort == "train" else Path(plan["evaluation"][args.cohort]["file"])
    frozen = json.loads(selection_file.read_text()) if selection_file else None
    if frozen and (
        frozen["sha256"] != plan["evaluation"][args.cohort]["selection_id"]
        or len(frozen["records"]) != plan["evaluation"][args.cohort]["clips"]
    ):
        raise ValueError("Frozen evaluation selection ID or context count changed from the P01 plan.")
    selection = select_contexts(runtime.data, args.cohort, selection=frozen, limit_contexts=args.limit_contexts)
    runtime.provenance.update(
        plan_sha256=file_sha256(args.plan),
        selection_file_sha256=file_sha256(selection_file) if selection_file else None,
        torch_version=str(torch.__version__),
        builder_driver_sha256=file_sha256(args.driver_file),
    )
    startup_seconds = time.perf_counter() - started
    report = build_shard(
        runtime,
        selection,
        args.output,
        shard=args.shard,
        shards=args.shards,
        batch_contexts=args.batch_contexts,
        resume=args.resume,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
    )
    report["session_seconds_including_startup"] = time.perf_counter() - started
    report["startup_seconds"] = startup_seconds
    directory = Path(args.output) / f"shard{args.shard:04d}"
    save_json(directory / "complete.json", report)
    save_json(directory / "progress.json", report)
    return report
