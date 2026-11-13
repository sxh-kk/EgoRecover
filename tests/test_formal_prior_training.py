import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from egorecover.codec import MotionCodec
from egorecover.diagnostics import synthetic_episode
from egorecover.distributed import DistributedContext
from egorecover.formal_prior_training import FormalPriorSampler, _train_formal_prior, validate_formal_bank
from egorecover.prior_data import concatenate_examples, prior_example, reconstructed_history_count
from egorecover.prior_training import PriorTrainOptions
from egorecover.state import state_map


def protocol():
    value = json.loads(Path("docs/reports/data/formal_p_training_plan_20261113.json").read_text())
    value["model"].update(width=24, layers=1, heads=4)
    value["loss"]["geometry_calibration"].update(samples=16, mature=12, early=4)
    return value


def make_bank(plan, split):
    episode = synthetic_episode(length=46)
    codec = MotionCodec(episode.stats)
    take = f"{split}-take"
    rows, examples = [], []
    for frame in range(20, 46):
        history = state_map(episode.teacher, lambda x: x[frame - 20 : frame][None])
        truth = state_map(episode.target(frame), lambda x: x[:, -1])
        examples.append(
            prior_example(codec, history, truth, generated_frames=reconstructed_history_count(frame - 20, 3, True))
        )
        rows.append(
            {
                "episode": take + "-context",
                "take": take,
                "source": take + "-source",
                "frame": frame,
                "L": frame - 20,
                "kind": "predicted",
                "window": 3,
                "timestamp_s": frame / 10,
                "floor_calibrated": True,
            }
        )
    p = plan["producer"]
    return {
        "format": "egorecover_prior_bank_v2",
        "split": split,
        "cohort": "train" if split == "train" else "D128",
        "limited_preflight": True,
        "provenance": {"producer_smpl_asset": "asset-free-fixture"},
        "stats": episode.stats,
        "producer_window": 3,
        "manifest_sha256": f"{split}-manifest",
        "selection_sha256": f"{split}-selection",
        "producer": {
            "checkpoint_sha256": p["sha256"],
            "step": p["step"],
            "window": p["window"],
            "shape_mode": p["shape_mode"],
            "source": p["source"],
            "NFE": p["NFE"],
            "training_manifest": p["native_checkpoint_manifest"],
        },
        "rows": rows,
        "tensors": concatenate_examples(examples),
    }


def test_formal_banks_bind_producer_and_local_source_flags():
    p = protocol()
    bank = make_bank(p, "train")
    validate_formal_bank(bank, "train", p)
    other = copy.deepcopy(bank)
    other["producer"]["checkpoint_sha256"] = "old-G"
    with pytest.raises(ValueError, match="producer mismatch"):
        validate_formal_bank(other, "train", p)
    other = copy.deepcopy(bank)
    other["tensors"]["body_valid"][0].fill_(True)
    with pytest.raises(ValueError, match="source flags"):
        validate_formal_bank(other, "train", p)
    with pytest.raises(ValueError, match="split mismatch"):
        validate_formal_bank(bank, "val", p)
    with pytest.raises(ValueError, match="limited preflight"):
        validate_formal_bank(bank, "train", p, cohort="train")
    val = make_bank(p, "val")
    with pytest.raises(ValueError, match="cohort mismatch"):
        validate_formal_bank(val, "val", p, cohort="D400", allow_limited=True)
    val["limited_preflight"] = False
    with pytest.raises(ValueError, match="selection mismatch"):
        validate_formal_bank(val, "val", p, cohort="D128")
    val["selection_sha256"] = p["evaluation"]["D128"]["selection_id"]
    with pytest.raises(ValueError, match="clip coverage"):
        validate_formal_bank(val, "val", p, cohort="D128")


def test_formal_sampler_has_exact_quota_take_balance_and_restorable_visitation():
    rows = []
    for take, count in [("one", 1), ("many", 11)]:
        for source in range(count):
            for frame in [20, 21, 40, 55]:
                rows.append(
                    {
                        "take": take,
                        "source": f"{take}-{source}",
                        "episode": f"{take}-{source}",
                        "frame": frame,
                        "kind": "predicted",
                        "window": 3,
                    }
                )
    sampler = FormalPriorSampler(rows)
    g = torch.Generator().manual_seed(62)
    seen = []
    for _ in range(800):
        indices = sampler.sample(g)
        assert sum(rows[i]["frame"] >= 40 for i in indices) == 6
        seen.extend(rows[i]["take"] == "one" for i in indices)
    assert abs(sum(seen) / len(seen) - 0.5) < 0.025
    rng, state = g.get_state(), sampler.state_dict()
    following = sampler.sample(g)
    other = FormalPriorSampler(rows)
    other.load_state_dict(state)
    g.set_state(rng)
    assert following == other.sample(g)
    assert other.state_dict() == sampler.state_dict()


class BetaFK:
    shape_mode = "predicted"

    def __call__(self, body):
        return body.joints[..., :3, 3] + body.auxiliary[..., -10:-9, None] * 0.01


@pytest.mark.parametrize(
    ("objective", "hold_beta"),
    [("raw_code", False), ("common_cv_valid_state", False), ("common_cv_valid_state", True)],
)
def test_formal_training_cpu_resume_preserves_lr_samples_optimizer_and_ema(tmp_path, monkeypatch, objective, hold_beta):
    import egorecover.formal_prior_training as training

    p = protocol()
    if objective != "raw_code":
        p["loss"].update(objective=objective, fixed_geometry_weight=0.008869197811349056)
        p["model"]["hold_beta"] = hold_beta
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(p))
    paths = {}
    for split in ("train", "val"):
        paths[split] = tmp_path / f"{split}.pt"
        torch.save(make_bank(p, split), paths[split])
    monkeypatch.setattr(training, "load_smpl", lambda device: None)
    monkeypatch.setattr(training, "FixedShapeFK", lambda *a, **kw: BetaFK())
    monkeypatch.setattr(training, "smpl_identity", lambda: "asset-free-fixture")
    monkeypatch.setattr(training, "code_identity", lambda: {"test": "frozen"})
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    options = PriorTrainOptions(
        output=tmp_path / "preflight-full",
        data_root=tmp_path,
        train_bank=paths["train"],
        val_bank=paths["val"],
        device="cpu",
        train_steps=2,
        eval_every=1,
        formal_protocol=protocol_file,
        formal_preflight=True,
        lr_decay_step=1,
    )
    _train_formal_prior(options, context)
    interrupted = replace(options, output=tmp_path / "preflight-resumed", stop_after=1)
    _train_formal_prior(interrupted, context)
    _train_formal_prior(replace(interrupted, resume=True, stop_after=None), context)

    def load(output):
        return torch.load(output / "resume.pt", weights_only=True, map_location="cpu")

    full, resumed = load(options.output), load(interrupted.output)
    assert full["identity"] == resumed["identity"]
    assert full["identity"]["objective"] == objective
    assert full["identity"]["hold_beta"] == hold_beta
    if objective != "raw_code":
        audit = full["identity"]["geometry_audit"]
        assert audit["weight"] == 0.008869197811349056
        assert audit["weight_mode"] == "P01_fixed_not_recalibrated"
        assert audit["target_weighted_geometry_ratio"] is None
    assert full["scheduler"] == resumed["scheduler"]
    assert full["step"] == resumed["step"] == 2
    assert full["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"]
    for key in ("model", "ema"):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
    for key, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
    for a, b in zip(full["rank_rng_states"], resumed["rank_rng_states"]):
        assert a["visited_indices"] == b["visited_indices"]
        assert a["sampling_counts"] == b["sampling_counts"]
        assert torch.equal(a["sample_rng"], b["sample_rng"]) and torch.equal(a["cpu_rng"], b["cpu_rng"])
    logs = [
        [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
        for out in (options.output, interrupted.output)
    ]
    assert [row["trace_hash"] for row in logs[0]] == [row["trace_hash"] for row in logs[1]]
    assert [row["lr"] for row in logs[0]] == pytest.approx([3e-5, 3e-6])
