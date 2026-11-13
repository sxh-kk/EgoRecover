"""P01 matched-cache evaluation, clip-weighted uncertainty, and fixed gates."""

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch

from .codec import BodyState
from .metrics import _similarity_align, foot_slide_uniegomotion
from .prior_data import TENSOR_FIELDS, predictor_forward, select_examples

PHASES = ("all", "mature", "startup")
METRICS = ("mpjpe_mm", "velocity_mm_s", "acceleration_mm_s2", "foot_slide_mm")
TEMPORAL_METRICS = METRICS[1:]
PARTS = {
    "upper_body_mm": (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    "lower_body_mm": (0, 1, 2, 4, 5, 7, 8, 10, 11),
    "wrist_mm": (20, 21),
    "feet_mm": (10, 11),
    "root_mm": (0,),
}


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _validate_bank(bank):
    if bank.get("format") != "egorecover_prior_bank_v2" or bank.get("producer_window") != 3:
        raise ValueError("Formal P evaluation requires a v2 bank from the single W3 producer.")
    rows, tensors = bank["rows"], bank["tensors"]
    if any(key not in tensors for key in TENSOR_FIELDS):
        raise ValueError("Formal P banks require all standard history/label tensor fields.")
    if any(
        not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != len(rows) for value in tensors.values()
    ):
        raise ValueError("Bank tensors and row counts disagree.")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("episode", "take")):
            raise ValueError("Bank rows require nonempty episode and take names.")
        _integer(row.get("frame"), "local frame", 20)
        time = row.get("timestamp_s")
        if isinstance(time, bool) or not isinstance(time, Real) or not math.isfinite(time):
            raise ValueError("Dense bank rows require finite timestamp_s values.")
        if not isinstance(row.get("floor_calibrated"), bool):
            raise ValueError("Dense bank rows require boolean floor_calibrated metadata.")
        if row.get("kind", "predicted") != "predicted" or row.get("window", 3) != 3:
            raise ValueError("Formal evaluation accepts only actual W3 predicted histories.")
        grouped[row["episode"]].append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: rows[index]["frame"])
        metadata = [rows[index] for index in indices]
        frames = [row["frame"] for row in metadata]
        if frames[0] != 20 or any(second - first != 1 for first, second in zip(frames, frames[1:])):
            raise ValueError("Dense evaluation requires consecutive, nonduplicated queries starting at local frame20.")
        if len({row["take"] for row in metadata}) != 1 or len({row["floor_calibrated"] for row in metadata}) != 1:
            raise ValueError("One episode must have one take and one floor-calibration status.")
        times = torch.tensor([row["timestamp_s"] for row in metadata], dtype=torch.float64)
        intervals = torch.diff(times)
        if len(intervals) and (
            not bool((intervals > 0).all())
            or not torch.allclose(intervals, intervals.mean().expand_as(intervals), atol=1e-7, rtol=2e-5)
        ):
            raise ValueError("Dense temporal evaluation requires increasing uniform timestamps.")
    return grouped


def formal_center_metrics(prediction, truth, times, floor_calibrated, *, include_pa=False, include_parts=False):
    """Score one clip phase; derivative stencils stay inside that phase."""
    prediction, truth, times = [value.detach().cpu().double() for value in (prediction, truth, times)]
    if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[1:] != (22, 3):
        raise ValueError("Centers and official truth must be matching [T,22,3] positions.")
    if not len(prediction) or times.shape != (len(prediction),):
        raise ValueError("A nonempty center trajectory needs one timestamp per frame.")
    if not all(bool(torch.isfinite(value).all()) for value in (prediction, truth, times)):
        raise ValueError("Nonfinite center evaluation input.")
    intervals = torch.diff(times)
    if len(intervals) and (
        not bool((intervals > 0).all())
        or not torch.allclose(intervals, intervals.mean().expand_as(intervals), atol=1e-7, rtol=2e-5)
    ):
        raise ValueError("Center trajectories require increasing uniform timestamps.")
    error = prediction - truth
    distances = error.norm(dim=-1)
    result = dict.fromkeys(METRICS)
    result["mpjpe_mm"] = float(distances.mean() * 1000)
    if len(intervals):
        dt = float(intervals.mean())
        result["velocity_mm_s"] = float((torch.diff(error, dim=0) / dt).norm(dim=-1).mean() * 1000)
        if len(prediction) >= 3:
            result["acceleration_mm_s2"] = float((torch.diff(error, n=2, dim=0) / dt**2).norm(dim=-1).mean() * 1000)
        if floor_calibrated:
            result["foot_slide_mm"] = foot_slide_uniegomotion(prediction)
    if include_pa:
        result["pa_mpjpe_mm"] = float((_similarity_align(prediction, truth) - truth).norm(dim=-1).mean() * 1000)
    if include_parts:
        result.update({name: float(distances[:, indices].mean() * 1000) for name, indices in PARTS.items()})
    return result


def _paired_metric(rows, name, samples, seed):
    selected = [row for row in rows if row["P"].get(name) is not None and row["CV"].get(name) is not None]
    if not selected:
        return None
    per_take = defaultdict(list)
    for row in selected:
        first, second = float(row["P"][name]), float(row["CV"][name])
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError("Nonfinite paired clip metric.")
        per_take[row["take"]].append((first, second))
    takes = sorted(per_take)
    count = np.asarray([len(per_take[take]) for take in takes], dtype=np.float64)
    sums = np.asarray([np.asarray(per_take[take], dtype=np.float64).sum(0) for take in takes])
    means = sums.sum(0) / count.sum()
    ci, identity = None, None
    if samples:
        indices = np.random.default_rng(seed).integers(0, len(takes), size=(samples, len(takes)))
        # Repeated take draws repeat every clip in that cluster. Divide by the
        # repeated clip count, not the number of takes: the estimand is clip-equal.
        deltas = (sums[:, 0] - sums[:, 1])[indices].sum(1) / count[indices].sum(1)
        ci = np.quantile(deltas, [0.025, 0.975]).tolist()
        identity = {
            "samples": samples,
            "seed": seed,
            "take_order": takes,
            "index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "degenerate_one_take": len(takes) == 1,
        }
    first, second = map(float, means)
    return {
        "P": first,
        "CV": second,
        "delta": first - second,
        "delta_ci95": ci,
        "regression_pct": 100 * (first - second) / second if second else (0.0 if not first else None),
        "clips": len(selected),
        "takes": len(takes),
        "bootstrap": identity,
    }


def summarize_formal_prior_rows(rows, *, bootstrap_samples=2000, bootstrap_seed=1062):
    """Paired clip means with take-cluster bootstrap; no W1 requirement."""
    samples = _integer(bootstrap_samples, "bootstrap_samples", 0)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral):
        raise ValueError("bootstrap_seed must be an integer.")
    rows = sorted(rows, key=lambda row: (row["episode"], row["phase"]))
    keys = [(row["episode"], row["phase"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate clip/phase in formal P evaluation.")
    if any(row["phase"] not in PHASES for row in rows):
        raise ValueError("Unknown formal P evaluation phase.")
    summary = {}
    for phase in PHASES:
        selected = [row for row in rows if row["phase"] == phase]
        names = sorted(set(METRICS) | {name for row in selected for name in row["P"]})
        summary[phase] = {
            "clips": len(selected),
            "takes": len({row["take"] for row in selected}),
            "frames": sum(row["frames"] for row in selected),
            "floor_calibrated_clips": sum(row["floor_calibrated"] for row in selected),
            "metrics": {name: _paired_metric(selected, name, samples, int(bootstrap_seed)) for name in names},
        }
    return summary


@torch.no_grad()
def evaluate_formal_prior(
    model,
    bank,
    codec,
    fk,
    *,
    device="cpu",
    batch_size=32,
    context=None,
    include_pa=False,
    include_parts=False,
    include_beta=False,
    include_shared_beta=False,
    bootstrap_samples=2000,
    bootstrap_seed=1062,
):
    """Compare P/CV on identical frozen W3 caches without any feedback.

    Real FK must use ``shape_mode='predicted'``. Asset-free diagnostic decoders
    without a shape_mode attribute remain supported for CPU preflight tests.
    With a context, complete episodes are sharded and clip rows gathered onto
    every worker before aggregation, including workers with no local episodes.
    """
    batch_size = _integer(batch_size, "batch_size", 1)
    _integer(bootstrap_samples, "bootstrap_samples", 0)
    if getattr(fk, "shape_mode", "predicted") != "predicted":
        raise ValueError("Formal P/CV FK must use each prediction's own beta.")
    grouped = _validate_bank(bank)
    rank, world = (context.rank, context.world_size) if context is not None else (0, 1)
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model.eval()
    rows = []
    for ordinal, (episode, indices) in enumerate(sorted(grouped.items())):
        if ordinal % world != rank:
            continue
        metadata = [bank["rows"][index] for index in indices]
        outputs = {name: [] for name in ("P", "CV", "truth")}
        betas = {name: [] for name in ("P", "CV")}
        shared = []
        for start in range(0, len(indices), batch_size):
            example = select_examples(bank["tensors"], indices[start : start + batch_size], device)
            prediction = predictor_forward(model, example)
            states = {
                name: codec.decode_current(codes[:, 0], example["reference"])
                for name, codes in (("P", prediction), ("CV", example["base"]))
            }
            for name, state in states.items():
                outputs[name].append(fk(state).detach().cpu())
                if include_beta:
                    betas[name].append(state.auxiliary[..., -10:].detach().cpu())
            if include_shared_beta:
                state = states["P"]
                auxiliary = torch.cat((state.auxiliary[..., :-10], states["CV"].auxiliary[..., -10:]), -1)
                shared.append(fk(BodyState(state.joints, state.reference, auxiliary)).detach().cpu())
            outputs["truth"].append(example["truth_xyz"].detach().cpu())
        outputs = {name: torch.cat(values) for name, values in outputs.items()}
        betas = {name: torch.cat(values).double() for name, values in betas.items()} if include_beta else None
        shared = torch.cat(shared) if include_shared_beta else None
        frames = torch.tensor([row["frame"] for row in metadata])
        times = torch.tensor([row["timestamp_s"] for row in metadata], dtype=torch.float64)
        floor = metadata[0]["floor_calibrated"]
        for phase, selected in (
            ("all", frames >= 20),
            ("mature", frames >= 40),
            ("startup", (frames >= 20) & (frames < 40)),
        ):
            if not bool(selected.any()):
                continue
            metrics = {
                name: formal_center_metrics(
                    outputs[name][selected],
                    outputs["truth"][selected],
                    times[selected],
                    floor,
                    include_pa=include_pa,
                    include_parts=include_parts,
                )
                for name in ("P", "CV")
            }
            for name in ("P", "CV"):
                if include_beta:
                    values = betas[name][selected]
                    metrics[name]["beta_norm"] = float(values.norm(dim=-1).mean())
                    metrics[name]["beta_change_per_s"] = (
                        float((torch.diff(values, dim=0) / torch.diff(times[selected])[:, None]).norm(dim=-1).mean())
                        if len(values) > 1
                        else None
                    )
                if include_shared_beta:
                    value = shared[selected] if name == "P" else outputs["CV"][selected]
                    metrics[name]["shared_beta_mpjpe_mm"] = float(
                        (value.double() - outputs["truth"][selected].double()).norm(dim=-1).mean() * 1000
                    )
            rows.append(
                {
                    "episode": episode,
                    "take": metadata[0]["take"],
                    "window": 3,
                    "phase": phase,
                    "frames": int(selected.sum()),
                    "floor_calibrated": floor,
                    **metrics,
                }
            )
    rows = sorted(
        context.gather_rows(rows) if context is not None else rows, key=lambda row: (row["episode"], row["phase"])
    )
    summary = summarize_formal_prior_rows(rows, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    metric = summary["mature"]["metrics"]["mpjpe_mm"]
    return {
        "status": "ok" if metric is not None else "no_mature_data",
        "selection_score": metric["P"] if metric is not None else None,
        "selection_key": "mature.mpjpe_mm.P",
        "best_key": "mature.mpjpe_mm.P",
        "summary": summary,
        "groups": summary,
        "rows": rows,
        "protocol": {
            "version": "P01_dense_clip_equal_v1",
            "producer_window": 3,
            "shape": "each prediction own beta; optional common CV beta auxiliary",
            "aggregation": "mean frames per clip phase, then equal clips",
            "uncertainty": "paired take-cluster percentile bootstrap with repeated clip counts retained",
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "phase_queries": {"all": ">=20", "mature": ">=40", "startup": "20..39"},
            "temporal": "actual uniform dt; forward differences entirely within each phase",
            "unknown_floor": "foot slide unavailable for both P and CV; available clip/take counts reported",
            "scope": "one-step predictions on the same frozen G histories; no P feedback or final-G gain claim",
            "parts": {name: list(indices) for name, indices in PARTS.items()} if include_parts else None,
            "diagnostic_only": bool(getattr(fk, "diagnostic_only", False)),
        },
    }


def _summary(report):
    if not isinstance(report, Mapping):
        return {}
    for key in ("summary", "groups", "report"):
        if key in report:
            return _summary(report[key])
    return report


def _metric(report, phase, name):
    metric = _summary(report).get(phase, {}).get("metrics", {}).get(name)
    if not isinstance(metric, Mapping):
        return None
    values = [metric.get(key) for key in ("P", "CV", "delta")]
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
        return None
    if values[0] < 0 or values[1] < 0:
        return None
    if not math.isclose(values[0] - values[1], values[2], rel_tol=1e-8, abs_tol=1e-7):
        return None
    return metric


def _temporal_checks(report, prefix):
    result = {}
    for phase in ("mature", "all"):
        for name in TEMPORAL_METRICS:
            metric = _metric(report, phase, name)
            result[f"{prefix}_{phase}_{name}_regression_le5pct"] = bool(
                metric is not None and metric["P"] >= 0 and metric["CV"] >= 0 and metric["P"] <= metric["CV"] * 1.05
            )
    return result


def _gate_result(checks, scope):
    passed = all(checks.values())
    return {
        "passed": passed,
        "gate_passed": passed,
        "checks": checks,
        "failures": [key for key, value in checks.items() if not value],
        "scope": scope,
    }


def pilot_gate(curve_or_reports, *, checks_passed=False):
    """P01 2k decision; external numerical/sampling/gradient checks fail closed."""
    if isinstance(curve_or_reports, Mapping):
        reports = {int(step): report for step, report in curve_or_reports.items()}
    else:
        reports = {int(row["step"]): row for row in curve_or_reports}
    checks = {"numerics_sampling_gradients": checks_passed is True}
    for step in (1500, 2000):
        metric = _metric(reports.get(step), "mature", "mpjpe_mm")
        checks[f"mature_delta_negative_at{step}"] = bool(metric is not None and metric["delta"] < 0)
    metric = _metric(reports.get(2000), "mature", "mpjpe_mm")
    checks["mature_gain_at2000_ge0_5mm"] = bool(metric is not None and metric["delta"] <= -0.5)
    checks.update(_temporal_checks(reports.get(2000), "D128_step2000"))
    return _gate_result(checks, "P01 continuation after2k; does not grant final acceptance")


def final_gate(d400_report, long_report):
    """P01 entry gate, using the D128-selected checkpoint exactly once."""
    mature = _metric(d400_report, "mature", "mpjpe_mm")
    all_frames = _metric(d400_report, "all", "mpjpe_mm")
    startup = _metric(d400_report, "startup", "mpjpe_mm")
    long_mature = _metric(long_report, "mature", "mpjpe_mm")
    ci = mature.get("delta_ci95") if mature is not None else None
    valid_ci = (
        isinstance(ci, (list, tuple))
        and len(ci) == 2
        and all(isinstance(value, Real) and math.isfinite(value) for value in ci)
        and ci[0] <= ci[1]
    )
    checks = {
        "D400_mature_gain_ge1mm": bool(mature is not None and mature["delta"] <= -1.0),
        "D400_mature_CI_upper_negative": bool(valid_ci and ci[1] < 0),
        "D400_all_improves": bool(all_frames is not None and all_frames["delta"] < 0),
        "D400_startup_regression_le1mm": bool(startup is not None and startup["delta"] <= 1.0),
        "long_mature_improves": bool(long_mature is not None and long_mature["delta"] < 0),
    }
    checks.update(_temporal_checks(d400_report, "D400"))
    checks.update(_temporal_checks(long_report, "long50"))
    return _gate_result(checks, "standalone P01 acceptance; final-G usefulness requires later matched G adaptation")
