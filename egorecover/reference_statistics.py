"""Paired reference diagnostics with take-cluster uncertainty estimates."""

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real

METRICS = (
    "predicted_mm",
    "reference_mm",
    "delta_mm",
    "response_mm",
    "same_shape_predicted_mm",
    "same_shape_reference_mm",
    "same_shape_delta_mm",
    "center_predicted_mm",
    "center_reference_mm",
    "center_delta_mm",
)
OPTIONAL_METRICS = ("seconds", "anchor_current_mm", "intervened_current_mm")
CLASSES = ("reference_better", "approximately_equal", "reference_worse")


def _mean(values):
    return math.fsum(values) / len(values)


def _classes(delta, tolerance):
    return dict(zip(CLASSES, (float(delta > tolerance), float(abs(delta) <= tolerance), float(delta < -tolerance))))


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def _bootstrap(per_take, samples, seed):
    if not samples:
        return {}, None
    takes = sorted(per_take)
    fields = ("predicted_mm", "reference_mm", "delta_mm", *CLASSES)
    values = [
        [row["means"][name] if name in METRICS else row["fractions"][name] for name in fields]
        for row in (per_take[t] for t in takes)
    ]
    rng = random.Random(f"reference_take_bootstrap_v1:{seed}")
    replicates = {name: [] for name in fields}
    fingerprint = hashlib.sha256(
        json.dumps({"takes": takes, "seed": seed, "samples": samples}, sort_keys=True).encode()
    )
    for _ in range(samples):
        indices = [rng.randrange(len(takes)) for _ in takes]
        fingerprint.update((",".join(map(str, indices)) + "\n").encode())
        for column, name in enumerate(fields):
            replicates[name].append(math.fsum(values[index][column] for index in indices) / len(takes))
    identity = {
        "algorithm": "paired_take_cluster_percentile_v1",
        "samples": samples,
        "seed": seed,
        "take_order": takes,
        "index_sha256": fingerprint.hexdigest(),
        "degenerate_one_take": len(takes) == 1,
    }
    return {name: [_quantile(series, 0.025), _quantile(series, 0.975)] for name, series in replicates.items()}, identity


def _table(rows, tolerance, sensitivities, samples, seed):
    metrics = (*METRICS, *(name for name in OPTIONAL_METRICS if all(name in row for row in rows)))
    anchors = defaultdict(list)
    for row in rows:
        anchors[(row["take"], row["episode"], row["anchor"])].append(row)
    episode_units = defaultdict(list)
    for (take, episode, anchor), draws in anchors.items():
        means = {name: _mean([float(row[name]) for row in draws]) for name in metrics}
        episode_units[(take, episode)].append({"anchor": anchor, "means": means})
    take_units = defaultdict(list)
    for (take, episode), units in episode_units.items():
        take_units[take].append(
            {
                "episode": episode,
                "means": {name: _mean([unit["means"][name] for unit in units]) for name in metrics},
                "fractions": {
                    str(tau): {
                        label: _mean([_classes(unit["means"]["delta_mm"], tau)[label] for unit in units])
                        for label in CLASSES
                    }
                    for tau in sensitivities
                },
            }
        )
    per_take = {}
    for take in sorted(take_units):
        units = take_units[take]
        fractions = {
            str(tau): {label: _mean([unit["fractions"][str(tau)][label] for unit in units]) for label in CLASSES}
            for tau in sensitivities
        }
        per_take[take] = {
            "means": {name: _mean([unit["means"][name] for unit in units]) for name in metrics},
            "fractions": fractions[str(tolerance)],
            "sensitivity": fractions,
            "episodes": len(units),
            "anchor_units": sum(key[0] == take for key in anchors),
            "raw_draws": sum(row["take"] == take for row in rows),
        }
    sensitivity = {
        str(tau): {
            label: _mean([value["sensitivity"][str(tau)][label] for value in per_take.values()]) for label in CLASSES
        }
        for tau in sensitivities
    }
    ci, bootstrap = _bootstrap(per_take, samples, seed)
    fractions = sensitivity[str(tolerance)]
    return {
        "counts": {
            "raw_draws": len(rows),
            "anchor_units": len(anchors),
            "episodes": len(episode_units),
            "takes": len(per_take),
        },
        "means": {name: _mean([value["means"][name] for value in per_take.values()]) for name in metrics},
        "fractions": fractions,
        "percentages": {name: 100 * value for name, value in fractions.items()},
        "per_take": per_take,
        "tolerance_mm": tolerance,
        "sensitivity": sensitivity,
        "ci95": ci,
        "bootstrap": bootstrap,
    }


def _tables(rows, tolerance, sensitivities, samples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["intervention"], row["horizon"])].append(row)
    result = {}
    for (intervention, horizon), selected in sorted(groups.items()):
        result.setdefault(intervention, {})[str(horizon)] = _table(selected, tolerance, sensitivities, samples, seed)
    return result


def summarize_reference_rows(rows, *, tolerance_mm=1.0, bootstrap_samples=2000, bootstrap_seed=1062):
    """Average paired draws before classifying anchors, then weight takes equally.

    Fractions and their CIs are on [0,1]; percentages are also supplied. Delta
    is E_pred-E_ref, so a positive delta favors the reference intervention.
    ``bootstrap_samples=0`` skips CIs for inexpensive progress reports.
    """
    if (
        isinstance(tolerance_mm, bool)
        or not isinstance(tolerance_mm, Real)
        or not math.isfinite(tolerance_mm)
        or tolerance_mm < 0
    ):
        raise ValueError("tolerance_mm must be a finite nonnegative number.")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, Integral) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral):
        raise ValueError("bootstrap_seed must be an integer.")
    tolerance, samples, seed = float(tolerance_mm), int(bootstrap_samples), int(bootstrap_seed)
    rows = list(rows)
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or not all(
            isinstance(row.get(name), str) and row[name] for name in ("take", "episode")
        ):
            raise ValueError("Reference rows require nonempty take and episode strings.")
        if row.get("intervention") not in ("body_pose", "full_state"):
            raise ValueError("Unknown reference intervention.")
        for name, minimum in (("anchor", 0), ("horizon", 1), ("draw", None)):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or (minimum is not None and value < minimum):
                raise ValueError(f"{name} must be an integer with a valid range.")
        key = tuple(row[name] for name in ("intervention", "horizon", "take", "episode", "anchor", "draw"))
        if key in seen:
            raise ValueError("Duplicate paired draw for one reference anchor.")
        seen.add(key)
        for name in (*METRICS, *(name for name in OPTIONAL_METRICS if name in row)):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Reference metric {name} must be finite.")
        for prefix in ("", "same_shape_", "center_"):
            expected = float(row[f"{prefix}predicted_mm"]) - float(row[f"{prefix}reference_mm"])
            if not math.isclose(float(row[f"{prefix}delta_mm"]), expected, rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError("Reference delta must equal predicted minus reference error.")
    sensitivities = sorted({0.0, 1.0, 5.0, tolerance})
    protocol = {
        "version": "reference_take_hierarchy_v1",
        "aggregation": "paired draw mean per anchor -> anchor mean per episode -> episode mean per take -> equal takes",
        "classification_unit": "anchor after paired draw averaging, before episode/take aggregation",
        "delta_definition": "predicted_mm - reference_mm; positive favors reference",
        "class_boundaries": {
            "reference_better": "delta > tolerance",
            "approximately_equal": "abs(delta) <= tolerance",
            "reference_worse": "delta < -tolerance",
        },
        "tolerance_mm": tolerance,
        "sensitivity_tolerances_mm": sensitivities,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence_interval": "95% paired take-cluster percentile, linearly interpolated quantiles; groups omit bootstrap",
        "fraction_units": "0..1; percentages also provided",
    }
    if not rows:
        return {"status": "empty", "protocol": protocol, "main": {}, "by_anchor": {}, "by_parent_task": {}}
    by_anchor = defaultdict(list)
    by_parent = defaultdict(list)
    parent_metadata = {}
    for row in rows:
        by_anchor[int(row["anchor"])].append(row)
        identifier = row.get("parent_task_id")
        name = row.get("parent_task_name")
        if identifier is not None:
            if isinstance(identifier, Integral):
                identifier = int(identifier)
            elif isinstance(identifier, Real) and math.isfinite(identifier):
                identifier = float(identifier)
            elif not isinstance(identifier, str):
                raise ValueError("parent_task_id must be a finite JSON scalar or None.")
        if name is not None and not isinstance(name, str):
            raise ValueError("parent_task_name must be a string or None.")
        key = f"id:{identifier}" if identifier is not None else f"name:{name}" if name else "unknown"
        by_parent[key].append(row)
        if key not in parent_metadata or parent_metadata[key]["parent_task_name"] is None:
            parent_metadata[key] = {"parent_task_id": identifier, "parent_task_name": name}
    return {
        "status": "ok",
        "protocol": protocol,
        "main": _tables(rows, tolerance, sensitivities, samples, seed),
        "by_anchor": {
            str(anchor): _tables(selected, tolerance, sensitivities, 0, seed)
            for anchor, selected in sorted(by_anchor.items())
        },
        "by_parent_task": {
            key: {**parent_metadata[key], "main": _tables(selected, tolerance, sensitivities, 0, seed)}
            for key, selected in sorted(by_parent.items())
        },
    }
