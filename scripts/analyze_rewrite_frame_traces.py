"""Relate observed corrections to simultaneous and next-frame temporal errors.

Committing the candidate cache occurs after the current output is generated.
Contemporaneous correlations therefore cannot establish a causal write effect.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .history_rewrite_metrics import THRESHOLDS, correlation
except ImportError:
    from history_rewrite_metrics import THRESHOLDS, correlation


def aligned_series(payload, metric, *, next_frame):
    trace = {k: np.asarray(v, dtype=float) for k, v in payload["traces"].items()}
    positions = np.asarray(payload["truth_joints"], dtype=float)
    dt = float(np.diff(np.asarray(payload["timestamps"], dtype=float)).mean())
    feet = metric == "foot_slide_transition_mm"
    correction = trace["lag1_feet_xy_correction_mm" if feet else "lag1_correction_mm"]
    # Errors all end at local scored frames2..59 (physical22..79). Current
    # corrections are at2..59; previous writes are at1..58 (physical21..78).
    x = correction[:-1] if next_frame else correction[1:]
    y = trace[metric] if "acceleration" in metric else trace[metric][1:]
    speed = np.linalg.norm(np.diff(positions, axis=0) / dt, axis=-1).mean(-1)[1:] * 1000
    acceleration = np.linalg.norm(np.diff(positions, n=2, axis=0) / dt**2, axis=-1).mean(-1) * 1000
    contact = (positions[1:-1, [7, 10, 8, 11], 2] < THRESHOLDS).mean(-1)
    if len(x) != len(y) or len(x) != len(speed):
        raise ValueError("Correction and error endpoints are misaligned.")
    return x, y, np.column_stack([speed, acceleration, contact])


def within_clip_association(series, *, draws=2000):
    """Retain frame autocorrelation by bootstrapping complete take clusters."""
    xs, ys, cs = [], [], []
    sums = {}
    for take, x, y, controls in series:
        x, y = np.asarray(x), np.asarray(y)
        xc, yc = x - x.mean(), y - y.mean()
        sufficient = np.array([np.dot(xc, yc), np.dot(xc, xc), np.dot(yc, yc)])
        sums[take] = sums.get(take, np.zeros(3)) + sufficient
        xs.append(np.log1p(x) - np.log1p(x).mean())
        ys.append(np.log1p(y) - np.log1p(y).mean())
        controls = np.log1p(controls)
        cs.append(controls - controls.mean(0))
    totals = np.stack(list(sums.values()))

    def rho(value):
        denominator = np.sqrt(value[..., 1] * value[..., 2])
        return np.divide(value[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 1e-15)

    indices = np.random.default_rng(1062).integers(len(totals), size=(draws, len(totals)))
    sampled = totals[indices].sum(1)
    valid = (sampled[:, 1] * sampled[:, 2]) > 1e-15
    samples = rho(sampled[valid])
    x, y, controls = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    residual_x = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    residual_y = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    return {
        "within_clip_pearson": float(rho(totals.sum(0))) if totals[:, 1:].sum(0).prod() > 1e-15 else None,
        "within_clip_pearson_ci95": np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) else None,
        "within_clip_log_partial_pearson_gt_motion_controls": correlation(residual_x, residual_y),
        "frames": sum(len(value[1]) for value in series),
        "clips": len(series),
        "takes": len(totals),
    }


def analyze(root):
    result = {}
    for source in ("gaussian", "prediction"):
        rows = json.loads((root / f"{source}_on/report.json").read_text())["rows"]
        buckets = {
            (key, following): []
            for key in ("velocity_error_mm_s", "acceleration_error_mm_s2", "foot_slide_transition_mm")
            for following in (False, True)
        }
        for row in rows:
            payload = torch.load(root / row["trace_file"], map_location="cpu", weights_only=True)
            for (key, following), bucket in buckets.items():
                bucket.append((row["take"], *aligned_series(payload, key, next_frame=following)))
        result[source] = {
            key + ("_next_frame" if following else "_same_frame"): within_clip_association(bucket)
            for (key, following), bucket in buckets.items()
        }
    report = {
        "status": "complete",
        "scope": "Rewrite ON; frame associations after removing each clip mean; take-cluster bootstrap2000.",
        "same_frame": "Corrections at physical22..79 vs temporal errors ending at22..79; observational compatibility only.",
        "next_frame": "Corrections at physical21..78 vs temporal errors ending at22..79; temporally ordered association, not causal identification.",
        "controls": "posthoc GT joint speed, acceleration, previous-frame foot contact fraction; log1p and clip fixed effects; adjusted point estimates without CI",
        "results": result,
    }
    (root / "frame-associations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
