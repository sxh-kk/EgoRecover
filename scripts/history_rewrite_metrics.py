"""Post-hoc diagnostics of W=3 history corrections and first-emission motion.

All positions are fixed-bootstrap-shape SMPL22 world positions in meters.
Labels are used only here, after the causal reconstruction has completed.
"""

import numpy as np

FEET = [7, 10, 8, 11]
THRESHOLDS = np.array([0.08, 0.04, 0.08, 0.04])


def contact_weights(positions):
    height = positions[..., FEET, 2]
    contact = height < THRESHOLDS
    return np.where(contact, np.abs(2 - np.exp2(np.where(contact, height / THRESHOLDS, 0))), 0)


def foot_contributions(previous, current):
    return contact_weights(previous) * np.linalg.norm(current[..., FEET, :2] - previous[..., FEET, :2], axis=-1)


def diagnostic_metrics(emitted, truth, windows, incoming, timestamps, *, rewrite):
    """Keep proposed corrections distinct from corrections actually committed.

    windows[t] contains the candidate poses for physical t-2,t-1,t. Incoming
    contains the actual two cached poses before that call. The first two
    bootstrap slots have no first emission and are excluded from corrections.
    """
    emitted, truth, windows, incoming, timestamps = [
        np.asarray(x, dtype=np.float64) for x in (emitted, truth, windows, incoming, timestamps)
    ]
    count = len(emitted)
    if (
        count < 3
        or emitted.shape != (count, 22, 3)
        or truth.shape != emitted.shape
        or windows.shape != (count, 3, 22, 3)
        or incoming.shape != (count, 2, 22, 3)
        or timestamps.shape != (count,)
    ):
        raise ValueError("Expected aligned first emissions and W=3 physical windows.")
    if not all(np.isfinite(x).all() for x in (emitted, truth, windows, incoming, timestamps)):
        raise ValueError("Diagnostic positions must be finite.")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals.mean(), rtol=2e-5, atol=1e-8):
        raise ValueError("Uniform increasing timestamps are required.")
    if not np.allclose(windows[:, -1], emitted, rtol=0, atol=2e-5):
        raise ValueError("Window terminal slots must be the actual first emissions.")
    # Use the exact native first-emission values at the current slots; batched
    # SMPL decoding of historical slots can differ by floating point roundoff.
    windows = windows.copy()
    windows[:, -1] = emitted
    dt = float(intervals.mean())
    indices = np.arange(count)[:, None] - np.array([2, 1])[None]
    valid = indices >= 0
    original = emitted[indices.clip(0)]
    update = windows[:, :2] - incoming
    cumulative = windows[:, :2] - original
    update_norm = np.linalg.norm(update, axis=-1) * 1000
    cumulative_norm = np.linalg.norm(cumulative, axis=-1) * 1000
    relative = cumulative - cumulative[..., :1, :]

    velocity = np.diff(emitted, axis=0) / dt
    gt_velocity = np.diff(truth, axis=0) / dt
    internal_velocity = (windows[1:, -1] - windows[1:, -2]) / dt
    acceleration = np.diff(emitted, n=2, axis=0) / dt**2
    gt_acceleration = np.diff(truth, n=2, axis=0) / dt**2
    internal_acceleration = (windows[2:, -1] - 2 * windows[2:, -2] + windows[2:, -3]) / dt**2
    velocity_gap = velocity - internal_velocity
    acceleration_gap = acceleration - internal_acceleration
    expected_v_gap = cumulative[1:, 1] / dt
    expected_a_gap = (2 * cumulative[2:, 1] - cumulative[2:, 0]) / dt**2
    if not np.allclose(velocity_gap, expected_v_gap, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Velocity seam identity failed: check physical frame alignment.")
    if not np.allclose(acceleration_gap, expected_a_gap, rtol=1e-10, atol=1e-9):
        raise RuntimeError("Acceleration seam identity failed: check physical frame alignment.")

    def magnitude(x):
        return np.linalg.norm(x, axis=-1).mean(-1) * 1000

    emitted_slide = foot_contributions(emitted[:-1], emitted[1:])
    internal_slide = foot_contributions(windows[1:, -2], windows[1:, -1])
    gt_weights = contact_weights(truth[:-1])
    fixed_contact_slide = gt_weights * np.linalg.norm(emitted[1:, FEET, :2] - emitted[:-1, FEET, :2], axis=-1)
    summary = {
        "candidate_cache_update_body_mm": float(update_norm[valid].mean()),
        "candidate_cache_update_lag1_body_mm": float(update_norm[1:, 1].mean()),
        "candidate_cache_update_lag2_body_mm": float(update_norm[2:, 0].mean()),
        "cumulative_lag2_deviation_from_first_mm": float(cumulative_norm[2:, 0].mean()),
        "candidate_cache_update_feet_xy_mm": float(np.linalg.norm(update[..., FEET, :2], axis=-1)[valid].mean() * 1000),
        "candidate_cache_update_root_mm": float(update_norm[..., 0][valid].mean()),
        "lag1_root_relative_deviation_mm": float(np.linalg.norm(relative[1:, 1], axis=-1).mean() * 1000),
        "applied_cache_update_body_mm": float(update_norm[valid].mean()) if rewrite else 0.0,
        "velocity_seam_gap_mm_s": float(magnitude(velocity_gap).mean()),
        "acceleration_seam_gap_mm_s2": float(magnitude(acceleration_gap).mean()),
        "first_emission_velocity_error_mm_s": float(magnitude(velocity - gt_velocity).mean()),
        "within_window_velocity_error_mm_s": float(magnitude(internal_velocity - gt_velocity).mean()),
        "first_emission_acceleration_error_mm_s2": float(magnitude(acceleration - gt_acceleration).mean()),
        "within_window_acceleration_error_mm_s2": float(magnitude(internal_acceleration - gt_acceleration).mean()),
        "first_emission_foot_slide_mm": float(emitted_slide.sum(0).mean() * 1000 / count),
        "within_window_foot_slide_mm": float(internal_slide.sum(0).mean() * 1000 / count),
        "gt_contact_weighted_first_emission_slide_mm": float(fixed_contact_slide.sum(0).mean() * 1000 / count),
        "gt_motion_speed_mm_s": float(magnitude(gt_velocity).mean()),
        "gt_motion_acceleration_mm_s2": float(magnitude(gt_acceleration).mean()),
        "gt_contact_fraction": float((gt_weights > 0).mean()),
        "valid_lag1_corrections": count - 1,
        "valid_lag2_corrections": count - 2,
    }
    traces = {
        "lag1_correction_mm": cumulative_norm[1:, 1].mean(-1),
        "lag1_feet_xy_correction_mm": np.linalg.norm(cumulative[1:, 1, FEET, :2], axis=-1).mean(-1) * 1000,
        "velocity_error_mm_s": magnitude(velocity - gt_velocity),
        "within_window_velocity_error_mm_s": magnitude(internal_velocity - gt_velocity),
        "acceleration_error_mm_s2": magnitude(acceleration - gt_acceleration),
        "within_window_acceleration_error_mm_s2": magnitude(internal_acceleration - gt_acceleration),
        "foot_slide_transition_mm": emitted_slide.mean(-1) * 1000,
        "within_window_foot_slide_transition_mm": internal_slide.mean(-1) * 1000,
        "velocity_seam_gap_mm_s": magnitude(velocity_gap),
        "acceleration_seam_gap_mm_s2": magnitude(acceleration_gap),
    }
    return summary, traces


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-15 else None


def ranks(x):
    x = np.asarray(x)
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    del values
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def paired_cluster(values, takes, *, draws=2000, seed=1062):
    values = np.asarray(values, dtype=float)
    takes = np.asarray(takes)
    groups = [np.flatnonzero(takes == name) for name in sorted(set(takes))]
    sums = np.array([values[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(groups), size=(draws, len(groups)))
    samples = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "delta": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "clips": len(values),
        "takes": len(groups),
        "fraction_improved": float((values < 0).mean()),
    }


def association(x, y, takes, controls=None, *, draws=2000, seed=1062):
    """Clip-level correlations with take-cluster uncertainty, not causality."""
    x, y, takes = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(takes)
    groups = [np.flatnonzero(takes == name) for name in sorted(set(takes))]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        value = correlation(ranks(x[indices]), ranks(y[indices]))
        if value is not None:
            samples.append(value)
    result = {
        "pearson": correlation(x, y),
        "spearman": correlation(ranks(x), ranks(y)),
        "spearman_ci95": np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
        "clips": len(x),
        "takes": len(groups),
    }
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        design = np.column_stack([np.ones(len(x)), np.log1p(controls)])
        rx = np.log1p(x) - design @ np.linalg.lstsq(design, np.log1p(x), rcond=None)[0]
        ry = np.log1p(y) - design @ np.linalg.lstsq(design, np.log1p(y), rcond=None)[0]
        result["log_partial_pearson_gt_motion_controls"] = correlation(rx, ry)
    centered_x, centered_y = x.copy(), y.copy()
    for group in groups:
        centered_x[group] -= x[group].mean()
        centered_y[group] -= y[group].mean()
    result["within_take_centered_pearson"] = correlation(centered_x, centered_y)
    return result
