import pytest

from run.select_budget import select


def test_budget_selection_requires_complete_common_dev_grid():
    summary = {
        "identity": {"group": "dev", "variants": ["clean", "freeze_3s", "drift_0p03mps"]},
        "takes": [f"take_{index}" for index in range(12)],
        "scores": {},
    }
    budgets = []
    for group in ("12take", "72take"):
        for steps in (400, 1200, 2400):
            label = f"{group}_{steps}"
            budgets.append(f"{label}={label}")
            score = 200 if steps == 1200 else 250
            summary["scores"][f"{label}/gaussian"] = {"macro_smpl22_mm": score}
            summary["scores"][f"{label}/history"] = {"macro_smpl22_mm": score + 20}
    result = select(summary, budgets)
    assert result["winners"]["12take"]["steps"] == 1200
    assert result["winners"]["72take"]["steps"] == 1200
    assert result["holdout_used"] is False
    with pytest.raises(ValueError, match="six cells"):
        select(summary, budgets[:-1])
    summary["takes"].pop()
    with pytest.raises(ValueError, match="12 common"):
        select(summary, budgets)
