from pathlib import Path

import pytest

from egorecover.checkpoint import resolve_prior_mode


@pytest.mark.parametrize(
    ("mode", "checkpoint", "expected"),
    [
        ("auto", None, "constant_velocity"),
        ("auto", Path("not-loaded-prior.pt"), "learned"),
        ("constant_velocity", None, "constant_velocity"),
        ("learned", Path("not-loaded-prior.pt"), "learned"),
    ],
)
def test_prior_resolution_selects_explicit_or_automatic_policy_without_loading_assets(mode, checkpoint, expected):
    assert resolve_prior_mode(mode, checkpoint) == expected


@pytest.mark.parametrize(
    ("mode", "checkpoint", "reason"),
    [
        ("learned", None, "requires --prior-checkpoint"),
        ("constant_velocity", Path("prior.pt"), "does not load --prior-checkpoint"),
    ],
)
def test_explicit_prior_policy_rejects_missing_or_conflicting_checkpoint(mode, checkpoint, reason):
    with pytest.raises(ValueError, match=reason):
        resolve_prior_mode(mode, checkpoint)


@pytest.mark.parametrize("mode", ["cv", "", None, "unrecognized"])
def test_unknown_prior_mode_is_rejected_instead_of_silently_falling_back(mode):
    with pytest.raises(ValueError, match="Prior mode must"):
        resolve_prior_mode(mode, None)
