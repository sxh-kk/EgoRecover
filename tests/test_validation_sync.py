from dataclasses import replace

import pytest
import torch

from egorecover.codec import BodyState
from egorecover.state import ObservationWindow, WindowCache


def test_masked_observations_allow_missing_nan_but_reject_available_nan():
    valid = torch.ones(2, 21, dtype=torch.bool)
    valid[:, 2] = False
    head, images = torch.zeros(2, 21, 9), torch.zeros(2, 21, 1024)
    head[:, 2] = float("nan")
    images[:, 2] = float("inf")
    observation = ObservationWindow(head, images, valid, valid, 20)
    observation.validate(2, 20)
    for name in ("head", "images"):
        payload = getattr(observation, name).clone()
        payload[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match=f"Nonfinite available observation: {name}"):
            replace(observation, **{name: payload}).validate(2, 20)


@pytest.mark.parametrize("name", ["joints", "reference", "auxiliary", "beta", "anchor"])
def test_combined_cache_check_still_rejects_each_nonfinite_field(name):
    cache = WindowCache(
        BodyState(torch.eye(4).repeat(2, 20, 22, 1, 1), torch.eye(4).repeat(2, 20, 1, 1), torch.zeros(2, 20, 36)),
        torch.zeros(2, 10),
        torch.eye(4).repeat(2, 1, 1),
        19,
    )
    cache.validate()
    tensor = (
        getattr(cache.body, name) if name in ("joints", "reference", "auxiliary") else getattr(cache, name)
    ).clone()
    tensor.flatten()[0] = float("nan")
    invalid = (
        replace(cache, body=replace(cache.body, **{name: tensor}))
        if name in ("joints", "reference", "auxiliary")
        else replace(cache, **{name: tensor})
    )
    with pytest.raises(ValueError):
        invalid.validate()
