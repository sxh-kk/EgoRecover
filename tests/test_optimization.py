import copy
import json
from types import SimpleNamespace

import pytest
import torch

from egorecover.optimization import make_optimizer, make_scheduler, optimization_protocol


def options(**changes):
    values = {
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "batch_size": 64,
        "lr_schedule": "uniegomotion",
        "lr_decay_step": None,
        "lr_decay_gamma": 0.1,
        "train_steps": 2400,
    }
    return SimpleNamespace(**(values | changes))


def test_official_epoch_reference_is_not_compressed_into_short_training_budget():
    reference = optimization_protocol(options(train_steps=12), reference_clips=143560)
    assert reference["scheduler"]["decay_step"] == 673200
    assert reference["reference"]["full_reference_updates"] == 785400
    assert reference == optimization_protocol(options(train_steps=785400), reference_clips=143560)
    assert json.loads(json.dumps(reference)) == reference
    assert reference["reference"]["commit"] in reference["reference"]["source_links"]["config"]


def test_reference_uses_effective_batch_size_and_rounds_up_partial_batch():
    small = optimization_protocol(options(batch_size=4), reference_clips=9)
    assert small["reference"]["reference_steps_per_epoch"] == 3
    assert small["scheduler"]["decay_step"] == 900
    assert small["reference"]["full_reference_updates"] == 1050
    larger = optimization_protocol(options(batch_size=8), reference_clips=9)
    assert larger["scheduler"]["decay_step"] == 600


def test_no_official_reference_keeps_lr_constant_unless_milestone_is_explicit():
    protocol = optimization_protocol(options())
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = make_optimizer([parameter], protocol)
    assert protocol["scheduler"]["decay_step"] is None
    assert protocol["reference"]["full_reference_updates"] is None
    assert make_scheduler(optimizer, protocol) is None
    constant = optimization_protocol(options(lr_schedule="constant"), reference_clips=143560)
    assert constant["scheduler"]["decay_step"] is None
    assert constant["reference"]["full_reference_updates"] == 785400
    assert make_scheduler(optimizer, constant) is None
    explicit = optimization_protocol(options(lr_decay_step=3))
    assert make_scheduler(optimizer, explicit) is not None
    assert optimization_protocol(SimpleNamespace())["scheduler"]["decay_step"] is None


@pytest.mark.parametrize(
    "changes",
    (
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"weight_decay": -0.01},
        {"weight_decay": float("nan")},
        {"grad_clip_norm": 0},
        {"grad_clip_norm": float("inf")},
        {"lr_decay_gamma": 0},
        {"lr_decay_gamma": 1},
        {"lr_decay_gamma": float("nan")},
        {"lr_decay_step": 0},
        {"lr_decay_step": 2.5},
        {"lr_decay_step": True},
        {"batch_size": 0},
        {"batch_size": True},
        {"lr_schedule": "cosine"},
        {"lr_schedule": "constant", "lr_decay_step": 3},
    ),
)
def test_invalid_optimization_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        optimization_protocol(options(**changes), reference_clips=10)


@pytest.mark.parametrize("reference", (0, -1, 1.5, True))
def test_official_reference_requires_a_positive_integer(reference):
    with pytest.raises(ValueError, match="reference_clips"):
        optimization_protocol(options(), reference_clips=reference)


def test_adamw_applies_the_explicit_learning_rate_and_decoupled_decay():
    protocol = optimization_protocol(options(learning_rate=0.1, weight_decay=0.2))
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = make_optimizer([parameter], protocol)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert parameter.item() == pytest.approx(1.96)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8


def test_lr_changes_after_the_declared_number_of_completed_optimizer_updates():
    protocol = optimization_protocol(options(learning_rate=0.1, lr_decay_step=3))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = make_optimizer([parameter], protocol)
    scheduler = make_scheduler(optimizer, protocol)
    rates = []
    for _ in range(7):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001])


def test_optimizer_and_scheduler_resume_crosses_decay_boundary_identically():
    protocol = optimization_protocol(options(learning_rate=0.1, lr_decay_step=3))
    parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = make_optimizer([parameter], protocol)
    scheduler = make_scheduler(optimizer, protocol)

    def advance(p, opt, schedule, first, count):
        rates = []
        for step in range(first, first + count):
            rates.append(opt.param_groups[0]["lr"])
            p.grad = torch.full_like(p, (step + 1) / 10)
            opt.step()
            schedule.step()
        return rates

    advance(parameter, optimizer, scheduler, 0, 2)
    saved = copy.deepcopy((parameter.detach(), optimizer.state_dict(), scheduler.state_dict()))
    expected_rates = advance(parameter, optimizer, scheduler, 2, 5)
    resumed = torch.nn.Parameter(saved[0].clone())
    restored_optimizer = make_optimizer([resumed], protocol)
    restored_scheduler = make_scheduler(restored_optimizer, protocol)
    restored_optimizer.load_state_dict(saved[1])
    restored_scheduler.load_state_dict(saved[2])
    actual_rates = advance(resumed, restored_optimizer, restored_scheduler, 2, 5)
    assert actual_rates == expected_rates
    torch.testing.assert_close(resumed, parameter, atol=0, rtol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
