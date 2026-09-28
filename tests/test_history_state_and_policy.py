import copy

import pytest
import torch

from egorecover.codec import BodyState, MotionCodec, rigid_transform
from egorecover.history import HistoryBuffer
from egorecover.prior import HistoryPrior
from egorecover.utility import UtilityPredictor, choose_action


def test_history_commits_once_and_candidates_have_no_side_effects():
    codec = MotionCodec(
        {
            "motion_mean": torch.zeros(243),
            "motion_std": torch.ones(243),
            "traj_mean": torch.zeros(18),
            "traj_std": torch.ones(18),
        }
    )
    identity = torch.eye(4)
    initial = BodyState(identity.expand(22, 4, 4).clone(), identity, torch.zeros(36))
    one = codec.encode_current(initial, identity)
    buffer = HistoryBuffer.from_bootstrap(codec, one.expand(20, -1).clone(), identity)
    before = buffer.encoded().clone()
    beta = buffer.beta_boot.clone()
    for _ in range(4):
        buffer.decode_candidate(one)
    assert list(buffer.frame_indices) == list(range(20))
    assert torch.equal(before, buffer.encoded())
    buffer.commit(one, 20)
    assert list(buffer.frame_indices) == list(range(1, 21))
    assert torch.equal(beta, buffer.beta_boot)
    with pytest.raises(ValueError, match="exactly once"):
        buffer.commit(one, 20)
    observation = {
        "aria_traj_obs": torch.tensor([[1.0, 0, 0, 0, 1, 0, 2, 3, 1.6]]),
        "img_feats": torch.randn(1, 1024),
        "img_available": torch.ones(1, dtype=torch.bool),
        "traj_available": torch.ones(1, dtype=torch.bool),
        "frame_id_30fps": torch.tensor([63]),
    }
    y = buffer.conditions(observation, action="a00")
    changed = copy.deepcopy(observation)
    changed["aria_traj_obs"].fill_(float("nan"))
    changed["img_feats"].fill_(float("nan"))
    other = buffer.conditions(changed, action="a00")
    assert all(torch.equal(y[key], other[key]) for key in y)
    changed["gt_body"] = torch.zeros(243)
    with pytest.raises(ValueError, match="documented fields"):
        buffer.conditions(changed)


def test_frozen_prior_stays_deterministic_after_parent_train():
    prior = HistoryPrior(width=32, layers=1, heads=4, dropout=0.5)
    history, valid, base = torch.randn(2, 20, 243), torch.ones(2, 20, dtype=torch.bool), torch.randn(2, 1, 243)
    assert torch.equal(prior(history, valid, base), base)
    with torch.no_grad():
        prior.output.weight.normal_(std=0.01)
    parent = torch.nn.Sequential(prior.freeze())
    parent.train()
    assert not prior.training and all(not p.requires_grad for p in prior.parameters())
    first = prior(history.requires_grad_(), valid, base)
    second = prior(history, valid, base)
    assert torch.equal(first, second) and not first.requires_grad


def test_q_missing_payload_isolation_and_action_canonicalization():
    q = UtilityPredictor(width=32, layers=1, heads=4, dropout=0.0).eval()
    with torch.no_grad():
        q.output[-1].weight.normal_(std=0.01)
    inputs = dict(
        history_motion=torch.randn(2, 20, 243),
        history_valid=torch.ones(2, 20, dtype=torch.bool),
        prior_mu=torch.randn(2, 1, 243),
        img_embs=torch.randn(2, 5, 1024),
        traj=torch.randn(2, 5, 18),
        img_available=torch.zeros(2, 5, dtype=torch.bool),
        traj_available=torch.ones(2, 5, dtype=torch.bool),
        observation_valid=torch.ones(2, 5, dtype=torch.bool),
    )
    with torch.no_grad():
        first = q(**inputs)
        inputs["img_embs"].fill_(float("nan"))
        assert torch.equal(first, q(**inputs))
    # a01 is equivalent to baseline when visual is already unavailable; its
    # large noisy gain must not override the canonical zero-gain baseline.
    gains = torch.tensor([[-1.0, 100.0, 90.0], [2.0, 100.0, -5.0]])
    action = choose_action(
        gains, img_available=torch.zeros(2, 1, dtype=torch.bool), traj_available=torch.ones(2, 1, dtype=torch.bool)
    )
    assert action.tolist() == [0, 1]


def test_offline_gains_use_one_shared_source_and_leave_inputs_unchanged():
    from egorecover.conditioning import build_conditioning
    from egorecover.history_flow import HistoryFlow
    from egorecover.utility import generate_utility_labels
    from mydiffusion.flow_matching import FlowMatching

    class KnownActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_sources = []

        def forward(self, x, t, y):
            if bool((t == 1).all()):
                self.first_sources.append(x.clone())
            value = 2 * (~y["img_mask"]).float() + 3 * (~y["traj_mask"]).float()
            return value[..., None].expand_as(x)

    model = KnownActionModel().eval()
    y = build_conditioning(
        history_motion=torch.zeros(1, 20, 243),
        history_valid=torch.ones(1, 20, dtype=torch.bool),
        prior_mu=torch.ones(1, 1, 243),
        traj=torch.zeros(1, 1, 18),
        img_embs=torch.zeros(1, 1, 1024),
        img_available=torch.ones(1, 1, dtype=torch.bool),
        traj_available=torch.ones(1, 1, dtype=torch.bool),
    )
    saved = copy.deepcopy(y)
    flow = HistoryFlow(sigma=0.3, flow=FlowMatching(num_steps=3))
    epsilon = torch.randn(1, 1, 243)
    result = generate_utility_labels(
        model, flow, y, epsilon, lambda value: (value - 2).abs().mean((1, 2)), checkpoint_id="test-only"
    )
    torch.testing.assert_close(result["errors"], torch.tensor([[3.0, 0, 1, 2]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result["gains"], torch.tensor([[3.0, 2, 1]]), atol=1e-6, rtol=0)
    assert len(model.first_sources) == 4
    assert all(torch.equal(value, model.first_sources[0]) for value in model.first_sources)
    assert all(torch.equal(y[name], saved[name]) for name in y)
