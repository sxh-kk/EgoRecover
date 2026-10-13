"""CPU integration checks for causal R0/R1 P/Q/G execution, without weights or GT."""

import pytest
import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.observation_reference import ObservationFlow, prepare
from egorecover.prior import HistoryPrior
from egorecover.rollout import run_episode
from mydiffusion.flow_matching import FlowMatching


@pytest.fixture
def setup():
    torch.set_num_threads(2)
    codec = MotionCodec(
        {
            name: torch.zeros(width) if name.endswith("mean") else torch.ones(width)
            for name, width in (("motion_mean", 243), ("motion_std", 243), ("traj_mean", 18), ("traj_std", 18))
        }
    )
    state = BodyState(torch.eye(4).expand(22, 4, 4).clone(), torch.eye(4), torch.zeros(36))
    raw = codec.denormalize(codec.encode_current(state, torch.eye(4)))
    raw[204] = 0.01
    bootstrap = codec.normalize(raw)

    class Initializer(torch.nn.Module):
        def forward(self, x, t, y):
            return bootstrap.expand_as(x)

    def make_observations(*, poison=False, missing=False):
        requests = []

        def observations(*, as_of, history_frames):
            requests.append((as_of, history_frames))
            indices = torch.arange(as_of - history_frames + 1, as_of + 1)
            head = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 1.6]).repeat(history_frames, 1)
            image = indices.float()[:, None].expand(-1, 1024).clone()
            available = indices < 20 if missing else torch.ones(history_frames, dtype=torch.bool)
            if poison:
                head[indices >= 20] = float("nan")
                image[indices >= 20] = float("inf")
            return {
                "aria_traj_obs": head,
                "img_feats": image,
                "img_available": available,
                "traj_available": available,
                "frame_id_30fps": indices * 3,
            }

        return observations, requests

    def make_prior():
        prior = HistoryPrior(width=32, layers=1, heads=4, dropout=0).freeze()
        prior.base_mode = "constant_velocity"
        prior.bases = []
        prior.register_forward_pre_hook(lambda module, args: module.bases.append(args[2].clone()))
        return prior

    return codec, Initializer().eval(), make_observations, make_prior


class Current(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.observation_length = length
        self.anchored = False
        self.inputs, self.noises = [], []

    def forward(self, x, t, y):
        clean = prepare(y)
        self.inputs.append(clean)
        if bool((t == 1).all()):
            self.noises.append(x.clone())
        result = clean["prior_mu"].clone()
        result[..., 204] += 0.02 * (~clean["img_mask"]).float()
        return result


class Utility(torch.nn.Module):
    def __init__(self, gains):
        super().__init__()
        self.gains = gains
        self.inputs = []

    def forward(self, **kwargs):
        self.inputs.append(kwargs)
        return torch.tensor([self.gains])


def execute(setup, length, *, scope="current", utility=None, poison=False, missing=False, action="a11"):
    codec, initializer, make_observations, make_prior = setup
    observations, requests = make_observations(poison=poison, missing=missing)
    model, prior = Current(length).eval(), make_prior()
    output = run_episode(
        model,
        ObservationFlow(action_scope=scope, flow=FlowMatching(num_steps=2)),
        initializer,
        FlowMatching(num_steps=1),
        codec,
        observations,
        num_frames=24,
        prior=prior,
        utility=utility,
        action=action,
        episode_id="synthetic_take/episode",
        seed=1062,
    )
    return output, model, prior, requests


@pytest.mark.parametrize("length", [21, 80])
@pytest.mark.parametrize("scope", ["current", "window"])
def test_memory_rollout_counts_calls_and_uses_action_scope_availability(setup, length, scope):
    utility = Utility([-1.0, 4.0, -1.0]).eval()
    output, model, prior, requests = execute(setup, length, scope=scope, utility=utility, missing=True, poison=True)
    assert requests == [(19, 20), (20, 1), (21, 1), (22, 1), (23, 1)]
    assert len(prior.bases) == len(utility.inputs) == 4 and len(model.inputs) == 8
    # Current-only actions collapse to a11 when both current sensors are absent.
    # Window actions remain distinct while previous actual observations exist.
    assert output["actions"] == ([0] * 4 if scope == "current" else [2] * 4)
    for frame, inputs in enumerate(utility.inputs, start=20):
        assert inputs["img_embs"].shape == (1, length, 1024)
        assert int(inputs["observation_valid"].sum()) == min(frame + 1, length)
        assert torch.isfinite(inputs["traj"]).all()
        assert torch.isfinite(inputs["img_embs"]).all()
        assert not inputs["img_available"][0, -1] and not inputs["traj_available"][0, -1]
    for y in model.inputs:
        assert y["valid_frames"].all() and y["history_valid"].all()
        assert y["memory_img_mask"].all() if scope == "window" else y["memory_img_mask"][0, -1]
    # Constant-velocity P sees the nonzero physical reference delta from startup.
    assert float(prior.bases[0][0, 0, 204]) == pytest.approx(0.01, abs=1e-6)
    assert output["protocol"]["action_scope"] == scope
    assert output["protocol"]["nfe_per_frame"] == 2
    assert output["protocol"]["prior_base_mode"] == "constant_velocity"


def test_window_mask_isolates_raw_nan_and_keeps_each_policy_history_independent(setup):
    ordinary, _, _, _ = execute(setup, 80, scope="window", action="a00")
    poisoned, _, _, _ = execute(setup, 80, scope="window", action="a00", poison=True)
    torch.testing.assert_close(ordinary["dense_world_joints"], poisoned["dense_world_joints"], rtol=0, atol=0)
    kept, keep_model, _, _ = execute(setup, 21, utility=Utility([-1.0, -1.0, -1.0]).eval())
    dropped, drop_model, _, _ = execute(setup, 21, utility=Utility([-1.0, -1.0, 5.0]).eval())
    assert not torch.equal(kept["dense_world_joints"], dropped["dense_world_joints"])
    assert not torch.equal(keep_model.inputs[-1]["history_motion"], drop_model.inputs[-1]["history_motion"])
    assert all(torch.equal(a, b) for a, b in zip(keep_model.noises, drop_model.noises))


def test_current_only_rollout_also_honors_calibrated_cv_baseline(setup):
    from egorecover.history_flow import HistoryFlow

    codec, initializer, make_observations, make_prior = setup
    observations, _ = make_observations()
    prior = make_prior()

    class CurrentOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t, y):
            return y["prior_mu"]

    run_episode(
        CurrentOnly().eval(),
        HistoryFlow(flow=FlowMatching(num_steps=1)),
        initializer,
        FlowMatching(num_steps=1),
        codec,
        observations,
        num_frames=21,
        prior=prior,
    )
    assert float(prior.bases[0][0, 0, 204]) == pytest.approx(0.01, abs=1e-6)


def test_observed_reference_is_rejected_by_prediction_reference_rollout(setup):
    codec, initializer, make_observations, make_prior = setup
    model = Current(21).eval()
    model.anchored = True
    observations, requests = make_observations()
    with pytest.raises(ValueError, match="prediction-defined"):
        run_episode(
            model,
            ObservationFlow(),
            initializer,
            FlowMatching(),
            codec,
            observations,
            num_frames=21,
            prior=make_prior(),
        )
    assert requests == []


def test_rollout_rejects_a_future_frame_packet_before_p_or_g(setup):
    codec, initializer, make_observations, make_prior = setup
    observations, _ = make_observations()
    model, prior = Current(21).eval(), make_prior()

    def future_packet(**kwargs):
        packet = observations(**kwargs)
        if kwargs["as_of"] >= 20:
            packet["frame_id_30fps"] += 3
        return packet

    with pytest.raises(ValueError, match="causal physical frame"):
        run_episode(
            model,
            ObservationFlow(),
            initializer,
            FlowMatching(num_steps=1),
            codec,
            future_packet,
            num_frames=21,
            prior=prior,
        )
    assert prior.bases == model.inputs == []
