import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.history_flow import HistoryFlow
from egorecover.rollout import run_episode
from mydiffusion.flow_matching import FlowMatching


def test_causal_rollout_calls_p_once_samples_one_action_and_ignores_masked_payload():
    codec = MotionCodec(
        {
            "motion_mean": torch.zeros(243),
            "motion_std": torch.ones(243),
            "traj_mean": torch.zeros(18),
            "traj_std": torch.ones(18),
        }
    )
    physical = BodyState(torch.eye(4).expand(22, 4, 4), torch.eye(4), torch.zeros(36))
    neutral = codec.encode_current(physical, torch.eye(4))

    class Initializer(torch.nn.Module):
        def forward(self, x, t, y):
            assert x.shape == (1, 20, 243)
            return neutral.expand_as(x)

    class Current(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, x, t, y):
            self.calls += 1
            assert x.shape == (1, 1, 243)
            assert y["img_mask"].all() and y["traj_mask"].all()
            assert y["history_motion"].shape == (1, 20, 243)
            return y["prior_mu"]

    class Prior(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, h, valid, mu):
            self.calls += 1
            return mu

    predictions = []
    for poison in (False, True):
        requests = []

        def observations(*, as_of, history_frames):
            requests.append((as_of, history_frames))
            count = min(history_frames, as_of + 1)
            packet = {
                "aria_traj_obs": torch.tensor([[1.0, 0, 0, 0, 1, 0, 0, 0, 1.6]]).repeat(count, 1),
                "img_feats": torch.ones(count, 1024),
                "img_available": torch.ones(count, dtype=torch.bool),
                "traj_available": torch.ones(count, dtype=torch.bool),
                "frame_id_30fps": torch.arange(as_of - count + 1, as_of + 1) * 3,
            }
            if poison and as_of >= 20:
                packet["aria_traj_obs"].fill_(float("nan"))
                packet["img_feats"].fill_(float("nan"))
            return packet

        model, prior = Current().eval(), Prior().eval()
        result = run_episode(
            model,
            HistoryFlow(flow=FlowMatching(num_steps=3)),
            Initializer().eval(),
            FlowMatching(num_steps=3),
            codec,
            observations,
            num_frames=25,
            prior=prior,
            action="a00",
        )
        assert requests == [(19, 20), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1)]
        assert prior.calls == 5 and model.calls == 15
        assert result["body_history_source"] == "model_predictions_only"
        assert result["frame_indices"] == list(range(20, 25))
        predictions.append(result["dense_world_joints"])
    assert torch.equal(*predictions)
    cached = run_episode(
        Current().eval(),
        HistoryFlow(flow=FlowMatching(num_steps=3)),
        Initializer().eval(),
        FlowMatching(num_steps=3),
        codec,
        observations,
        num_frames=25,
        prior=Prior().eval(),
        action="a00",
        startup={
            "normalized_motion": result["bootstrap_motion"],
            "initial_reference": result["bootstrap_initial_reference"],
            "beta_boot": result["beta_boot"],
            "floor_estimate_m": result["floor_estimate_m"],
            "sampling_seed": result["bootstrap_sampling_seed"],
            "references": result["bootstrap_references"],
            "world_joints": result["bootstrap_world_joints"],
        },
    )
    torch.testing.assert_close(cached["dense_world_joints"], result["dense_world_joints"])
    prior_model = Current().eval()
    prior_only = run_episode(
        prior_model,
        HistoryFlow(flow=FlowMatching(num_steps=3)),
        Initializer().eval(),
        FlowMatching(num_steps=3),
        codec,
        observations,
        num_frames=25,
        prior=Prior().eval(),
        action="a00",
        prior_only=True,
    )
    assert prior_model.calls == 0
    assert torch.isfinite(prior_only["dense_world_joints"]).all()
    torch.testing.assert_close(prior_only["dense_world_joints"], result["dense_world_joints"])
