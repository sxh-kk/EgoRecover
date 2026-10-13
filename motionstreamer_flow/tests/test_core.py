from dataclasses import replace

import pytest
import torch

from motionstreamer_flow.data import Episodes, Normalizer, online_inputs
from motionstreamer_flow.demo import make_dataset, synthetic_offsets, tiny_config
from motionstreamer_flow.export_egorecover import body_to_pose_root, convert_packet
from motionstreamer_flow.geometry import (
    axis_angle_to_matrix,
    fk,
    head_features,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    to_world,
)
from motionstreamer_flow.model import CausalTAE, MotionFlow, euler_sample, windows
from motionstreamer_flow.objective import replacement_mask, temporarily_eval, two_forward_loss
from motionstreamer_flow.streaming import StreamingSession, rollout


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.fixture
def setup(tmp_path):
    cfg = tiny_config()
    episodes = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    norm = Normalizer(cfg)
    for key, mask in (("pose", "valid"), ("root", "valid"), ("head", "head_valid")):
        norm.fit(key, (episodes.load(r)[key][episodes.load(r)[mask]] for r in episodes.split("train")))
    tae, flow = CausalTAE(cfg), MotionFlow(cfg)
    # Give the conditioner a nonzero gradient path rather than testing a freshly zero-initialized head only.
    torch.nn.init.normal_(flow.output[-1].weight, std=0.02)
    for block in flow.blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.01)
    return cfg, episodes, norm, tae, flow


def arguments(batch, cfg):
    state = torch.randn(*batch["pose"].shape[:2], cfg.state_dim)
    return [
        state,
        batch["images"],
        batch["head"],
        batch["image_valid"],
        batch["head_valid"],
        batch["valid"],
        batch["beta"],
    ]


def test_tae_stride_one_and_future_invariance(setup):
    cfg, episodes, norm, tae, _ = setup
    p = episodes.load(episodes.split("train")[0])
    x = norm.normalize("pose", p["pose"])[None]
    mean, _ = tae.encode(x, p["beta"][None])
    assert mean.shape == (1, len(x[0]), cfg.latent_dim)
    changed = x.clone()
    changed[:, 10:] += 50
    after, _ = tae.encode(changed, p["beta"][None])
    torch.testing.assert_close(mean[:, :10], after[:, :10], rtol=0, atol=0)
    decoded = tae.decode(mean, p["beta"][None])
    changed_latent = mean.clone()
    changed_latent[:, 10:] += 50
    after_decode = tae.decode(changed_latent, p["beta"][None])
    torch.testing.assert_close(decoded[:, :10], after_decode[:, :10], rtol=0, atol=0)


def test_tae_padding_equals_real_causal_start(setup):
    cfg, _, _, tae, _ = setup
    # Trainable affine norm biases must not turn virtual padding into history.
    for module in tae.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.bias, std=0.1)
    x, beta = torch.randn(2, 12, 152), torch.randn(2, 10)
    normal, _ = tae.encode(x, beta)
    padded = torch.cat((torch.zeros(2, 7, 152), x), 1)
    mask = torch.cat((torch.zeros(2, 7, dtype=torch.bool), torch.ones(2, 12, dtype=torch.bool)), 1)
    result, _ = tae.encode(padded, beta, mask)
    torch.testing.assert_close(normal, result[:, 7:], atol=1e-6, rtol=1e-5)
    z = torch.randn(2, 12, cfg.latent_dim)
    recon = tae.decode(z, beta)
    result = tae.decode(torch.cat((torch.zeros(2, 7, cfg.latent_dim), z), 1), beta, mask)
    torch.testing.assert_close(recon, result[:, 7:], atol=1e-6, rtol=1e-5)


def test_causal_conditioner_excludes_current_body_and_future_observations(setup):
    cfg, episodes, _, _, flow = setup
    batch = episodes.sample(torch.Generator().manual_seed(7))
    args = arguments(batch, cfg)
    expected = flow.conditioner(*args)
    args[0] = args[0].clone()
    args[0][:, 5:] += 100
    args[1] = args[1].clone()
    args[1][:, 6:] += 100
    args[2] = args[2].clone()
    args[2][:, 6:] += 100
    actual = flow.conditioner(*args)
    torch.testing.assert_close(expected[:, :6], actual[:, :6], atol=0, rtol=0)
    assert not torch.allclose(expected[:, 6:], actual[:, 6:])


def test_history_window_matches_streaming_and_does_not_expand_with_layers(setup):
    cfg, episodes, _, _, flow = setup
    batch = episodes.sample(torch.Generator().manual_seed(2))
    args = arguments(batch, cfg)
    expected = flow.conditioner(*args)[:, -1:]
    cropped = [x[:, -(cfg.history + 1) :] if x.ndim >= 2 and x.shape[1] == len(batch["pose"][0]) else x for x in args]
    actual = flow.conditioner(*cropped, last_only=True)
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
    args[0][:, : -(cfg.history + 1)] += 100
    torch.testing.assert_close(expected, flow.conditioner(*args, last_only=True), atol=1e-6, rtol=1e-5)


def test_actions_gate_entire_window_and_mask_nan(setup):
    cfg, episodes, _, _, flow = setup
    args = arguments(episodes.sample(torch.Generator().manual_seed(7)), cfg)
    actions = torch.ones(2, 8, 2, dtype=torch.bool)
    actions[:, -1] = False
    a = flow.conditioner(*args, actions, last_only=True)
    args[1] = torch.full_like(args[1], float("nan"))
    args[2] = torch.full_like(args[2], float("nan"))
    b = flow.conditioner(*args, actions, last_only=True)
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    args[3].zero_()
    args[4].zero_()
    assert torch.isfinite(flow.conditioner(*args)).all()


def test_qk_scale_gets_gradient(setup):
    cfg, episodes, _, _, flow = setup
    args = arguments(episodes.sample(torch.Generator().manual_seed(3)), cfg)
    flow.conditioner(*args).square().mean().backward()
    gradient = flow.conditioner.history_blocks[0].attention.log_scale.grad
    assert gradient is not None and gradient.abs() > 0


@pytest.mark.parametrize("steps", (1, 3, 10))
def test_x0_euler_integrates_constant_endpoint_exactly(steps):
    target, noise = torch.randn(2, 17), torch.randn(2, 17)
    called_times = []

    def endpoint(x, tau, condition):
        called_times.append(tau.clone())
        return condition

    actual = euler_sample(endpoint, target, noise, steps)
    torch.testing.assert_close(target, actual, atol=1e-6, rtol=1e-5)
    assert len(called_times) == steps and all((time > 0).all() for time in called_times)


def test_rotations_fk_gradient_and_fixed_anchor():
    axis = (torch.randn(2, 21, 3) * 0.05).requires_grad_()
    pose = torch.cat((matrix_to_rotation_6d(axis_angle_to_matrix(axis)).flatten(1), torch.zeros(2, 26)), -1)
    root = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 1.0]).expand(2, -1).clone().requires_grad_()
    joints, rotations = fk(pose, root, synthetic_offsets())
    joints[:, 15].square().sum().backward()
    assert axis.grad.abs().sum() > 0 and root.grad[:, 6:].abs().sum() > 0
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2, 22), atol=1e-5, rtol=1e-5)
    zero_rotation = rotation_6d_to_matrix(torch.zeros(3, 6))
    torch.testing.assert_close(torch.linalg.det(zero_rotation), torch.ones(3))
    anchor = torch.eye(4)
    anchor[:3, :3] = axis_angle_to_matrix(torch.tensor([0.0, 0, 0.5]))
    anchor[:3, 3] = torch.tensor([1.0, 2, 0])
    restored = (to_world(joints, anchor) - anchor[:3, 3]) @ anchor[:3, :3]
    torch.testing.assert_close(joints, restored)


def test_head_features_are_backward_causal_and_handle_gaps():
    transforms = torch.eye(4).expand(10, 4, 4).clone()
    transforms[:, 0, 3] = torch.arange(10)
    valid = torch.ones(10, dtype=torch.bool)
    a = head_features(transforms, torch.eye(4), valid)
    transforms[5:, 0, 3] += 50
    b = head_features(transforms, torch.eye(4), valid)
    torch.testing.assert_close(a[:5], b[:5])
    torch.testing.assert_close(a[1:, 15], torch.ones(9))
    valid[3] = False
    transforms[3] = float("nan")
    c = head_features(transforms, torch.eye(4), valid)
    assert torch.isfinite(c).all()
    torch.testing.assert_close(c[4, 9:], torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 0]))


def test_two_forward_schedule_startup_and_invalid_frames():
    valid = torch.ones(3, 12, dtype=torch.bool)
    valid[:, -1] = False
    assert not replacement_mask(valid, 4, 0).any()
    selected = replacement_mask(valid, 4, 0.5)
    assert not selected[:, :4].any() and not selected[:, -1].any()
    assert torch.equal(selected[0], selected[1])
    selected = replacement_mask(valid, 4, 1)
    torch.testing.assert_close(selected[:, 4:], valid[:, 4:])


def test_first_pass_detached_second_pass_gradients_and_modes_restored(setup):
    cfg, episodes, norm, tae, flow = setup
    tae.eval().requires_grad_(False)
    batch = episodes.sample(torch.Generator().manual_seed(7))
    seen = []
    handle = flow.register_forward_hook(
        lambda module, args, result: seen.append((torch.is_grad_enabled(), module.training, result))
    )
    loss, stats = two_forward_loss(flow, tae, norm, batch, cfg, cfg.total_steps)
    handle.remove()
    assert [(grad, mode) for grad, mode, _ in seen] == [(False, False), (True, True)]
    assert not seen[0][2].requires_grad and seen[1][2].requires_grad
    assert stats["replaced_fraction"] == 1
    loss.backward()
    assert all(parameter.grad is None for parameter in tae.parameters())
    assert flow.output[-1].weight.grad.abs().sum() > 0
    assert flow.conditioner.image[0].weight.grad.abs().sum() > 0
    assert flow.conditioner.head[0].weight.grad.abs().sum() > 0
    assert flow.training


def test_frozen_decoder_propagates_geometry_gradient(setup):
    cfg, episodes, norm, tae, _ = setup
    tae.requires_grad_(False)
    batch = episodes.sample(torch.Generator().manual_seed(4))
    latent = torch.randn(2, 8, cfg.latent_dim, requires_grad=True)
    states = torch.randn(2, 8, cfg.state_dim)
    decoded = tae.decode_queries(states, latent, batch["beta"], batch["valid"])
    joints, _ = fk(norm.restore("pose", decoded), batch["root"], batch["offsets"])
    joints.square().mean().backward()
    assert latent.grad.abs().sum() > 0
    assert all(p.grad is None for p in tae.parameters())


def test_two_forward_decoder_uses_second_pass_mixed_history_and_original_targets(setup, monkeypatch):
    cfg, episodes, norm, tae, flow = setup
    tae.eval().requires_grad_(False)
    batch = episodes.sample(torch.Generator().manual_seed(10))
    histories, predictions, decoder_contexts = [], [], []

    def record_forward(module, args, output):
        histories.append(args[2].detach().clone())
        predictions.append(output.detach().clone())

    handle = flow.register_forward_hook(record_forward)
    original_decode = tae.decode_queries

    def record_decode(past, latent, beta, valid):
        decoder_contexts.append(past.detach().clone())
        return original_decode(past, latent, beta, valid)

    monkeypatch.setattr(tae, "decode_queries", record_decode)
    _, stats = two_forward_loss(flow, tae, norm, batch, cfg, cfg.total_steps)
    handle.remove()
    torch.testing.assert_close(histories[0][:, : cfg.bootstrap], histories[1][:, : cfg.bootstrap])
    expected_mixed = norm.project_state(predictions[0], cfg.latent_dim)
    torch.testing.assert_close(histories[1][:, cfg.bootstrap :], expected_mixed[:, cfg.bootstrap :])
    torch.testing.assert_close(
        decoder_contexts[0][..., : cfg.latent_dim], norm.restore("latent", histories[1][..., : cfg.latent_dim])
    )
    target, _ = tae.encode(norm.normalize("pose", batch["pose"]), batch["beta"], batch["valid"])
    expected_latent_loss = (
        (predictions[1][:, cfg.bootstrap :, : cfg.latent_dim] - norm.normalize("latent", target)[:, cfg.bootstrap :])
        .square()
        .mean()
    )
    torch.testing.assert_close(stats["latent"], expected_latent_loss)


def test_two_forward_can_be_disabled_for_ablation(setup):
    cfg, episodes, norm, tae, flow = setup
    cfg = replace(cfg, two_forward=False)
    tae.eval().requires_grad_(False)
    calls = []
    handle = flow.register_forward_hook(lambda module, args, output: calls.append(torch.is_grad_enabled()))
    loss, stats = two_forward_loss(
        flow, tae, norm, episodes.sample(torch.Generator().manual_seed(1)), cfg, cfg.total_steps
    )
    handle.remove()
    assert calls == [True] and stats["gamma"] == 0 and stats["replaced_fraction"] == 0
    loss.backward()


def test_eval_context_restores_nested_modes_after_error(setup):
    flow = setup[-1]
    flow.train()
    flow.conditioner.image.eval()
    with pytest.raises(RuntimeError):
        with temporarily_eval(flow):
            assert not flow.training
            raise RuntimeError("intentional")
    assert flow.training and not flow.conditioner.image.training


def test_rollout_ignores_truth_and_failed_step_is_atomic(setup):
    cfg, episodes, norm, tae, flow = setup
    p = episodes.load(episodes.split("test")[0])
    first = rollout(flow, tae, norm, online_inputs(p), seed=123)
    p["pose"][:] += 50
    p["root"][:] += 50
    p["joints"][:] += 50
    second = rollout(flow, tae, norm, online_inputs(p), seed=123)
    torch.testing.assert_close(first["joints"], second["joints"], rtol=0, atol=0)
    inputs = online_inputs(p)
    inputs = {
        key: value[: cfg.bootstrap] if key in ("images", "head", "image_valid", "head_valid") else value
        for key, value in inputs.items()
    }
    session = StreamingSession(flow, tae, norm, **inputs)
    history, frames = session.states.clone(), session.frames
    with pytest.raises(ValueError, match="observation"):
        session.step(torch.full((cfg.image_dim,), float("nan")), torch.zeros(18))
    torch.testing.assert_close(session.states, history)
    assert session.frames == frames
    with pytest.raises(FloatingPointError):
        session.step(torch.zeros(cfg.image_dim), torch.zeros(18), noise=torch.full((cfg.state_dim,), float("nan")))
    torch.testing.assert_close(session.states, history)
    assert session.frames == frames


def test_converter_uses_predicted_anchor_not_gt_reference(setup):
    cfg, episodes, _, _, _ = setup
    p = episodes.load(episodes.split("train")[0])
    joints, rotations = fk(p["pose"], p["root"], p["offsets"])
    transforms = torch.eye(4).expand(len(joints), 22, 4, 4).clone()
    transforms[:, :, :3, :3], transforms[:, :, :3, 3] = rotations, joints
    body = {
        "joints": transforms,
        "reference": torch.eye(4).expand(len(joints), 4, 4).clone(),
        "auxiliary": torch.cat((p["pose"][:, 126:], torch.zeros(len(joints), 10)), -1),
    }
    packet = {
        "truth": {**body, "reference": torch.full_like(body["reference"], float("nan"))},
        "teacher": body,
        "beta_boot": p["beta"],
        "images": torch.nan_to_num(p["images"]),
        "head": p["root"],
        "record": {
            "floor_height": 0.0,
            "base_seq_name": f"take___0___{3*(len(p['pose'])-1)}",
            "episode_start_motion_idx": 0,
            "num_frames": len(p["pose"]),
        },
    }
    converted = convert_packet(packet, cfg, p["offsets"])
    torch.testing.assert_close(converted["pose"], p["pose"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(converted["root"], p["root"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(converted["joints"], p["joints"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable; CPU checks cannot establish GPU validity"
)
def test_cuda_bf16_objective(setup):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("This GPU lacks bf16 support")
    cfg, episodes, norm, tae, flow = setup
    tae, flow, norm = tae.cuda().requires_grad_(False), flow.cuda(), norm.cuda()
    batch = {key: value.cuda() for key, value in episodes.sample(torch.Generator().manual_seed(7)).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = two_forward_loss(flow, tae, norm, batch, cfg, cfg.total_steps)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flow.output[-1].weight.grad).all()
