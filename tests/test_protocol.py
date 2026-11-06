"""Protocol tests exercise physical state, causality and full feedback gradients."""

import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from egorecover import ModelConfig, ObservationWindow, WindowCache, WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.config import TrainOptions
from egorecover.data import PreparedWindowData, keyed_noise
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.evaluation import evaluate_episode, paired_horizons
from egorecover.experiments import experiment_matrix, interaction
from egorecover.flow import build_source, solve
from egorecover.io import file_sha256
from egorecover.losses import frozen_ema, paired_feedback, training_objective, update_ema
from egorecover.models import HistoryPrior, MotionModel
from egorecover.state import FIELDS, state_map
from egorecover.system import WindowResult


@pytest.fixture(autouse=True, scope="module")
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def episode():
    return synthetic_episode()


def make_system(episode, **kwargs):
    config = WindowConfig(**kwargs)
    return WindowReconstructor(MotionCodec(episode.stats), None, config)


class CenterEndpoint(nn.Module):
    """Simple physical predictor with a measurable full-history dependency."""

    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, tau, y):
        return y["center"] + self.shift


@pytest.mark.parametrize("window", [1, 3, 5, 10])
def test_world_window_roundtrip_and_cache_rewrite(episode, window):
    system = make_system(episode, window=window)
    cache = episode.cache()
    before = state_map(cache.body, lambda x: x.clone())
    y, reference = system.prepare(cache, episode.observations(20))
    assert y["fixed_motion"].shape == (1, 21 - window, 243)
    target = episode.target(20, window)
    restored = system.decode(system.encode(target, reference), reference)
    torch.testing.assert_close(restored.joints, target.joints, atol=2e-6, rtol=1e-5)
    for revise in (False, True):
        updated = cache.advance(restored, revise_recent=revise)
        updated.validate()
        assert updated.frame == 20
        torch.testing.assert_close(updated.body.joints[:, -1], target.joints[:, -1])
        if revise:
            torch.testing.assert_close(updated.body.joints[:, -window:], restored.joints)
        else:
            torch.testing.assert_close(updated.body.joints[:, :-1], before.joints[:, 1:])
    torch.testing.assert_close(cache.body.joints, before.joints, atol=0, rtol=0)


@pytest.mark.parametrize("steps", [1, 2, 4, 10])
def test_euler_endpoint_and_differentiable_steps(steps):
    source = torch.randn(1, 3, 243, requires_grad=True)
    endpoint = torch.full_like(source, 2.0)
    calls = []

    def model(x, tau, y):
        calls.append(tau.clone())
        return x * 0.2 + endpoint

    result = solve(model, source, {}, steps=steps)
    assert len(calls) == steps
    assert calls[0].item() == 1
    if steps == 1:
        torch.testing.assert_close(result, source * 0.2 + endpoint)
    result.sum().backward()
    assert torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0


def test_prediction_and_gaussian_share_center_and_scaled_standard_noise():
    center, epsilon = torch.randn(2, 5, 243), torch.randn(2, 5, 243)
    gaussian = build_source(center, epsilon, mode="gaussian", sigma=0.3)
    prediction = build_source(center, epsilon, mode="prediction", sigma=0.3)
    torch.testing.assert_close(prediction - gaussian, center)


def test_frozen_prior_transmits_input_gradients():
    prior = HistoryPrior(width=24, layers=1, heads=4, dropout=0.5).freeze()
    with torch.no_grad():
        prior.output.weight.normal_(std=0.02)
    history = torch.randn(1, 20, 243, requires_grad=True)
    valid = torch.ones(1, 20, dtype=torch.bool)
    first = prior(history, valid, torch.zeros(1, 1, 243))
    torch.testing.assert_close(first, prior(history, valid, torch.zeros(1, 1, 243)), atol=0, rtol=0)
    first.square().sum().backward()
    assert history.grad.abs().sum() > 0
    assert all(p.grad is None for p in prior.parameters())


@pytest.mark.parametrize("source", ["gaussian", "prediction"])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_feedback_only_current_receives_gradient_and_ema_is_frozen(episode, source, steps):
    system = make_system(episode, source=source, steps=steps)
    cache = episode.cache()
    y, reference = system.prepare(cache, episode.observations(20))
    codes = (y["center"] + 0.02).detach().requires_grad_()
    body = system.decode(codes, reference)
    current = WindowResult(codes, body, cache.advance(body), reference, y["center"])
    truth = state_map(episode.target(20), lambda x: x[:, -1].clone().requires_grad_())
    teacher = frozen_ema(CenterEndpoint())
    loss = paired_feedback(
        system, teacher, current, truth, episode.observations(21), torch.randn(2, 1, 3, 243), DenseJoints()
    )
    assert loss > 0
    loss.backward()
    assert codes.grad[:, -1].abs().sum() > 0
    assert torch.count_nonzero(codes.grad[:, :-1]) == 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(getattr(truth, field).grad is None for field in FIELDS)


def test_identical_current_writes_give_zero_paired_loss(episode):
    system = make_system(episode)
    current = system.reconstruct(CenterEndpoint(), episode.cache(), episode.observations(20), torch.randn(1, 3, 243))
    loss = paired_feedback(
        system,
        frozen_ema(CenterEndpoint()),
        current,
        current.current,
        episode.observations(21),
        torch.randn(2, 1, 3, 243),
        DenseJoints(),
    )
    assert loss.item() == 0


def test_each_branch_recomputes_reference_center_and_sensor_conditions(episode):
    system = make_system(episode)
    current = system.reconstruct(CenterEndpoint(), episode.cache(), episode.observations(20), torch.randn(1, 3, 243))
    a = current.cache.replace_current(current.current)
    b = current.cache.replace_current(state_map(episode.target(20), lambda x: x[:, -1]))
    ya, ra = system.prepare(a, episode.observations(21))
    yb, rb = system.prepare(b, episode.observations(21))
    assert not torch.allclose(ra, rb)
    assert not torch.allclose(ya["center"], yb["center"])
    assert not torch.allclose(ya["traj"], yb["traj"])
    torch.testing.assert_close(ya["images"], yb["images"])


@pytest.mark.parametrize("mode", ["none", "paired", "direct", "feature", "geometry"])
def test_real_window_network_joint_training_backward(episode, mode):
    torch.manual_seed(62)
    system = make_system(episode, feedback=mode)
    model = MotionModel(system.config, ModelConfig.compact())
    ema = frozen_ema(model)
    result = training_objective(
        system,
        model,
        ema,
        episode.cache(),
        episode.observations(20),
        episode.target(20, 3),
        torch.randn(1, 3, 243),
        torch.tensor([0.4]),
        DenseJoints(),
        next_observations=episode.observations(21),
        next_epsilon=torch.randn(1, 1, 3, 243),
        truth_next=state_map(episode.target(21), lambda x: x[:, -1]),
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.output_process.weight.grad.abs().sum() > 0
    assert model.mu_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ema.parameters())
    before = ema.output_process.weight.clone()
    with torch.no_grad():
        model.output_process.weight.add_(0.1)
    update_ema(ema, model, 0.9)
    torch.testing.assert_close(ema.output_process.weight, before + 0.01)


def test_no_body_feedback_control_is_independent_of_cache(episode):
    system = make_system(episode, body_feedback=False)
    model = MotionModel(system.config, ModelConfig.compact()).eval()
    cache = episode.cache()
    changed_body = state_map(cache.body, lambda x: x.clone())
    changed_body.joints[..., :3, 3] += 100
    changed_body.reference[..., :2, 3] -= 50
    changed = WindowCache(changed_body, cache.beta, cache.anchor, cache.frame)
    obs, epsilon = episode.observations(20), torch.randn(1, 3, 243)
    first = system.online(model, cache, obs, epsilon)
    second = system.online(model, changed, obs, epsilon)
    torch.testing.assert_close(first.body.joints, second.body.joints, atol=0, rtol=0)
    row = evaluate_episode(system, model, episode, DenseJoints())
    assert row["center_mm"] is None


def test_future_observations_and_labels_cannot_affect_current_output(episode):
    system = make_system(episode)
    model = CenterEndpoint().eval()
    epsilon = torch.randn(1, 3, 243)
    first = system.online(model, episode.cache(), episode.observations(20), epsilon)
    episode.head[21:] = float("nan")
    episode.images[21:] = float("nan")
    episode.truth.joints[:] = float("nan")
    second = system.online(model, episode.cache(), episode.observations(20), epsilon)
    torch.testing.assert_close(first.codes, second.codes, atol=0, rtol=0)
    with pytest.raises(ValueError, match="physical frame"):
        system.prepare(episode.cache(), replace(episode.observations(20), frame=21))


def test_missing_sensors_are_sanitized_before_geometry_and_projection(episode):
    system = make_system(episode)
    model = MotionModel(system.config, ModelConfig.compact()).eval()
    obs = episode.observations(20)
    obs = ObservationWindow(
        torch.full_like(obs.head, float("nan")),
        torch.full_like(obs.images, float("nan")),
        torch.zeros_like(obs.head_valid),
        torch.zeros_like(obs.image_valid),
        obs.frame,
    )
    result = system.online(model, episode.cache(), obs, torch.randn(1, 3, 243))
    assert torch.isfinite(result.codes).all()


def test_paired_multistep_diagnostic_and_verified_timebase(episode):
    system = make_system(episode)
    values = paired_horizons(system, CenterEndpoint().eval(), episode, DenseJoints())
    assert set(values) == {"1", "2", "4", "10"}
    assert values["10"]["seconds"] == pytest.approx(1.0)
    assert values["1"]["seconds"] == pytest.approx(0.1)


def test_noise_keeps_current_slot_matched_across_windows():
    a = keyed_noise("a", 62, 20, 1, "cpu")
    b = keyed_noise("a", 62, 20, 10, "cpu")
    torch.testing.assert_close(a[:, -1], b[:, -1], atol=0, rtol=0)


def write_prepared(root, episode, *, split="train", take="one", identifier="one", floor=1.25):
    record = {
        "group": split,
        "base_take_name": take,
        "episode_id": identifier,
        "base_seq_name": f"{take}___0___{(len(episode)-1)*3}",
        "episode_start_motion_idx": 0,
        "num_frames": len(episode),
        "floor_height": floor,
    }
    directory = root / "data/episodes"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = root / "data/manifest.json"
    manifest.write_text(json.dumps({"records": [record]}))
    packet = {
        "record": record,
        "stats": episode.stats,
        "truth": {k: getattr(episode.truth, k) for k in FIELDS},
        "teacher": {k: getattr(episode.teacher, k) for k in FIELDS},
        "beta_boot": episode.beta,
        "head": episode.head,
        "images": episode.images,
    }
    path = directory / f"{identifier}.pt"
    torch.save(packet, path)
    path.with_suffix(".json").write_text(
        json.dumps({"completed": True, "manifest_sha256": file_sha256(manifest), "sha256": file_sha256(path)})
    )
    return path, record


def test_prepared_adapter_checks_identity_timing_and_corruption(episode, tmp_path):
    path, _ = write_prepared(tmp_path, episode)
    data = PreparedWindowData(tmp_path, "train")
    torch.testing.assert_close(data.load(0).teacher.joints, episode.teacher.joints)
    world = data.load(0).restore_world(episode.truth.joints[..., :3, 3])
    torch.testing.assert_close(world[..., :2], episode.truth.joints[..., :2, 3])
    torch.testing.assert_close(world[..., 2], episode.truth.joints[..., 2, 3] + 1.25)
    assert data.load(0).timestamps[21] - data.load(0).timestamps[20] == pytest.approx(0.1)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        PreparedWindowData(tmp_path, "train")


def test_take_leakage_rejected_before_reading_packets(episode, tmp_path):
    _, record = write_prepared(tmp_path, episode)
    path = tmp_path / "data/manifest.json"
    path.write_text(json.dumps({"records": [record, {**record, "group": "dev", "episode_id": "two"}]}))
    with pytest.raises(ValueError, match="leakage"):
        PreparedWindowData(tmp_path, "train")


@pytest.mark.parametrize("floor", [None, float("nan"), True])
def test_missing_or_invalid_floor_is_never_assumed_zero(episode, tmp_path, floor):
    write_prepared(tmp_path, episode, floor=floor)
    with pytest.raises(ValueError):
        PreparedWindowData(tmp_path, "train")


def test_experiment_matrix_matches_embedded_tables():
    matrix = {r["name"]: r for r in experiment_matrix()}
    assert {matrix[f"architecture/W{w}"]["protocol"]["window"] for w in (1, 3, 5, 10)} == {1, 3, 5, 10}
    for label, (source, feedback) in zip(
        "ABCD", [("gaussian", "none"), ("prediction", "none"), ("gaussian", "paired"), ("prediction", "paired")]
    ):
        cfg = matrix[f"combination/{label}"]["protocol"]
        assert (cfg["source"], cfg["feedback"]) == (source, feedback)
    assert matrix["feedback/paired"]["initialize_from"] == matrix["feedback/direct"]["initialize_from"]
    assert matrix["feedback/paired"]["initialize_from"] == matrix["source/gaussian"]["initialize_from"]
    assert matrix["combination/D"]["initialize_from"] == matrix["source/prediction"]["initialize_from"]
    assert interaction({"A": 100, "B": 90, "C": 80, "D": 65})["interaction"] == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 2},
        {"history": 21},
        {"steps": 3},
        {"sigma": 0},
        {"feedback": "bad"},
        {"source": "prediction", "body_feedback": False},
    ],
)
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        WindowConfig(**kwargs)


def test_training_resume_is_exact_and_feedback_teacher_stays_fixed(tmp_path):
    from egorecover.training import train

    config = tmp_path / "paired.json"
    config.write_text(json.dumps(WindowConfig(feedback="paired").to_dict()))
    args = TrainOptions(
        config=config,
        device="cpu",
        synthetic=True,
        compact=True,
        data_root=None,
        prior_checkpoint=None,
        e7_checkpoint=None,
        initialize=None,
        seed=62,
        train_steps=2,
        batch_size=1,
        learning_rate=3e-5,
        rollin_frames=1,
        eval_every=1,
        dev_limit=None,
        stop_after=None,
        resume=False,
        output=tmp_path / "full",
    )
    train(args)
    args.output, args.stop_after = tmp_path / "resumed", 1
    train(args)
    args.resume, args.stop_after = True, None
    train(args)
    full = torch.load(tmp_path / "full/resume.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/resume.pt", weights_only=True)
    for group in ("model", "ema", "feedback_model"):
        for key in full[group]:
            torch.testing.assert_close(full[group][key], resumed[group][key], atol=0, rtol=0)
    torch.testing.assert_close(full["sample_rng"], resumed["sample_rng"], atol=0, rtol=0)
    torch.manual_seed(62)
    initial = MotionModel(WindowConfig(feedback="paired"), ModelConfig.compact())
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(full["feedback_model"][key], value, atol=0, rtol=0)
    assert not torch.equal(full["model"]["output_process.weight"], full["feedback_model"]["output_process.weight"])


def test_full_e7_checkpoint_initializes_active_window(episode):
    from egorecover.checkpoint import LEGACY_TEXT, load_e7_weights

    torch.manual_seed(62)
    system = make_system(episode, window=5)
    model = MotionModel(system.config).eval()
    source = {key: value.clone() for key, value in model.state_dict().items() if key not in model.NEW_PARAMETER_NAMES}
    source.update(
        {
            "embed_text_cond.weight": torch.zeros(768, 768),
            "embed_text_cond.bias": torch.zeros(768),
            "mask_tokens.text": torch.zeros(768),
        }
    )
    report = load_e7_weights(model, source, weight_source="model")
    assert set(report.new_parameters) == model.NEW_PARAMETER_NAMES
    assert set(report.ignored_parameters) == LEGACY_TEXT
    torch.testing.assert_close(model.output_process.weight, source["output_process.weight"], atol=0, rtol=0)
    conditions, _ = system.prepare(episode.cache(), episode.observations(20))
    with torch.no_grad():
        result = model(torch.randn(1, 5, 243), torch.ones(1), conditions)
    assert result.shape == (1, 5, 243) and torch.isfinite(result).all()
