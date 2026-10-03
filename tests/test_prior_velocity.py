"""Velocity feature physics, causal deployment, matched initialization and resume."""
import json
import math
import sys
import pytest
import torch
from egorecover.codec import MotionCodec, planar_reference, rigid_transform
from egorecover.prior_velocity import create_prior, history_velocities, PARENTS
from egorecover.prior_cv_residual import predict, physical_batch, state_map
from egorecover.evaluation_protocol import file_sha256
from test_prior_two_forward import stats, trajectory, cache
from test_smplx_evaluation import SyntheticLayer
from utils.rotation_conversions import axis_angle_to_matrix


def encode(c, states):
    return c.encode_history(states, planar_reference(states.reference[:, 0]))


def features(c, states):
    x = encode(c, states)
    return history_velocities(c, x, torch.ones(x.shape[:2], dtype=torch.bool))


def test_physical_velocity_units_and_first_frame():
    c, body = MotionCodec(stats()), trajectory(frames=20)
    f = features(c, body)
    assert f.shape == (1, 20, 70)
    assert torch.count_nonzero(f[:, 0]) == 0
    torch.testing.assert_close(f[:, 1:, 0], torch.full((1, 19), .3), atol=1e-5, rtol=1e-5)
    assert f[:, 1:, 1:69].abs().max() < 1e-5
    assert bool((f[:, 1:, 69] == 1).all())


def test_root_rotation_and_local_articulation_are_separate():
    c, body = MotionCodec(stats()), trajectory(frames=20)
    angles = torch.arange(20) * .01
    root = axis_angle_to_matrix(torch.stack((angles * 0, angles * 0, angles), -1))
    body.joints[..., :3, :3] = root[None, :, None]
    f = features(c, body)
    torch.testing.assert_close(f[:, 1:, 5], torch.full((1, 19), .1 / math.pi), atol=1e-5, rtol=1e-5)
    assert f[:, 1:, 6:69].abs().max() < 1e-5
    # Joint21 is a leaf: varying it changes only its own local angular velocity.
    flex = axis_angle_to_matrix(torch.stack((angles * 2, angles * 0, angles * 0), -1))
    body.joints[:, :, 21, :3, :3] = root @ flex
    f = features(c, body)
    torch.testing.assert_close(f[:, 1:, 66], torch.full((1, 19), .2 / math.pi), atol=1e-5, rtol=1e-5)


def test_reference_changes_global_transform_and_causality():
    c, body = MotionCodec(stats()), trajectory(frames=20)
    original = features(c, body)
    # Changing the internal reference leaves physical velocities unchanged.
    angle = torch.arange(20) * .03
    body.reference[..., :3, :3] = axis_angle_to_matrix(torch.stack((angle * 0, angle * 0, angle), -1))
    torch.testing.assert_close(features(c, body), original, atol=3e-5, rtol=1e-5)
    rotation = axis_angle_to_matrix(torch.tensor([0., 0., .8]))
    transform = rigid_transform(rotation, torch.tensor([3., -2., 1.]))
    moved = type(body)(transform @ body.joints, transform @ body.reference, body.auxiliary)
    torch.testing.assert_close(features(c, moved), original, atol=3e-5, rtol=1e-5)
    body.joints[:, 15:, :, 0, 3] += 3
    changed = features(c, body)
    torch.testing.assert_close(changed[:, :15], original[:, :15], atol=3e-5, rtol=1e-5)
    assert changed[:, 15, 0] > 20


def test_shared_initialization_rng_and_learnable_branch():
    c, body = MotionCodec(stats()), trajectory(frames=21)
    h, t = state_map(body, lambda x: x[:, :20]), state_map(body, lambda x: x[:, 20])
    batch = physical_batch(c, h, t, torch.zeros(1, 10))
    torch.manual_seed(62)
    control = create_prior(stats(), False, width=16, layers=1, heads=4)
    rng = torch.get_rng_state()
    torch.manual_seed(62)
    velocity = create_prior(stats(), True, width=16, layers=1, heads=4)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in control.state_dict().items():
        torch.testing.assert_close(velocity.state_dict()[k], v, atol=0, rtol=0)
    velocity.eval()
    torch.testing.assert_close(predict(velocity, batch), batch['velocity_mu'], atol=0, rtol=0)
    optimizer = torch.optim.AdamW(velocity.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (predict(velocity, batch) - (batch['target'] + .1)).square().mean()
        loss.backward()
        optimizer.step()
    assert velocity.velocity_input.weight.abs().sum() > 0
    assert torch.isfinite(velocity.velocity_input.weight.grad).all()


def test_invalid_history_rejected():
    c, body = MotionCodec(stats()), trajectory(frames=20)
    x = encode(c, body)
    mask = torch.ones(1, 20, dtype=torch.bool)
    mask[:, 5] = False
    with pytest.raises(ValueError, match='complete'):
        history_velocities(c, x, mask)


def test_velocity_training_exact_resume(tmp_path, monkeypatch):
    from run import train_prior_velocity as runner
    from run.complete_stages import Task, verify_task
    folder, output = tmp_path / 'data', tmp_path / 'run'
    payload, save = cache(folder)
    asset = tmp_path / 'asset'
    asset.mkdir()
    (asset / 'SMPLX_NEUTRAL.npz').write_bytes(b'asset')
    payload['identity']['smplx_asset_sha256'] = file_sha256(asset / 'SMPLX_NEUTRAL.npz')
    save()
    monkeypatch.setenv('SMPLX_MODEL_PATH', str(asset))
    monkeypatch.setattr(runner, 'get_smpl', SyntheticLayer)
    monkeypatch.setattr(runner, 'create_prior', lambda *args: create_prior(*args, width=16, layers=1, heads=4))
    monkeypatch.setattr(runner, 'append_log', lambda *args: None)
    argv = ['train', '--data', str(folder), '--output', str(output), '--device', 'cpu',
            '--steps', '2', '--eval-every', '1', '--batch-size', '8', '--velocity-input']
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    assert verify_task(Task('test', [], output, 'prior'))['velocity_input'] is True
    expected = torch.load(output / 'resume.pt', weights_only=True)
    interrupted = tmp_path / 'interrupted'
    changed = [str(interrupted) if x == str(output) else x for x in argv]
    def stop(title, lines):
        if title == '独立 P 选点':
            raise RuntimeError('interrupted')
    monkeypatch.setattr(runner, 'append_log', stop)
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.main()
    monkeypatch.setattr(runner, 'append_log', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', changed + ['--resume'])
    runner.main()
    actual = torch.load(interrupted / 'resume.pt', weights_only=True)
    for k, v in expected['model'].items():
        torch.testing.assert_close(actual['model'][k], v, atol=0, rtol=0)
    assert expected['curve'] == actual['curve']
    monkeypatch.setattr(sys, 'argv', [x for x in argv if x != '--velocity-input'] + ['--resume'])
    with pytest.raises(ValueError, match='protocol'):
        runner.main()


def test_queue_and_summary(tmp_path):
    from run.prior_velocity_experiment import tasks_for, collect_results, write_results
    tasks = tasks_for(tmp_path)
    assert len(tasks) == 6
    assert [t.gpu for t in tasks] == [4, 5, 6, 7, 4, 5]
    def result(score):
        v = {'mean': score, 'per_take': {'a': score - 1, 'b': score + 1}}
        return {'single': {k: v for k in ['fk22_mm', 'pa_mpjpe_mm', 'same_shape_mpjpe_mm']},
                'rollout': {'1000': v}}
    for task in tasks:
        case, seed = task.name.rsplit('_s', 1)
        enabled = case == 'velocity'
        task.output.mkdir(parents=True)
        state = {'weight': torch.tensor(int(seed))}
        if enabled:
            state['velocity_input.weight'] = torch.zeros(2)
        torch.save({'state_dict': state}, task.output / 'initial.pt')
        torch.save({'velocity_input': enabled, 'base_mode': 'constant_velocity', 'velocity_schema': 'test'}, task.output / 'prior.pt')
        report = {k: 1 for k in ['data_sha256', 'code_sha256', 'steps', 'batch_size', 'eval_every', 'lr',
                                'weight_decay', 'fk_weight', 'geometry_weight', 'reference_mode', 'selection']}
        report.update(velocity_input=enabled, base_mode='constant_velocity', velocity_schema='test',
                      training_seed=int(seed), selected_step=100, compute={}, final_single={},
                      results={'prior': result(34 if enabled else 36)})
        if task.name == 'control_s62':
            report['results'].update(constant_velocity=result(36.75), hold=result(50))
        task.report.write_text(json.dumps(report))
    summary = collect_results(tasks)
    assert summary['comparisons']['control']['fk22_mm']['mean'] == -2
    assert summary['comparisons']['control']['fk22_mm']['take_bootstrap_95ci'] == [-2, -2]
    write_results(tmp_path, summary)
    assert (tmp_path / 'RESULTS.md').exists()
