"""Physical CV baselines, angular/PA metrics, rollout and resume identity."""
import json
import sys
import pytest
import torch
from egorecover.codec import MotionCodec
from egorecover.prior import HistoryPrior
from egorecover.prior_cv_residual import (predict, measure, rotation_angle, evaluate_rollout,
                                         PriorSequences, physical_batch, state_map)
from egorecover.evaluation_protocol import file_sha256
from test_prior_two_forward import stats, trajectory, cache
from test_smplx_evaluation import SyntheticLayer


def test_zero_residual_and_causal_baseline():
    c, full = MotionCodec(stats()), trajectory(frames=21)
    history, target = state_map(full, lambda x: x[:, :20]), state_map(full, lambda x: x[:, 20])
    batch = physical_batch(c, history, target, torch.zeros(1, 10))
    p = HistoryPrior(width=16, layers=1, heads=4).eval()
    p.base_mode = 'constant_velocity'
    prediction = predict(p, batch)
    torch.testing.assert_close(prediction, batch['velocity_mu'])
    decoded = c.decode_current(prediction[:, 0], batch['previous_reference'])
    torch.testing.assert_close(decoded.joints, target.joints, atol=1e-5, rtol=1e-5)
    batch['target'].fill_(100)
    torch.testing.assert_close(predict(p, batch), prediction)
    p.base_mode = 'hold'
    torch.testing.assert_close(predict(p, batch), batch['base_mu'])
    p.base_mode = 'invalid'
    with pytest.raises(ValueError):
        predict(p, batch)


def test_cv_rotation_extrapolation_and_angles():
    c, full = MotionCodec(stats()), trajectory(frames=3)
    def rz(angle):
        t = torch.tensor(angle)
        z, o = torch.tensor(0.), torch.tensor(1.)
        return torch.stack((torch.stack((t.cos(), -t.sin(), z)),
                            torch.stack((t.sin(), t.cos(), z)), torch.stack((z, z, o))))
    for i, angle in enumerate((0., .2, .4)):
        full.joints[:, i, :, :3, :3] = rz(angle)
        full.reference[:, i, :3, :3] = rz(angle)
    previous, last = (state_map(full, lambda x: x[:, i]) for i in (0, 1))
    encoded = c.constant_velocity_prior(previous, last)
    pred = c.decode_current(encoded, last.reference)
    torch.testing.assert_close(pred.joints[:, :, :3, :3], full.joints[:, 2, :, :3, :3], atol=1e-5, rtol=1e-5)
    assert rotation_angle(rz(0), rz(torch.pi / 2)).item() == pytest.approx(90, abs=1e-4)
    assert rotation_angle(rz(0), rz(torch.pi)).item() == pytest.approx(180, abs=1e-4)


def test_pa_alignment_and_same_shape():
    c, full = MotionCodec(stats()), trajectory(frames=21)
    history, target = state_map(full, lambda x: x[:, :20]), state_map(full, lambda x: x[:, 20])
    batch = physical_batch(c, history, target, torch.zeros(1, 10))
    values, _, _ = measure(c, batch['target'], batch, target, SyntheticLayer())
    assert values['same_shape_mpjpe_mm'].max() < .01
    assert values['local_rotation_deg'].max() < .01
    shifted = state_map(target, lambda x: x.clone())
    shifted.joints[..., :3, 3] += torch.tensor([1., 2., 3.])
    values, _, _ = measure(c, c.encode_current(shifted, batch['previous_reference'])[:, None], batch, target, SyntheticLayer())
    assert values['fk22_mm'].min() > 3000
    assert values['pa_mpjpe_mm'].max() < .05
    from eval.metrics import reconstruction_error
    points = torch.randn(2, 22, 3).numpy()
    rotated = points @ torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).numpy()
    assert reconstruction_error(2 * rotated + 4, points) < 1e-5


def test_cv_rollout_matches_zero_residual(tmp_path):
    cache(tmp_path / 'data')
    data = PriorSequences(tmp_path / 'data', 'dev')
    p = HistoryPrior(width=16, layers=1, heads=4).eval()
    p.base_mode = 'constant_velocity'
    reference, arrays = evaluate_rollout('constant_velocity', data, MotionCodec(stats()), SyntheticLayer(), 'cpu')
    learned, predictions = evaluate_rollout(p, data, MotionCodec(stats()), SyntheticLayer(), 'cpu')
    assert learned == reference
    assert learned['1000']['mean'] < .1
    torch.testing.assert_close(arrays['joints'][10], predictions['joints'][10])


def test_cpu_training_resume_and_mode_identity(tmp_path, monkeypatch):
    from run import train_prior_cv_residual as runner
    from run.complete_stages import Task, verify_task
    folder, output = tmp_path / 'data', tmp_path / 'run'
    payload, save = cache(folder)
    asset = tmp_path / 'asset'
    asset.mkdir()
    (asset / 'SMPLX_NEUTRAL.npz').write_bytes(b'test asset')
    payload['identity']['smplx_asset_sha256'] = file_sha256(asset / 'SMPLX_NEUTRAL.npz')
    save()
    monkeypatch.setenv('SMPLX_MODEL_PATH', str(asset))
    monkeypatch.setattr(runner, 'get_smpl', SyntheticLayer)
    monkeypatch.setattr(runner, 'HistoryPrior', lambda: HistoryPrior(width=16, layers=1, heads=4))
    monkeypatch.setattr(runner, 'append_log', lambda *a: None)
    argv = ['train', '--data', str(folder), '--output', str(output), '--device', 'cpu',
            '--steps', '2', '--eval-every', '1', '--batch-size', '8', '--base-mode', 'constant_velocity', '--fk-weight', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    report = verify_task(Task('test', [], output, 'prior'))
    assert report['base_mode'] == 'constant_velocity'
    expected = torch.load(output / 'resume.pt', weights_only=True)
    interrupted = tmp_path / 'interrupted'
    changed = [str(interrupted) if a == str(output) else a for a in argv]
    def stop(title, lines):
        if title == '独立 P 选点':
            raise RuntimeError('interrupted')
    monkeypatch.setattr(runner, 'append_log', stop)
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.main()
    monkeypatch.setattr(runner, 'append_log', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', changed + ['--resume'])
    runner.main()
    actual = torch.load(interrupted / 'resume.pt', weights_only=True)
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], atol=0, rtol=0)
    monkeypatch.setattr(sys, 'argv', ['hold' if a == 'constant_velocity' else a for a in argv] + ['--resume'])
    with pytest.raises(ValueError, match='protocol'):
        runner.main()


def test_queue_matrix(tmp_path):
    from run.prior_cv_residual_experiment import tasks_for
    tasks = tasks_for(tmp_path)
    assert len(tasks) == 12
    assert [t.gpu for t in tasks] == [4, 5, 6, 7] * 3
    assert sum('--baselines' in t.argv for t in tasks) == 1


def test_summary_pairs_seeds_and_distinguishes_baseline_gain(tmp_path):
    from run.prior_cv_residual_experiment import tasks_for, collect_results, write_results, CASES
    tasks = tasks_for(tmp_path)
    def value(score):
        return {'mean': score, 'per_take': {'a': score - 1, 'b': score + 1}}
    def result(score):
        return {'single': {k: value(score) for k in ('fk22_mm', 'pa_mpjpe_mm', 'same_shape_mpjpe_mm')},
                'rollout': {str(h): value(score + h / 10) for h in (100, 200, 400, 800, 1000)}}
    for task in tasks:
        case, seed = task.name.rsplit('_s', 1)
        mode, fk = CASES[case]
        score = {'H0_hold_dense': 42., 'V0_cv_dense': 34., 'H1_hold_fk': 40., 'V1_cv_fk': 33.}[case]
        task.output.mkdir(parents=True)
        torch.save({'state_dict': {'weight': torch.tensor(int(seed))}}, task.output / 'initial.pt')
        torch.save({'base_mode': mode}, task.output / 'prior.pt')
        report = {k: 1 for k in ('data_sha256', 'code_sha256', 'steps', 'batch_size', 'eval_every', 'lr',
                                 'weight_decay', 'reference_mode', 'smplx_asset_sha256', 'selection')}
        report.update(base_mode=mode, training_seed=int(seed), selected_step=100,
                      results={'prior': result(score)}, final_single=result(score)['single'])
        if task.name == 'H0_hold_dense_s62':
            report['results'].update(hold=result(50.), constant_velocity=result(36.75))
        task.report.write_text(json.dumps(report))
    summary = collect_results(tasks)
    assert summary['candidate'] == 'V1_cv_fk'
    paired = summary['comparisons']['V0_cv_dense_minus_H0_hold_dense']['fk22_mm']
    assert paired['mean'] == -8
    assert paired['take_bootstrap_95ci'] == [-8., -8.]
    assert summary['comparisons']['V1_cv_fk_minus_constant_velocity']['fk22_mm']['mean'] == -3.75
    write_results(tmp_path, summary)
    assert (tmp_path / 'RESULTS.md').exists()
