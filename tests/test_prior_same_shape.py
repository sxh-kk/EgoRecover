"""Same-shape supervision isolates shape, preserves scoring, and resumes exactly."""
import json
import sys
import pytest
import torch
from egorecover.codec import MotionCodec
from egorecover.prior import HistoryPrior
from egorecover.prior_same_shape import physical_objective, same_shape_position_errors
from egorecover.losses import physical_objective as original_objective, fk_position_errors
from egorecover.prior_cv_residual import physical_batch, state_map, predict
from egorecover.evaluation_protocol import file_sha256
from test_prior_two_forward import stats, trajectory, cache
from test_smplx_evaluation import SyntheticLayer


class ShapeLayer(SyntheticLayer):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        result.joints[:, 1:] += kwargs['betas'][:, None, 3:6]
        return result


def example():
    codec, body = MotionCodec(stats()), trajectory(frames=21)
    history, target = state_map(body, lambda x: x[:, :20]), state_map(body, lambda x: x[:, 20])
    beta = torch.zeros(1, 10);beta[:, 3] = .1
    return codec, physical_batch(codec, history, target, beta)


def test_same_pose_zero_despite_wrong_shape_and_original_metric_unchanged():
    c, batch = example();smpl = ShapeLayer()
    before = batch['target_joints'].clone()
    prediction = batch['target'].clone()
    original = fk_position_errors(c, prediction, batch, smpl)
    assert original.norm(dim=-1).mean() > .09
    assert same_shape_position_errors(c, prediction, batch, smpl).abs().max() < 1e-6
    torch.testing.assert_close(batch['target_joints'], before, atol=0, rtol=0)
    torch.testing.assert_close(fk_position_errors(c, prediction, batch, smpl), original, atol=0, rtol=0)


def test_gt_beta_is_ignored_and_target_has_no_gradient():
    c, batch = example();smpl = ShapeLayer()
    prediction = batch['velocity_mu'].clone().requires_grad_()
    batch['target'] = batch['target'].clone().requires_grad_()
    first = same_shape_position_errors(c, prediction, batch, smpl)
    with torch.no_grad():batch['target'][..., -10:] += 123
    second = same_shape_position_errors(c, prediction, batch, smpl)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    (second.square().sum() + second.sum()).backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert batch['target'].grad is None


def test_zero_weight_matches_old_objective_and_weight_scaling():
    c, batch = example();smpl = ShapeLayer();prediction = batch['base_mu'].clone()
    zero = physical_objective(c,prediction,batch,same_shape_weight=0,smpl=smpl)
    old = original_objective(c,prediction,batch,fk_weight=0,smpl=smpl)
    torch.testing.assert_close(zero,old,atol=0,rtol=0)
    one = physical_objective(c,prediction,batch,same_shape_weight=1,smpl=smpl,return_components=True)
    low = physical_objective(c,prediction,batch,same_shape_weight=.1,smpl=smpl,return_components=True)
    assert one['weighted_same_shape_fk'] > 0
    torch.testing.assert_close(low['weighted_same_shape_fk'],one['weighted_same_shape_fk']*.1)
    with pytest.raises(ValueError):physical_objective(c,prediction,batch,fk_weight=1,smpl=smpl)
    batch['beta_boot_is_model'].fill_(False)
    with pytest.raises(ValueError,match='bootstrap'):same_shape_position_errors(c,prediction,batch,smpl)


def test_training_resume_and_supervision_identity(tmp_path,monkeypatch):
    from run import train_prior_same_shape as runner
    from run.complete_stages import Task,verify_task
    folder,output=tmp_path/'data',tmp_path/'run';payload,save=cache(folder)
    asset=tmp_path/'asset';asset.mkdir();(asset/'SMPLX_NEUTRAL.npz').write_bytes(b'asset')
    payload['identity']['smplx_asset_sha256']=file_sha256(asset/'SMPLX_NEUTRAL.npz');save()
    monkeypatch.setenv('SMPLX_MODEL_PATH',str(asset));monkeypatch.setattr(runner,'get_smpl',ShapeLayer)
    monkeypatch.setattr(runner,'HistoryPrior',lambda:HistoryPrior(width=16,layers=1,heads=4))
    monkeypatch.setattr(runner,'append_log',lambda *args:None)
    argv=['train','--data',str(folder),'--output',str(output),'--device','cpu','--steps','2','--eval-every','1',
          '--batch-size','8','--same-shape-weight','0.3']
    monkeypatch.setattr(sys,'argv',argv);runner.main()
    assert verify_task(Task('test',[],output,'prior'))['same_shape_weight']==.3
    expected=torch.load(output/'resume.pt',weights_only=True)
    interrupted=tmp_path/'interrupted';changed=[str(interrupted) if x==str(output) else x for x in argv]
    def stop(title,lines):
        if title=='独立 P 选点':raise RuntimeError('interrupted')
    monkeypatch.setattr(runner,'append_log',stop);monkeypatch.setattr(sys,'argv',changed)
    with pytest.raises(RuntimeError,match='interrupted'):runner.main()
    monkeypatch.setattr(runner,'append_log',lambda *args:None);monkeypatch.setattr(sys,'argv',changed+['--resume']);runner.main()
    actual=torch.load(interrupted/'resume.pt',weights_only=True)
    for k,v in expected['model'].items():torch.testing.assert_close(actual['model'][k],v,atol=0,rtol=0)
    assert expected['curve']==actual['curve']
    monkeypatch.setattr(sys,'argv',['1.0' if x=='0.3' else x for x in argv]+['--resume'])
    with pytest.raises(ValueError,match='protocol'):runner.main()


def test_matrix_and_summary(tmp_path):
    from run.prior_same_shape_experiment import tasks_for,collect_results,write_results,CASES
    tasks=tasks_for(tmp_path);assert len(tasks)==12
    assert [t.gpu for t in tasks]==[4,5,6,7]*3
    def result(score):
        val={'mean':score,'per_take':{'a':score-1,'b':score+1}}
        return {'single':{k:val for k in ['fk22_mm','pa_mpjpe_mm','same_shape_mpjpe_mm','local_rotation_deg']},'rollout':{'1000':val}}
    for t in tasks:
        case,seed=t.name.rsplit('_s',1);t.output.mkdir(parents=True)
        torch.save({'state_dict':{'weight':torch.tensor(int(seed))}},t.output/'initial.pt')
        identity={'same_shape_weight':CASES[case],'base_mode':'constant_velocity','supervision_schema':'test'}
        torch.save(identity,t.output/'prior.pt')
        r={k:1 for k in ['data_sha256','code_sha256','steps','batch_size','eval_every','lr','weight_decay','fk_weight',
                         'geometry_weight','velocity_input','reference_mode','smplx_asset_sha256','selection']}
        r.update(identity,training_seed=int(seed),selected_step=100,final_single={},compute={},results={'prior':result(36-CASES[case])})
        if t.name=='control_s62':r['results'].update(hold=result(50),constant_velocity=result(36.75))
        t.report.write_text(json.dumps(r))
    summary=collect_results(tasks)
    assert summary['candidate']=='pose10'
    assert summary['comparisons']['pose10']['control']['fk22_mm']['mean']==-1
    write_results(tmp_path,summary);assert (tmp_path/'RESULTS.md').exists()
