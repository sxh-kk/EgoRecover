import unittest
from types import SimpleNamespace
import torch
from run.prepare_prior_expanded import select_clips, select_takes, interval
from egorecover.prior_expanded import ExpandedPriorSequences

class ExpandedTests(unittest.TestCase):
    def test_overlapping_source_sequences_and_retained_clip(self):
        names=['a___0___5997','a___300___6297']
        source=SimpleNamespace(motion={n:{'num_frames':2000} for n in names})
        original=dict(base_seq_name=names[0],episode_start_motion_idx=119,num_frames=200,episode_id='original')
        clips=select_clips(source,'a',names,original,4)
        self.assertEqual(clips[0],original);self.assertEqual(len(clips),4)
        for i,a in enumerate(clips):
            for b in clips[i+1:]:
                self.assertTrue(interval(a)[1]<=interval(b)[0] or interval(b)[1]<=interval(a)[0])
        self.assertEqual(clips,select_clips(source,'a',names,original,4))

    def test_take_selection_excludes_all_reserved_and_non_train(self):
        names=list('abcdefg');meta={n:{'take_uid':n,'task_name':str(i%2)} for i,n in enumerate(names)}
        split={'train':['a'],'dev':['b'],'holdout':['c']}
        official={n:('val' if n=='d' else 'train') for n in names}
        got=select_takes(dict.fromkeys(names),meta,official,split,4)
        self.assertEqual(got,list('aefg'))
        with self.assertRaises(ValueError): select_takes(dict.fromkeys(names),meta,official,split,5)

    def test_sampling_equal_take_probability_with_unequal_clips(self):
        dataset=ExpandedPriorSequences.__new__(ExpandedPriorSequences);dataset.group='train'
        dataset.take_episodes=[torch.tensor([0]),torch.tensor([1,2,3,4])]
        a=dataset.sample(20000,torch.Generator().manual_seed(62))
        b=dataset.sample(20000,torch.Generator().manual_seed(62))
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.dtype,torch.int64)
        self.assertLess(abs(float((a[:,0]==0).float().mean())-.5),.02)
        self.assertGreaterEqual(int(a[:,1].min()),40);self.assertLess(int(a[:,1].max()),200)

if __name__=='__main__':unittest.main()


def expanded_cache(tmp_path):
    import json
    from test_prior_two_forward import cache
    from egorecover.prior_expanded import SCOPE
    from egorecover.evaluation_protocol import file_sha256
    original=tmp_path/'original';payload,save=cache(original)
    asset=tmp_path/'asset';asset.mkdir();(asset/'SMPLX_NEUTRAL.npz').write_bytes(b'test asset')
    payload['identity']['smplx_asset_sha256']=file_sha256(asset/'SMPLX_NEUTRAL.npz');save()
    folder=tmp_path/'expanded';folder.mkdir()
    payload={**payload,'identity':{**payload['identity'],'scope':SCOPE,'original_data':str(original)},
             'episode_ids':['episode'+str(i) for i in range(len(payload['take_names']))]}
    def save_expanded():
        torch.save(payload,folder/'sequences.pt')
        (folder/'report.json').write_text(json.dumps({'completed':True,'identity':payload['identity'],
            'cache_sha256':file_sha256(folder/'sequences.pt')}))
    save_expanded()
    return folder,asset,payload,save_expanded


def test_dev_tensor_change_is_rejected(tmp_path):
    import pytest
    folder,_,payload,save=expanded_cache(tmp_path)
    ExpandedPriorSequences(folder,'dev')
    i=payload['take_names'].index(payload['identity']['splits']['dev'][0])
    payload['joints'][i,0,0,0,3]+=1;save()
    with pytest.raises(ValueError,match='dev tensors changed'): ExpandedPriorSequences(folder,'dev')


def test_expanded_training_resume_reproduces_parameters(tmp_path,monkeypatch):
    import sys
    import pytest
    from run import train_prior_expanded as runner
    from egorecover.prior import HistoryPrior
    from test_smplx_evaluation import SyntheticLayer
    from run.complete_stages import Task,verify_task
    folder,asset,_,_=expanded_cache(tmp_path)
    monkeypatch.setenv('SMPLX_MODEL_PATH',str(asset))
    monkeypatch.setattr(runner,'get_smpl',SyntheticLayer)
    monkeypatch.setattr(runner,'HistoryPrior',lambda:HistoryPrior(width=16,layers=1,heads=4))
    monkeypatch.setattr(runner,'append_log',lambda *a:None)
    output=tmp_path/'run'
    argv=['train','--data',str(folder),'--output',str(output),'--device','cpu','--steps','2','--eval-every','1','--batch-size','8']
    monkeypatch.setattr(sys,'argv',argv);runner.main()
    verify_task(Task('test',[],output,'prior'))
    expected=torch.load(output/'resume.pt',weights_only=True)
    interrupted=tmp_path/'interrupted';changed=[str(interrupted) if a==str(output) else a for a in argv]
    def stop(title,lines):
        if title=='独立 P 选点': raise RuntimeError('interrupted')
    monkeypatch.setattr(runner,'append_log',stop);monkeypatch.setattr(sys,'argv',changed)
    with pytest.raises(RuntimeError,match='interrupted'):runner.main()
    monkeypatch.setattr(runner,'append_log',lambda *a:None);monkeypatch.setattr(sys,'argv',changed+['--resume']);runner.main()
    actual=torch.load(interrupted/'resume.pt',weights_only=True)
    for k in expected['model']:torch.testing.assert_close(actual['model'][k],expected['model'][k],atol=0,rtol=0)


def test_summary_keeps_fixed_budget_scores_and_paired_differences(tmp_path,monkeypatch):
    import json
    from run import prior_expanded_experiment as runner
    from run.complete_stages import Task
    base=tmp_path/'base';monkeypatch.setattr(runner,'BASE',base)
    tasks=[]
    for s in runner.SEEDS:
        old=base/f'V0_cv_dense_s{s}';new=tmp_path/f'D_s{s}'
        old.mkdir(parents=True);new.mkdir()
        state={'state_dict':{'weight':torch.tensor(s)}}
        for path in (old,new):torch.save(state,path/'initial.pt')
        def scores(x):return {k:{'mean':x,'per_take':{'a':x-1,'b':x+1}} for k in ('fk22_mm','pa_mpjpe_mm','same_shape_mpjpe_mm')}
        report={k:1 for k in ('stats_sha256','smplx_asset_sha256','reference_mode','base_mode','geometry_weight','fk_weight','batch_size','lr','weight_decay','eval_every')}
        report.update(data_sha256='original',original_cache_sha256='original',selected_step=4800,curve=[{'step':x,'dev':scores(34)} for x in (2400,4800,9600)],final_single=scores(34))
        (old/'report.json').write_text(json.dumps({**report,'results':{'prior':{'single':scores(36)}}}))
        (new/'report.json').write_text(json.dumps({**report,'results':{'prior':{'single':scores(34)}}}))
        tasks.append(Task(f'D_s{s}',[],new,'prior'))
    result=runner.collect_results(tasks)
    assert result['differences']['fk22_mm']['mean']==-2
    assert result['differences']['fk22_mm']['take_bootstrap_95ci']==[-2.,-2.]
    assert set(result['fixed_budget_dev']['62'])=={'2400','4800','9600'}
