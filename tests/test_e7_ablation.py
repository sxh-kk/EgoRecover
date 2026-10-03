import torch

from egorecover.e7_ablation import matrix, bounds, draw_noise, conditions
from egorecover.codec import MotionCodec


def sample():
    head=torch.tensor([1.,0,0,0,1,0,0,0,1.7]).repeat(200,1)
    head[:,6]=torch.arange(200)*.01
    kp=torch.zeros(200,22,3);kp[...,2]=.05
    case={'observations':{'aria_traj_obs':head,'img_feats':torch.zeros(200,1024),
                          'img_available':torch.ones(200,dtype=torch.bool),
                          'traj_available':torch.ones(200,dtype=torch.bool)},
          'supervision':{'floor_height':torch.tensor(.1),'kp3d':kp},
          'startup':{'floor_estimate_m':.2}}
    stats={f'{k}_{s}':(torch.zeros(n) if s=='mean' else torch.ones(n))
           for k,n in [('motion',243),('traj',18)] for s in ('mean','std')}
    return case,MotionCodec(stats)


def test_four_cells_isolate_two_factors():
    rows=matrix()
    assert len(rows)==4 and len({(r['window'],r['floor']) for r in rows})==4
    for field,value in [('encoder','native'),('decoder','native'),('beta','window'),
                         ('noise','independent'),('padding','pad80')]:
        assert all(r[field]==value for r in rows)


def test_causal_future_and_padding_isolation():
    case,codec=sample();row=matrix()[1]
    y,anchor,floor=conditions(case,*bounds(20,'causal'),row,codec)
    case['observations']['aria_traj_obs'][21:]=float('nan')
    case['observations']['img_feats'][21:]=float('nan')
    after,other,other_floor=conditions(case,*bounds(20,'causal'),row,codec)
    assert all(torch.equal(y[k],after[k]) for k in y)
    assert torch.equal(anchor,other) and floor==other_floor
    assert int(y['valid_frames'].sum())==21 and y['traj'].shape==(80,18)
    assert y['traj'][21:].count_nonzero()==0


def test_floor_changes_only_height_channel():
    case,codec=sample()
    a,_,_=conditions(case,0,80,matrix()[0],codec)
    b,_,_=conditions(case,0,80,matrix()[2],codec)
    diff=b['traj']-a['traj']
    assert torch.allclose(diff[:,8],torch.full((80,),-.1),atol=1e-6)
    diff[:,8]=0
    assert diff.abs().max()<1e-6


def test_no_duplicate_target_and_matching_endpoint_windows():
    for t in range(20,200):
        first,last=bounds(t,'offline')
        assert last-first==80 and first<=t<last
        first,last=bounds(t,'causal')
        assert last==t+1 and 1<=last-first<=80
    for t in (79,159,199):
        assert bounds(t,'causal')==bounds(t,'offline')


def test_floor_pairs_share_noise_and_padding_prefix():
    a=draw_noise('take',1062,0,21,'independent',80)
    b=draw_noise('take',1062,0,21,'independent',21)
    assert torch.equal(a[:21],b)
    assert not torch.equal(a,draw_noise('take',1063,0,21,'independent',80))


def test_summary_pairs_by_take_and_reports_interaction(tmp_path, monkeypatch):
    import json
    import pytest
    import run.e7_ablation as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'LOG.md').write_text('# Test log\n')
    output=tmp_path/'results';ident={'test':'synthetic'}
    plan={'takes':['a','b'],'draws':[1,2]}
    for take in plan['takes']:
        for draw in plan['draws']:
            metrics={}
            for name,value in [('E0',10),('E1',20),('E2',30),('E3',45)]:
                metrics[name]={k:torch.full((179 if k=='velocity_error_mmps' else 180,),float(value))
                               for k in ['body_mpjpe_mm','pa_mpjpe_mm','root_mm','velocity_error_mmps']}
            runner.save(output/'dev'/take/str(draw)/'metrics.pt',
                        {'identity':ident,'times':torch.arange(20,200),'metrics':metrics})
    runner.summarize(plan,output,ident)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['interaction_E3_minus_E1_minus_E2_plus_E0']['all']['difference_of_differences_mm']==5
    assert len(summary['paired_edges'])==16  # Four edges, four time subsets.
    assert 'RESULTS.md' in (tmp_path/'LOG.md').read_text()
    (output/'dev'/'b'/'2'/'metrics.pt').unlink()
    with pytest.raises(FileNotFoundError):
        runner.summarize(plan,output,ident)


def test_data_producer_is_preserved_across_consumer_code_changes(tmp_path):
    import pytest
    import run.e7_ablation as runner
    old={'plan_sha256':'same','checkpoint_sha256':'same','code_sha256':{'runner':'old'}}
    new={**old,'code_sha256':{'runner':'new'}}
    path=tmp_path/'data/packets.pt'
    runner.save(path,{'identity':old,'packs':{}})
    runner.write_json(tmp_path/'data/report.json',{'identity':old,'packet_sha256':runner.sha(path)})
    assert runner.load_data(tmp_path,new)['identity']==old
    with pytest.raises(ValueError):
        runner.load_data(tmp_path,{**new,'checkpoint_sha256':'changed'})
    runner.save(path,{'identity':old,'packs':{'tampered':True}})
    with pytest.raises(ValueError):
        runner.load_data(tmp_path,new)
