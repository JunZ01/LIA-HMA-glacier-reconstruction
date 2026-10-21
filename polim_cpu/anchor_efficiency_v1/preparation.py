"""One scalar-only historical scheduling analysis; stop at first divergence."""
from .control import *
import math,statistics
from .efficiency import History
P57=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v57_anchor_reuse_v1'
R57=ROOT/'polim_cpu/runs/v57-anchor-reuse-v1'
STEP57=R57/'N/native/step-1553.0000-1553.2500'

def coefficients():
    result=read(STEP57/'result.json');events=[json.loads(x) for x in (STEP57/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    builds=[x for x in events if x['event']=='anchor-built'];times=[];before=0.
    for b in builds:
        total=b['work']['seconds']['J_build'];times.append(total-before);before=total
    w=result['work'];assert len(times)==8 and w['counts']['V_requests']==256 and before==w['seconds']['J_build']
    assert w['seconds']['step_phase']>before and all(t>0 for t in times)
    runner=ROOT/'polim_cpu/anchor_reuse_v1/runner.py';coupling=ROOT/'polim_cpu/anchor_reuse_v1/coupling.py'
    assert "with w.span('step_phase'):r=attempt(" in runner.read_text(encoding='utf-8')
    assert "with w.span('J_build'):" in coupling.read_text(encoding='utf-8')
    return dict(c_V=(w['seconds']['step_phase']-before)/256,c_J=statistics.median(times[1:8]),
        source_hashes=hashes([STEP57/'result.json',STEP57/'events.jsonl',runner,coupling]),
        original_step_seconds=w['seconds']['step_phase'],original_J_parent_seconds=before,anchor_J_seconds=times,
        nesting_verified=True,meaning='Frozen non-J step amortization perV and median complete J build2..8, NOT AMG kernel tariff, no linear-cost guarantee.')

def first_historical_trigger(coeff):
    events=[json.loads(x) for x in (STEP57/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    states={x['name']:x for x in events if x['event']=='joint-state'};regions={x['name']:x for x in events if x['event']=='joint-state-region'}
    starts=[x for x in events if x['event']=='anchor-direction-start'];adopted={x['state']:x for x in events if x['event']=='anchor-internal-adopted'}
    models={x['direction']:x for x in events if x['event']=='anchor-model'};branch={};bid=0
    branch_reasons={'wet-branch','donor-branch','outlet-branch','reflection-branch','formal-discrete-revision','no-anchor'}
    for d in starts:
        if branch_reasons.intersection(d['refresh_reasons']):bid+=1
        branch[d['parent']]=str(bid)
    history=None;rows=[];found=None
    for d in starts:
        n=d['number'];s=states[d['parent_name']];r=regions[d['parent_name']];orig=d['refresh_reasons'];point=d['parent']
        other=bool(s['local']<=1e-8 and s['fixed_point']<=1e-8 and s['mass']<=1e-10 and s['annual_cumulative_mass_relative']<=1e-10
            and s['inherited_cumulative_mass_relative']<=1e-10 and s['physical_path'] and s['all_finite'] and s['H_min']>=0)
        if history is not None:
            decision=history.decide(point=point,branch=branch[point],f=r['F_norm'],force=r['force_scale'],bp=s['bp'],other_pass=other,coeff=coeff,original=orig)
            decision.update(direction=n,state=d['parent_name']);rows.append(decision)
            if decision['trigger']:found=decision;break
        if orig:
            v=next(x['V_used'] for x in events if x['event']=='FGMRES-cycle-start' and x['direction']==f'direction-{n:02d}-anchor-linear')
            history=History(d['anchor'],point,branch[point],r['F_norm'],v)
        m=models.get(n)
        if m and m['child'] in adopted:
            child=states[m['child']];cr=regions[m['child']];cp=child['point']
            # Branch labels only from already recorded exact v57 refresh tests;
            # no current model or array evaluation is needed.
            cb=branch.get(cp,'not-recorded');V=256-m['budget_remaining']['V_requests']
            history.adopt(point,cp,branch[point],cb,r['F_norm'],cr['F_norm'],V,d['reused'],'parent')
    return dict(status='first-trigger-found' if found else 'not-found',first_trigger=found,examined=rows,
        source_hashes=hashes([STEP57/'events.jsonl']),stopped_at_first_different_refresh=True,post_trigger_simulated_savings=None,
        limitation='Scalar log reconstruction only; cannot use the old suffix as the new strategy trajectory. Dry projection follows verified old candidate construction.',
        model_calls=0,assemblies=0,linear_solves=0,time_steps=0)
