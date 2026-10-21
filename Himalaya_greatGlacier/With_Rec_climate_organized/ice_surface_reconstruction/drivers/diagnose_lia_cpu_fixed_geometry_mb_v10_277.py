"""CPU policy-v2 reassessment and fixed-geometry surface-MB diagnosis only.

Never constructs or advances a flowline dynamics model. No source file writes.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'validation_data/results/paleo_surface_validation_v0'
TX=BASE/'lia_cpu_fixed_climate_forward_v10_275_transaction_v1'
OLD=BASE/'lia_cpu_fixed_climate_forward_analysis_v10_276_v1'
OUT=BASE/'lia_cpu_direction_and_fixed_geometry_mb_v10_277_v1'
MODULE='Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.diagnose_lia_cpu_fixed_geometry_mb_v10_277'


def now(): return datetime.now(timezone.utc).isoformat()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest().upper()


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def ref(p):
    p=Path(p)
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)}


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())


def verify(item):
    if sha(ROOT/item['path']) != item['sha256'].upper():
        raise RuntimeError('source-hash-changed:'+item['path'])


def validate():
    r=subprocess.run([sys.executable,str(Path(__file__).with_name('validate_lia_execution_state_v1.py'))],capture_output=True,text=True,cwd=ROOT)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    return json.loads(r.stdout)


def prepare():
    validation=validate()
    if OUT.exists(): raise RuntimeError('immutable-output-already-exists')
    matrix=read(TX/'matrix.json')
    original=read(TX/'transaction_manifest.json')
    for x in original['code']+[original['matrix'],original['climates'],original['sources'],original['population_ledger']]: verify(x)
    members=[]
    for m in matrix:
        for p,h in [('source_state','source_state_sha256'),('source_checkpoint','source_checkpoint_sha256')]:
            verify({'path':m[p],'sha256':m[h]})
        rp=(ROOT/m['source_checkpoint']).parents[2]/'result.json'
        r=read(rp)
        assert r['fingerprint']==m['source_spinup_fingerprint'] and r['lia_unique_id']==m['lia_unique_id']
        climate=r['climate']
        verify({'path':climate['climate_file'],'sha256':climate['sha256']})
        assert climate['month_count']==360
        members.append({'key':m['key'],'lia_unique_id':m['lia_unique_id'],
                        'spinup_result':ref(rp),'spinup_climate':ref(Path(climate['climate_file']))})
    climates=read(TX/'climate_inventory.json')
    for c in climates.values():
        for f in c['files']: verify(f)
    assert len(members)==1393 and len({m['lia_unique_id'] for m in members})==173
    files=[Path(__file__),TX/'matrix.json',TX/'climate_inventory.json',TX/'population_191.json',
           TX/'curve_standard.json',OLD/'scored_cells.json',OLD/'glacier_summary.json',
           OLD/'terminal_summary.json',BASE/'lia_gpu_offline_direction_policy_v2/transaction.json',
           Path(sys.executable),Path('D:/Anaconda/envs/oggm_env2/Lib/site-packages/oggm/params.cfg')]
    OUT.mkdir()
    write(OUT/'input_inventory.json',members)
    tx={'schema':'lia-cpu-offline-policy-and-fixed-geometry-mb-v10.277','created_at':now(),
        'authorization':'User ok开始 authorizes the first two recommended actions: CPU direction-policy reassessment and three-case fixed-geometry MB diagnosis; no dynamics or parameter search.',
        'model_launch_authorized':False,'offline_analysis_authorized':True,'accepted_ledger_effect':0,
        'inputs':[ref(p) for p in files]+[ref(OUT/'input_inventory.json')],
        'production_code':original['code'],'members':1393,'systems':173,'full_population':191,
        'policy':{'version':'CPU-direction-v2','retrospective_policy_revision':True,
                  'remove_only':'old sign(L2000-L1470) direction hard gate; preserve diagnostically',
                  'unchanged':['10% global LIA peak','10% RGI2000 endpoint','minimum amplitude','v3 lag','v3 sustained phases','v3 artifacts'],
                  'no_new_gate':True,'old_scores_immutable':True,'independent_reproduction':'not-authorized'},
        'diagnosis':{'geometry':'every saved year900 state, unchanged, mapped to1470',
            'surface_domain':'existing ice-covered bins using OGGM bin_area_m2; no arbitrary thickness cutoff',
            'A':'original1441-1470 repeating climate with inherited b+s',
            'B':'same repeating climate with inherited b; only remove spinup s',
            'C':'existing1470-2000 historical climate with inherited b and q=0',
            'A_B_model_years_inclusive':[900,929],
            'C_windows_inclusive':[[1470,1499],[1500,1529],[1970,1999]],
            'operator':'unchanged OGGM MonthlyTIModel.get_annual_mb(add_climate=True); snowfall and potential melt on frozen surface, bias=0 as in production',
            'outputs':'area-weighted kg m-2 yr-1 (numerically mm w.e. yr-1), surface temperature degC and implied ice-volume source m3 yr-1',
            'decomposition':'B-A parameter-withdrawal effect; C-B climate-period effect; C-A total under the same frozen geometry',
            'limitations':['Not a dynamics run or observed MB.','Fixed seasonal-end geometry need not have exactly zero30y mean MB even if a periodic evolving trajectory is stable.',
              'No changing surface/area, flux transport, new ice-covered area, melt supply limit or new equilibrium is computed.',
              'Does not decompose530-year length response or establish climate/geometry causation.'],
            'new_scientific_thresholds':None,'technical_member_failure':'isolate, record, no automatic retry'},
        'runtime':{'python':sys.executable,'processes':1,'all_internal_threads':1,'dynamics_steps':0},
        'prior_state':ref(ROOT/'LIA_CURRENT_STATE.json'),'prior_facts':ref(ROOT/'LIA_PROJECT_FACTS.md')}
    write(OUT/'transaction.json',tx);write(OUT/'validator-before.json',validation)
    print(json.dumps({'prepared':True,'transaction':ref(OUT/'transaction.json'),'time':now()}),flush=True)


def policy():
    original=read(OLD/'scored_cells.json');out=[]
    for r in original:
        q=copy.deepcopy(r)
        q['direction_gate_v2_role']='diagnostic-only'
        q['provisional_curve_pass_v2']=all(r[k] is True for k in
             ['lia_gate','rgi_gate','amplitude_gate','lag_gate_pass','bidirectional_multistage_gate_pass','artifact_gate_pass'])
        q['score_policy_version']='CPU-direction-v2-retrospective'
        assert {k:q[k] for k in r}==r
        out.append(q)
    ids=sorted({r['lia_unique_id'] for r in out if r['provisional_curve_pass_v2']})
    old_count=sum(r['provisional_curve_pass'] for r in out)
    new_count=sum(r['provisional_curve_pass_v2'] for r in out)
    assert old_count==32 and new_count==77 and len(ids)==9
    write(OUT/'rescored_cells_1393.json',out)
    population=read(OLD/'glacier_summary.json')
    for g in population:
        group=[r for r in out if r['lia_unique_id']==g['lia_unique_id']]
        g['provisional_curve_pass_v2_count']=sum(r['provisional_curve_pass_v2'] for r in group)
        g['provisional_delta_t_eq_v2_degC']=[r['delta_t_eq_degC'] for r in group if r['provisional_curve_pass_v2']]
    write(OUT/'population_191_v2.json',population)
    summary={'status':'complete','old_provisional_cells':old_count,'old_provisional_systems':8,
             'new_provisional_cells':new_count,'new_provisional_systems':len(ids),'new_provisional_system_ids':ids,
             'newly_provisional_cells':new_count-old_count,'old_metrics_and_six_gates_unchanged':True,
             'accepted_ledger_effect':0,'dynamics_steps':0,'retrospective_policy_revision':True}
    write(OUT/'policy_summary.json',summary)
    return summary


def run():
    started=time.perf_counter();v=validate();state=read(ROOT/'LIA_CURRENT_STATE.json')
    assert state['current_focus'].get('offline_transaction_sha256')==sha(OUT/'transaction.json')
    assert state['current_focus'].get('offline_analysis_authorized') is True
    if (OUT/'active.json').exists(): raise RuntimeError('unique offline execution already started')
    tx=read(OUT/'transaction.json')
    for x in tx['inputs']+tx['production_code']: verify(x)
    write(OUT/'active.json',{'started_at':now(),'pid':os.getpid(),'command':sys.argv,'model_launch':False,'transaction':ref(OUT/'transaction.json')})
    psummary=policy()
    print(json.dumps({'policy':psummary}),flush=True)
    import numpy as np
    from .lia_elevation_band_spinup import _configure_oggm
    from .lia_fixed_climate_forward_v10_275 import source_state
    _configure_oggm(OUT/'work',OUT/'work')
    from oggm import cfg,utils
    from oggm.core.massbalance import MonthlyTIModel
    matrix=read(TX/'matrix.json');inventory={m['key']:m for m in read(OUT/'input_inventory.json')}
    climates=read(TX/'climate_inventory.json')
    original_scores={r['key']:r for r in read(OLD/'scored_cells.json')}
    physical_cfg={p:cfg.PARAMS[p] for p in ['temp_default_gradient','temp_all_solid','temp_all_liq','ice_density']}
    write(OUT/'runtime_physics.json',{'parameters':physical_cfg,'seconds_per_year':cfg.SEC_IN_YEAR,
          'melt_factor_monthly_conversion':'melt_f *365/12, unchanged production MonthlyTIModel',
          'original_production_mb_code_sha256':sha(Path('D:/Anaconda/envs/oggm_env2/Lib/site-packages/oggm/core/massbalance.py'))})
    cache={};completed=[];failures=[]
    def mb(gdir,m,suffix,start,end,repeat,bias):
        cfg.PARAMS['temp_melt']=m['temp_melt_degC']
        return MonthlyTIModel(gdir,filename='gcm_data',input_filesuffix=suffix,
             melt_f=m['melt_f'],prcp_fac=m['prcp_fac'],temp_bias=bias,bias=0,
             check_calib_params=False,ys=start,ye=end,repeat=repeat)
    for i,m in enumerate(matrix):
        key=m['key'];row_start=time.perf_counter()
        try:
            _,fls=source_state(m)
            arrays_before=[(fl.section.copy(),fl.surface_h.copy(),fl.widths_m.copy()) for fl in fls]
            heights=np.concatenate([fl.surface_h for fl in fls]);weights=np.concatenate([fl.bin_area_m2 for fl in fls])
            covered=weights>0;heights=heights[covered];weights=weights[covered]
            area=float(weights.sum());assert area>0
            np.testing.assert_allclose(area,float(m['area_m2']),rtol=1e-12,atol=0)
            item=inventory[key];verify(item['spinup_climate'])
            spinfile=ROOT/item['spinup_climate']['path']
            rgi=m['rgi_id'];spinroot=spinfile.parents[3]
            if m['lia_unique_id'] not in cache:
                sg=utils.GlacierDirectory(rgi,base_dir=str(spinroot))
                hg=utils.GlacierDirectory(rgi,base_dir=str(ROOT/'w275/g'))
                cache[m['lia_unique_id']]=(sg,hg,sha(spinfile))
            sg,hg,sh=cache[m['lia_unique_id']]
            assert sh==sha(spinfile),'Different spinup climate within one glacier'
            hist=climates[str(m['lia_unique_id'])]
            a_mb=mb(sg,m,'_batch1_1441_1470_prcp_off',1441,1470,True,m['base_temp_bias_degC']+m['delta_t_eq_degC'])
            b_mb=mb(sg,m,'_batch1_1441_1470_prcp_off',1441,1470,True,m['base_temp_bias_degC'])
            c_mb=mb(hg,m,hist['suffix'],1470,2000,False,m['base_temp_bias_degC'])
            assert a_mb.ref_hgt==b_mb.ref_hgt==c_mb.ref_hgt
            assert a_mb.rho==b_mb.rho==c_mb.rho==cfg.PARAMS['ice_density']
            # Same calendar1470 input must be identical after removing s.
            np.testing.assert_array_equal(b_mb.temp[-12:],c_mb.temp[:12])
            np.testing.assert_array_equal(b_mb.prcp[-12:],c_mb.prcp[:12])
            def evaluate(obj,years):
                annual=[];max_residual=0.
                for y in years:
                    net,t,tm,p,solid=obj.get_annual_mb(heights,year=y,add_climate=True)
                    melt=tm*obj.monthly_melt_f
                    net_kg=net*cfg.SEC_IN_YEAR*obj.rho
                    residual=float(np.max(np.abs(net_kg-(solid-melt-obj.bias))))
                    max_residual=max(max_residual,residual)
                    vals=[float(np.average(x,weights=weights)) for x in (solid,melt,net_kg,t,p)]
                    assert np.isfinite(vals).all()
                    annual.append({'evaluation_year':y,'solid_precip_kg_m2_yr':vals[0],
                        'potential_melt_kg_m2_yr':vals[1],'net_surface_mb_kg_m2_yr':vals[2],
                        'surface_temp_degC':vals[3],'total_precip_kg_m2_yr':vals[4]})
                means={k:float(np.mean([r[k] for r in annual])) for k in annual[0] if k!='evaluation_year'}
                means['implied_ice_volume_source_m3_yr']=means['net_surface_mb_kg_m2_yr']*area/obj.rho
                return {'annual':annual,'mean':means,'maximum_component_identity_residual_kg_m2_yr':max_residual}
            cases={'A':evaluate(a_mb,range(900,930)),'B':evaluate(b_mb,range(900,930)),
                   'C_first30':evaluate(c_mb,range(1470,1500)),
                   'C_second30':evaluate(c_mb,range(1500,1530)),
                   'C_final30':evaluate(c_mb,range(1970,2000))}
            A=cases['A']['mean']['net_surface_mb_kg_m2_yr'];B=cases['B']['mean']['net_surface_mb_kg_m2_yr']
            effects={'withdrawal_B_minus_A_kg_m2_yr':B-A}
            for w in ['first30','second30','final30']:
                C=cases['C_'+w]['mean']['net_surface_mb_kg_m2_yr']
                effects[w+'_climate_C_minus_B_kg_m2_yr']=C-B
                effects[w+'_total_C_minus_A_kg_m2_yr']=C-A
                effects[w+'_ref_temp_C_minus_A_degC']=cases['C_'+w]['mean']['surface_temp_degC']-cases['A']['mean']['surface_temp_degC']
            for fl,before in zip(fls,arrays_before):
                for arr,old in zip((fl.section,fl.surface_h,fl.widths_m),before): np.testing.assert_array_equal(arr,old)
            verify({'path':m['source_state'],'sha256':m['source_state_sha256']})
            row={'key':key,'lia_unique_id':m['lia_unique_id'],'rgi_id':rgi,'status':'completed',
                 'delta_t_eq_degC':m['delta_t_eq_degC'],'base_temp_bias_degC':m['base_temp_bias_degC'],
                 'fixed_area_m2':area,'occupied_bin_count':len(heights),'cases':cases,'effects':effects,
                 'old_total_length_change_m':original_scores[key]['total_length_change_m'],
                 'geometry_and_source_unchanged':True,'dynamics_steps':0,'runtime_seconds':time.perf_counter()-row_start}
            write(OUT/'members'/f'{key}.json',row);completed.append(row)
        except Exception as e:
            failure={'key':key,'lia_unique_id':m['lia_unique_id'],'status':'diagnostic-technical-failure',
                     'error':repr(e),'traceback':traceback.format_exc(),'source_outputs_unchanged':True,'model_failure':False,'retry':False}
            write(OUT/'members'/f'{key}.json',failure);failures.append(failure)
        if (i+1)%100==0:
            print(json.dumps({'processed':i+1,'completed':len(completed),'diagnostic_failures':len(failures),'total':1393}),flush=True)
    for x in tx['inputs']+tx['production_code']: verify(x)
    for c in climates.values():
        for f in c['files']:verify(f)
    def scalar(r,case):return r['cases'][case]['mean']['net_surface_mb_kg_m2_yr']
    med={c:float(np.median([scalar(r,c) for r in completed])) if completed else None for c in ['A','B','C_first30','C_second30','C_final30']}
    effect_keys=list(completed[0]['effects']) if completed else []
    emed={k:float(np.median([r['effects'][k] for r in completed])) for k in effect_keys}
    system_rows=[]
    for sid in read(TX/'population_191.json')['all_ids']:
        group=[r for r in completed if r['lia_unique_id']==sid]
        system_rows.append({'lia_unique_id':sid,'completed_diagnostics':len(group),
            'status':'no-stable-parent-retained' if sid in read(TX/'population_191.json')['no_stable_ids'] else ('complete' if len(group)==sum(m['lia_unique_id']==sid for m in matrix) else 'partial-diagnostic-failure'),
            'median_fixed_geometry_net_mb':{c:float(np.median([scalar(r,c) for r in group])) if group else None for c in med},
            'median_effects':{k:float(np.median([r['effects'][k] for r in group])) if group else None for k in effect_keys}})
    write(OUT/'fixed_geometry_glaciers_191.json',system_rows)
    positive={c:sum(scalar(r,c)>0 for r in completed) for c in med}
    joint_growth=[r for r in completed if r['old_total_length_change_m']>0]
    summary={'schema':'lia-cpu-policy-and-fixed-geometry-mb-terminal-v10.277','status':'completed' if not failures else 'completed-with-isolated-diagnostic-failures',
        'finished_at':now(),'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started,'policy':psummary,
        'diagnostic_members_completed':len(completed),'diagnostic_technical_failures':len(failures),'diagnostic_failure_details':failures,
        'population':191,'forward_systems':173,'no_stable_systems_retained':18,'dynamics_steps':0,'new_model_trajectories':0,
        'member_weighted_median_fixed_geometry_net_mb_kg_m2_yr':med,
        'member_weighted_median_effects_kg_m2_yr_or_degC':emed,
        'positive_fixed_geometry_net_mb_members':positive,
        'equal_glacier_weight_median_of_within_glacier_medians':{c:float(np.median([r['median_fixed_geometry_net_mb'][c] for r in system_rows if r['completed_diagnostics']])) if completed else None for c in med},
        'equal_glacier_weight_median_effects':{k:float(np.median([r['median_effects'][k] for r in system_rows if r['completed_diagnostics']])) for k in effect_keys},
        'growth_association':{'existing_net_growth_members':len(joint_growth),
             'of_those_positive_first30_fixed_geometry_MB':sum(scalar(r,'C_first30')>0 for r in joint_growth),
             'of_those_positive_withdrawal_effect':sum(r['effects']['withdrawal_B_minus_A_kg_m2_yr']>0 for r in joint_growth),
             'interpretation':'association on initial surface, not causal decomposition of530-year geometry evolution'},
        'maximum_component_identity_residual_kg_m2_yr':max((c['maximum_component_identity_residual_kg_m2_yr'] for r in completed for c in r['cases'].values()),default=None),
        'sources_reverified_unchanged':True,'accepted_ledger_effect':0,'launch_authorized':False,
        'transaction':ref(OUT/'transaction.json'),'rescored_cells':ref(OUT/'rescored_cells_1393.json'),
        'population_policy':ref(OUT/'population_191_v2.json'),'diagnostic_population':ref(OUT/'fixed_geometry_glaciers_191.json'),
        'members':[ref(OUT/'members'/f'{m["key"]}.json') for m in matrix],
        'limitations':tx['diagnosis']['limitations'],'next_legal_action':'Review these offline results and prepare a separately authorized coarse physical-parameter matrix; no automatic model launch.'}
    write(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='members'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
