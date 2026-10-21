"""Static input binding and tests. Never advances a dynamics model."""
from __future__ import annotations
from . import lia_cpu_melt_factor_v10_279 as r
import copy,json,shutil,subprocess,sys
from pathlib import Path
D=r.RES/'lia_cpu_melt_factor_design_v10_278_v1'
def audit(proposed):
    wanted={}
    for row in proposed:wanted.setdefault(int(row['lia_unique_id']),[]).append(row)
    listing=subprocess.run(['rg','--files','--hidden','-g','!**/.git/**','-g','checkpoint.json','-g','c.json','-g','result.json','-g','*rules*.json'],cwd=r.ROOT,capture_output=True,text=True,check=True)
    groups={};others=[]
    for name in listing.stdout.splitlines():
        p=Path(name);s=p.as_posix()
        if '/checkpoints/' in s:groups.setdefault(s.split('/checkpoints/')[0],p)
        elif '/cp/' in s:groups.setdefault(s.split('/cp/')[0],p)
        else:others.append(p)
    matched=[];unreadable=[];identified=0;fingerprints=set()
    for p in list(groups.values())+others:
        try:o=r.read(r.ROOT/p)
        except Exception as e:unreadable.append({'path':str(p),'error':repr(e)});continue
        if not isinstance(o,dict):continue
        if o.get('fingerprint'):fingerprints.add(o['fingerprint'])
        sid=o.get('lia_unique_id');params=o.get('parameters',o)
        if sid is None or not isinstance(params,dict):continue
        if 'melt_f' not in params:continue
        identified+=1
        for row in wanted.get(int(sid),[]):
            # A physical match is not dismissed just because driver versions differ.
            if all(k in params and abs(float(params[k])-float(row[k]))<=max(abs(float(row[k]))*1e-12,1e-30) for k in r.PARAMS):
                matched.append({'candidate':row['proposal_key'],'evidence':r.ref(r.ROOT/p),'schema':o.get('schema'),'status':o.get('status')})
    report={'schema':'lia-prior-executed-member-cache-audit-v10.279','listed_files':len(listing.stdout.splitlines()),'checkpoint_chains':len(groups),'other_manifests':len(others),'identified_physical_records':identified,'physical_matches':matched,'unreadable':unreadable,'fingerprints':sorted(fingerprints),'scope':'All workspace checkpoint/result and rules manifests found by rg including hidden files, checkpoint metadata grouped by chain; immutable full900 baseline additionally bound exhaustively by design inventory. No equivalence inferred from scientific parameters alone.','closed_mechanism_review':'Current full-grid temperature-only branch completed; new consistent melt factors are the explicitly approved next physical axis. Cancelled v10.260 adaptive melt proposal is not an executed prior batch; independent GPU mechanism design is not launched or promoted.'}
    r.write(r.TX/'cache_audit.json',report)
    if matched or unreadable:raise ValueError('prior-cache-match-or-unreadable-requires-resolution')
    print('cache audit',identified,'physical records, zero candidate matches',flush=True)
def tests(rows,inputs):
    import numpy as np
    from . import lia_all191_adaptive_spinup_v10_260 as base
    root=r.OUT/'test';root.mkdir();base._configure_oggm(root/'w',root/'w')
    sample=sorted({x['lia_unique_id']:x for x in rows}.values(),key=lambda x:inputs[str(x['lia_unique_id'])]['nx'])
    checks=[]
    for i in (0,len(sample)//2,len(sample)-1):
        row=sample[i];item=inputs[str(row['lia_unique_id'])];fl=base._read_flowline(r.ROOT/item['geometry']['path'])
        model=base.make_model([fl],r.mb_for(row,'spinup'),row['glen_a_pa_3_s_1'],0)
        saved=r.save_cp(root/row['key'],model,0,row,[base.snapshot(model,0)],{'accepted_steps':0})
        c,restored,h=r.load_cp(saved,row)
        for a,b in zip(model.fls,restored):
            for n in ('thick','section','bed_h','surface_h','widths_m'):np.testing.assert_array_equal(getattr(a,n),getattr(b,n))
        bad={**row,'fingerprint':'bad'}
        try:r.load_cp(saved,bad)
        except ValueError:pass
        else:raise AssertionError('bad-identity-not-rejected')
        bad={**row,'melt_f':row['melt_f']+1}
        try:r.load_cp(saved,bad)
        except ValueError:pass
        else:raise AssertionError('bad-parameters-not-rejected')
        forward_mb=r.mb_for(row,'forward')
        assert forward_mb.melt_f==model.mb_model.melt_f==row['melt_f']
        assert forward_mb.temp_bias==row['base_temp_bias_degC']
        assert model.yr==0
        checks.append({'key':row['key'],'lia_unique_id':row['lia_unique_id'],'nx':fl.nx,'roundtrip_exact':True,'identity_parameter_corruption_rejected':True,'same_melt_f_both_stages':True,'steps':0})
    calls=[]
    def locked(src,dst):calls.append(1);raise PermissionError('injected-status-lock')
    fallback=r.status(root/'status.json',{'year':0},locked)
    assert fallback and len(calls)==4 and r.read(fallback)['payload']['year']==0
    assert r.status(root/'status.json',{'year':30}) and r.read(root/'status.json')['year']==30
    fixture=r.RES/'lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1/summary.json'
    assert r.read(fixture)['preflight_result']['resume_check']['passed']
    # Replay all existing CPU scores: changes to the adapter must not change gates.
    old=r.read(r.RES/'lia_cpu_direction_and_fixed_geometry_mb_v10_277_v1/rescored_cells_1393.json')
    count=0
    gates=('lia_gate','rgi_gate','amplitude_gate','lag_gate_pass','bidirectional_multistage_gate_pass','artifact_gate_pass')
    for item in old:
        row={**item,'target_lia_length_m':item['lia_reference_m'],'target_rgi_length_m':item['rgi_reference_m'],'melt_f_multiplier':1.0}
        rp=r.ROOT/'w275/m'/row['key']/'result.json'
        result={**r.read(rp),'annual_history':r.ref(r.ROOT/'w275/m'/row['key']/'cp/2000/h.json')}
        got=r.score_record(row,result,rp)
        if any(got[k]!=item[k] for k in gates) or got['provisional_curve_pass']!=item['provisional_curve_pass_v2']:raise AssertionError('score-replay:'+row['key'])
        count+=got['provisional_curve_pass']
    assert count==77
    result={'status':'pass','static_checkpoint_cases':checks,'status_lock_fallback_and_subsequent_progress':True,'prior_real_30_year_recovery_fixture':r.ref(fixture),'score_replay_members':len(old),'score_replay_passes':count,'all_six_gates_identical':True,'new_dynamics_steps':0,'new_scientific_preflight':False}
    r.write(r.TX/'technical_tests.json',result);return result
def main():
    import numpy as np,pandas as pd,xarray as xr,psutil
    from . import lia_all191_adaptive_spinup_v10_260 as base
    validator=subprocess.run([r.PYTHON,str(r.DRIVERS/'validate_lia_execution_state_v1.py')],cwd=r.ROOT,capture_output=True,text=True,check=True)
    if r.TX.exists() or r.OUT.exists():raise FileExistsError('immutable-root-exists')
    r.TX.mkdir();r.OUT.mkdir();r.write(r.TX/'validator-before.json',json.loads(validator.stdout))
    proposed=r.read(D/'spinup_matrix_proposed_7644.json');pop=r.read(D/'population_191.json')
    ids={x['lia_unique_id'] for x in proposed};retained=[x for x in pop if x['lia_unique_id'] not in ids]
    assert len(proposed)==7644 and len(ids)==182 and len(pop)==191 and len(retained)==9
    assert sum(x['provisional_curve_pass_v2_count'] for x in retained)==77
    for sid in ids:
        sub=[x for x in proposed if x['lia_unique_id']==sid]
        assert {(x['melt_f_multiplier'],x['delta_t_eq_degC']) for x in sub}=={(m,t/10) for m in (.7,1.3) for t in range(-10,11)}
        assert all(x['fs']==0 for x in sub)
    audit(proposed)
    registry=r.read(r.RES/'lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json')
    global_inputs={k:registry['global_inputs'][k] for k in ('population_selection','length_references','climate_pixels','reconstruction_climate')}
    for x in global_inputs.values():r.check(x)
    old=r.read(r.RES/'lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1/transaction_manifest.json')
    inventory=r.read(r.RES/'lia_cpu_fixed_climate_forward_v10_275_transaction_v1/climate_inventory.json')
    pixels=pd.read_csv(global_inputs['climate_pixels']['path'])
    population={x['lia_unique_id']:x for x in pop};inputs={};sources=[];copied=0;generated=0
    base._configure_oggm(r.OUT/'prep',r.OUT/'prep')
    from oggm import utils
    for sid in sorted(ids):
        row=next(x for x in proposed if x['lia_unique_id']==sid);rgi=row['rgi_id']
        r.check(row['baseline_reference']);r.check(row['baseline_parameters_reference'])
        geo=r.RES/f'lia_elevation_band_core191_length_corrected_v1/{sid}/elevation_band_flowline_lia.pkl.gz'
        assert r.sha(geo)==registry['member_file_hashes'][str(sid)]['corrected_flowline'].upper()
        src=base._gdir_path(Path(old['inputs']['source_gdir_root']),rgi);dst=base._gdir_path(r.OUT/'g',rgi);dst.mkdir(parents=True)
        for name in ('outlines.tar.gz','glacier_grid.json','climate_historical.nc','mb_calib.json'):
            shutil.copy2(src/name,dst/name);assert r.sha(src/name)==r.sha(dst/name)
        for file,key in (('climate_historical.nc','source_climate_historical'),('mb_calib.json','source_mb_calib')):assert r.sha(dst/file)==registry['member_file_hashes'][str(sid)][key].upper()
        original=r.read(r.ROOT/row['baseline_reference']['path'])['climate']
        cp=Path(original['climate_file']);assert r.sha(cp)==original['sha256'].upper();shutil.copy2(cp,dst/cp.name)
        spin_suffix='_batch1_1441_1470_prcp_off';forward_suffix='_batch1_1470_2000_prcp_off'
        assert cp.name=='gcm_data'+spin_suffix+'.nc'
        if str(sid) in inventory:
            prior=next(x for x in inventory[str(sid)]['files'] if Path(x['path']).name=='gcm_data'+forward_suffix+'.nc');r.check(prior)
            shutil.copy2(r.ROOT/prior['path'],dst/Path(prior['path']).name);copied+=1;forward_source=prior
        else:
            pixel=pixels[pixels.lia_unique_id.astype(int).eq(sid)&pixels.rgi_id.eq(rgi)];assert len(pixel)==1
            gdir=utils.GlacierDirectory(rgi,base_dir=str(r.OUT/'g'))
            suffix,audit_result=base._write_case_climate(gdir,Path(global_inputs['reconstruction_climate']['path']),pixel.iloc[0],'off',1470,2000)
            assert suffix==forward_suffix;generated+=1;forward_source={'new_same_contract_climate_file':True,'audit':audit_result}
        with xr.open_dataset(dst/cp.name,decode_times=False) as a,xr.open_dataset(dst/('gcm_data'+forward_suffix+'.nc'),decode_times=False) as b:
            assert a.sizes['time']==360 and b.sizes['time']==6372 and a.attrs['ref_hgt']==b.attrs['ref_hgt']
            for name in ('temp','prcp','temp_std'):
                if name in a or name in b:np.testing.assert_array_equal(a[name].values[-12:],b[name].values[:12])
        fl=base._read_flowline(geo)
        item={'lia_unique_id':sid,'rgi_id':rgi,'geometry':r.ref(geo),'nx':int(fl.nx),'files':[r.ref(p) for p in sorted(dst.iterdir()) if p.is_file()],
            'spinup_suffix':spin_suffix,'forward_suffix':forward_suffix,'spinup_original':r.ref(cp),'forward_original':forward_source,'baseline_parameters':row['baseline_parameters_reference']}
        r.write(r.TX/'g'/f'{sid}.json',item);inputs[str(sid)]=item
        if len(inputs)%25==0:print('inputs',len(inputs),'/182',flush=True)
    assert copied==164 and generated==18
    rules=r.read(D/'rules-proposed.json');rules.update(status='user-confirmed-numerical-contract-v10.279',multiplier_status='User 确定 approves 0.7 and1.3',launch_authorized=True,execution_pending=[],new_dynamic_preflight=False)
    rules['runtime'].update(workers=20,memory_above80='pause-new-dispatch-for-user-decision; active workers continue',invalid_rate_above15_after120='advisory-only-no-auto-stop',automatic_retry=False)
    r.write(r.TX/'rules.json',rules)
    code_files=[Path(__file__),Path(r.__file__),Path(base.__file__),r.DRIVERS/'lia_elevation_band_spinup.py',r.DRIVERS/'lia_first10_spinup_batch.py',r.DRIVERS/'score_lia_all191_forward_essential_standard_v3.py']
    import oggm.core.flowline,oggm.core.massbalance,oggm.cfg
    code_files.extend(Path(m.__file__) for m in (oggm.core.flowline,oggm.core.massbalance,oggm.cfg));code=[r.ref(p) for p in code_files]
    rows=[]
    for p in proposed:
        sid=p['lia_unique_id'];item=inputs[str(sid)];population_row=population[sid]
        row={k:p[k] for k in ('lia_unique_id','rgi_id','melt_f_multiplier')+r.PARAMS}
        row.update(key=p['proposal_key'],stage='spinup',inputs=r.ref(r.TX/'g'/f'{sid}.json'),initial_state_hash=item['geometry']['sha256'],target_lia_length_m=population_row['lia_reference_m'],target_rgi_length_m=population_row['rgi_reference_m'])
        payload={'lia_unique_id':sid,'rgi_id':p['rgi_id'],'parameters':{k:row[k] for k in r.PARAMS},'geometry':item['geometry'],'inputs':row['inputs'],'solver':'OGGM-SemiImplicit-108000sec-fs0','years':[0,900],'rules':r.ref(r.TX/'rules.json'),'code':code}
        row.update(fingerprint_payload=payload,fingerprint=r.digest(payload));rows.append(row);r.write(r.TX/'rows'/(row['key']+'.json'),row)
    rows.sort(key=lambda x:(x['melt_f_multiplier'],x['delta_t_eq_degC'],x['lia_unique_id']))
    assert len({x['fingerprint'] for x in rows})==7644
    assert not set(x['fingerprint'] for x in rows).intersection(r.read(r.TX/'cache_audit.json')['fingerprints'])
    r.write(r.TX/'spinup_matrix.json',rows)
    r.write(r.TX/'inputs.json',{'per_glacier':{sid:r.ref(r.TX/'g'/f'{sid}.json') for sid in inputs},'globals':global_inputs,'copied_forward_climates':copied,'new_same_contract_forward_climates':generated})
    report=tests(rows,inputs)
    maximum=max(len(str(r.OUT/'s'/x['key']/'cp/.t2000.999999.9999999999999999999/c.json')) for x in rows);assert maximum<160
    manifest={'schema':'lia-cpu-melt-factor-transaction-v10.279','transaction_id':'lia-cpu-melt-v10.279','created_at':r.now(),'user_authorization':'确定; approve melt_f0.7/1.3 for182systems x21 spinup offsets, CPU20, stable states to fixed historical forward','code':code,'matrix':r.ref(r.TX/'spinup_matrix.json'),'inputs':r.ref(r.TX/'inputs.json'),'rules':r.ref(r.TX/'rules.json'),'technical_tests':r.ref(r.TX/'technical_tests.json'),'cache_audit':r.ref(r.TX/'cache_audit.json'),'population':r.ref(D/'population_191.json'),'baseline_reuse':r.ref(D/'baseline_reuse_4011.json'),'retained77':r.ref(D/'retained_provisional_candidates_77.json'),'runtime_evidence':r.ref(D/'runtime_evidence.json'),'source_design':r.ref(D/'design_evidence.json'),'output_root':str(r.OUT),'workers':20,'internal_threads':r.THREADS,'spinup_cells':7644,'forward_cells_maximum':7644,'forward_actual_count':None,'forward_matrix':'materialized separately from technically valid actual stable parent result/state hashes','checkpoint_interval_years':30,'maximum_temporary_path_chars':maximum,'resource_before':{'memory_percent':psutil.virtual_memory().percent,'available_memory_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(r.OUT).free},'accepted_ledger_effect':0,'gpu_affected':False,'automatic_retry':False,'new_dynamic_preflight':False}
    r.write(r.TX/'transaction_manifest.json',manifest)
    print(json.dumps({'prepared':True,'cells':len(rows),'systems':len(inputs),'technical_tests':report['status'],'transaction':r.ref(r.TX/'transaction_manifest.json')}),flush=True)
if __name__=='__main__':main()
