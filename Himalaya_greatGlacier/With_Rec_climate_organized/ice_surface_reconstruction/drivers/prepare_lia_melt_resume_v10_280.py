"""Freeze exact residual queue and validate user-authorized saved states, no dynamics."""
from . import lia_cpu_melt_resume_v10_280 as r
from . import lia_cpu_melt_factor_v10_279 as old
from pathlib import Path
import ast,copy,inspect,json,os,subprocess
def main():
    import numpy as np,psutil
    from . import lia_all191_adaptive_spinup_v10_260 as base
    validation=subprocess.run([r.PYTHON,str(r.DRIVERS/'validate_lia_execution_state_v1.py')],cwd=r.ROOT,capture_output=True,text=True,check=True)
    old.contract()
    live=[{'pid':p.pid,'command':p.info.get('cmdline')} for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info.get('name') or '').startswith('python')]
    assert not live,'unexpected-active-python-review-before-preparation'
    assert not r.TX.exists() and not r.OUT.exists(),'version-root-collision'
    r.TX.mkdir();r.OUT.mkdir();r.write(r.TX/'validator-before.json',json.loads(validation.stdout))
    rows=r.read(old.TX/'spinup_matrix.json');completed=[];partial=[];unstarted=[];invalid=[];failures=[];queue=[];index={};temps=[]
    for row in rows:
        key=row['key'];rp=old.OUT/'s'/key/'result.json';directory=rp.parent
        if rp.exists():
            result=r.read(rp)
            assert result['key']==key and result['fingerprint']==row['fingerprint']
            if result['technical_failure']:failures.append({'key':key,'source':r.ref(rp)});continue
            assert result['terminal'] and result['year']==900
            for name in ('final_checkpoint','annual_history','final_state'):r.check(result[name])
            cp=r.read(r.ROOT/result['final_checkpoint']['path']);h=r.read(r.ROOT/result['annual_history']['path'])
            assert cp['fingerprint']==row['fingerprint'] and cp['parameters']=={n:row[n] for n in r.PARAMS}
            assert [x['year'] for x in h]==list(range(901)) and cp['history_sha256']==r.sha(r.ROOT/result['annual_history']['path'])
            assert cp['state_sha256']==r.sha(r.ROOT/result['final_state']['path'])
            completed.append({'key':key,'fingerprint':row['fingerprint'],'lia_unique_id':row['lia_unique_id'],'stable':result['stable'],'result':r.ref(rp),'checkpoint':result['final_checkpoint'],'state':result['final_state'],'history':result['annual_history'],'runtime_seconds':result['runtime_seconds']})
        else:
            binding={'key':key,'fingerprint':row['fingerprint'],'lia_unique_id':row['lia_unique_id'],'checkpoint':None,'action':'never-started','start_year':0}
            if directory.exists():
                paths=sorted([p for p in (directory/'cp').glob('*/c.json') if p.parent.name.isdigit()],key=lambda p:int(p.parent.name),reverse=True)
                assert paths,'started-without-committed-checkpoint'
                chosen=paths[0];meta=r.read(chosen)
                # Only local, prior-controller-generated, hash-verified checkpoints
                # named in the explicit resume queue are deserialized.
                cp,fls,h=r.load_cp(chosen.parent,row)
                assert cp['year']%30==0 and cp['year']<900
                binding.update(checkpoint=r.ref(chosen),state=r.ref(chosen.parent/'s.gz'),history=r.ref(chosen.parent/'h.json'),action='resume-exact-checkpoint',start_year=cp['year'])
                partial.append(binding)
            else:unstarted.append(key)
            queue.append(binding);index[key]=binding
        if directory.exists():temps.extend(str(p.relative_to(r.ROOT)) for p in (directory/'cp').glob('.t*'))
    assert len(completed)==1152 and not any(x['stable'] for x in completed)
    assert len(partial)==19 and len(unstarted)==6473 and len(queue)==6492 and not failures
    assert len(completed)+len(queue)==7644 and not({x['key'] for x in completed}&set(index))
    queue.sort(key=lambda x:(x['action']!='resume-exact-checkpoint',next(i for i,a in enumerate(rows) if a['key']==x['key'])))
    r.write(r.TX/'recovery_queue.json',queue);r.write(r.TX/'queue_index.json',index);r.write(r.TX/'reused_completed.json',completed)
    parent=r.read(old.OUT/'r/parent.json');status=r.read(old.OUT/'r/status.json');logs=[]
    for p in sorted((old.OUT/'l').glob('*.err.log')):
        text=p.read_text(encoding='utf-8-sig',errors='replace')
        logs.append({'file':r.ref(p),'traceback':'Traceback (most recent call last)' in text,'error_marker':'Error:' in text})
    assert not any(x['traceback'] or x['error_marker'] for x in logs)
    terminal={'status':'process-tree-vanished-ungraceful-interruption','observed_at':r.now(),'source_parent':r.ref(old.OUT/'r/parent.json'),'source_last_status':r.ref(old.OUT/'r/status.json'),'parent_pid':parent['pid'],'parent_alive':False,'workers_alive':0,'last_status_at':status['updated_at'],'completed_valid':1152,'stable':0,'nonconverged':1152,'member_technical_failures':0,'interrupted':19,'never_started':6473,'remaining':6492,'forward_started':False,'source_results':r.ref(r.TX/'reused_completed.json'),'recovery_queue':r.ref(r.TX/'recovery_queue.json'),'uncommitted_temporary_directories_preserved':temps,'stderr':logs,'parent_stderr':r.ref(old.OUT/'parent.err.log'),'root_cause':'No caught Python traceback or recorded member technical failure; exact external actor is unknown. User was informed and explicitly requested continuation. Similar process-session loss was documented in Section564. This is control/lifecycle interruption, not failed scientific convergence.','gpu_affected':False,'accepted_ledger_effect':0}
    r.write(r.TX/'source_terminal.json',terminal)
    base._configure_oggm(r.OUT/'test/w',r.OUT/'test/w');tests=[]
    for binding in partial:
        row=old.row_for('spinup',binding['key']);r.check(binding['checkpoint']);r.check(binding['state']);r.check(binding['history'])
        cp,fls,h=r.load_cp((r.ROOT/binding['checkpoint']['path']).parent,row)
        model=base.make_model(fls,r.mb_for(row,'spinup'),row['glen_a_pa_3_s_1'],cp['year'])
        for a,b in zip(fls,model.fls):
            for n in ('thick','section','bed_h','surface_h','widths_m'):np.testing.assert_array_equal(getattr(a,n),getattr(b,n))
        got=base.snapshot(model,cp['year'])
        for n in ('length_m','volume_m3','area_m2','minimum_thickness_m','maximum_thickness_m'):np.testing.assert_allclose(got[n],h[-1][n],rtol=1e-7,atol=0)
        saved=r.save_cp(r.OUT/'test'/row['key'],model,cp['year'],row,h,cp['counters']);c,ff,hh=r.load_cp(saved,row)
        assert hh==h and c['counters']==cp['counters'] and model.yr==cp['year']
        wrong={**row,'fingerprint':'INVALID'}
        try:r.load_cp(saved,wrong)
        except ValueError:pass
        else:raise AssertionError('corrupt-identity-accepted')
        tests.append({'key':row['key'],'resumed_year':cp['year'],'all_state_arrays_exact':True,'complete_history_preserved':True,'snapshot_restore_rtol':1e-7,'zero_dynamics_steps':True})
    def numerical_loop(function):
        tree=ast.parse(inspect.getsource(function))
        return next(ast.dump(x,include_attributes=False) for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='year')
    assert numerical_loop(r.worker)==numerical_loop(old.worker)
    assert r.mb_for is old.mb_for and r.score_record is old.score_record and r.save_cp is old.save_cp and r.load_cp is old.load_cp
    calls=[]
    def locked(src,dst):calls.append(1);raise PermissionError('injected-lock-static-test')
    fallback=r.status(r.OUT/'test/status.json',{'year':0},locked);assert fallback and len(calls)==4
    assert r.status(r.OUT/'test/status.json',{'year':30})
    evidence={'status':'pass','resume_cases':tests,'numerical_loop_ast_identical':True,'same_MB_checkpoint_restore_and_scorer_functions':True,'status_lock_fallback_then_progress':True,'prior_real_continuation_and_scoring_tests':r.ref(old.TX/'technical_tests.json'),'new_dynamics_steps':0,'new_scientific_preflight':False}
    r.write(r.TX/'technical_tests.json',evidence)
    code=[r.ref(Path(__file__)),r.ref(Path(r.__file__))]
    mean=sum(x['runtime_seconds'] for x in completed)/len(completed);equivalents=sum((900-x['start_year'])/900 for x in queue)
    manifest={'schema':'lia-cpu-melt-exact-recovery-v10.280','transaction_id':'lia-cpu-melt-resume-v10.280','created_at':r.now(),'user_authorization':'那继续 after being informed the process tree exited without a recorded cause','original_transaction':r.ref(old.TX/'transaction_manifest.json'),'source_terminal':r.ref(r.TX/'source_terminal.json'),'matrix':r.ref(r.TX/'recovery_queue.json'),'queue_index':r.ref(r.TX/'queue_index.json'),'reused_completed':r.ref(r.TX/'reused_completed.json'),'technical_tests':r.ref(r.TX/'technical_tests.json'),'code':code,'workers':20,'internal_threads':r.THREADS,'remaining_spinups':6492,'resume_members':19,'never_started':6473,'reused_completed_cells':1152,'scientific_fingerprint_unchanged':True,'changed_scientific_parameters':[],'source_readonly':str(old.OUT),'output_root':str(r.OUT),'checkpoint_interval_years':30,'forward':'all technically valid stable parents to fixed historical1470-2000 after spinup matrix completes, unchanged six gates','source_stable_cells':0,'retained_provisional_systems':9,'retained_provisional_cells':77,'baseline_one_rerun':False,'new_dynamic_preflight':False,'automatic_retry':False,'runtime':{'observed_mean_member_seconds':mean,'remaining900_equivalent_members':equivalents,'ideal_spinup_hours_cpu20':mean*equivalents/20/3600,'caveat':'Changed temperature/melt mix and host load may alter runtime; not an upper bound. Forward time additional.'},'process_ownership':'Windows-WMI-created-hidden-supervisor-with-OS-writer-lock; no detached Start-Process, no service or recurring schedule. Existing project supervisory pattern reused without importing or affecting GPU code.','stop_policy':'No automatic batch stop for scientific nonconvergence; invalid>15percent after120 advisory only; memory>80percent or total status storage unavailable pauses dispatch for user decision; single member technical failures isolated. No automatic restart on supervisor/controller exit.','resource_before':{'available_memory_bytes':psutil.virtual_memory().available,'memory_percent':psutil.virtual_memory().percent},'accepted_ledger_effect':0,'gpu_affected':False}
    r.write(r.TX/'transaction_manifest.json',manifest);r.contract()
    print(json.dumps({'prepared':True,'reused':1152,'resume':19,'new':6473,'tests':'pass','ideal_spinup_hours':manifest['runtime']['ideal_spinup_hours_cpu20'],'manifest':r.ref(r.TX/'transaction_manifest.json')}),flush=True)
if __name__=='__main__':main()
