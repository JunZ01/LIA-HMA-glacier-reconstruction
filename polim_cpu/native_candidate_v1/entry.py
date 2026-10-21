"""Independent v35 candidate: check CONFIG; run CONFIG; resume CONFIG; status OUTPUT."""
from .control import ROOT,RUN,E,PLAN,INPUT,read,write,digest,arrays,gate
from .state import identity,load_bundle,ledger,seal
from polim_cpu import packed_storage_v1 as st
from polim_cpu.c_control_repair_v1.process import launch,native_result
from polim_cpu.local_b_batch_v1.gpu_launch import env as gpu_env
import argparse,sys,time,traceback,os,json
from pathlib import Path
import numpy as np

def check_config(path,registered=False):
    c=read(path)
    if c['case'] not in ('A','R'):raise ValueError('Only frozen A/R cases are supported')
    if (c['start_year'],c['end_year'])!=((1552,1554) if c['case']=='A' else (1553,1554)):raise ValueError('Window is outside this candidate registration')
    if c['parameters']!=read(ROOT/'polim_cpu/runs/v35-annual-checkpoint-v1/1551-exact/config.json')['parameters']:raise ValueError('Physics parameters differ from original v35')
    years=arrays(c['climate'])['years']
    if not all(np.count_nonzero(years==y)==1 for y in range(c['start_year'],c['end_year'])):raise ValueError('Missing or duplicated climate year; extrapolation forbidden')
    ident=identity(c['mesh'],c['climate'],c['parameters'])
    if c['case']=='A':
        a,m=st.restore(c['initial_frame'],c['initial_contract'])
        if m['year']!=c['start_year'] or m['row']['BP_residual']>1e-6 or m['row']['mass_relative']>1e-10:raise ValueError('Native initial checkpoint not accepted at requested time')
    elif Path(c['resume_bundle']).exists():
        a,m,b=load_bundle(c['resume_bundle'],ident)
        if m['year']!=c['start_year']:raise ValueError('Resume start year mismatch')
    elif registered:raise ValueError('Registered dependency checkpoint missing')
    else:return dict(passed=True,dependency='A accepted1553 bundle, not yet generated',identity=ident,model_work=0)
    return dict(passed=True,year=m['year'],H_sha256=st.array_hash(a['h']),u_sha256=st.array_hash(a['u']),identity=ident,model_work=0)

def classify(exc):
    text=str(exc)
    if isinstance(exc,(FileNotFoundError,ValueError)):return 'input-error'
    if any(x in text for x in ('budget','CFL','Newton','Line search','CG','donor limiting','mass','numerical acceptance')):return 'numerical-stop'
    return 'environment-or-technical-error'

def execute(path,mode):
    c=read(path);case=c['case'];root=Path(c['output']);root.mkdir(exist_ok=False)
    start=time.perf_counter();command_epoch=float(os.environ.get('NATIVE_CANDIDATE_STARTED',time.time()));last=c['start_year'];results=[];failure=None;eng=True;bundle_path=None
    write(root/'request.json',dict(mode=mode,config=str(Path(path).resolve()),start_epoch=time.time()))
    try:
        p=gate('entry:'+case,True)
        if digest(path)!=p['case_config_hashes'][case]:raise ValueError('Config not frozen')
        if mode!=('run' if case=='A' else 'resume'):raise ValueError('Wrong entry operation')
        check_config(path,True);ident=identity(c['mesh'],c['climate'],c['parameters']);token=digest(PLAN)
        case_deadline=min(p['batch_start_epoch']+2400,command_epoch+(600 if case=='A' else 300))
        z=arrays(c['mesh']);area=z['geo_dual_area'];(root/'checkpoints').mkdir()
        if case=='A':
            a,m=st.restore(c['initial_frame'],c['initial_contract']);origin=dict(year=1552,V0=float(area@a['h']),inherited_native_source=m['smb_volume'],inherited_native_export=m['exported'],initial_frame=c['initial_frame'])
            window=ledger(origin['V0'],origin['V0'],0.,0.);frame=c['initial_frame'];contract=c['initial_contract'];initial_bundle=None
        else:
            a,m,b=load_bundle(c['resume_bundle'],ident);origin=b['window_origin'];window=b['window_ledger'];frame=b['frame'];contract=b['contract'];initial_bundle=b
            write(root/'restored-bundle.json',dict(source=c['resume_bundle'],sha256=digest(c['resume_bundle']),bundle=b))
        for year in range(c['start_year'],c['end_year']):
            nr=None
            yr_start=time.perf_counter();year_epoch=command_epoch if year==c['start_year'] else time.time();deadline=min(case_deadline,year_epoch+300)
            gate('year:'+case,False)
            if time.time()>=deadline:raise RuntimeError('Annual command budget before initialization')
            yr=root/f'year-{year}';native=yr/'native';native.mkdir(parents=True)
            write(native/'contract.json',dict(contract=token))
            # Rebind only the contract of the validated native frame, retaining all scientific metadata.
            st.save(native,f"frame-{m['step']:06d}",a['h'],a['u'],m,token)
            loaded,lm=st.restore(native/f"frame-{m['step']:06d}",token)
            if not np.array_equal(loaded['h'],a['h']) or not np.array_equal(loaded['u'],a['u']):raise ValueError('Native annual import differs')
            source=native/'source';source.mkdir();write(source/'contract.json',dict(contract=token))
            os.link(c['mesh'],source/'input.npz');os.link(c['climate'],source/'climate.npz')
            cache=c.get('shared_forcing') if case=='R' else None
            cfg=dict(source_root=str(source),start_year=year,end_year=year+1,mode='exact',parameters=c['parameters'],shared_forcing=cache)
            write(native/'config.json',cfg)
            write(yr/'import.json',dict(source_frame=frame,source_contract=contract,metadata_before=m,metadata_after=lm,H_sha256=st.array_hash(a['h']),u_sha256=st.array_hash(a['u']),window_ledger=window,identity=ident))
            env=gpu_env();env['CN_C_DEADLINE']=str(deadline)
            process=launch([p['v35']['runtime'],'-m','polim_cpu.native_candidate_v1.worker',str(native)],native,native,deadline,env)
            nr=native_result(native)
            if process['exit_code'] or nr['status']!='completed':raise RuntimeError('Native '+str(nr.get('error',nr['status'])))
            data=arrays(native/'annual-ledger.npz');audit=read(native/'endpoint-audit.json');newmeta=audit['metadata'];newrows=[r for r in nr['rows'] if year<r['year']<=year+1]
            current=ledger(float(area@a['h']),float(area@data['H']),float(data['actual_nodes'].sum()),float(data['boundary_volume'].sum()))
            window=ledger(origin['V0'],float(area@data['H']),newmeta['smb_volume']-origin['inherited_native_source'],newmeta['exported']-origin['inherited_native_export'])
            native_cumulative=ledger(newmeta['V0'],float(area@data['H']),newmeta['smb_volume'],newmeta['exported'])
            threshold=64*np.finfo(np.float64).eps*np.maximum(1.,np.maximum(abs(z['bed']),abs(z['bed']+data['H'])))
            dry=np.repeat(data['H']<=threshold,4)
            proof=dict(year=year+1,BP=audit['BP'],native_inline_rows_pass=bool(newrows) and all(r['BP_residual']<=1e-6 and r['mass_relative']<=1e-10 and not r['skipped'] for r in newrows),annual=current,window=window,inherited=native_cumulative,finite=bool(np.isfinite(data['H']).all() and np.isfinite(data['u']).all()),nonnegative=bool((data['H']>=0).all()),actual_source_valid=bool((data['unmet_nodes']>=-1e-8).all()),source_native_increment_error=abs(current['source']-(newmeta['smb_volume']-m['smb_volume'])),export_native_increment_error=abs(current['exported']-(newmeta['exported']-m['exported'])))
            proof['dry_velocity_zero']=bool((data['u'][dry]==0).all())
            proof['passed']=proof['BP']<=1e-6 and proof['native_inline_rows_pass'] and proof['finite'] and proof['nonnegative'] and proof['dry_velocity_zero'] and proof['actual_source_valid'] and all(proof[k]['mass_relative']<=1e-10 for k in ('annual','window','inherited')) and max(proof['source_native_increment_error'],proof['export_native_increment_error'])<=1e-10*max(current['V0'],1.)
            write(yr/'acceptance.json',proof)
            if not proof['passed']:raise RuntimeError('Native numerical acceptance failed')
            frame=native/f"frame-{newmeta['step']:06d}";contract=token
            aa,mm=st.restore(frame,contract)
            if not np.array_equal(aa['h'],data['H']) or not np.array_equal(aa['u'],data['u']):raise ValueError('Endpoint frame and audit arrays mismatch')
            forcing=read(native/'forcing.json');entry=read(native/'entry-audit.json')
            if entry['H_sha256']!=st.array_hash(a['h']) or entry['u_sha256']!=st.array_hash(a['u']) or entry['metadata']!=lm:raise ValueError('Actual worker restore metadata mismatch')
            bundle_path=root/'checkpoints'/f'accepted-{year+1}.json';seal(bundle_path,frame,contract,ident,window,origin,forcing,native/'annual-ledger.npz')
            from polim_cpu.cn_validation_v1.evaluate import annual
            l=dict(window,exported=window['exported']);evaluation=annual(yr,year+1,data['H'],data['u'],l,z,1552,1552,audit['BP']);eng=eng and evaluation['passed']
            row=dict(year=year+1,status='accepted',command_gpu=process,year_complete_seconds=time.time()-year_epoch,year_body_seconds=time.perf_counter()-yr_start,acceptance=proof,engineering=evaluation,work=read(native/'native-costs.json'),checkpoint=str(bundle_path),native_result=nr)
            write(yr/'year-result.json',row);results.append(row);last=year+1;a,m=aa,mm
            print(json.dumps(dict(case=case,accepted=last,annual_engineering=evaluation['passed'],seconds=row['year_complete_seconds'])),flush=True)
            if time.time()>=deadline:raise RuntimeError('Annual300s budget after necessary output')
    except BaseException as exc:
        current_native=locals().get('nr') or {}
        failure=dict(category=classify(exc),error=repr(exc),traceback=traceback.format_exc(),native_output=str(locals().get('native','not-started')),last_durable_year=last,last_accepted_native_row=(current_native.get('rows') or [None])[-1]);write(root/'failure.json',failure)
    finally:
        out=dict(case=case,status='completed' if failure is None and last==c['end_year'] else 'stopped',last_accepted_year=last,checkpoint=str(bundle_path) if bundle_path else None,engineering_passed=eng and failure is None,years=results,failure=failure,worker_wall_seconds=time.perf_counter()-start)
        write(root/'science-result.json',out)
        try:gate('exit:'+case,True)
        except BaseException as ex:
            out.update(status='stopped',protection_error=repr(ex));write(root/'protection-failure.json',out)
        write(root/'result.json',out)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    for x in ('check','run','resume','status'):sub.add_parser(x).add_argument('path',type=Path)
    a=p.parse_args()
    if a.operation=='check':r=check_config(a.path)
    elif a.operation=='status':
        r=read(a.path/'result.json') if (a.path/'result.json').exists() else read(a.path/'science-result.json') if (a.path/'science-result.json').exists() else dict(status='no-completed-result',path=str(a.path))
    else:r=execute(a.path,a.operation)
    print(json.dumps({k:v for k,v in r.items() if k!='years'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
