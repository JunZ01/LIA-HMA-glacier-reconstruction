"""No-model exercise of the actual repaired orchestration, not a parallel toy writer."""
from .control import ROOT,E,read,write,digest
from . import entry
from .state import seal,ledger
from polim_cpu import packed_storage_v1 as st
from polim_cpu.c_control_repair_v1.process import launch
from pathlib import Path
from unittest.mock import patch
import numpy as np
import sys,time,json,traceback

def run():
    start=time.perf_counter();tests={};records=[];out=E/'preparation-artifacts';out.mkdir(exist_ok=False)
    try:
        h=np.array([1.,3.]);u=np.zeros(8);a={'h':h,'u':u};ident={'test':'no-model'}
        meta=dict(year=1553.,step=34,last_solve=1553.,V0=4.,smb_volume=0.,exported=0.,row=dict(BP_residual=0.,mass_relative=0.),window_rows=[])
        st.save(out,'frame-000034',h,u,meta,'synthetic')
        window=ledger(4.,4.,0.,0.);origin=dict(year=1552,V0=4.,inherited_native_source=0.,inherited_native_export=0.)
        seal(out/'accepted-1553.json',out/'frame-000034','synthetic',ident,window,origin)
        for i,frame in enumerate((str(out/'frame-000034'),out/'frame-000034')):
            entry.record_import(out/f'import-{i}.json',frame,'synthetic',meta,meta,a,window,ident)
            tests[f'actual-import-boundary-{i}']=read(out/f'import-{i}.json')['source_frame']==str(frame)
        try:json.dumps({'source_frame':out/'frame-000034'})
        except TypeError:tests['original-error-reproduced-without-model']=True
        np.savez(out/'mesh.npz',geo_dual_area=np.ones(2),bed=np.zeros(2))
        np.savez(out/'climate.npz',years=np.array([1553]))
        write(out/'mock-plan.json',{'test':'only synthetic'})
        current=['success']
        def fake_launch(command,records,output,deadline,env):
            native=Path(output);token=read(native/'contract.json')['contract'];frame=next(native.glob('frame-*'));aa,mm=st.restore(frame,token)
            write(native/'launch.json',dict(command=command,model_work=0));(native/'worker.log').write_text('Synthetic native outputs; no model solved.\n')
            if current[0]=='failure':raise RuntimeError('Injected child failure')
            H=aa['h']+.5;U=aa['u'].copy();nm=dict(mm,year=1554.,step=mm['step']+1,last_solve=1554.,smb_volume=mm['smb_volume']+1.)
            row=dict(year=1554.,BP_residual=0.,mass_relative=0.,skipped=False);nm['row']=row;nm['window_rows']=[row]
            st.save(native,f"frame-{nm['step']:06d}",H,U,nm,token);_,nm=st.restore(native/f"frame-{nm['step']:06d}",token)
            np.savez(native/'annual-ledger.npz',H=H,u=U,actual_nodes=np.full(2,.5),unmet_nodes=np.zeros(2),boundary_volume=np.zeros(2))
            np.savez(native/'forcing.npz',smb=np.full(2,.5),surface=aa['h'])
            write(native/'forcing.json',dict(path=str(native/'forcing.npz'),sha256=digest(native/'forcing.npz')))
            write(native/'entry-audit.json',dict(H_sha256=st.array_hash(aa['h']),u_sha256=st.array_hash(aa['u']),metadata=mm))
            write(native/'endpoint-audit.json',dict(BP=0.,metadata=nm));write(native/'native-costs.json',dict(model_work=0))
            write(native/'result-mock.json',dict(status='completed',rows=[row]));return dict(exit_code=0,seconds=0.,mock=True)
        for name,case in [('A','A'),('R','R'),('failed','A')]:
            current[0]='failure' if name=='failed' else 'success'
            cfg=dict(case=case,start_year=1553,end_year=1554,output=str(out/name),resume_bundle=str(out/'accepted-1553.json'),mesh=str(out/'mesh.npz'),climate=str(out/'climate.npz'),parameters={})
            cp=out/f'config-{name}.json';write(cp,cfg)
            p=dict(case_config_hashes={case:digest(cp)},batch_start_epoch=time.time(),v35={'runtime':'NO-MODEL'})
            with patch.object(entry,'PLAN',out/'mock-plan.json'),patch.object(entry,'gate',return_value=p),patch.object(entry,'identity',return_value=ident),patch.object(entry,'check_config',return_value={'passed':True}),patch.object(entry,'gpu_env',return_value={}),patch.object(entry,'launch',side_effect=fake_launch),patch('polim_cpu.cn_validation_v1.evaluate.annual',return_value={'passed':True,'synthetic':True}):
                r=entry.execute(cp,'run' if case=='A' else 'resume')
            tests[name+'-actual-orchestration']=r['status']==('stopped' if name=='failed' else 'completed')
            tests[name+'-evidence-written']=(out/name/'result.json').exists() and (out/name/'year-1553/import.json').exists()
            if name=='failed':tests['failed-prefix-preserved']=r['last_accepted_year']==1553 and (out/name/'failure.json').exists()
        tests['endpoint-ledger']=read(out/'A/checkpoints/accepted-1554.json')['window_ledger']['mass_relative']==0.
        for mode in ('outer','outer-fail'):
            d=out/mode;(d/'supervisor').mkdir(parents=True);(d/'native').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(d),mode],d/'supervisor',d,time.time()+20)
            records.append(r);tests[mode+'-subprocess']=(r['exit_code']==0)==(mode=='outer')
            tests[mode+'-layout']=all((d/l/f).exists() for l in ('supervisor','native') for f in ('launch.json','worker.log'))
        d=out/'collision';d.mkdir();(d/'launch.json').write_text('{}')
        r=launch([sys.executable,'-c','raise Exception("must not launch")'],d,d,time.time()+10);records.append(r)
        tests['exclusive-create']=r['exit_code']==-2 and 'pid' not in r
        for case in ('A','R'):tests['actual-config-'+case]=entry.check_config(Path(__file__).parent/'configs'/f'{case}.json',True)['passed']
        for name in ('worker.py','state.py'):tests[name+'-byte-identical']=digest(Path(__file__).parent/name)==digest(ROOT/'polim_cpu/native_candidate_v1'/name)
    except BaseException as ex:
        tests['unexpected-error']=False;write(out/'unexpected-error.json',dict(error=repr(ex),traceback=traceback.format_exc()))
    result=dict(passed=all(tests.values()),tests=tests,processes=records,seconds=time.perf_counter()-start,real_BP=0,real_transport=0,real_residual=0,real_matrix=0,real_steps=0)
    write(E/'preparation.json',result)
    assert result['passed'],tests

if __name__=='__main__':run()
