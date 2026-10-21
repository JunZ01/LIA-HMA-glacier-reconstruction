import os,sys,time,traceback,json
from pathlib import Path
import numpy as np
from .control import ROOT,RUN,OLD,PLAN,INPUT,read,write,arrays,digest,ahash,gate
from polim_cpu.c_control_repair_v1.process import launch,native_result
SOURCE=INPUT.parent

def common(p):
    from polim_cpu.geometry_window_v1.ledger import restore
    t=time.perf_counter();root=RUN/'common';root.mkdir();h,m=restore(p['common_initial']);fr=read(OLD/'A/forcing-1553.json')
    assert m['year']==1553. and m['ledger']['cumulative_mass_relative']<=1e-10
    assert fr['H_sha256']==ahash(h['H']) and digest(p['common_forcing'])==fr['sha256']
    assert read(OLD/'A/annual-1553.json')['original_BP']<=1e-6
    os.link(p['common_forcing'],root/'forcing-1553.npz');fr={**fr,'path':str(root/'forcing-1553.npz')}
    write(root/'forcing.json',fr);write(root/'inherited-1552-history.json',m)
    write(root/'preparation.json',dict(seconds=time.perf_counter()-t,source=p['common_initial'],H=ahash(h['H']),u=ahash(h['u']),forcing=fr,
        BP_pair='v48 accepted1553 originalBP evidence; no new equilibrium solve',new_smb_generated=False))
    return fr

def native_case(case,p,shared,deadline):
    from polim_cpu.cn_validation_v1.ledger import restore,checkpoint
    from polim_cpu.cn_validation_v1.evaluate import annual
    from polim_cpu import packed_storage_v1 as st
    started=time.perf_counter();native_deadline=min(deadline,time.time()+300);root=RUN/'C/native';source=root/'source';source.mkdir()
    result={};reason=None
    try:
        common,cm=restore(OLD/'A/checkpoints/accepted-1553.0000.npz');prior,pm=restore(OLD/'A/checkpoints/accepted-1552.7500.npz')
        token=digest(PLAN);write(source/'contract.json',dict(contract=token));write(root/'contract.json',dict(contract=token))
        for name in ['input.npz','climate.npz']:os.link(SOURCE/name,source/name)
        st.save(source,'frame-common',common['H'],common['u'],dict(year=1553.,step=0),token)
        st.save(source,'frame-prior',prior['H'],prior['u'],dict(year=1552.75,step=0),token)
        cfg=dict(source_root=str(source),start_frame='frame-common',previous_frame='frame-prior',start_year=1553.,end_year=1554.,mode='exact',parameters=p['parameters'],shared_forcing=shared['path'])
        write(root/'config.json',cfg)
        env=os.environ.copy();site=Path(p['v35']['torch_site']);env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env.update(PYTHONPATH=str(site)+os.pathsep+str(ROOT),POLIM_TORCH_SITE=str(site),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CN_C_DEADLINE=str(native_deadline),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[p['v35']['runtime'],'-m','polim_cpu.cn_validation_v1.native_v35',str(root)]
        initialization_restore=time.perf_counter()-started
        process=launch(cmd,root,root,native_deadline,env)
        code=process['exit_code'];result=native_result(root)
        if code or result['status']!='completed':raise RuntimeError(str(result.get('error',result['status'])))
        evaluation_start=time.perf_counter();data=arrays(root/'annual-ledger.npz');audit=read(root/'endpoint-audit.json');m=audit['metadata'];area=arrays(INPUT)['geo_dual_area']
        ledger=dict(V0=m['V0'],volume=float(area@data['H']),source=float(data['actual_nodes'].sum()),potential=float(data['potential_nodes'].sum()),
            unmet=float(data['unmet_nodes'].sum()),exported=float(data['boundary_volume'].sum()),cumulative_mass_relative=m['row']['mass_relative'])
        ledger['cumulative_mass_error']=ledger['volume']-ledger['V0']-ledger['source']+ledger['exported']
        assert abs(ledger['exported']-m['exported'])<=1e-10*ledger['V0'] and abs(ledger['source']-m['smb_volume'])<=1e-10*ledger['V0']
        meta=dict(year=1554.,window_start=1553,ledger=ledger,forcing=shared,forcing_used=shared,contract_sha256=token,original_BP=audit['BP'])
        seal=checkpoint(root/'annual-1554.npz',data['H'],data['u'],{k:v for k,v in data.items() if k not in ['H','u']},meta)
        write(root/'annual-seal-1554.json',seal);evaluation=annual(root,1554,data['H'],data['u'],ledger,arrays(INPUT),1553,1552,audit['BP'])
        assert evaluation['passed'], 'Native annual engineering acceptance failed'
        assert result['year']==1554. and audit['year']==1554.
        assert audit['BP']<=1e-6 and ledger['cumulative_mass_relative']<=1e-10
        assert np.isfinite(data['H']).all() and np.isfinite(data['u']).all() and (data['H']>=0).all()
        evaluation_seconds=time.perf_counter()-evaluation_start
        if time.time()>=native_deadline:raise TimeoutError('C300s includes endpoint writing/evaluation')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(root/'stop.json',reason)
    costs=read(root/'native-costs.json') if (root/'native-costs.json').exists() else {}
    out=dict(case='C',status='completed' if reason is None else 'stopped',last_accepted_year=result.get('year',1553.),new_real_wall=time.perf_counter()-started,
        counts=costs.get('counts',{}),work=costs,native=result,reason=reason,timers_overlap=True,output_policy='Native annual cadence with required ledger/audit',initialization_restore_seconds=locals().get('initialization_restore'),annual_evaluation_seconds=locals().get('evaluation_seconds'))
    write(root/'execution.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['native','work']}),flush=True);return out

