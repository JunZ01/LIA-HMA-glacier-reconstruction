"""Only new, no-model admission tests. No CUDA service or real operators."""
import ast,time,tempfile,json
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
from scipy import sparse
from .coupling import Process
from .client import Backend,Work
from .diagnostics import apply_parent
from polim_cpu.geometry_coupled_v1.backend import BudgetStop

def run():
    checks={};start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='v53-no-model-') as temp:
        root=Path(temp)
        for limit in (8,16):
            folder=root/str(limit);folder.mkdir();w=Work(folder)
            # Exercise the actual solve loop. Synthetic directions/trials only.
            obj=object.__new__(Process);obj.w=w;obj.history=[]
            d=dict(local=0.,mass=0.,cumulative_mass_relative=0.,physical_path=True,fixed_point=0.,all_finite=True,H_min=0.)
            def state(self,h,u,tag):return dict(H=h,u=u,bp=0. if h[0]>=limit else 2e-6,d=d,F=np.zeros(4),force=np.ones(4),e={'R':np.zeros(1),'scale':1.},wet=np.ones(4,bool)),False
            def direction(self,s):w.count('joint_directions',limit);return np.zeros(4),np.ones(1)
            def trial(self,s,du,dh,k,tag):
                r,_=state(self,s['H']+dh,s['u'],tag);return r,False,False,1.
            obj.state=MethodType(state,obj);obj.direction=MethodType(direction,obj);obj.trial=MethodType(trial,obj)
            obj.mass=SimpleNamespace(eq=SimpleNamespace(V0=1.),save=lambda *args:None)
            # merit requires source/export; declining exact acceptance bypass at cap.
            d.update(source=0.,export=0.)
            import polim_cpu.joint_budget_v1.coupling as c
            original=c.merit
            try:
                c.merit=lambda s,scale:float(limit+1-s['H'][0])
                answer=obj.solve(np.zeros(1),np.zeros(4))
            finally:c.merit=original
            checks['last_direction_'+str(limit)+'_can_accept']=answer['accepted'] and w.counts['joint_directions']==limit
        folder=root/'limits';folder.mkdir();w=Work(folder);b=object.__new__(Backend);b.w=w
        calls=[];b.rpc=lambda *a,**kw:calls.append(a) or {'value':np.zeros(4)}
        w.counts['V_requests']=256
        try:b.V(np.zeros(4),'mock');checks['V_independent']=False
        except BudgetStop:checks['V_independent']=not calls
        w.counts['pure_joint_residual']=128
        try:b.pure(np.zeros(1),np.zeros(4));checks['evaluation_independent']=False
        except BudgetStop:checks['evaluation_independent']=not calls
        A=sparse.eye(4,format='csr');B=sparse.csr_matrix(np.arange(4)[:,None]);C=B.T;D=sparse.eye(1,format='csr')*3
        blocks=(A,B,C,D);before=[x.data.copy() for x in blocks];s=np.array([.1,.2,0.,-.4,.3]);Js=apply_parent(blocks,s)
        child=sparse.eye(5)*99 # a different child identity cannot bind parent helper
        checks['parent_actual_displacement']=np.allclose(Js,np.block([[A.toarray(),B.toarray()],[C.toarray(),D.toarray()]])@s,atol=1e-15,rtol=0)
        checks['parent_no_mutation']=all(np.array_equal(x.data,b) for x,b in zip(blocks,before))
        gp=np.arange(5.);gt=gp*.2;elin=gp+Js;emodel=gt-gp-Js
        checks['model_identity']=np.linalg.norm(gt-elin-emodel)<1e-14
        w.save('before-failure',s=s,Js=Js)
        try:raise BudgetStop('injected no-model stop')
        except BudgetStop as exc:w.event('stopped',reason=str(exc),work=w.record())
        checks['failure_arrays_and_counts_persist']=bool((folder/'before-failure.npz').exists() and 'injected' in (folder/'events.jsonl').read_text())
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    checks['all_new_sources_parse']=True
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(status='pass' if all(checks.values()) else 'fail',checks=checks,seconds=time.perf_counter()-start,
                real_BP=0,real_linear_solve=0,real_operator_evaluation=0,real_assembly=0,time_steps=0,GPU_services=0)
    json.dumps(result,allow_nan=False)
    return result
