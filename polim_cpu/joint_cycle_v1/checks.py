"""Small synthetic recurrence/budget checks only; no real model or CUDA."""
import ast,json,time
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
from .algebra import fgmres,require_block_V,RESTART,MAX_CYCLES,MAX_ARNOLDI
from .client import Work
from .coupling import Process
from .control import E
from polim_cpu.geometry_coupled_v1.backend import BudgetStop,LinearFailure

def run():
    start=time.perf_counter();checks={};records=[];root=E/'preparation-data';root.mkdir()
    # Fixed diagonal target; synthetic flexible preconditioning becomes exact
    # at a chosen request. This deliberately tests restart, not real AMG quality.
    diag=np.geomspace(1.,1e8,64);rhs=np.ones(64)
    def case(name,exact_at,initial_V=0,expect_failure=None):
        folder=root/name;folder.mkdir();w=Work(folder);w.counts['V_requests']=initial_V;requests=[0]
        def pre(v):
            require_block_V(w)
            w.count('V_requests',256);w.count('V_requests',256);requests[0]+=1
            return v/diag if requests[0]>=exact_at else v.copy()
        status='pass';error=None;info=None
        try:
            x,info=fgmres(lambda v:diag*v,pre,rhs,w,'linear')
            true=float(np.linalg.norm(diag*x-rhs)/np.linalg.norm(rhs))
        except (BudgetStop,LinearFailure) as exc:status=type(exc).__name__;error=str(exc);true=None
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
        rec=dict(name=name,status=status,error=error,info=info,true=true,counts=dict(w.counts),seconds=dict(w.times),requests=requests[0]);records.append(rec)
        return folder,w,events,rec
    folder,w,events,r=case('third-cycle',17)
    starts=[x for x in events if x['event']=='FGMRES-cycle-start'];ends=[x for x in events if x['event']=='FGMRES-cycle-end']
    checks['enters_third_after_two_failures']=len(starts)==3 and all(x['true_relative']>.1 for x in ends[:2])
    checks['third_early_exit']=r['status']=='pass' and r['info']['iterations']==17 and r['true']<=.1
    with np.load(folder/'linear-cycle-2-end.npz') as a,np.load(folder/'linear-cycle-3-start.npz') as b:
        checks['third_inherits_cumulative_solution']=np.array_equal(a['solution'],b['solution']) and np.linalg.norm(b['solution'])>0
        checks['third_inherits_true_residual']=np.array_equal(a['true_residual'],b['true_residual'])
    _,_,_,r=case('last-Arnoldi-and-V',24,208)
    checks['24th_Arnoldi_can_pass_at_256V']=r['status']=='pass' and r['info']['iterations']==24 and r['counts']['V_requests']==256
    _,_,_,r=case('two-V-not-direction-reservation',1,254)
    checks['remaining_two_V_can_converge']=r['status']=='pass' and r['counts']['V_requests']==256 and r['requests']==1
    folder,w,events,r=case('one-V-left',1,255,BudgetStop)
    checks['no_partial_block_over_budget']=r['status']=='BudgetStop' and r['counts']['V_requests']==255 and r['requests']==0
    checks['budget_stop_saved']=bool((folder/'linear-interrupted.npz').exists()) and any(x['event']=='FGMRES-interrupted' for x in events)
    folder,w,events,r=case('24-step-failure',100)
    checks['24_stop_no_fourth_cycle']=r['status']=='LinearFailure' and r['counts']['joint_Arnoldi']==24 and len([x for x in events if x['event']=='FGMRES-cycle-start'])==3
    checks['failure_residual_saved']=bool((folder/'linear.npz').exists()) and events[-1]['event']=='joint-linear-failure'
    # Actual outer solve loop, synthetic states only: a direction at the cap
    # still forms/adopts its candidate and executes final acceptance.
    folder=root/'last-direction';folder.mkdir();w=Work(folder);obj=object.__new__(Process);obj.w=w;obj.history=[]
    d=dict(local=0.,mass=0.,cumulative_mass_relative=0.,physical_path=True,fixed_point=0.,all_finite=True,H_min=0.,source=0.,export=0.)
    def state(self,h,u,tag):return dict(H=h,u=u,bp=0. if h[0]>=16 else 2e-6,d=d,F=np.zeros(4),force=np.ones(4),e={'R':np.zeros(1),'scale':1.},wet=np.ones(4,bool)),False
    def direction(self,s):w.count('joint_directions',16);return np.zeros(4),np.ones(1)
    def trial(self,s,du,dh,k,tag):
        r,_=state(self,s['H']+dh,s['u'],tag);return r,False,False,1.
    obj.state=MethodType(state,obj);obj.direction=MethodType(direction,obj);obj.trial=MethodType(trial,obj)
    obj.mass=SimpleNamespace(eq=SimpleNamespace(V0=1.),save=lambda *args:None)
    from . import coupling
    original=coupling.merit
    try:
        coupling.merit=lambda s,scale:float(17-s['H'][0]);answer=obj.solve(np.zeros(1),np.zeros(4))
    finally:coupling.merit=original
    checks['16th_direction_final_acceptance']=answer['accepted'] and w.counts['joint_directions']==16
    checks['constants_consistent']=(RESTART,MAX_CYCLES,MAX_ARNOLDI)==(8,3,24)
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    checks['syntax']=True;checks={k:bool(v) for k,v in checks.items()}
    result=dict(status='pass' if all(checks.values()) else 'fail',checks=checks,records=records,seconds=time.perf_counter()-start,
        real_BP=0,real_linear_solve=0,real_operator_evaluation=0,real_assembly=0,time_steps=0,GPU_services=0)
    json.dumps(result,allow_nan=False);return result
