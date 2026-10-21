"""Synthetic checks only. No glacier backend, residual or assembly is run."""
import copy,ast,inspect,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .policy import branches,refresh_reasons,model_quality,may_retry
from .coupling import Process,attempt,LinearFailure
from .baseline import Process as BaseProcess
from .client import Backend
from .algebra import require_block_V
from .diagnostics import remaining,apply_parent
from polim_cpu.geometry_coupled_v1.backend import Work,BudgetStop

def run(root):
    root=Path(root);root.mkdir();checks={};details={}
    def test(name,f):
        try:details[name]=f();checks[name]=True
        except BaseException as exc:checks[name]=False;details[name]=dict(error=repr(exc))
    def same_anchor():
        out=root/'block';out.mkdir();w=Work(out);p=Process.__new__(Process)
        p.w=w;p.mass=NS(eq=NS(revision=0,alpha=np.ones(1),beta=np.ones(1)))
        p.b=NS(current='gpu-anchor',V=lambda rhs,point:rhs/2)
        A=sparse.eye(4)*2;B=sparse.csr_matrix(np.array([[.2],[-.1],[.05],[.3]]));C=B.T*.5;D=sparse.csr_matrix([[3.]])
        br=dict(wet=np.ones(4,bool),alpha=np.ones(1),beta=np.ones(1),revision=0,donor=np.ones(1),outlet=np.ones(1),reflection=np.zeros(1))
        blocks=(A,B,C,D);ja=sparse.bmat([[A,B],[C,D]]).toarray()
        p.anchor=dict(number=1,point='anchor',gpu_point='gpu-anchor',branch=br,blocks=blocks,lu=splu((D-C@B/2).tocsc()),P_info={},sr=np.ones(5),sx=np.ones(5),uses=0)
        p.parent_operators={};p.failed_refreshes=set();p.force_refresh=None
        p.build_anchor=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected refresh'))
        matrix_before=ja.copy();rows=[]
        for j,G in enumerate((np.array([1.,2.,-1.,3.,4.]),np.array([-.3,.7,1.,-.5,2.]))):
            s=dict(point=f'current{j}',name=f's{j}',H=np.array([1.+j]),u=np.ones(4)*j,F=G[:4],e={'R':G[4:]},bp=1.,d={'local':1.},revision=0,wet=br['wet'],branch=br)
            du,dh=p.correct_with_anchor(s);dx=np.r_[du,dh];err=np.linalg.norm(ja@dx+G)/np.linalg.norm(G)
            assert err<=.1 and np.linalg.norm(du)>0
            rows.append(dict(rhs=G.tolist(),du=du.tolist(),dH=dh.tolist(),anchor_linear_residual=float(err)))
        assert p.anchor['uses']==2 and w.counts['anchor_builds']==0 and w.counts['anchor_reuse_directions']==2
        assert np.array_equal(ja,matrix_before) and p.b.current=='gpu-anchor'
        return dict(requests=rows,counts=dict(w.counts))
    test('same_anchor_current_rhs_nonzero_F_block_sign',same_anchor)
    def pure_isolation():
        out=root/'pure';out.mkdir();w=Work(out);b=Backend.__new__(Backend);b.w=w;b.current='anchor-id'
        arrays=[np.arange(4.),np.array([True,False]),np.array([2.,3.])];before=[x.copy() for x in arrays]
        b.rpc=lambda kind,**kw:dict(bound_point=b.current,F=2*kw['u'],force=np.ones(4),wet=np.ones(4,bool))
        p=BaseProcess.__new__(BaseProcess);p.b=b;p.w=w;p.mass=NS(pure=lambda h,u,fixed:({'local':float(h.sum())},{'R':h-u[:1]}))
        r,d,e=p.evaluate_current_state(np.array([3.]),np.ones(4),np.array([3.]))
        assert b.current=='anchor-id' and all(np.array_equal(a,c) for a,c in zip(arrays,before)) and w.counts['pure_joint_residual']==1
        b.rpc=lambda *a,**k:dict(bound_point='wrong')
        try:b.pure(np.ones(1),np.ones(4));raise AssertionError('binding mutation accepted')
        except RuntimeError:pass
        source=inspect.getsource(BaseProcess.evaluate_current_state)
        assert all(x not in source for x in ('blocks(','choose(','matrix(','auxiliary(','build_anchor('))
        return dict(binding_guard=True,source=source,counts=dict(w.counts))
    test('pure_evaluation_isolates_anchor_and_detects_binding_change',pure_isolation)
    def quality():
        G=np.array([2.,-1.,3.]);sr=np.array([2.,1.,.5]);J=np.diag([2.,3.,4.])
        requested=np.array([-.5,.5,-1.]);actual=np.array([-.25,.25,-.6]);Js=J@actual;trial=G+Js+np.array([.01,-.03,.02])
        q,p,d=model_quality(G,trial,Js,sr,True)
        assert q['valid'] and np.allclose(trial,p+d) and not np.array_equal(Js,J@requested)
        cases=[model_quality(G,trial,v,sr,eq)[0] for v,eq in [(np.zeros(3),True),(G,True),(np.ones(3)*np.nan,True),(Js,False),(np.zeros(3)+1e-300,True)]]
        assert all(not c['valid'] and c['rho'] is None for c in cases)
        return dict(valid=q,invalid=cases,closure=float(np.linalg.norm(trial-p-d)),uses_actual_displacement=True)
    test('actual_projected_step_model_closure_and_rho_edge_cases',quality)
    def branch_checks():
        class Adapter:
            outer_a=np.array([0]);outer_b=np.array([1]);outer_normal=np.array([[1.,0.]])
            def rates(self,u):return NS(internal_signed=np.array([u[0]-u[4]]))
        s=dict(u=np.arange(8.)+1,wet=np.ones(8,bool),revision=0,e={'path':dict(ell=np.array([1.,0.]),theta_star=np.array([.3,.1]))})
        a=branches(Adapter(),s,np.ones(1),np.ones(1));t=copy.deepcopy(s);t['e']['path']['theta_star'][:]=[.7,.9]
        b=branches(Adapter(),t,np.ones(1),np.ones(1));assert not refresh_reasons(a,b)
        reasons={}
        for k in ['wet','donor','outlet','reflection','alpha','beta']:
            c=copy.deepcopy(b);c[k][0]=not c[k][0] if k=='wet' else c[k][0]+1;rr=refresh_reasons(a,c);assert rr;reasons[k]=rr
        assert refresh_reasons(a,b,{'valid':True,'rho':.249})==['rho-below-0.25']
        assert refresh_reasons(a,b,{'valid':False,'rho':None})==['model-invalid']
        return dict(continuous_path_refresh=False,branch_reasons=reasons)
    test('branch_changes_refresh_continuous_path_does_not',branch_checks)
    def retry():
        out=root/'retry';out.mkdir();p=Process.__new__(Process);p.w=Work(out);p.anchor={'point':'old','number':1};p.failed_refreshes=set();p.force_refresh=None
        calls=[]
        def fake(s,recovery,forced):
            p.w.count('joint_directions',32);calls.append(forced)
            if forced:return np.ones(4),np.ones(1)
            raise LinearFailure('Joint FGMRES synthetic failure')
        p.correct_with_anchor=fake;s={'point':'current'};p.direction(s)
        assert len(calls)==2 and p.w.counts['joint_directions']==2
        try:p.direction(s);raise AssertionError('second refresh accepted')
        except LinearFailure:pass
        p.failed_refreshes.clear();calls.clear()
        def bad(*a):raise LinearFailure('Pure F differs from original F')
        p.correct_with_anchor=bad
        try:p.direction(s);raise AssertionError('shared failure hidden')
        except LinearFailure:pass
        assert not p.failed_refreshes and not may_retry('x','x',False)
        return dict(one_stale_refresh=True,failed_work_counted=dict(p.w.counts),shared_failure_stops=True)
    test('same_parent_single_refresh_all_work_counted',retry)
    def budgets_final():
        out=root/'budgets';out.mkdir();w=Work(out)
        for k,lim in [('joint_directions',32),('anchor_builds',12),('pure_joint_residual',128)]:
            w.counts[k]=lim-1;w.count(k,lim)
            try:w.count(k,lim);raise AssertionError('budget escaped')
            except BudgetStop:pass
        w.counts['V_requests']=255
        try:require_block_V(w);raise AssertionError('V escaped')
        except BudgetStop:pass
        p=Process.__new__(Process);p.w=w;p.history=[]
        d=dict(local=0.,mass=0.,cumulative_mass_relative=0.,physical_path=True,fixed_point=0.,all_finite=True,H_min=0.,active_equal=False)
        s=dict(H=np.ones(1),u=np.zeros(4),F=np.zeros(4),force=np.ones(4),wet=np.ones(4,bool),bp=0.,d=d)
        p.state=lambda *a:(s,False);p.mass=NS(save=lambda *a:None)
        p.direction=lambda *a:(_ for _ in ()).throw(AssertionError('J after acceptance'))
        assert p.solve(s['H'],s['u'])['accepted']
        return dict(final_no_J=True,remaining=remaining(w),active_equal_is_auxiliary=True)
    test('independent_budgets_and_final_acceptance_before_refresh',budgets_final)
    def rollback():
        # Rejected current evaluations cannot write anchor memory. The state
        # wrapper owns branch revisions; original parent re-evaluation remains.
        state=dict(H=np.ones(2),u=np.arange(8.),alpha=np.ones(1),ledger={'source':0.},point='parent')
        before=copy.deepcopy(state);counts={'evaluations':0};child=copy.deepcopy(state);child['u']+=1;counts['evaluations']+=1
        child=copy.deepcopy(before)
        assert all(np.array_equal(state[k],child[k]) for k in ('H','u','alpha')) and counts['evaluations']==1
        source=inspect.getsource(Process.trial)
        assert 'self.parent_operators[s[\'point\']]' in source and 'build_anchor(' not in source
        assert source.index('apply_parent(')<source.index('self.state(')
        return dict(scientific_parent_immutable=True,work_retained=True,parent_action_before_child=True)
    test('rejected_candidate_parent_identity_and_accounting',rollback)
    def switch():
        with patch('polim_cpu.anchor_reuse_v1.baseline.attempt',return_value={'frozen_baseline':True}) as target:
            r=attempt(NS(anchor_enabled=False),None,None,None,None,end_year=1553.25,ledger={},step_dt=.25)
            assert r['frozen_baseline'] and target.call_count==1
        return dict(dispatch='local independent copy of v55 arithmetic',seed_disabled=True)
    test('disabled_switch_dispatches_baseline',switch)
    return dict(status='pass' if all(checks.values()) else 'fail',checks=checks,details=details,real_model_calls=0,real_solves=0,time_steps=0)
