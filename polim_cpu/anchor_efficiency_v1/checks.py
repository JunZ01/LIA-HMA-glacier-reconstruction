"""Only new scalar lifetime, priority and accounting checks; no real solves."""
import math,copy,inspect,ast
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .efficiency import History,branch_id,nonmomentum_pass
from .coupling import Process,LinearFailure
from polim_cpu.geometry_coupled_v1.backend import Work

COEFF={'c_V':.026,'c_J':.51}
def fixture(failed=0):
    h=History(1,'a','b',100.,0);values=[100.,100*math.exp(-1),100*math.exp(-1.1),100*math.exp(-1.2)]
    h.adopt('a','1','b','b',values[0],values[1],10,False,'parent')
    h.adopt('1','2','b','b',values[1],values[2],20+failed,True,'parent')
    h.adopt('2','3','b','b',values[2],values[3],30+failed,True,'parent')
    args=dict(point='3',branch='b',f=values[3],force=1e6,bp=values[3]/1e6,other_pass=True,coeff=COEFF)
    return h,args

def run(root):
    root=Path(root);root.mkdir();checks={};detail={}
    def test(name,f):
        try:detail[name]=f();checks[name]=True
        except BaseException as exc:detail[name]={'error':repr(exc)};checks[name]=False
    def formula():
        h,a=fixture();q=h.decide(**a);assert q['trigger'] and abs(q['g_f']-.1)<1e-14 and abs(q['g_s']-.01)<1e-14
        assert q['W_stale']%2==q['W_fresh']%2==0
        h2,b=fixture(8);r=h2.decide(**b);assert abs(r['g_s']-.2/28)<1e-14 and h2.recent[0]['delta_V']==18
        q2=h.decide(**{**a,'force':1e5,'bp':a['f']/1e5});assert q2['g_f']==q['g_f'] and q2['g_s']==q['g_s'] and q2['L']!=q['L']
        return dict(base=q,with_failed_work=r,force_change_only_affects_target=True)
    test('raw_norm_efficiency_even_work_and_failed_V_intervals',formula)
    def identity():
        h,a=fixture();h.adopt('3','4','b','newbranch',a['f'],a['f']*.9,40,True,'parent')
        q=h.decide(**{**a,'point':'4','branch':'newbranch','f':a['f']*.9});assert not q['available']
        h,a=fixture();h.adopt('unrelated','4','b','b',a['f'],a['f']*.9,40,True,'parent');assert h.first is None
        fresh=History(2,'4','b',a['f'],40);assert fresh.first is None and not fresh.recent
        h,a=fixture();assert not h.decide(**{**a,'point':'other'})['available']
        return dict(branch=q,noncontiguous_clears_history=True,refresh_clears_history=True)
    test('same_anchor_same_branch_contiguous_adoptions_only',identity)
    def invalid():
        results=[]
        for fb in [0.,-1.,math.nan,math.inf,100.,101.]:
            h=History(1,'a','b',100.,0);h.adopt('a','1','b','b',100.,fb,10,False,'parent')
            q=h.decide(point='1',branch='b',f=1.,force=1.,bp=1.,other_pass=True,coeff=COEFF)
            assert not q['available'] and not q['trigger'];results.append(q['reason'])
        h,a=fixture();h.recent=[];assert not h.decide(**a)['available']
        h,a=fixture();h.first['g_f']=math.inf;assert not h.decide(**a)['available']
        h,a=fixture();h.recent[0]['delta_V']=-10;h.recent[1]['delta_V']=10;assert not h.decide(**a)['available']
        return dict(invalid_cases=results,insufficient_history_and_bad_denominators=True)
    test('nonpositive_nonfinite_zero_and_missing_history_unavailable',invalid)
    def applicability():
        h,a=fixture();q=h.decide(**{**a,'bp':.5e-6,'other_pass':False});assert 'BP-already-passed' in q['reason'] and not q['trigger']
        assert not h.decide(**{**a,'other_pass':False})['trigger']
        q=h.decide(**{**a,'original':['wet-branch','rho-below-0.25']});assert q['reason']=='original-refresh-priority' and not q['trigger']
        disabled=h.decide(**{**a,'enabled':False});assert not disabled['trigger']
        return dict(BP_done_is_not_applicable=True,original_priority=q,disabled=disabled['reason'])
    test('original_priority_BP_applicability_and_v57_switch',applicability)
    def no_mutation():
        h,a=fixture();before=copy.deepcopy(h.__dict__);h.decide(**a);assert h.__dict__==before
        d=dict(local=0.,fixed_point=0.,mass=0.,annual_cumulative_mass_relative=0.,inherited_cumulative_mass_relative=0.,cumulative_mass_relative=0.,physical_path=True,all_finite=True,H_min=0.)
        s=dict(u=np.zeros(4),wet=np.ones(4,bool),d=d);assert nonmomentum_pass(s)
        for key in ['local','fixed_point','mass','annual_cumulative_mass_relative','inherited_cumulative_mass_relative']:
            t=copy.deepcopy(s);t['d'][key]=1.;assert not nonmomentum_pass(t)
        return dict(pure_scalar_decision=True,all_nonmomentum_gates=True)
    test('decision_preserves_science_ledger_and_history',no_mutation)
    def final_and_single_parent():
        out=root/'control';out.mkdir();w=Work(out);p=Process.__new__(Process);p.w=w;p.anchor=None;p.built_points={'same'}
        try:p.build_anchor({'point':'same'},['economic-efficiency-degradation']);raise AssertionError('duplicate allowed')
        except LinearFailure:pass
        assert w.counts['anchor_builds']==1 and not w.counts['J_build']
        p.history=[];d=dict(local=0.,mass=0.,cumulative_mass_relative=0.,physical_path=True,fixed_point=0.,all_finite=True,H_min=0.)
        s=dict(H=np.ones(1),u=np.zeros(4),F=np.zeros(4),force=np.ones(4),wet=np.ones(4,bool),bp=0.,d=d)
        p.state=lambda *a:(s,False);p.mass=NS(save=lambda *a:None);p.direction=lambda *a:(_ for _ in ()).throw(AssertionError('unneeded refresh'))
        assert p.solve(s['H'],s['u'])['accepted']
        return dict(one_parent_rebuild_guard=True,failed_request_counted=True,final_acceptance_no_J=True)
    test('same_parent_once_and_accepted_state_no_rebuild',final_and_single_parent)
    def interfaces():
        base=Path(__file__).parent;old=base.parent/'anchor_reuse_v1'
        for name in ['device.py','service.py','client.py','algebra.py','baseline.py','policy.py','structure.py','diagnostics.py']:
            a=(base/name).read_text(encoding='utf-8').replace('polim_cpu.anchor_efficiency_v1','polim_cpu.anchor_reuse_v1')
            b=(old/name).read_text(encoding='utf-8');assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b)),name
        source=inspect.getsource(Process.efficiency_decision)
        assert 'build_anchor(' not in source and 'self.b.' not in source
        return dict(unchanged_numerical_and_GPU_modules=True,CPU_scalar_policy_only=True)
    test('original_GPU_isolation_and_numerical_interfaces_unchanged',interfaces)
    return dict(status='pass' if all(checks.values()) else 'fail',checks=checks,details=detail,real_BP=0,real_residual=0,real_assembly=0,real_linear_solve=0,time_steps=0)
