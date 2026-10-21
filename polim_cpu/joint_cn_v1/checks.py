"""Focused v51 admission: block signs, paired displacement and actual joint path."""
import tempfile,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from polim_cpu.geometry_coupled_v1.backend import Work
from polim_cpu.geometry_coupled_v2.velocity_candidate import VelocityCandidate
from polim_cpu.cn_flux_time_v1.checks import Adapter
from polim_cpu.large_step_transport_v1 import FixedVelocityRates
from polim_cpu.geometry_probe_v2.response import wet
from .coupling import attempt,accepted

def run(root):
    root=Path(root);root.mkdir();checks={};evidence={}
    Fu=np.array([[3.,.2],[.2,2.]]);FH=np.array([[.1,.4],[-.2,.3]]);Ru=np.array([[.5,-.1],[.3,.2]]);RH=np.array([[2.,-.1],[.2,1.5]])
    F=np.array([.3,-.2]);R=np.array([.2,.4]);c=np.linalg.solve(Fu,F)
    K=-Ru@np.linalg.solve(Fu,FH);dh=np.linalg.solve(RH+K,-R+Ru@c);du=-np.linalg.solve(Fu,F+FH@dh)
    block=np.block([[Fu,FH],[Ru,RH]]);sol=np.linalg.solve(block,-np.r_[F,R])
    checks['block_reduction_nonzero_F']=np.allclose(np.r_[du,dh],sol,rtol=0,atol=1e-14)
    evidence['block_error']=float(np.linalg.norm(np.r_[du,dh]-sol))
    # Use the production paired-update helper with nonzero F, alpha<1,
    # actual projected displacement and an explicit dry-node projection.
    class L:
        h=np.array([1.,2.]);u=np.arange(8,dtype=float)/100;a={'wet':np.ones(8,bool)}
        def fh(self,d):return np.repeat(d,4)*.1
        def solve(self,b,label):return b/2
    linear=L();cc=np.arange(8)/1000+.01;delta=np.array([-4.,.3]);vc=VelocityCandidate(linear,cc,np.zeros(2,bool),np.zeros(2))
    vc.capture(delta,-linear.fh(delta)/2);alpha=.5;ht=np.maximum(linear.h+alpha*delta,0.)
    u,disp,pred,meta=vc.propose(ht,delta,alpha);expect=-alpha*cc-linear.fh(ht-linear.h)/2
    checks['projected_nonzero_F_alpha']=np.allclose(pred,expect,rtol=0,atol=1e-14) and np.all(u[:4]==0) and meta['projected']
    ht=linear.h+.125*delta;u,disp,pred,meta=vc.propose(ht,delta,.125)
    checks['paired_common_alpha']=np.allclose(pred,.125*(-linear.fh(delta)/2-cc),rtol=0,atol=1e-14) and meta['cache_hit']
    linear.h=linear.h+.01
    try:vc.propose(ht,delta,.125);checks['stale_branch_refused']=False
    except RuntimeError:checks['stale_branch_refused']=True

    class A(Adapter):
        def __init__(self):
            super().__init__(3,[(0,1),(1,2)],(1,2));self.M=np.zeros((4,12))
            self.M[0,0]=1;self.M[1,4]=1;self.M[2,4]=1;self.M[3,8]=1
            self.outer_a=np.array([1]);self.outer_b=np.array([2]);self.outer_normal=np.array([[1.,0.]])
        def rates(self,u):return super().rates(self.M@u)
    ad=A()
    class Toy:
        def __init__(self,w):self.w=w;self.mesh={'bed':np.zeros(3)};self.point=0;self.current_arrays=None
        def evaluate(self,h,u,tangent=False):
            self.point+=1;self.current_arrays=(h.copy(),u.copy());f=np.zeros(12);f[::4]=.3+.05*h
            return 2*u-f,None,np.ones(12),dict(wet=np.ones(12,bool))
        def ru(self,h,u,du,weights):
            wa,wb,wo=weights;r=ad.rates(u);donor=r.donor_receiver_rate()[0];dc=ad.M@du;q=h[donor]*dc[:2];o=h[[1,2]]*dc[2:]
            out=np.zeros(3);np.add.at(out,[0,1],wa*q);np.add.at(out,[1,2],-wb*q);np.add.at(out,[1,2],wo*o);return out
        def linearize(self,h,u):
            F,_,force,a=self.evaluate(h,u);toy=self
            class Lin:
                def solve(self,b,label):toy.w.count('tangent',64);return b/2
                def fh(self,dh):v=np.zeros(12);v[::4]=-.05*dh;return v
                def feedback(self,dh,weights):v=self.solve(-self.fh(dh),'feedback');return v,toy.ru(h,u,v,weights)
            l=Lin();l.h=h.copy();l.u=u.copy();l.F=F;l.f=force;l.a=a;l.point=str(self.point);return l
        def full(self,*a,**kw):raise AssertionError('Nested full BP is prohibited')
    w=Work(root/'joint');w.root.mkdir();h0=np.array([2.,1.,1.5]);u0=np.zeros(12);u0[::4]=(.3+.05*h0)/2
    led=dict(V0=float(h0.sum()),volume=float(h0.sum()),source=0.,exported=0.,potential=0.,unmet=0.)
    result=attempt(Toy(w),ad,h0,u0,np.array([0.,0.,0.]),end_year=.25,ledger=led,step_dt=.25)
    checks['actual_joint_execution']=result['accepted'] and w.counts['joint_Newton_updates']<=8 and w.counts['tangent']<=64
    checks['no_independent_BP']=w.counts['full_BP']==0
    evidence['joint']=result;evidence['work']=w.record()
    events=[json.loads(s) for s in (w.root/'events.jsonl').read_text().splitlines()]
    checks['candidate_and_branch_recorded']=any(s['event']=='joint-trial-result' and 'branch' in s for s in events)
    # Failure at a real-path linear solve must leave initial candidate and F/R.
    wf=Work(root/'injected');wf.root.mkdir()
    class Fail(Toy):
        def linearize(self,h,u):raise RuntimeError('Injected before joint linear solve; no model')
    try:attempt(Fail(wf),ad,h0,u0,np.zeros(3),end_year=.25,ledger=led,step_dt=.25)
    except RuntimeError as exc:
        (wf.root/'failure.json').write_text(json.dumps(dict(error=str(exc),work=wf.record())))
    checks['failure_state_retained']=(wf.root/'candidate-00.npz').exists() and (wf.root/'momentum-00.npz').exists() and (wf.root/'failure.json').exists()
    return dict(status='pass' if all(checks.values()) else 'fail',checks={k:bool(v) for k,v in checks.items()},evidence=evidence,full_BP=0,real_tangent=0,real_time_steps=0)
