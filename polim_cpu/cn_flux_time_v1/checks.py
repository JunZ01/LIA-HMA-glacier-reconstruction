"""One concentrated admission batch for new time reconstruction only."""
import numpy as np
from scipy.linalg import expm
from contextlib import contextmanager
from collections import defaultdict
from polim_cpu.large_step_transport_v1 import TransportTopology,FixedVelocityRates,implicit_conservative_step
from .path import polynomial,roots,absolute_integral,reconstruct
from .transport import Equation

class Work:
    def __init__(self):self.counts=defaultdict(int);self.events=[]
    @contextmanager
    def span(self,key):self.count(key);yield
    def count(self,key):self.counts[key]+=1
    def event(self,key,**data):self.events.append(dict(event=key,**data))

class Adapter:
    def __init__(self,n,edges,boundary=()):self.topology=TransportTopology(np.ones(n),np.array(edges,dtype=int).reshape(-1,2),np.array(boundary,dtype=int))
    def rates(self,u):
        ne=len(self.topology.internal_edges);return FixedVelocityRates(self.topology,np.asarray(u[:ne]),np.maximum(u[ne:],0.))

def run():
    results={};evidence={};w=Work()
    def check(name,value,**data):results[name]=bool(value);evidence[name]=data
    p=polynomial([1.],[-2.5],[0.],[-10.],.25)
    check('pure_ablation',p['physical'] and p['H_path'][0]==0 and p['ell'][0]==1.5 and abs(p['dry_lo'][0]*.25-.1)<1e-15)
    vals=[]
    for incoming in [.5,1.,2.]:
        p=polynomial([0.],[incoming-1],[0.],[-1.],1.)
        vals.append(dict(incoming=incoming,H=float(p['H_path'][0]),ell=float(p['ell'][0]),dry=bool(p['dry'][0])))
    check('initial_dry_concurrent_supply',vals==[dict(incoming=.5,H=0.,ell=.5,dry=True),dict(incoming=1.,H=0.,ell=0.,dry=True),dict(incoming=2.,H=1.,ell=0.,dry=False)],cases=vals)
    p=polynomial([0.],[-1.],[1.],[-1.],1.)
    check('dry_then_wet',p['physical'] and p['H_path'][0]==.25 and p['ell'][0]==.25 and p['dry_hi'][0]==.5,
        H=float(p['H_path'][0]),actual_source=-1+float(p['ell'][0]),unmet=float(p['ell'][0]))
    p=polynomial([0.],[-1.],[0.],[0.],1.)
    check('no_SMB_cannot_absorb_outflow',not p['physical'] and p['invalid_nodes']==[0])
    ad=Adapter(3,[(0,1),(1,2)])
    p=reconstruct(ad,np.array([1.,0.,0.]),np.zeros(3),1.,np.ones(2),np.zeros(0),np.ones(2),np.zeros(0),np.ones(2),np.zeros(0))
    check('dry_throughflow_detected',1 in p['invalid_nodes'] and 1 in p['dry_out_edges'],bad_nodes=p['invalid_nodes'])
    # CN oscillation empties a donor; monotone limiter must repair, not book transport as SMB.
    ad=Adapter(2,[(0,1)]);eq=Equation(ad,np.array([1.,0.]),np.array([10.]),np.zeros(2),1.,w)
    h,e=eq.limited(np.array([10.]),np.array([1.,0.]));ref=implicit_conservative_step(eq.h0,eq.smb,1.,ad.rates(np.array([10.])))
    check('limiter_repairs_transport',np.all(eq.alpha==0) and e['path']['physical'] and np.allclose(h,ref.thickness,atol=1e-12) and not np.any(e['U']),alpha=eq.alpha.tolist(),H=h.tolist())
    # Smooth wet cycle, exact matrix exponential; no SMB, outlet or limiter activity.
    ad=Adapter(3,[(0,1),(1,2),(2,0)]);u=np.array([.7,.5,.4]);h0=np.array([1.2,1.,.8]);L=ad.rates(u).outward_operator().toarray()
    exact=expm(-L)@h0;errors=[]
    for dt in [.25,.125,.0625]:
        h=h0.copy();alphas=[]
        for _ in range(round(1/dt)):
            eq=Equation(ad,h,u,np.zeros(3),dt,w);h,e=eq.limited(u,h);alphas.append(bool((eq.alpha==1).all()))
        errors.append(float(np.linalg.norm(h-exact)))
        check('wet_CN_retained_'+str(dt),all(alphas) and abs(h.sum()-h0.sum())<1e-12)
    orders=np.log2(np.array(errors[:-1])/errors[1:]);check('smooth_second_order',bool(np.all((orders>1.9)&(orders<2.1))),errors=errors,orders=orders.tolist())
    # All-BE exact complementarity degeneration.
    ad=Adapter(2,[(0,1)],(1,));u=np.array([.7,.3]);eq=Equation(ad,np.array([.1,1.]),u,np.array([-3.,-.2]),.25,w)
    eq.alpha[:]=0;eq.beta[:]=0;old=implicit_conservative_step(eq.h0,eq.smb,.25,ad.rates(u));e=eq.evaluate(old.thickness,u)
    check('all_BE_equivalence',e['path']['physical'] and e['local']<1e-14 and np.max(abs(e['U']-old.unmet_ablation_volume))<1e-14)
    # Endpoint sign reversal retains gross absolute transport, outlet integrated separately.
    ad=Adapter(2,[(0,1)],(1,));eq=Equation(ad,np.array([3.,3.]),np.array([1.,.2]),np.zeros(2),.25,w)
    e=eq.evaluate(np.array([3.,3.]),np.array([-1.,.4]))
    check('reversal_and_outlet',abs(e['qi'][0])<1e-15 and abs(e['absolute_q'][0]-.375)<1e-15 and abs(e['oi'][0]-.225)<1e-14,
        signed_internal=e['qi'].tolist(),absolute_internal=e['absolute_q'].tolist(),outlet=e['oi'].tolist())
    check('internal_cancellation',abs(float(e['R'].sum()+e['source'].sum()-e['oi'].sum()))<1e-14)
    # Smooth branch derivatives in H and new endpoint C; old endpoint fixed.
    cases=[('no_reflection',[4.,4.],[.1],[.2],[0.,0.],[4.,4.]),
           ('end_dry',[0.,4.],[0.],[-.1],[-1.,0.],[0.,4.]),
           ('dry_rewet',[0.,10.],[0.],[-.2],[-1.,0.],[.25,10.])]
    for name,h0,u0,u,smb,h in cases:
        ad=Adapter(2,[(0,1)]);eq=Equation(ad,np.array(h0),np.array(u0),np.array(smb),1.,w)
        h=np.array(h);u=np.array(u);e=eq.evaluate(h,u);dh=np.array([.13,-.07]);dc=np.array([.03]);eps=1e-5
        if name=='end_dry':dh[0]=0.
        finite=(eq.evaluate(h+eps*dh,u)['R']-eq.evaluate(h-eps*dh,u)['R'])/(2*eps)
        predicted=eq.matrix(u,e)@dh
        donor=ad.rates(u).donor_receiver_rate()[0];dq=dc*h[donor]
        finite_u=(eq.evaluate(h,u+eps*dc)['R']-eq.evaluate(h,u-eps*dc)['R'])/(2*eps)
        pred_u=eq.weighted(dq,np.zeros(0),e)
        err=float(np.linalg.norm(finite-predicted)/max(np.linalg.norm(predicted),1.));erru=float(np.linalg.norm(finite_u-pred_u)/max(np.linalg.norm(pred_u),1.))
        check('derivative_'+name,err<1e-8 and erru<1e-8,H_error=err,u_error=erru,ell=e['path']['ell'].tolist(),theta=e['path']['theta_star'].tolist())
    rr=roots(1.,-1e16,1.);double=roots(.25,-1.,1.)
    check('stable_roots_and_double_root',abs(rr[0]-1e-16)<1e-30 and rr[1]==1e16 and double==[.5],roots=rr,double=double)
    p=polynomial([.25],[-1.],[1.],[0.],1.)
    check('isolated_zero_not_dry_interval',not bool(p['dry'][0]) and p['ell'][0]==0.)
    p=polynomial([0.],[-1.],[1.],[-1.],1.)
    S=-1.+float(p['ell'][0]);I=-1.;H=float(p['H_path'][0])
    check('final_state_source_ledger',H==.25 and abs(H+I-S)<1e-15 and S==-.75)
    return dict(status='pass' if all(results.values()) else 'fail',checks=results,evidence=evidence,counts=dict(w.counts),events=w.events,
        full_BP=0,tangent=0,real_time_steps=0,switches_not_used_as_smooth_derivative_cases=True)
