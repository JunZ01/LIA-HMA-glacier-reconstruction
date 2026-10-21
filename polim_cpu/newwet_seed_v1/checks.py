"""New graph/transaction/budget tests only; no real model work."""
import copy,time,tempfile,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .harmonic import predict
from .seed import after_adopt,scientific_hash
from polim_cpu.geometry_probe_v2.response import wet
from polim_cpu.geometry_coupled_v1.backend import Work

def run():
    t=time.perf_counter();checks={};xy=np.array([[0.,0.],[3.,0.],[1.,0.],[2.,0.]])
    tri=np.array([[0,2,3],[2,3,1]]);h=np.array([2.,3.,.2,.4]);u=np.tile([1.,-2.,3.,-4.],(4,1))*h[:,None];u[2:]=0
    ow=np.array([True,True,False,False]);nw=np.ones(4,dtype=bool)
    pred,info,dat=predict(xy,tri,h,u.ravel(),ow,nw)
    checks['multi_node_harmonic_4rhs']=bool(info['factorizations']==1 and info['linear_RHS']==4 and np.allclose(pred.reshape(-1,4)[2:],h[2:,None]*[1,-2,3,-4],atol=1e-14))
    checks['H_existing_wet_unchanged']=bool(np.array_equal(pred.reshape(-1,4)[:2],u[:2]) and np.array_equal(h,[2,3,.2,.4]))
    free=np.ones((4,4),bool);free[2,1]=False
    constrained,_,_=predict(xy,tri,h,u.ravel(),ow,nw,free);checks['Dirichlet_free_projection']=constrained.reshape(-1,4)[2,1]==u[2,1]
    basal=np.zeros((4,3,2));basal[:,1:,:]=pred.reshape(4,2,2);checks['eliminated_basal_zero']=bool(np.all(basal[:,0,:]==0))
    no,ii,_=predict(xy,tri,h,u.ravel(),np.zeros(4,bool),nw);checks['no_boundary']=ii['factorizations']==0 and np.array_equal(no,u.ravel())
    zero,ii,_=predict(xy,tri,h,np.zeros(16),ow,nw);checks['zero_boundary']=bool(np.all(zero==0) and not ii['effective'])
    mixed=u.copy();mixed[1]*=-1;pp,ii,_=predict(xy,tri,h,mixed.ravel(),ow,nw);checks['mixed_direction']=bool(np.isfinite(pp).all() and ii['graph_relative_residual']<1e-12)
    thin=h.copy();thin[2:]=[1e-12,1e-13];pp,_,_=predict(xy,tri,thin,u.ravel(),ow,wet(thin,np.zeros(4)));checks['thin_original_wet']=bool(np.isfinite(pp).all() and not wet(np.array([1e-20]),np.zeros(1))[0])
    with tempfile.TemporaryDirectory() as temp:
      for mode in ('reject','accept','exception','budget','no-newwet','disabled'):
        root=Path(temp)/mode;root.mkdir();w=Work(root);eq=NS(h0=h.copy(),smb=np.zeros(4),dt=.25,q0=np.ones(2),o0=np.ones(1),alpha=np.ones(2),beta=np.ones(1),revision=0,area=np.ones(4),V0=10.)
        d=dict(local=1e-3,mass=1e-4,cumulative_mass_relative=1e-4,fixed_point=1e-3,scale=10.,source=0.,export=0.,physical_path=True,all_finite=True,H_min=.2)
        base=dict(H=h.copy(),u=u.ravel().copy(),F=np.ones(16),force=np.ones(16),wet=np.repeat(nw,4),d=d,e=dict(R=np.ones(4),source=np.ones(4),U=np.zeros(4),qi=np.zeros(2),oi=np.zeros(1),scale=10.),bp=.25,point='base',name='base')
        parent={**base,'wet':np.repeat(ow,4),'name':'parent'}
        mass=NS(eq=eq,last={'a':np.ones(2)},last_h=h.copy(),last_u=u.ravel().copy(),ledger={'x':1},inherited={'x':2},selection=False)
        p=NS(w=w,z=dict(xy=xy,tri=tri),mass=mass,b=NS(current='parent-bound'),seed_seen=False,seed_enabled=mode!='disabled')
        def state(hh,uu,tag,**kw):
            w.count('pure_joint_residual',128);out=copy.deepcopy(base);out['u']=uu.copy();out['name']='trial';out['locked']=dict(ok=True)
            mass.last={'a':np.zeros(2)}
            if mode=='exception':eq.alpha[0]=0;raise ValueError('injected before judgment')
            if mode=='accept':
                for k in ('local','mass','cumulative_mass_relative','fixed_point'):out['d'][k]*=.1
                out['F']*=.1;out['bp']*=.1;out['e']['R']*=.1
            else:out['d']['mass']*=2
            return out,False
        p.state=state;sig=scientific_hash(p,base)
        if mode=='no-newwet':parent['wet']=base['wet'].copy()
        if mode=='budget':w.counts['pure_joint_residual']=128
        try:result=after_adopt(p,parent,base,7 if mode=='reject' else 1)
        except ValueError:
            checks['exception_persist_restore']=scientific_hash(p,base)==sig and 'seed-exception' in (root/'events.jsonl').read_text();continue
        if mode=='reject':checks['rollback_all_fields']=scientific_hash(p,result)==sig and np.array_equal(mass.last['a'],np.ones(2)) and p.b.current=='parent-bound';checks['shared_budget']=w.counts['seed_trials']==1
        if mode=='accept':checks['first_good_only']=w.counts['seed_trials']==1 and w.counts['seed_adopted']==1
        if mode in ('budget','no-newwet','disabled'):checks[mode]=result is base and w.counts['seed_trials']==0 and w.counts['seed_graph_construction']==0
    return dict(status='pass' if all(checks.values()) else 'fail',checks={k:bool(v) for k,v in checks.items()},seconds=time.perf_counter()-t,real_BP=0,real_joint_solve=0,real_model_evaluation=0,time_steps=0)
