"""Focused admission. Saved-state operators/algebra only; no BP/tangent solves."""
import json,time,tempfile,shutil,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from .algebra import fgmres,flux_matrix,auxiliary,column_WB
from .coupling import matrices,Mass,Process,identity,accepted
from .client import Backend,Work
from .control import ROOT,E,INPUT,read,write,arrays,ahash,ref
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
from polim_cpu.geometry_probe_v2.response import wet,flux
from polim_cpu.cn_validation_v1.ledger import zero_ledger
from polim_cpu.c_control_repair_v1.process import launch

STANDARDS=dict(F=dict(relative=1e-10,absolute=0.,scale='max(norm(force),1N)'),
    Fu=dict(relative=1e-9,absolute=1e-8,scale='norm(original Fu direction)'),
    FH=dict(relative=1e-8,absolute=1e-8,scale='norm(original FH direction)'),
    Ru=dict(relative=1e-10,absolute=1e-10,scale='norm(original Ru direction)'),
    flux=dict(relative=1e-10,absolute=1e-10,scale='norm(original signed common flux, outward flux)'),
    joint_defect=1e-9,memory_fraction=.10)

def compare(root,name,reference,current,force=None):
    rule=STANDARDS[name];den=max(float(np.linalg.norm(force)),1.) if name=='F' else float(np.linalg.norm(reference))
    error=float(np.linalg.norm(current-reference));tol=rule['absolute']+rule['relative']*den
    result=dict(absolute_error=error,denominator=den,relative_error=error/den if den else None,absolute_tolerance=tol,rule=rule,passed=error<=tol)
    with (root/(name+'.npz')).open('xb') as f:np.savez(f,reference=reference,current=current,difference=current-reference)
    write(root/(name+'.json'),result)
    return result

def synthetic(root):
    root.mkdir();w=Work(root);rng=np.random.default_rng(5201)
    q=rng.normal(size=(8,8));A=q.T@q+8*np.eye(8);B=rng.normal(size=(8,2))*.1;C=rng.normal(size=(2,8))*.1;D=np.diag([2.,3.])
    F=rng.normal(size=8);R=rng.normal(size=2);J=np.block([[A,B],[C,D]]);G=np.r_[F,R]
    wb,factors=column_WB(sparse.csr_matrix(A),sparse.csr_matrix(B));P=D-C@wb
    V=lambda r:r/np.diag(A)
    def pre(q):
        z=V(q[:8]);h=np.linalg.solve(P,q[8:]-C@z);return np.r_[z-V(B@h),h]
    sr=np.r_[np.ones(8)/3,np.ones(2)/2];sx=np.r_[np.ones(8)*2,np.ones(2)*3]
    dx,info=fgmres(lambda x:sr*(J@(sx*x)),lambda y:pre(y/sr)/sx,-sr*G,w,'synthetic-block')
    exact=np.linalg.solve(J,-G);c=np.linalg.solve(A,F);h=np.linalg.solve(D-C@np.linalg.solve(A,B),-R+C@c);u=-np.linalg.solve(A,F+B@h)
    checks=dict(block_sign=np.linalg.norm(exact-np.r_[u,h])<1e-12,scaled_true=info['true_relative']<=.1,
        column_factors=np.linalg.norm(factors@factors.transpose(0,2,1)-np.stack((A[:4,:4],A[4:,4:])))<1e-12)
    # Recovery scheduler: the first internal improvement is still worse than
    # the parent and must not be adopted. The second crosses the parent merit.
    def state(f):return dict(H=np.ones(2),u=np.zeros(8),F=np.array([f]),e={'R':np.zeros(2)},
        d=dict(cumulative_mass_relative=0.,fixed_point=0.,physical_path=True,all_finite=True,H_min=0.,local=1.,mass=0.),bp=f)
    class Toy:
        def __init__(self):self.w=w;self.k=0
        def direction(self,s,recovery):self.k+=1;w.count('recovery_directions');return np.zeros(8),np.zeros(2)
        def trial(self,*args):return state(1.5 if self.k==1 else .5),False,True,1.
    toy=Toy();s,restart=Process.recover(toy,state(2.),state(1.),1.,dict(force=1.,local=1.,mass=1.))
    checks['bounded_newwet_parent_comparison']=toy.k==2 and s['F'][0]==.5 and restart is None
    token=identity(np.ones(2),np.zeros(8),np.ones(8,bool),np.ones(1),np.ones(1))
    checks['branch_token_invalidated']=token!=identity(np.ones(2),np.zeros(8),np.r_[np.zeros(4,bool),np.ones(4,bool)],np.ones(1),np.ones(1))
    good=dict(local=0,mass=0,cumulative_mass_relative=0,physical_path=True,fixed_point=0,all_finite=True,H_min=0,active_equal=False)
    checks['original_accepted_no_active_equal']=bool(accepted(good,0))
    out=dict(checks={k:bool(v) for k,v in checks.items()},fgmres=info,work=w.record());write(root/'result.json',out)
    return all(checks.values()),out

def run():
    start=time.perf_counter();E.mkdir(exist_ok=True);checks={};details={}
    write(E/'preparation-standards.json',STANDARDS) # immutable before comparisons
    checks['synthetic'],details['synthetic']=synthetic(E/'synthetic')
    layouts=[]
    with tempfile.TemporaryDirectory(prefix='v52-') as td:
        for mode in ('outer','outer-fail'):
            root=Path(td)/mode;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(root),mode],root/'supervisor',root/'native',time.time()+20)
            shutil.copytree(root,E/('layout-'+mode));layouts.append(r)
        checks['supervision']=layouts[0]['exit_code']==0 and layouts[1]['exit_code']!=0
    details['layouts']=layouts
    z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);base=ROOT/'polim_cpu/runs/v51-joint-cn-v1/N/native/step-1553.0000-1553.2500'
    states=[('v51-parent',base/'candidate-01.npz'),('v51-newwet',base/'trial-mass-02.npz'),('D-accepted',ROOT/'polim_cpu/runs/v48-cn-cuda-v1/D/checkpoints/accepted-1482.5000.npz')]
    write(E/'preparation-inputs.json',[dict(label=label,input=ref(p)) for label,p in states])
    backend=None;rows=[]
    root=E/'operators';root.mkdir();w=Work(root)
    try:
        backend=Backend(z,w,E/'GPU-preparation',prepare_only=True);backend.begin(w)
        try:backend.rpc('synthetic-error')
        except RuntimeError:checks['GPU_exception_saved']=bool(list((E/'GPU-preparation').glob('failure-*.json')))
        previous=None
        for label,path in states:
            sub=root/label;sub.mkdir();data=arrays(path);h,u=data['H'],data['u'];smb=np.zeros_like(h)
            m=Mass(ad,h,u,smb,w,.25,zero_ledger(float(ad.topology.dual_area@h)))
            if 'alpha' in data:m.eq.alpha[:]=data['alpha'];m.eq.beta[:]=data['beta']
            snapshot=(m.eq.alpha.copy(),m.eq.beta.copy(),m.eq.revision,m.eq.h0.copy(),m.eq.smb.copy(),dict(m.ledger))
            original=backend.blocks(h,u);A,B=matrices(z,original);pure=backend.pure(h,u)
            e=m.eq.evaluate(h,u);weights=m.eq.weights(e);C=flux_matrix(ad,h,u,weights,original['wet'][::4]);D=m.eq.matrix(u,e).tocsr()
            du=np.sin(np.arange(len(u))*.71)*original['wet'];dh=np.cos(np.arange(len(h))*.37)*original['wet'][::4]
            other=backend.rpc('operator',point=backend.current,du=du,dH=dh,weights=weights)
            q,o=flux(ad,h,u);out={}
            for name,refval,val in [('F',original['F'],pure['F']),('Fu',other['Fu'],A@du),('FH',other['FH'],B@dh),('Ru',other['Ru'],C@du),('flux',other['flux'],np.r_[q,o])]:
                out[name]=compare(sub,name,refval,val,original['force'])
            # Pure evaluation must leave the bound matrix/projection untouched,
            # including when evaluating a different H on a new wet branch.
            before=backend.rpc('identity',point=backend.current)
            shifted=arrays(states[1][1]);backend.pure(shifted['H'],shifted['u'])
            after=backend.rpc('identity',point=backend.current)
            out['pure_immutable']=bool(all(np.array_equal(before[k],after[k]) for k in ('H','u','wet','A_values')) and before['point']==after['point'] and
                np.array_equal(snapshot[0],m.eq.alpha) and np.array_equal(snapshot[1],m.eq.beta) and snapshot[2]==m.eq.revision and snapshot[5]==m.ledger)
            P,lu,pi=auxiliary(A,B,C,D,w)
            # Algebraic residual of actual nonsymmetric auxiliary solve, not a
            # BP equilibrium or true Fu tangent solve.
            b=np.sin(np.arange(len(h))*.3);x=lu.solve(b);pdef=float(np.linalg.norm(P@x-b)/np.linalg.norm(b))
            v=backend.V(du,backend.current)
            out['V_projection']=bool(np.isfinite(v).all() and (v[~original['wet']]==0).all());out['P_true_defect']=pdef
            out['P']=pi;out['memory']=dict(peak=original['peak_memory'],quota=original['memory_quota'],passed=original['peak_memory']<=original['memory_quota'])
            out['point']=original['point'];out['H_sha256']=ahash(h);out['u_sha256']=ahash(u)
            sparse.save_npz(sub/'A.npz',A);sparse.save_npz(sub/'B.npz',B);sparse.save_npz(sub/'C.npz',C);sparse.save_npz(sub/'D.npz',D);sparse.save_npz(sub/'P.npz',P)
            with (sub/'state.npz').open('xb') as f:np.savez(f,H=h,u=u,wet=original['wet'],du=du,dH=dh,force=original['force'])
            if label=='v51-newwet':
                nw=original['wet']&~previous['wet'];out['newwet_DOF_count']=int(nw.sum());out['newwet_A_rows_nonzero']=bool(np.all(np.asarray(abs(A[nw]).sum(1)).ravel()>0))
                out['newwet_V_nonzero']=bool(np.linalg.norm(v[nw])>0);checks['newwet_entered_J_V']=out['newwet_DOF_count']>0 and out['newwet_A_rows_nonzero'] and out['newwet_V_nonzero']
            previous=original
            passed=all(out[k]['passed'] for k in STANDARDS if k in ('F','Fu','FH','Ru','flux')) and out['pure_immutable'] and out['V_projection'] and pdef<=1e-9 and out['memory']['passed']
            out['passed']=passed;write(sub/'result.json',out);rows.append(out);checks[label]=passed
            if not passed:break
    except BaseException as exc:
        import traceback
        details['failure']=dict(error=repr(exc),traceback=traceback.format_exc());checks['execution']=False
    finally:
        if backend:backend.close()
    details['operators']=rows;details['work']=w.record()
    out=dict(status='pass' if all(checks.values()) else 'fail',checks=checks,details=details,seconds=time.perf_counter()-start,real_BP=0,exact_Fu_solves=0,time_steps=0)
    write(E/'preparation.json',out);return out
