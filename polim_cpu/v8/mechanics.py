"""P1 triangle x P1 vertical BP with arbitrary vertical nodes.

Assemble each prism's 12 x 12 matrix, retaining both horizontal components.
No full-Stokes physics. Units m, Pa, year; A is supplied in Pa^-3 year^-1.
"""
import time,json,sys
from pathlib import Path
import numpy as np
from numba import njit,prange
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve,cg
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.runtime/pyamg-5.2.1'))
import pyamg
from .boundary import constraints,loads


def prepare(mesh,layers):
    xy=mesh['xy'];tri=np.asarray(mesh['triangles'],dtype=np.int32)
    H=mesh['thickness'];bed=mesh['bed'];nk=layers-1
    if layers<3 or layers>33 or np.any(H<0) or not all(np.isfinite(x).all() for x in (xy,H,bed)):raise ValueError('Invalid geometry or vertical count')
    vertices=xy[tri];jac=np.stack([vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],axis=2)
    det=np.linalg.det(jac)
    if np.any(det<=0) or np.any(H[tri].mean(axis=1)<=0):raise ValueError('Degenerate ice element')
    grad=np.einsum('ij,tjk->tik',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(jac))
    nodes=np.concatenate([tri[:,None,:]*layers+np.arange(nk)[None,:,None],tri[:,None,:]*layers+np.arange(1,layers)[None,:,None]],axis=2).reshape(-1,6)
    ids=np.stack([nodes*2,nodes*2+1],axis=2).reshape(-1,12).astype(np.int32)
    if len(ids)>1000000:raise MemoryError('Registered prism assembly budget exceeded')
    return dict(H=H[tri],bed=bed[tri],grad=grad,area=det*.5,ids=ids,ndof=len(xy)*layers*2,layers=layers,
        rows=np.broadcast_to(ids[:,:,None],(len(ids),12,12)).ravel().copy(),
        cols=np.broadcast_to(ids[:,None,:],(len(ids),12,12)).ravel().copy())


@njit(cache=True,parallel=True)
def kernel(local,H,bed,grad,area,layers,A,rho,g,reg,tangent):
    nk=layers-1;count=len(H)*nk;r=np.zeros((count,12));f=np.zeros((count,12));energies=np.zeros(count)
    K=np.zeros((count,12,12)) if tangent else np.zeros((0,12,12))
    coeff=A**(-1./3.)
    for p in prange(count):
        e=p//nk;k=p%nk;gs=np.zeros(2)
        for j in range(3):
            for axis in range(2):gs[axis]+=(bed[e,j]+H[e,j])*grad[e,j,axis]
        for b in range(3):
            lam=np.full(3,1./6.);lam[b]=2./3.;height=0.
            for j in range(3):height+=H[e,j]*lam[j]
            dz=height/nk
            for q in range(2):
                t=(1.+(2*q-1)/np.sqrt(3.))*.5;sigma=(k+t)/nk;zx=np.zeros(2)
                for j in range(3):
                    for axis in range(2):zx[axis]+=(bed[e,j]+sigma*H[e,j])*grad[e,j,axis]
                basis=np.zeros((5,12));shape=np.zeros(12)
                for end in range(2):
                    for j in range(3):
                        dof=(end*3+j)*2;vertical=-1. if end==0 else 1.;phi=(1.-t) if end==0 else t
                        gz=vertical*lam[j]/dz;gx=phi*grad[e,j,0]-gz*zx[0];gy=phi*grad[e,j,1]-gz*zx[1]
                        basis[0,dof]=gx;basis[1,dof+1]=gy;basis[2,dof]=gy;basis[2,dof+1]=gx
                        basis[3,dof]=gz;basis[4,dof+1]=gz;shape[dof]=phi*lam[j];shape[dof+1]=phi*lam[j]
                strain=basis@local[p]
                stress=np.array([4*strain[0]+2*strain[1],2*strain[0]+4*strain[1],strain[2],strain[3],strain[4]])
                Bu=basis.T@stress
                e2=strain[0]**2+strain[1]**2+strain[0]*strain[1]+.25*(strain[2]**2+strain[3]**2+strain[4]**2)+reg
                eta=.5*coeff*e2**(-1./3.);weight=area[e]*dz/6.
                energies[p]+=weight*1.5*coeff*(e2**(2./3.)-reg**(2./3.))
                for i in range(12):
                    force=-rho*g*gs[i%2]*shape[i]*weight
                    f[p,i]+=force;r[p,i]+=weight*eta*Bu[i]-force;energies[p]-=force*local[p,i]
                    if tangent:
                        for j in range(12):
                            form=(4*basis[0,i]*basis[0,j]+4*basis[1,i]*basis[1,j]+2*(basis[0,i]*basis[1,j]+basis[1,i]*basis[0,j])
                                  +basis[2,i]*basis[2,j]+basis[3,i]*basis[3,j]+basis[4,i]*basis[4,j])
                            K[p,i,j]+=weight*eta*(form-Bu[i]*Bu[j]/(6*e2))
    return r,f,energies,K


def evaluate(a,u,A,rho,g,reg,tangent=False):
    r,f,e,K=kernel(u[a['ids']],a['H'],a['bed'],a['grad'],a['area'],a['layers'],A,rho,g,reg,tangent)
    residual=np.bincount(a['ids'].ravel(),weights=r.ravel(),minlength=a['ndof'])
    force=np.bincount(a['ids'].ravel(),weights=f.ravel(),minlength=a['ndof'])
    matrix=coo_matrix((K.ravel(),(a['rows'],a['cols'])),shape=(a['ndof'],a['ndof'])).tocsr() if tangent else None
    return residual,float(e.sum()),matrix,force


def initial_sia(mesh,layers,A,rho,g):
    tri=mesh['triangles'];p=mesh['xy'][tri];jac=np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0]],axis=2)
    area=np.linalg.det(jac)*.5;grad=np.einsum('ij,tjk->tik',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(jac))
    slope=np.einsum('ti,tij->tj',(mesh['bed']+mesh['thickness'])[tri],grad);nn=len(mesh['xy'])
    weights=np.bincount(tri.ravel(),weights=np.repeat(area,3),minlength=nn);averaged=np.zeros((nn,2))
    for axis in range(2):averaged[:,axis]=np.bincount(tri.ravel(),weights=np.repeat(area*slope[:,axis],3),minlength=nn)/weights
    peak=-.5*A*(rho*g)**3*mesh['thickness'][:,None]**4*np.sum(averaged**2,axis=1)[:,None]*averaged
    return (peak[:,None,:]*(1-(1-np.linspace(0,1,layers))**4)[None,:,None]).ravel()


def solve(mesh,bc,*,layers=9,A=1e-16,rho=910.,g=9.81,regularization=1e-20,
          initial=None,checkpoint=None,iteration_offset=0,max_iterations=80,tolerance=1e-7,max_seconds=1800):
    start=time.perf_counter();a=prepare(mesh,layers);T,lift,cmeta=constraints(mesh,layers,bc)
    external,drag,lmeta=loads(mesh,layers,bc,rho,g);norms=np.asarray(T.power(2).sum(axis=0)).ravel()
    if not T.shape[1]:raise ValueError('No free unknowns')
    desired=initial_sia(mesh,layers,A,rho,g) if initial is None else np.asarray(initial).ravel()
    q=np.asarray(T.T@(desired-lift)).ravel()/norms;history=[]
    def reduced(q,tangent=False):
        u=T@q+lift;r,e,K,f=evaluate(a,u,A,rho,g,regularization,tangent)
        r+=drag@u-external;e+=.5*float(u@(drag@u))-float(external@u)
        rr=np.asarray(T.T@r).ravel();KK=(T.T@(K+drag)@T).tocsr() if tangent else None
        return rr,e,KK,f,u
    _,_,_,force,_=reduced(q)
    scale=max(float(np.linalg.norm(T.T@(force+external))),1.)
    for it in range(iteration_offset,max_iterations+1):
        r,energy,K,_,u=reduced(q,True);residual=float(np.linalg.norm(r)/scale)
        info=dict(iteration=it,relative_residual=residual,energy=energy,elapsed_seconds=time.perf_counter()-start)
        history.append(info)
        if checkpoint:checkpoint(it,u.reshape(-1,layers,2),info)
        print(json.dumps(info),flush=True)
        if not np.isfinite(residual):raise RuntimeError('Nonfinite state')
        if residual<tolerance:break
        if it==max_iterations or time.perf_counter()-start>max_seconds:raise RuntimeError('Newton/time budget reached')
        if len(q)<12000:delta=spsolve(K.tocsc(),-r);linear_count=1
        else:
            d=1/np.sqrt(K.diagonal());D=diags(d);scaled=(D@K@D).tocsr()
            np.random.seed(2387)
            # Per-level/component near-null candidates, transformed by constraints.
            fullmodes=np.zeros((a['ndof'],2*(layers-1)))
            for level in range(1,layers):
                for axis in range(2):fullmodes[level*2+axis::layers*2,(level-1)*2+axis]=1
            modes=np.asarray(T.T@fullmodes)/norms[:,None]/d[:,None]
            keep=np.linalg.norm(modes,axis=0)>1e-20;modes=modes[:,keep];modes/=np.linalg.norm(modes,axis=0)[None,:]
            ml=pyamg.smoothed_aggregation_solver(scaled,B=modes,symmetry='symmetric',max_coarse=160,max_levels=12)
            count=[0]
            def callback(_):count[0]+=1
            target=min(.03,max(1e-6,.02*np.sqrt(residual)));y=None
            for attempt in range(5):
                y,status=cg(scaled,-d*r,x0=y,M=ml.aspreconditioner(),rtol=target,atol=0,maxiter=max(1,500-count[0]),callback=callback)
                ratio=float(np.linalg.norm(K@(d*y)+r)/np.linalg.norm(r))
                if status or ratio<=.05 or count[0]>=500:break
                target=max(1e-14,target*min(.1,.02/ratio))
            if status or ratio>.05:raise RuntimeError(f'Linear convergence failed: {status}, {ratio}')
            delta=d*y;linear_count=count[0];del ml,scaled,fullmodes,modes
        descent=float(r@delta)
        if not np.isfinite(delta).all() or descent>=0:raise RuntimeError('Invalid Newton direction')
        alpha=1.
        for ls in range(35):
            rt,et,_,_,_=reduced(q+alpha*delta)
            resolved=abs(1e-4*alpha*descent)>64*np.finfo(float).eps*max(abs(energy),1.)
            decrease=np.linalg.norm(rt)<(1-1e-4*alpha)*np.linalg.norm(r)
            if (resolved and et<=energy+1e-4*alpha*descent) or ((not resolved) and decrease) or np.linalg.norm(rt)<.5*np.linalg.norm(r):
                q+=alpha*delta;break
            alpha*=.5
        else:raise RuntimeError('Line search failed')
        info.update(step_length=alpha,linear_iterations=linear_count)
    return u.reshape(-1,layers,2),dict(status='converged',relative_residual=residual,iterations=it,history=history,
        unknowns=a['ndof'],free_unknowns=T.shape[1],boundary_constraints=cmeta,boundary_loads=lmeta,
        wall_seconds=time.perf_counter()-start,vertical_nodes=layers)
