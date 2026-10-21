"""Global P1-triangle x P1-vertical Blatter-Pattyn finite elements, CPU float64.

Unknowns are u and v at every 3-D node. The weak form is assembled once over
the entire connected domain; tributary labels do not appear in the equations.
SI lengths, Pa, years. Basal linear drag is defined per horizontal bed area.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

RHO=900.;G=9.80665;YEAR=31536000.


def assemble(mesh,layers=3,beta=1e4):
    xy=mesh['xy'];tri=mesh['triangles'];B=mesh['bed'];H=mesh['thickness']
    if len(xy)*layers*2>20000:raise MemoryError('Prototype sparse-assembly budget exceeded; real native mesh requires a separately designed scalable solver')
    if layers<3 or np.any(H<=0):raise ValueError('Positive geometry and >=3 layers required')
    vertices=xy[tri];jac=np.stack([vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],axis=2)
    det=np.linalg.det(jac)
    if np.any(det<=0):raise ValueError('Inverted triangle')
    # grad lambda_i = reference grad lambda_i @ inverse horizontal Jacobian.
    grad=np.einsum('ij,tjk->tik',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(jac))
    gs=np.einsum('ti,tij->tj',(B+H)[tri],grad)
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    ids=[];forms=[];weights=[];loads=[];shape_store=[]
    for k in range(layers-1):
        nodes=np.concatenate([tri*layers+k,tri*layers+k+1],axis=1)
        dofs=np.stack([nodes*2,nodes*2+1],axis=2).reshape(len(tri),12)
        for lam in bary:
            height=H[tri]@lam;dz=height/(layers-1)
            for t in ((1-1/np.sqrt(3))/2,(1+1/np.sqrt(3))/2):
                sigma=(k+t)/(layers-1);zx=np.einsum('ti,tij->tj',(B+sigma*H)[tri],grad)
                shape=np.r_[(1-t)*lam,t*lam]
                gz=np.r_[-lam,lam][None,:]/dz[:,None]
                horizontal=np.concatenate([(1-t)*grad,t*grad],axis=1)-gz[:,:,None]*zx[:,None,:]
                ux=np.zeros((len(tri),12));uy=ux.copy();uz=ux.copy();vx=ux.copy();vy=ux.copy();vz=ux.copy()
                ux[:,::2]=horizontal[:,:,0];uy[:,::2]=horizontal[:,:,1];uz[:,::2]=gz
                vx[:,1::2]=horizontal[:,:,0];vy[:,1::2]=horizontal[:,:,1];vz[:,1::2]=gz
                outer=lambda a,b:np.einsum('ti,tj->tij',a,b)
                form=4*outer(ux,ux)+4*outer(vy,vy)+2*(outer(ux,vy)+outer(vy,ux))+outer(uy+vx,uy+vx)+outer(uz,uz)+outer(vz,vz)
                weight=det*.5*dz/6
                force=np.zeros((len(tri),12));force[:,::2]=-RHO*G*gs[:,0,None]*shape;force[:,1::2]=-RHO*G*gs[:,1,None]*shape
                ids.append(dofs);forms.append(form);weights.append(weight);loads.append(force*weight[:,None]);shape_store.append(shape)
    ids=np.concatenate(ids);forms=np.concatenate(forms);weights=np.concatenate(weights);local_f=np.concatenate(loads)
    ndof=len(xy)*layers*2;force=np.bincount(ids.ravel(),weights=local_f.ravel(),minlength=ndof)
    beta=np.broadcast_to(np.asarray(beta,dtype=float),(len(xy),))
    if np.any(beta<=0) or not np.isfinite(beta).all():raise ValueError('Positive finite basal friction required')
    basal=np.zeros((len(tri),6,6));bids=np.stack([tri*layers*2,tri*layers*2+1],axis=2).reshape(len(tri),6)
    for lam in bary:
        for axis in range(2):
            phi=np.zeros(6);phi[axis::2]=lam
            basal+=(det*.5/3*(beta[tri]@lam))[:,None,None]*np.outer(phi,phi)[None,:,:]
    rows=np.broadcast_to(bids[:,:,None],basal.shape).ravel();cols=np.broadcast_to(bids[:,None,:],basal.shape).ravel()
    drag=coo_matrix((basal.ravel(),(rows,cols)),shape=(ndof,ndof)).tocsr()
    return dict(ids=ids,forms=forms,weights=weights,force=force,drag=drag,ndof=ndof,layers=layers,
                rows=np.broadcast_to(ids[:,:,None],forms.shape).ravel(),cols=np.broadcast_to(ids[:,None,:],forms.shape).ravel())


def solve(mesh,*,layers=3,beta=1e4,A=2.4e-24*YEAR,n=3.,fixed_nodes=None,tolerance=1e-6,
          max_iterations=60,initial=None,checkpoint=None,iteration_offset=0,regularization=1e-20):
    if not np.isfinite([A,n,tolerance,regularization]).all() or A<=0 or n<1 or tolerance<=0 or regularization<=0:raise ValueError('Invalid physical/solver parameters')
    a=assemble(mesh,layers,beta);ids=a['ids'];B=a['forms'];weight=a['weights'];f=a['force'];drag=a['drag'];ndof=a['ndof']
    fixed=np.zeros(ndof,dtype=bool)
    if fixed_nodes is not None:
        nodes=np.asarray(fixed_nodes,dtype=int)
        if np.any((nodes<0)|(nodes>=len(mesh['xy']))):raise ValueError('Invalid boundary node')
        fixed[(nodes[:,None]*layers*2+np.arange(layers*2)[None,:]).ravel()]=True
    free=np.flatnonzero(~fixed);coeff=A**(-1/n);power=(1-n)/(2*n)
    if not len(free):raise ValueError('No free velocity unknowns')
    def matrix(K):return coo_matrix((K.ravel(),(a['rows'],a['cols'])),shape=(ndof,ndof)).tocsr()+drag
    def evaluate(u,tangent=False):
        local=u[ids];Bu=np.einsum('qij,qj->qi',B,local)
        raw=np.sum(local*Bu,axis=1)/4
        if np.min(raw)<-1e-10*max(1.,float(np.max(abs(raw)))):raise RuntimeError('Invalid strain energy')
        e=np.maximum(raw,0)+regularization;eta=.5*coeff*e**power
        r=np.bincount(ids.ravel(),weights=(weight[:,None]*eta[:,None]*Bu).ravel(),minlength=ndof)+drag@u-f
        potential=weight*2*n/(n+1)*coeff*(e**((n+1)/(2*n))-regularization**((n+1)/(2*n)))
        energy=float(np.sum(potential)-np.dot(f,u)+.5*np.dot(u,drag@u))
        if not tangent:return r,energy
        K=weight[:,None,None]*eta[:,None,None]*(B+(power/(2*e))[:,None,None]*np.einsum('qi,qj->qij',Bu,Bu))
        return r,energy,K
    norm=max(float(np.linalg.norm(f[free])),1.);history=[]
    u=np.zeros(ndof) if initial is None else np.asarray(initial,dtype=float).reshape(ndof).copy();u[fixed]=0
    for it in range(max_iterations+1):
        r,energy,K=evaluate(u,True);residual=float(np.linalg.norm(r[free])/norm)
        if not np.isfinite(residual):raise RuntimeError('Nonfinite momentum state')
        history.append(dict(iteration=it+iteration_offset,relative_residual=residual,energy=energy))
        if checkpoint:checkpoint(it+iteration_offset,u.reshape(-1,layers,2),history[-1])
        if residual<=tolerance:break
        if it==max_iterations:raise RuntimeError(f'BP Newton failed: {residual}')
        tangent=matrix(K);delta=np.zeros(ndof);delta[free]=spsolve(tangent[free][:,free].tocsc(),-r[free])
        if not np.isfinite(delta).all():raise RuntimeError('Nonfinite sparse solve')
        descent=float(r@delta);alpha=1.
        for ls in range(35):
            trial=u+alpha*delta;rt,et=evaluate(trial)
            if et<=energy+1e-4*alpha*descent or np.linalg.norm(rt[free])<np.linalg.norm(r[free])*.5:
                u=trial;break
            alpha*=.5
        else:raise RuntimeError('BP line search failed')
    return u.reshape(-1,layers,2),dict(converged=True,relative_residual=residual,iterations=it,
        history=history,unknowns=ndof,free_unknowns=len(free),quadrature_points=len(ids),
        formulation='global vector BP P1 triangle x P1 vertical',stress_continuity='weak-form internal traction balance',
        numerical_stress_flux_error_evaluated=False)
