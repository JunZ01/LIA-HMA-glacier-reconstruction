"""BP evolution kernel derived from frozen v7r1; local thermal fluidity and dry-node elimination.

Three vertical nodes; basal u=v=0 exactly. Sub-ULP momentum geometry is dry; conserved H unchanged.
Units: m, Pa, year. Triangle kernels avoid storing every quadrature matrix.
"""
import sys,time
from pathlib import Path
import numpy as np
from numba import njit,prange
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg,spsolve

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.runtime/pyamg-5.2.1'))
import pyamg

YEAR=31536000.; A_SI=7.538801981725612e-23
RHO=900.; G=9.80665; REG=1e-20


def prepare(mesh):
    alltri=np.asarray(mesh['triangles'],dtype=np.int32);xy=mesh['xy']
    rawH=np.asarray(mesh['thickness']);bed=np.asarray(mesh['bed'])
    # Below vertical coordinate roundoff, S=B+H is not meaningfully resolved.
    # Keep transport/state H exactly; this local copy is momentum geometry only.
    threshold=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(bed),abs(bed+rawH)))
    H=np.where(rawH>threshold,rawH,0.)
    active=np.sum(H[alltri],axis=1)>0;tri=alltri[active]
    if not all(np.isfinite(a).all() for a in (xy,rawH,bed)) or np.any(rawH<0):
        raise ValueError('Invalid geometry')
    vertices=xy[tri]
    jac=np.stack([vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],axis=2)
    det=np.linalg.det(jac)
    if np.any(det<=0):raise ValueError('Inverted triangle')
    grad=np.einsum('ij,tjk->tik',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(jac))
    ids=(tri[:,:,None]*4+np.arange(4,dtype=np.int32)).reshape(-1,12)
    rows=np.broadcast_to(ids[:,:,None],(len(tri),12,12)).ravel().copy()
    cols=np.broadcast_to(ids[:,None,:],(len(tri),12,12)).ravel().copy()
    return dict(wet=np.repeat(H>0,4),tri=tri,H=H[tri].copy(),bed=bed[tri].copy(),grad=grad,area=det*.5,
                ids=ids,rows=rows,cols=cols,ndof=len(xy)*4)


@njit(cache=True,parallel=True)
def kernel(local,H,bed,grad,area,A,regularization,tangent):
    nt=len(H);r=np.zeros((nt,12));f=np.zeros((nt,12));energies=np.zeros(nt)
    K=np.zeros((nt,12,12)) if tangent else np.zeros((0,12,12))
    power=-1./3.
    for e in prange(nt):
        gs=np.zeros(2)
        for j in range(3):
            for axis in range(2):gs[axis]+=(bed[e,j]+H[e,j])*grad[e,j,axis]
        for k in range(2):
            for b in range(3):
                lam=np.full(3,1./6.);lam[b]=2./3.
                height=0.
                for j in range(3):height+=H[e,j]*lam[j]
                dz=height*.5
                for q in range(2):
                    t=(1.+(2*q-1)/np.sqrt(3.))*.5;sigma=(k+t)*.5
                    zx=np.zeros(2)
                    for j in range(3):
                        for axis in range(2):zx[axis]+=(bed[e,j]+sigma*H[e,j])*grad[e,j,axis]
                    basis=np.zeros((5,12));shape=np.zeros(12)
                    for j in range(3):
                        for end in range(2):
                            level=k+end
                            if level==0:continue
                            dof=j*4+(level-1)*2
                            vertical=-1. if end==0 else 1.
                            phi=(1.-t) if end==0 else t
                            gz=vertical*lam[j]/dz
                            gx=phi*grad[e,j,0]-gz*zx[0]
                            gy=phi*grad[e,j,1]-gz*zx[1]
                            basis[0,dof]=gx;basis[1,dof+1]=gy
                            basis[2,dof]=gy;basis[2,dof+1]=gx
                            basis[3,dof]=gz;basis[4,dof+1]=gz
                            shape[dof]=phi*lam[j];shape[dof+1]=phi*lam[j]
                    strain=basis@local[e]
                    stress=np.array([4*strain[0]+2*strain[1],2*strain[0]+4*strain[1],strain[2],strain[3],strain[4]])
                    Bu=basis.T@stress
                    e2=(strain[0]*strain[0]+strain[1]*strain[1]+strain[0]*strain[1]
                        +.25*(strain[2]**2+strain[3]**2+strain[4]**2))+regularization
                    fluidity=0.
                    for node in range(3):fluidity+=lam[node]*((1-t)*A[e,node,k]+t*A[e,node,k+1])
                    coeff=fluidity**(-1./3.)
                    eta=.5*coeff*e2**power;weight=area[e]*dz/6.
                    energies[e]+=weight*1.5*coeff*(e2**(2./3.)-regularization**(2./3.))
                    for i in range(12):
                        force=-RHO*G*gs[i%2]*shape[i]*weight
                        f[e,i]+=force;r[e,i]+=weight*eta*Bu[i]-force
                        energies[e]-=force*local[e,i]
                        if tangent:
                            for j in range(12):
                                form=(4*basis[0,i]*basis[0,j]+4*basis[1,i]*basis[1,j]
                                    +2*(basis[0,i]*basis[1,j]+basis[1,i]*basis[0,j])
                                    +basis[2,i]*basis[2,j]+basis[3,i]*basis[3,j]+basis[4,i]*basis[4,j])
                                K[e,i,j]+=weight*eta*(form+power/(2*e2)*Bu[i]*Bu[j])
    return r,f,energies,K


def evaluate(a,u,tangent=False,A=A_SI*YEAR,regularization=REG):
    r,f,e,K=kernel(u[a['ids']],a['H'],a['bed'],a['grad'],a['area'],np.broadcast_to(A,(a['ndof']//4,3))[a['tri']],regularization,tangent)
    r=np.bincount(a['ids'].ravel(),weights=r.ravel(),minlength=a['ndof'])
    f=np.bincount(a['ids'].ravel(),weights=f.ravel(),minlength=a['ndof'])
    matrix=None
    if tangent:
        matrix=coo_matrix((K.ravel(),(a['rows'],a['cols'])),shape=(a['ndof'],a['ndof'])).tocsr()
        W=diags(a['wet'].astype(float));matrix=(W@matrix@W+diags((~a['wet']).astype(float))).tocsr()
        matrix.eliminate_zeros()
    r[~a['wet']]=u[~a['wet']];f[~a['wet']]=0.
    return r,float(e.sum())+.5*float(u[~a['wet']]@u[~a['wet']]),matrix,f


def sia_initial(mesh,A=A_SI*YEAR):
    a=prepare(mesh);tri=a['tri'];H=np.where(a['wet'][::4],mesh['thickness'],0.)
    slope=np.einsum('ti,tij->tj',a['bed']+a['H'],a['grad'])
    g=np.zeros((len(H),2));weight=np.bincount(tri.ravel(),weights=np.repeat(a['area'],3),minlength=len(H))
    for axis in range(2):g[:,axis]=np.bincount(tri.ravel(),weights=np.repeat(slope[:,axis]*a['area'],3),minlength=len(H))/np.maximum(weight,1e-300)
    surface=-.5*np.mean(np.broadcast_to(A,(len(H),3)),axis=1)[:,None]*(RHO*G)**3*H[:,None]**4*np.sum(g*g,axis=1)[:,None]*g
    out=surface[:,None,:]*(1-(1-np.array([.5,1.]))**4)[None,:,None]
    return out.ravel()


def solve(mesh,initial=None,iteration_offset=0,checkpoint=None,tolerance=1e-6,max_iterations=80,
          max_seconds=7200,direct=False,A=A_SI*YEAR,regularization=REG):
    start=time.perf_counter();a=prepare(mesh);history=[]
    u=sia_initial(mesh,A) if initial is None else np.asarray(initial).reshape(-1).copy()
    u[~a['wet']]=0.
    if u.size!=a['ndof'] or not np.isfinite(u).all():raise ValueError('Invalid initial state')
    for it in range(iteration_offset,max_iterations+1):
        r,energy,K,f=evaluate(a,u,True,A,regularization)
        residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))
        info=dict(iteration=it,relative_residual=residual,energy=energy,elapsed_seconds=time.perf_counter()-start)
        history.append(info)
        if checkpoint:checkpoint(it,u.reshape(-1,2,2),info)
        print(__import__('json').dumps(info),flush=True)
        if not np.isfinite(residual):raise RuntimeError('Nonfinite momentum')
        if residual<=tolerance:break
        if it==max_iterations or time.perf_counter()-start>max_seconds:raise RuntimeError('Registered Newton/runtime budget reached')
        if direct:delta=spsolve(K.tocsc(),-r);iterations=[1]
        else:
            diagonal=K.diagonal()
            if np.any(diagonal<=0):raise RuntimeError('Nonpositive tangent diagonal')
            d=1/np.sqrt(diagonal);D=diags(d);scaled=(D@K@D).tocsr()
            # Four block modes cover separate u/v at each free vertical level.
            modes=np.zeros((a['ndof'],4))
            for j in range(4):modes[j::4,j]=1/d[j::4]
            modes/=np.linalg.norm(modes,axis=0)[None,:]
            np.random.seed(2387)
            ml=pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,
                symmetry='symmetric',max_levels=12,max_coarse=160,
                presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),
                postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
            iterations=[0]
            def count(_):iterations[0]+=1
            linear_tolerance=min(.1,max(1e-4,.05*np.sqrt(residual)))
            y=None;linear_ratio=float('inf')
            for correction in range(5):
                y,status=cg(scaled,-d*r,x0=y,M=ml.aspreconditioner(),rtol=linear_tolerance,atol=0,maxiter=max(1,500-iterations[0]),callback=count)
                linear_ratio=float(np.linalg.norm(K@(d*y)+r)/np.linalg.norm(r))
                if status or linear_ratio<=.1 or iterations[0]>=500:break
                linear_tolerance=max(1e-14,linear_tolerance*min(.1,.05/linear_ratio))
            if linear_ratio>.1:raise RuntimeError(f'Physical linear residual criterion failed: {linear_ratio}')
            print(__import__('json').dumps(dict(iteration=it,cg_iterations=iterations[0],cg_status=int(status),physical_linear_residual_ratio=linear_ratio,operator_complexity=ml.operator_complexity())),flush=True)
            if status:raise RuntimeError(f'AMG-CG failed at iteration {it}, status {status}')
            delta=d*y
            del ml,scaled,D,modes
        if not np.isfinite(delta).all():raise RuntimeError('Nonfinite linear correction')
        descent=float(r@delta)
        if descent>=0:raise RuntimeError('Newton direction not descending')
        alpha=1.
        for ls in range(35):
            trial=u+alpha*delta;rt,et,_,_=evaluate(a,trial,False,A,regularization)
            energy_resolved=abs(1e-4*alpha*descent)>64*np.finfo(float).eps*max(abs(energy),1.)
            residual_decreased=np.linalg.norm(rt)<(1-1e-4*alpha)*np.linalg.norm(r)
            if (energy_resolved and et<=energy+1e-4*alpha*descent) or ((not energy_resolved) and residual_decreased) or np.linalg.norm(rt)<np.linalg.norm(r)*.5:
                u=trial;break
            alpha*=.5
        else:raise RuntimeError('Newton line search failed')
        info.update(cg_iterations=iterations[0],step_length=alpha)
    full=np.zeros((len(mesh['xy']),3,2));full[:,1:,:]=u.reshape(-1,2,2)
    return full,dict(converged=True,relative_residual=residual,iterations=it,history=history,
                    unknowns=len(u)*3//2,free_unknowns=len(u),wall_seconds=time.perf_counter()-start)


def stress_centroids(mesh,velocity,A=A_SI*YEAR,regularization=REG):
    """Physical strain and deviatoric stresses at sigma .25/.75, triangle centroid.
    No nodal stress smoothing; stress need not be pointwise continuous for P1 FE.
    """
    a=prepare(mesh);tri=a['tri'];uv=velocity[tri];H=a['H'];bed=a['bed'];grad=a['grad']
    results=[]
    for k in range(2):
        sigma=(k+.5)*.5;dz=H.mean(axis=1)*.5
        zx=np.einsum('ti,tij->tj',bed+sigma*H,grad)
        gz=np.tile(np.r_[-np.ones(3)/3,np.ones(3)/3],(len(tri),1))/dz[:,None]
        horizontal=np.concatenate([.5*grad,.5*grad],axis=1)-gz[:,:,None]*zx[:,None,:]
        local=np.concatenate([uv[:,:,k,:],uv[:,:,k+1,:]],axis=1)
        deriv=np.einsum('tna,tnb->tab',local,np.concatenate([horizontal,gz[:,:,None]],axis=2))
        ux=deriv[:,0,0];uy=deriv[:,0,1];uz=deriv[:,0,2]
        vx=deriv[:,1,0];vy=deriv[:,1,1];vz=deriv[:,1,2]
        e2=ux*ux+vy*vy+ux*vy+.25*((uy+vx)**2+uz**2+vz**2)+regularization
        local_A=np.broadcast_to(A,(len(mesh['xy']),3))[tri]
        mid_A=(local_A[:,:,k]+local_A[:,:,k+1]).mean(axis=1)*.5
        eta=.5*mid_A**(-1/3)*e2**(-1/3)
        results.append(np.column_stack([2*eta*ux,2*eta*vy,eta*(uy+vx),eta*uz,eta*vz,
                                       -2*eta*(ux+vy),np.sqrt(e2),eta]))
    return np.stack(results,axis=1)
