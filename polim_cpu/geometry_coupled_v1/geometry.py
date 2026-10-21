"""Analytic FH action of v13r1.bp.kernel on a fixed momentum wet branch.

Differentiate surface driving gradient, sigma-coordinate strain basis, dz,
quadrature weight, and nonlinear viscosity. A, bed and horizontal mesh fixed.
No nonlinear BP solve and no dense nodal geometry Jacobian is constructed.
"""
import numpy as np
from numba import njit,prange
from polim_cpu.v13r1.bp import RHO,G,REG

@njit(cache=True,parallel=True)
def kernel(local,H,dH,bed,grad,area,A,regularization):
    nt=len(H);out=np.zeros((nt,12))
    for e in prange(nt):
        gs=np.zeros(2);dgs=np.zeros(2)
        for j in range(3):
            for axis in range(2):
                gs[axis]+=(bed[e,j]+H[e,j])*grad[e,j,axis]
                dgs[axis]+=dH[e,j]*grad[e,j,axis]
        for k in range(2):
            for b in range(3):
                lam=np.full(3,1./6.);lam[b]=2./3.
                height=H[e]@lam;dheight=dH[e]@lam;dz=.5*height;ratio=dheight/height
                for q in range(2):
                    t=(1.+(2*q-1)/np.sqrt(3.))*.5;sigma=(k+t)*.5
                    zx=np.zeros(2);dzx=np.zeros(2)
                    for j in range(3):
                        for axis in range(2):
                            zx[axis]+=(bed[e,j]+sigma*H[e,j])*grad[e,j,axis]
                            dzx[axis]+=sigma*dH[e,j]*grad[e,j,axis]
                    B=np.zeros((5,12));dB=np.zeros((5,12));shape=np.zeros(12)
                    for j in range(3):
                        for end in range(2):
                            level=k+end
                            if level==0:continue
                            dof=j*4+(level-1)*2;vertical=-1. if end==0 else 1.;phi=1.-t if end==0 else t
                            gz=vertical*lam[j]/dz;dgz=-gz*ratio
                            gx=phi*grad[e,j,0]-gz*zx[0];gy=phi*grad[e,j,1]-gz*zx[1]
                            dx=-dgz*zx[0]-gz*dzx[0];dy=-dgz*zx[1]-gz*dzx[1]
                            B[0,dof]=gx;B[1,dof+1]=gy;B[2,dof]=gy;B[2,dof+1]=gx;B[3,dof]=gz;B[4,dof+1]=gz
                            dB[0,dof]=dx;dB[1,dof+1]=dy;dB[2,dof]=dy;dB[2,dof+1]=dx;dB[3,dof]=dgz;dB[4,dof+1]=dgz
                            shape[dof]=phi*lam[j];shape[dof+1]=phi*lam[j]
                    strain=B@local[e];ds=dB@local[e]
                    stress=np.array([4*strain[0]+2*strain[1],2*strain[0]+4*strain[1],strain[2],strain[3],strain[4]])
                    dstress=np.array([4*ds[0]+2*ds[1],2*ds[0]+4*ds[1],ds[2],ds[3],ds[4]])
                    Bu=B.T@stress;dBu=dB.T@stress+B.T@dstress
                    e2=strain[0]**2+strain[1]**2+strain[0]*strain[1]+.25*(strain[2]**2+strain[3]**2+strain[4]**2)+regularization
                    de2=.5*(stress@ds);fluidity=0.
                    for node in range(3):fluidity+=lam[node]*((1-t)*A[e,node,k]+t*A[e,node,k+1])
                    eta=.5*fluidity**(-1./3.)*e2**(-1./3.);deta=-eta*de2/(3*e2);weight=area[e]*dz/6.
                    for i in range(12):
                        out[e,i]+=weight*((ratio*eta+deta)*Bu[i]+eta*dBu[i])+RHO*G*shape[i]*weight*(dgs[i%2]+gs[i%2]*ratio)
    return out

def action(a,u,dh,A):
    if np.any(dh[~a['wet'][::4]]!=0):raise ValueError('FH smooth action cannot change dry momentum nodes')
    v=kernel(u[a['ids']],a['H'],dh[a['tri']],a['bed'],a['grad'],a['area'],np.broadcast_to(A,(a['ndof']//4,3))[a['tri']],REG)
    out=np.bincount(a['ids'].ravel(),weights=v.ravel(),minlength=a['ndof']);out[~a['wet']]=0.
    return out
