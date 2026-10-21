"""Minimum surface-gradient reconstruction with a bed obstacle and fixed anchors.

PalaeoIce2 motivates interpolating S before forming H=S-B. This implementation
is a disclosed graph obstacle problem, not a port of ArcGIS TopoToRaster.
"""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def reconstruct(mask,bed,oldH,anchors,max_iterations=100,tolerance_m=1e-5):
    rows,cols=np.nonzero(mask);n=len(rows);ids=np.full(mask.shape,-1,int);ids[mask]=np.arange(n)
    east=mask[:,:-1]&mask[:,1:];south=mask[:-1,:]&mask[1:,:]
    a=np.r_[ids[:,:-1][east],ids[:-1,:][south]];b=np.r_[ids[:,1:][east],ids[1:,:][south]]
    diagonal=np.bincount(np.r_[a,b],minlength=n).astype(float)
    L=coo_matrix((np.r_[diagonal,-np.ones(2*len(a))],(np.r_[np.arange(n),a,b],np.r_[np.arange(n),b,a])),shape=(n,n)).tocsr()
    boundary=mask&~binary_erosion(mask);fixed=(boundary|anchors)[mask];B=bed[mask];S=B+oldH[mask];active=np.zeros(n,bool);history=[]
    def energy(s):return float(.5*np.sum((s[a]-s[b])**2))
    old_energy=energy(S)
    for iteration in range(max_iterations):
        fixed_all=fixed|active;free=~fixed_all;target=np.where(fixed,B+oldH[mask],B)
        S[fixed_all]=target[fixed_all]
        if free.any():S[free]=spsolve(L[free][:,free],-(L[free][:,fixed_all]@S[fixed_all]))
        grad=L@S;candidate=~fixed&((S-B-grad/np.maximum(diagonal,1))<=0)
        projected=S-B-np.maximum(0,S-B-grad/np.maximum(diagonal,1));projected[fixed]=0
        residual=float(np.max(abs(projected)));history.append(dict(iteration=iteration,active_cells=int(active.sum()),projected_residual_m=residual,minimum_H_m=float((S-B).min())))
        if residual<tolerance_m and np.min(S-B)>-tolerance_m:break
        active=candidate
    else:raise RuntimeError('Surface obstacle active set did not converge')
    H=np.maximum(S-B,0);H[fixed]=oldH[mask][fixed];S=B+H
    result=np.zeros(mask.shape);result[mask]=H
    err=float(np.max(abs(H[fixed]-oldH[mask][fixed])))
    if err>1e-8 or not np.isfinite(H).all() or np.max(H)>2000 or energy(S)>old_energy*(1+1e-10):raise ValueError('Surface geometry check failed')
    return result,dict(status='converged',iterations=len(history),history=history,fixed_cells=int(fixed.sum()),anchor_cells=int((anchors&mask).sum()),zero_H_cells=int((H==0).sum()),constraint_error_m=err,old_surface_gradient_energy=old_energy,new_surface_gradient_energy=energy(S),max_H_m=float(H.max()),projected_residual_m=residual)

def fixture():
    y,x=np.mgrid[0:31,0:41];mask=np.ones(x.shape,bool);S=100+.5*x+.2*y;h=20*np.sin(np.pi*x/40)**2*np.sin(np.pi*y/30)**2;bed=S-h;anchors=np.zeros_like(mask);anchors[:,20]=True
    derived,info=reconstruct(mask,bed,h,anchors);error=float(np.max(abs(derived-h)))
    if error>1e-7:raise ValueError('Planar-surface reconstruction fixture failed')
    obstacle=bed.copy();obstacle[10:15,10:15]=S[10:15,10:15]+10;h2=np.maximum(S-obstacle,0);_,hill=reconstruct(mask,obstacle,h2,anchors)
    return dict(status='pass',analytic_planar_surface_max_error_m=error,obstacle_projected_residual_m=hill['projected_residual_m'],unchanged_BP_and_restart_evidence='Reuse v8 BP benchmarks and v12 independent H,A,u restart; no suite replay')

def map_to_mesh(mesh,values):
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];corners=mesh['pixel_corner_nodes'];H=np.zeros(len(mesh['xy']));H[nodes]=values[rows,cols]
    sums=np.bincount(corners.ravel(),weights=np.repeat(values[rows,cols],4),minlength=len(H));counts=np.bincount(corners.ravel(),minlength=len(H));use=counts>0;H[use]=sums[use]/counts[use]
    return H
