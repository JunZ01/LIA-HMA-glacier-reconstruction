"""BP with per-geometry quadrature caching and within-solve AMG reuse.

Same quadrature, rheology, true Jacobian and stopping criteria as v13r1.
The velocity-dependent terms are always evaluated at the current iterate.
"""
import time
import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from .v13r1 import bp as original


@njit(cache=True, parallel=True)
def geometry(H, bed, grad, area, A):
    nt = len(H)
    bases = np.zeros((nt, 12, 5, 12))
    forces = np.zeros((nt, 12, 12))
    coefficients = np.zeros((nt, 12))
    weights = np.zeros((nt, 12))
    for e in prange(nt):
        gs = np.zeros(2)
        for j in range(3):
            for axis in range(2):
                gs[axis] += (bed[e,j]+H[e,j])*grad[e,j,axis]
        for k in range(2):
            for b in range(3):
                lam = np.full(3, 1./6.); lam[b] = 2./3.
                height = 0.
                for j in range(3): height += H[e,j]*lam[j]
                dz = height*.5
                for q in range(2):
                    point = k*6+b*2+q
                    t = (1.+(2*q-1)/np.sqrt(3.))*.5; sigma = (k+t)*.5
                    zx = np.zeros(2)
                    for j in range(3):
                        for axis in range(2):
                            zx[axis] += (bed[e,j]+sigma*H[e,j])*grad[e,j,axis]
                    shape = np.zeros(12)
                    for j in range(3):
                        for end in range(2):
                            level = k+end
                            if level == 0: continue
                            dof = j*4+(level-1)*2
                            vertical = -1. if end == 0 else 1.
                            phi = (1.-t) if end == 0 else t
                            gz = vertical*lam[j]/dz
                            gx = phi*grad[e,j,0]-gz*zx[0]
                            gy = phi*grad[e,j,1]-gz*zx[1]
                            bases[e,point,0,dof] = gx; bases[e,point,1,dof+1] = gy
                            bases[e,point,2,dof] = gy; bases[e,point,2,dof+1] = gx
                            bases[e,point,3,dof] = gz; bases[e,point,4,dof+1] = gz
                            shape[dof] = phi*lam[j]; shape[dof+1] = phi*lam[j]
                    fluidity = 0.
                    for node in range(3):
                        fluidity += lam[node]*((1-t)*A[e,node,k]+t*A[e,node,k+1])
                    coefficients[e,point] = fluidity**(-1./3.)
                    weight = area[e]*dz/6.; weights[e,point] = weight
                    for i in range(12):
                        forces[e,point,i] = -original.RHO*original.G*gs[i%2]*shape[i]*weight
    return bases, forces, coefficients, weights


@njit(cache=True, parallel=True)
def kernel(local, bases, forces, coefficients, weights, regularization, tangent):
    nt = len(local)
    r = np.zeros((nt,12)); f = np.zeros((nt,12)); energies = np.zeros(nt)
    K = np.zeros((nt,12,12)) if tangent else np.zeros((0,12,12))
    power = -1./3.
    for e in prange(nt):
        for p in range(12):
            basis = bases[e,p]
            strain = basis@local[e]
            stress = np.array([4*strain[0]+2*strain[1],2*strain[0]+4*strain[1],strain[2],strain[3],strain[4]])
            Bu = basis.T@stress
            e2 = (strain[0]*strain[0]+strain[1]*strain[1]+strain[0]*strain[1]
                  +.25*(strain[2]**2+strain[3]**2+strain[4]**2))+regularization
            coeff = coefficients[e,p]; weight = weights[e,p]
            eta = .5*coeff*e2**power
            energies[e] += weight*1.5*coeff*(e2**(2./3.)-regularization**(2./3.))
            for i in range(12):
                force = forces[e,p,i]
                f[e,i] += force; r[e,i] += weight*eta*Bu[i]-force
                energies[e] -= force*local[e,i]
                if tangent:
                    for j in range(12):
                        form = (4*basis[0,i]*basis[0,j]+4*basis[1,i]*basis[1,j]
                                +2*(basis[0,i]*basis[1,j]+basis[1,i]*basis[0,j])
                                +basis[2,i]*basis[2,j]+basis[3,i]*basis[3,j]+basis[4,i]*basis[4,j])
                        K[e,i,j] += weight*eta*(form+power/(2*e2)*Bu[i]*Bu[j])
    return r, f, energies, K


class HorizontalCache:
    def __init__(self, mesh):
        self.xy = mesh['xy'].copy(); self.bed = mesh['bed'].copy()
        self.tri = mesh['triangles'].copy()
        self.base = original.prepare({**mesh, 'thickness': np.ones(len(self.xy))})

    def prepare(self, mesh):
        if not (np.array_equal(mesh['xy'],self.xy) and np.array_equal(mesh['bed'],self.bed)
                and np.array_equal(mesh['triangles'],self.tri)):
            raise ValueError('Horizontal cache identity changed')
        raw = np.asarray(mesh['thickness'])
        if not np.isfinite(raw).all() or np.any(raw < 0): raise ValueError('Invalid H')
        threshold = 64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(self.bed),abs(self.bed+raw)))
        H = np.where(raw>threshold,raw,0.)
        base = self.base; active = np.sum(H[base['tri']],axis=1)>0
        a = {k:base[k][active] for k in ['tri','bed','grad','area','ids']}
        for k in ['rows','cols']: a[k] = base[k].reshape(-1,144)[active].ravel()
        a.update(H=H[a['tri']].copy(),wet=np.repeat(H>0,4),ndof=base['ndof'])
        return a


def prepare(mesh, A, cache=None):
    a = original.prepare(mesh) if cache is None else cache.prepare(mesh)
    a['quadrature'] = geometry(a['H'],a['bed'],a['grad'],a['area'],
                               np.broadcast_to(A,(a['ndof']//4,3))[a['tri']])
    return a


def evaluate(a, u, tangent=False, regularization=original.REG):
    r,f,e,K = kernel(u[a['ids']],*a['quadrature'],regularization,tangent)
    r = np.bincount(a['ids'].ravel(),weights=r.ravel(),minlength=a['ndof'])
    f = np.bincount(a['ids'].ravel(),weights=f.ravel(),minlength=a['ndof'])
    matrix = None
    if tangent:
        matrix = coo_matrix((K.ravel(),(a['rows'],a['cols'])),shape=(a['ndof'],a['ndof'])).tocsr()
        W = diags(a['wet'].astype(float))
        matrix = (W@matrix@W+diags((~a['wet']).astype(float))).tocsr()
        matrix.eliminate_zeros()
    r[~a['wet']] = u[~a['wet']]; f[~a['wet']] = 0.
    return r,float(e.sum())+.5*float(u[~a['wet']]@u[~a['wet']]),matrix,f


def solve(mesh, initial, A, checkpoint=None, cache=None, iteration_offset=0,
          tolerance=1e-6, max_iterations=80, max_seconds=600):
    start = time.perf_counter(); times = dict(prepare=0.,assembly=0.,amg=0.,cg=0.,linesearch=0.,checkpoint=0.)
    t = time.perf_counter(); a = prepare(mesh,A,cache); times['prepare'] += time.perf_counter()-t
    u = np.asarray(initial).reshape(-1).copy(); u[~a['wet']] = 0.
    if u.size != a['ndof'] or not np.isfinite(u).all(): raise ValueError('Invalid initial state')
    history = []; ml = None; cached_trial = None; previous_cg = 0; builds = 0; matrices = 0
    for it in range(iteration_offset,max_iterations+1):
        t = time.perf_counter()
        if cached_trial is not None:
            r,energy,f = cached_trial
            residual = float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))
        else: residual = float('inf')
        # The line search has already evaluated this exact accepted iterate.
        if residual > tolerance:
            r,energy,K,f = evaluate(a,u,True); matrices += 1
            residual = float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))
        times['assembly'] += time.perf_counter()-t
        info = dict(iteration=it,relative_residual=residual,energy=energy,elapsed_seconds=time.perf_counter()-start)
        history.append(info)
        t = time.perf_counter()
        if checkpoint: checkpoint(it,u.reshape(-1,2,2),info)
        times['checkpoint'] += time.perf_counter()-t
        if not np.isfinite(residual): raise RuntimeError('Nonfinite momentum')
        if residual <= tolerance: break
        if it == max_iterations or time.perf_counter()-start > max_seconds:
            raise RuntimeError('Registered Newton/runtime budget reached')
        diagonal = K.diagonal()
        if np.any(diagonal <= 0): raise RuntimeError('Nonpositive tangent diagonal')
        d = 1/np.sqrt(diagonal); D = diags(d); scaled = (D@K@D).tocsr()
        # Reuse only inside a geometry-frozen nonlinear solve; restart starts fresh.
        if ml is None or previous_cg > 20:
            t = time.perf_counter(); modes = np.zeros((a['ndof'],4))
            for j in range(4): modes[j::4,j] = 1/d[j::4]
            modes /= np.linalg.norm(modes,axis=0)[None,:]
            np.random.seed(2387)
            ml = original.pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,
                symmetry='symmetric',max_levels=12,max_coarse=160,
                presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),
                postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
            times['amg'] += time.perf_counter()-t; builds += 1
        count = [0]
        def inc(_): count[0] += 1
        t = time.perf_counter(); linear_tolerance = min(.1,max(1e-4,.05*np.sqrt(residual)))
        y = None; linear_ratio = float('inf')
        for correction in range(5):
            y,status = cg(scaled,-d*r,x0=y,M=ml.aspreconditioner(),rtol=linear_tolerance,
                          atol=0,maxiter=max(1,500-count[0]),callback=inc)
            linear_ratio = float(np.linalg.norm(K@(d*y)+r)/np.linalg.norm(r))
            if status or linear_ratio <= .1 or count[0] >= 500: break
            linear_tolerance = max(1e-14,linear_tolerance*min(.1,.05/linear_ratio))
        times['cg'] += time.perf_counter()-t; previous_cg = count[0]
        if status or linear_ratio > .1: raise RuntimeError('AMG-CG true residual failed')
        delta = d*y
        if not np.isfinite(delta).all(): raise RuntimeError('Nonfinite correction')
        descent = float(r@delta)
        if descent >= 0: raise RuntimeError('Newton direction not descending')
        alpha = 1.; t = time.perf_counter()
        for ls in range(35):
            trial = u+alpha*delta; rt,et,_,ft = evaluate(a,trial,False)
            resolved = abs(1e-4*alpha*descent)>64*np.finfo(float).eps*max(abs(energy),1.)
            decreased = np.linalg.norm(rt)<(1-1e-4*alpha)*np.linalg.norm(r)
            if ((resolved and et<=energy+1e-4*alpha*descent) or ((not resolved) and decreased)
                    or np.linalg.norm(rt)<np.linalg.norm(r)*.5):
                u = trial; cached_trial = (rt,et,ft); break
            alpha *= .5
        else: raise RuntimeError('Newton line search failed')
        times['linesearch'] += time.perf_counter()-t
        info.update(cg_iterations=count[0],step_length=alpha,physical_linear_residual_ratio=linear_ratio)
    full = np.zeros((len(mesh['xy']),3,2)); full[:,1:,:] = u.reshape(-1,2,2)
    return full,dict(converged=True,relative_residual=residual,iterations=it,history=history,
                    free_unknowns=len(u),wall_seconds=time.perf_counter()-start,timers=times,
                    amg_builds=builds,tangent_assemblies=matrices)
