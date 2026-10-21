"""v23 BP nonlinear solver with SuperLU MMD_AT_PLUS_A linear solves."""
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .bp_accelerated_v1 import HorizontalCache, prepare, evaluate
def solve(mesh, initial, A, checkpoint=None, cache=None, iteration_offset=0,
          tolerance=1e-6, max_iterations=80, max_seconds=600):
    start = time.perf_counter(); times = dict(prepare=0.,assembly=0.,amg=0.,cg=0.,linesearch=0.,checkpoint=0.,factor=0.,direct_solve=0.)
    t = time.perf_counter(); a = prepare(mesh,A,cache); times['prepare'] += time.perf_counter()-t
    u = np.asarray(initial).reshape(-1).copy(); u[~a['wet']] = 0.
    if u.size != a['ndof'] or not np.isfinite(u).all(): raise ValueError('Invalid initial state')
    history = []; ml = None; cached_trial = None; previous_cg = 0; builds = 0; matrices = 0; factorizations = 0
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
        t = time.perf_counter()
        diagonal = K.diagonal()
        if np.any(diagonal <= 0): raise RuntimeError('Nonpositive tangent diagonal')
        d = 1/np.sqrt(diagonal); D = diags(d); scaled = (D@K@D).tocsc()
        factor = splu(scaled,permc_spec='MMD_AT_PLUS_A')
        times['factor'] += time.perf_counter()-t; factorizations += 1
        t = time.perf_counter(); delta = d*factor.solve(-d*r)
        linear_ratio = float(np.linalg.norm(K@delta+r)/np.linalg.norm(r))
        times['direct_solve'] += time.perf_counter()-t
        if linear_ratio > 1e-8: raise RuntimeError('Direct true linear residual failed')
        del factor,scaled,D
        count = [0]
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
                    amg_builds=builds,tangent_assemblies=matrices,direct_factorizations=factorizations)
