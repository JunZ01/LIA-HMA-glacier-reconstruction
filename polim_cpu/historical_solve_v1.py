"""Same BP Newton method with bounded work for changing ice support."""
from .gpu_cached_v1 import torch,time,T

def solve(bp,u,mg,checkpoint=None,refresh=None):
    start=time.perf_counter();u=u.clone();u[~bp.wet]=0.;history=[]
    for it in range(81):
        r,e=bp.evaluate(u,True);res=float(r.norm()/bp.force_norm);history.append(dict(iteration=it,residual=res))
        if res<=1e-6:return u,dict(converged=True,iterations=it,relative_residual=res,history=history,seconds=time.perf_counter()-start)
        if it==80 or time.perf_counter()-start>120:raise RuntimeError('Historical Newton budget (80 iterations/120s)')
        try:delta,k,true=bp.pcg.solve(r)
        except RuntimeError as exc:
            if str(exc) not in ['200CG budget','True linear residual failed'] or refresh is None:raise
            # Rebuild from the already computed current exact tangent once.
            # No new geometry, time advancement or convergence relaxation.
            refresh(it)
            delta,k,true=bp.pcg.solve(r)
        rn=r.norm();descent=float(r@delta);alpha=1.;energy=float(e)
        if descent>=0:raise RuntimeError('Non-descent Newton direction')
        for _ in range(35):
            trial=u+alpha*delta;rt,et=bp.evaluate(trial)
            resolved=abs(1e-4*alpha*descent)>64*torch.finfo(T).eps*max(abs(energy),1.)
            decreased=float(rt.norm())<(1-1e-4*alpha)*float(rn)
            if (resolved and float(et)<=energy+1e-4*alpha*descent) or (not resolved and decreased) or float(rt.norm())<.5*float(rn):u=trial;break
            alpha*=.5
        else:raise RuntimeError('Line search failed')
        history[-1].update(cg_iterations=k,linear_residual=true,step_length=alpha)
        if checkpoint and (it+1)%5==0:checkpoint(it+1,u,history[-1])
