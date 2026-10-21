"""Exact BP Newton with a checked terminal residual and reusable scaling.

The terminal line-search residual is already the exact nonlinear residual;
do not assemble another unused tangent when it satisfies the original gate.
"""
from .gpu_cached_v1 import torch, time, T


def solve(bp, u, refresh, checkpoint=None):
    start = time.perf_counter()
    u = u.clone()
    u[~bp.wet] = 0.
    history = []
    for it in range(81):
        r, e = bp.evaluate(u, True)
        res = float(r.norm() / bp.force_norm)
        history.append(dict(iteration=it, residual=res))
        if res <= 1e-6:
            return u, dict(iterations=it, relative_residual=res, history=history)
        if it == 80 or time.perf_counter() - start > 120:
            raise RuntimeError('Adaptive Newton budget')
        # A fixed SPD V-cycle remains a preconditioner after geometry changes.
        # Positive congruence scaling changes conditioning, not the BP operator.
        diagonal = bp.values[bp.diagonal_positions]
        bp.pcg.mg.d.copy_(torch.where(bp.wet, diagonal, torch.ones_like(diagonal)).rsqrt())
        try:
            delta, k, true = bp.pcg.solve(r)
        except RuntimeError as exc:
            if str(exc) not in ['200CG budget', 'True linear residual failed', 'Nonpositive CG curvature']:
                raise
            refresh(it)
            delta, k, true = bp.pcg.solve(r)
        rn = float(r.norm())
        descent = float(r @ delta)
        if descent >= 0:
            raise RuntimeError('Non-descent Newton direction')
        alpha = 1.
        energy = float(e)
        for _ in range(35):
            trial = u + alpha * delta
            rt, et = bp.evaluate(trial)
            resolved = abs(1e-4 * alpha * descent) > 64 * torch.finfo(T).eps * max(abs(energy), 1.)
            decreased = float(rt.norm()) < (1 - 1e-4 * alpha) * rn
            if (resolved and float(et) <= energy + 1e-4 * alpha * descent) or (not resolved and decreased) or float(rt.norm()) < .5 * rn:
                u = trial
                break
            alpha *= .5
        else:
            raise RuntimeError('Line search failed')
        history[-1].update(cg_iterations=k, linear_residual=true, step_length=alpha)
        residual = float(rt.norm() / bp.force_norm)
        if residual <= 1e-6:
            return u, dict(iterations=it + 1, relative_residual=residual, history=history)
        if checkpoint and (it + 1) % 5 == 0:
            checkpoint(it + 1, u, history[-1])
