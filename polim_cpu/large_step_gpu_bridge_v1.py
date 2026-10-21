"""Adapter from the phase-1 outer correction to the retained exact GPU BP.

Construction is intentionally explicit and lazy: importing this module does
not initialize CUDA.  A future registered worker can create the bridge inside
the existing isolated GPU process and pass it directly as ``bp_solve`` to
``solve_coupled_large_step``.
"""
from __future__ import annotations

from pathlib import Path
import gc

import numpy as np


class ProjectedAdaptiveBPSolver:
    """Reuse the v34/v35 projected AMG and exact terminal BP residual."""

    def __init__(self, arrays, cache_root: str | Path):
        # Keep the CUDA/OpenMP isolation rules of the existing workers: these
        # imports occur only after their launcher has prepared the environment.
        from .gpu_energy_worker_v1 import tensor, torch
        from .gpu_allocator_safe_v1 import BP, PCG
        from .gpu_allocator_recovery_worker_v1 import Hierarchy

        self._tensor = tensor
        self._torch = torch
        self._PCG = PCG
        self._Hierarchy = Hierarchy
        self._ProjectedMultigrid = __import__(
            "polim_cpu.projected_amg_v1", fromlist=["ProjectedMultigrid"]
        ).ProjectedMultigrid
        self._solve = __import__(
            "polim_cpu.adaptive_bp_solve_v1", fromlist=["solve"]
        ).solve
        self.bp = BP(arrays)
        positions = []
        for row in range(self.bp.ndof):
            start, end = arrays["csr_indptr"][row : row + 2]
            match = np.flatnonzero(arrays["csr_indices"][start:end] == row)
            if len(match) != 1:
                raise RuntimeError("BP CSR diagonal is not unique")
            positions.append(int(start + match[0]))
        self.bp.diagonal_positions = tensor(positions, True)
        root = Path(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        self.hierarchy = Hierarchy(root, arrays)
        self.mg = None
        self.build_index = len(list(root.glob("hierarchy-*")))
        self.bp_calls = 0
        self.rebuilds = 0

    def _release_pcg(self) -> None:
        if hasattr(self.bp, "pcg"):
            self._torch.cuda.synchronize()
            graph = self.bp.pcg.graph
            if graph is not None:
                graph.reset()
            del self.bp.pcg
        self.mg = None
        gc.collect()
        self._torch.cuda.empty_cache()

    def _refresh(self, _iteration: int) -> None:
        self._release_pcg()
        self.build_index += 1
        self.mg = self.hierarchy.build(self.bp, self.build_index)
        projected = self._ProjectedMultigrid(self.mg, self.bp)
        self.bp.pcg = self._PCG(self.bp, projected, True)
        self.rebuilds += 1

    def __call__(self, thickness, velocity_guess):
        h = self._tensor(np.asarray(thickness, dtype=np.float64))
        u = self._tensor(np.asarray(velocity_guess, dtype=np.float64))
        self.bp.prepare(h)
        u = u.clone()
        u[~self.bp.wet] = 0.0
        if self.mg is None:
            self.bp.evaluate(u, True)
            self._refresh(0)
        u, info = self._solve(self.bp, u, self._refresh)
        self.bp_calls += 1
        info = dict(info)
        info["amg_rebuilds_total"] = self.rebuilds
        return u.detach().cpu().numpy(), info

    def close(self) -> None:
        self._release_pcg()
        self.hierarchy.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
