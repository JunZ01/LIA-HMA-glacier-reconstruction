"""SciPy-signature compatibility wrapper for the versioned 2321 real step.

V1 stopped before solving because this environment's ``cg`` accepts ``tol``
rather than ``rtol``.  This wrapper changes only that keyword and directs the
otherwise identical run to a new output directory.
"""

from __future__ import annotations

from pathlib import Path

from . import run_paired_cell_hybrid_graph_real_step_2321_v1 as implementation


_cg = implementation.cg


def compatible_cg(A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, M=None, callback=None):
    return _cg(A, b, x0=x0, tol=rtol, atol=atol, maxiter=maxiter, M=M, callback=callback)


implementation.cg = compatible_cg
implementation.OUTPUT = (
    Path(implementation.PROJECT)
    / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_real_step_2321_v2"
)


if __name__ == "__main__":
    implementation.main()
