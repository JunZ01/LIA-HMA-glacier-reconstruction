"""Bounded one-step reproduction of the 1686 active-set failure."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

from . import network_monotone_positive_conservative_system2005_failure_state_step_v1 as positive
from . import run_system2005_disconnection_diagnostic_length30_historical_v3_convergence_adaptive_checkpoint as launcher
from . import run_system2005_topology_adaptive_historical_v6_single_step_diagnostic as single_step


def diagnostic_active_set(
    matrix,
    rhs,
    constraint_matrix,
    lower,
    volume_row,
    volume_target,
    tolerance=1e-10,
    maximum_iterations=200,
):
    unconstrained = np.asarray(spsolve(matrix, rhs), dtype=float)
    residual = np.asarray(constraint_matrix @ unconstrained, dtype=float) - lower
    active = sorted(np.flatnonzero(residual < -tolerance).astype(int).tolist())
    seen = {}
    add_count = 0
    drop_count = 0
    last = None
    for iteration in range(maximum_iterations):
        key = tuple(active)
        if key in seen:
            raise RuntimeError(
                "affine-active-set-cycle:"
                f"first_iteration={seen[key] + 1}:repeat_iteration={iteration + 1}:"
                f"period={iteration - seen[key]}:active_count={len(active)}:"
                f"adds={add_count}:drops={drop_count}"
            )
        seen[key] = iteration
        active_matrix = (
            constraint_matrix[active, :].tocsr()
            if active
            else csr_matrix((0, matrix.shape[0]))
        )
        zero_aa = csr_matrix((len(active), len(active)))
        zero_av = csr_matrix((len(active), 1))
        zero_va = csr_matrix((1, len(active)))
        zero_vv = csr_matrix((1, 1))
        kkt = bmat(
            [
                [matrix, -active_matrix.T, volume_row.T],
                [active_matrix, zero_aa, zero_av],
                [volume_row, zero_va, zero_vv],
            ],
            format="csr",
        )
        solution = np.asarray(
            spsolve(kkt, np.r_[rhs, lower[active], volume_target]), dtype=float
        )
        candidate = solution[: matrix.shape[0]]
        multipliers = solution[matrix.shape[0] : matrix.shape[0] + len(active)]
        residual = np.asarray(constraint_matrix @ candidate, dtype=float) - lower
        inactive = np.ones(constraint_matrix.shape[0], dtype=bool)
        inactive[active] = False
        violated = np.flatnonzero(inactive & (residual < -tolerance))
        negative_positions = np.flatnonzero(multipliers < -tolerance)
        last = (
            len(active),
            float(np.min(residual)),
            None if not len(multipliers) else float(np.min(multipliers)),
            len(violated),
            len(negative_positions),
        )
        if len(negative_positions):
            position = int(
                negative_positions[np.argmin(multipliers[negative_positions])]
            )
            active.pop(position)
            drop_count += 1
            continue
        if len(violated):
            constraint = int(violated[np.argmin(residual[violated])])
            active.append(constraint)
            active.sort()
            add_count += 1
            continue
        return candidate, unconstrained, active, multipliers, float(solution[-1]), []
    raise RuntimeError(
        "affine-active-set-no-cycle-within-limit:"
        f"iterations={maximum_iterations}:unique_sets={len(seen)}:"
        f"adds={add_count}:drops={drop_count}:last={last}"
    )


def main():
    original_solver = positive.solve_affine_active_set
    original_adaptive = launcher.adaptive
    positive.solve_affine_active_set = diagnostic_active_set
    launcher.adaptive = single_step
    try:
        launcher.main()
    finally:
        launcher.adaptive = original_adaptive
        positive.solve_affine_active_set = original_solver


if __name__ == "__main__":
    main()
