"""Numerically scaled QP launcher for the system-5770 vertical-datum audit."""

from __future__ import annotations

import json as std_json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize as scipy_minimize

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    audit_system5770_branch_vertical_datum_v1 as implementation,
)


PROJECT = Path(__file__).resolve().parents[4]
implementation.OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_branch_rigid_vertical_datum_5770_v3"
)
_ORIGINAL_DUMPS = std_json.dumps
_ORIGINAL_MINIMIZE = implementation.minimize


def _numpy_dumps(value, *args, **kwargs):
    kwargs.setdefault("default", lambda item: item.item() if isinstance(item, np.generic) else str(item))
    return _ORIGINAL_DUMPS(value, *args, **kwargs)


def _scaled_minimize(_fun, x0, _args=(), method=None, bounds=None, constraints=(), **kwargs):
    weights = np.asarray(constraints[0].A, dtype=float).reshape(-1)
    normalized = weights / weights.sum()
    scaled_constraints = [LinearConstraint(normalized, 0.0, 0.0), *constraints[1:]]
    options = dict(kwargs.get("options") or {})
    options.update({"ftol": 1e-12, "maxiter": 5000, "disp": False})
    return scipy_minimize(
        lambda c: float(np.sum(normalized * c * c)),
        np.asarray(x0, dtype=float),
        jac=lambda c: 2.0 * normalized * c,
        method="SLSQP",
        bounds=bounds,
        constraints=scaled_constraints,
        options=options,
    )


def main() -> None:
    implementation.json.dumps = _numpy_dumps
    implementation.minimize = _scaled_minimize
    try:
        implementation.main()
    finally:
        implementation.minimize = _ORIGINAL_MINIMIZE
        implementation.json.dumps = _ORIGINAL_DUMPS


if __name__ == "__main__":
    main()
