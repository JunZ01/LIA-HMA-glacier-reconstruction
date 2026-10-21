"""SciPy-compatibility wrapper for the unchanged v1 semi-implicit fixtures."""

from __future__ import annotations

import json

from . import planform_2d_frozen_diffusivity_semiimplicit_core_v1 as core


_scipy_cg = core.cg


def _cg_compat(operator, rhs, rtol=1e-5, atol=0.0, maxiter=None, M=None):
    return _scipy_cg(operator, rhs, tol=rtol, atol=atol, maxiter=maxiter, M=M)


core.cg = _cg_compat

from .planform_2d_frozen_diffusivity_fixtures_v1 import (  # noqa: E402
    PROJECT,
    manufactured_conservation_fixture,
    oggm_infinitesimal_consistency_fixture,
    zero_diffusivity_fixture,
)


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_fixtures_v2"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "zero_diffusivity_identity": zero_diffusivity_fixture(),
        "manufactured_mass_conservation": manufactured_conservation_fixture(),
        "oggm_infinitesimal_consistency": oggm_infinitesimal_consistency_fixture(),
    }
    passed = all(item["passed"] for item in fixtures.values())
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-fixtures-v2",
        "status": "fixture-pass-real-system-authorized" if passed else "fixture-fail-mechanism-closed",
        "supersedes": "v1-technical-failure-scipy-cg-parameter-name-only",
        "scientific_algorithm_changed": False,
        "fixtures": fixtures,
        "real_system_run": False,
        "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
