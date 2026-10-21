"""Reduced technical retry of the system-84 localized transport LP.

Scientific constraints and the 2.5 km locality radius are unchanged.  Only
the deterministic audit discretization is reduced from five to three source
surface groups per node and from 16 to eight nearest candidate destinations;
HiGHS dual simplex is selected explicitly.  This is the sole retry after the
v1 time-limit result.
"""

from pathlib import Path

from . import audit_system84_local_group_transport_lp_v1 as implementation


implementation.GROUPS_PER_NODE = 3
implementation.K_NEAREST = 8
implementation.OUTPUT = implementation.PROJECT if hasattr(implementation, "PROJECT") else Path(__file__).resolve().parents[4]
implementation.OUT = Path(__file__).resolve().parents[4] / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_group_transport_lp_v2"
_original_linprog = implementation.linprog


def _dual_simplex(*args, **kwargs):
    kwargs["method"] = "highs-ds"
    kwargs["options"] = {"time_limit": 300.0, "presolve": True}
    return _original_linprog(*args, **kwargs)


implementation.linprog = _dual_simplex


if __name__ == "__main__":
    implementation.main()
