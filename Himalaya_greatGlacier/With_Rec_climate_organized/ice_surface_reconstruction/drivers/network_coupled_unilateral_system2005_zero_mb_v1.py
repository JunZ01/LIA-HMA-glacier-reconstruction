"""Rollback-safe 0.1y launcher using the unilateral active-set step."""

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_coupled_unilateral_system2005_step_v1 import unilateral_one_step


implementation.one_step = unilateral_one_step

from . import network_coupled_semiimplicit_system2005_zero_mb_v3 as adaptive


if __name__ == "__main__":
    adaptive.main()
