"""Versioned connected-front runner using exact-zero capacity arithmetic."""

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_capacity_connected_front_step_v2 import connected_front_step_v2

implementation.one_step = connected_front_step_v2
from . import network_coupled_semiimplicit_system2005_zero_mb_v3 as adaptive

if __name__ == "__main__":
    adaptive.main()

