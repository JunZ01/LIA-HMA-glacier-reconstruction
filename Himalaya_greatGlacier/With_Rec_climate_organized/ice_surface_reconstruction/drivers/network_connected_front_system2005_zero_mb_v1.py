"""0.1y zero-MB wrapper for the connected-front capacity step."""

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_capacity_connected_front_step_v1 import connected_front_step


implementation.one_step = connected_front_step
from . import network_coupled_semiimplicit_system2005_zero_mb_v3 as adaptive


if __name__ == "__main__":
    adaptive.main()
