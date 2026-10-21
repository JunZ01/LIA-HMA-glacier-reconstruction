"""Serialization-only compatibility launcher for the recovered 5-year gate.

The v1 integration completed and saved the five-year state, then failed while
formatting the saved-state path because the restored base module exposes
``PROJECT`` rather than ``ROOT``.  V2 aliases that path name before invoking
the unchanged v1 integration and gate logic.  No numerical or scientific
parameter is changed.
"""

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_5y_v1 as implementation


def main() -> None:
    base.ROOT = base.PROJECT
    implementation.main()


if __name__ == "__main__":
    main()

