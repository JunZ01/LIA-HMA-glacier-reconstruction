"""Package-safe launcher for the frozen system-3660 steady v5 adapter."""
from __future__ import annotations

import sys

from . import synthetic_lia_conservative_steady_50y_registered_floor_v1 as steady


sys.modules.setdefault("synthetic_lia_conservative_steady_50y_registered_floor_v1", steady)

from . import run_steady_3660_root_coarsen2x_v5 as target  # noqa: E402


if __name__ == "__main__":
    target.main()
