"""Ten-year preflight with isolated-clone climate-cache write compatibility."""

from __future__ import annotations

import stat
from pathlib import Path

from . import run_system2005_policy5pct_historical_10y_preflight_v1 as preflight_v1
from . import run_system2005_topology_adaptive_historical_v1 as historical


def main() -> None:
    original = historical._ramp_climate

    def isolated_clone_writable_ramp(gdir, config, pixel):
        for path in Path(gdir.dir).glob("gcm_data*.nc"):
            path.chmod(path.stat().st_mode | stat.S_IWRITE | stat.S_IREAD)
        return original(gdir, config, pixel)

    historical._ramp_climate = isolated_clone_writable_ramp
    try:
        preflight_v1.main()
    finally:
        historical._ramp_climate = original


if __name__ == "__main__":
    main()
