"""Registered v3/v4 surface-coverage wrapper for anchor-guided pilot v1."""

from . import synthetic_lia_system_geometry_preflight_v1 as geometry_v1
from . import synthetic_lia_system_geometry_preflight_v3 as _geometry_v3  # noqa: F401
from . import synthetic_lia_system_geometry_preflight_v4 as _geometry_v4  # noqa: F401
from . import synthetic_lia_anchor_guided_centerline_pilot_v1 as pilot


pilot.run_native_geometry = geometry_v1.run


if __name__ == "__main__":
    pilot.main()
