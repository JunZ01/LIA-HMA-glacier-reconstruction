"""Versioned Section-182 geometry run with OGGM-native 1.5 degree filtering."""

from __future__ import annotations

from oggm import cfg

from . import synthetic_lia_system_geometry_preflight_v1 as v1
from . import synthetic_lia_system_geometry_preflight_v3 as v3


_configure_v3 = v3._configure_oggm


def _configure_oggm(output_root, flowline_dx):
    version = _configure_v3(output_root, flowline_dx)
    cfg.PARAMS["min_slope_flowline_filter"] = 1.5
    return version


v1._configure_oggm = _configure_oggm


if __name__ == "__main__":
    v1.main()
