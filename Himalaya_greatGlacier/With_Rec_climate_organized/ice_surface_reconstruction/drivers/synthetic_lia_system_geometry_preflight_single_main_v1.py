"""OGGM-native single-main-flowline static geometry wrapper.

This is a topology representation pilot.  It keeps the synthetic LIA outline,
paired surface, regional DEM, map/flowline spacing, and all hard gates unchanged,
and changes only OGGM's native ``use_multiple_flowlines`` switch to ``False``.
"""

from __future__ import annotations

from oggm import cfg

from . import synthetic_lia_system_geometry_preflight_v1 as v1
from . import synthetic_lia_system_geometry_preflight_v4 as v4


_configure_v4 = v4._configure_oggm


def _configure_oggm(output_root, flowline_dx):
    version = _configure_v4(output_root, flowline_dx)
    cfg.PARAMS["use_multiple_flowlines"] = False
    return version


v1._configure_oggm = _configure_oggm


if __name__ == "__main__":
    v1.main()
