"""Windows-compatible wrapper for the Section-181 geometry preflight.

V1 reached GlacierDirectory creation but OGGM's tar-shapefile write hit the
Windows long-path limit.  V2 changes only shapefile storage to unpacked files;
the scientific geometry contract is unchanged.
"""

from __future__ import annotations

from oggm import cfg

from . import synthetic_lia_system_geometry_preflight_v1 as v1


_configure_v1 = v1._configure_oggm


def _configure_oggm(output_root, flowline_dx):
    version = _configure_v1(output_root, flowline_dx)
    cfg.PARAMS["use_tar_shapefiles"] = False
    return version


v1._configure_oggm = _configure_oggm


if __name__ == "__main__":
    v1.main()
