"""Versioned compatibility wrapper enabling unpacked synthetic shapefiles."""

from __future__ import annotations

from oggm import cfg

from . import scan_synthetic_lia_native_junction_pix_v1 as v1


_configure_v1 = v1._configure


def _configure(output_root):
    _configure_v1(output_root)
    cfg.PARAMS["use_tar_shapefiles"] = False


v1._configure = _configure


if __name__ == "__main__":
    v1.main()
