"""Invocation shim for the Cycle-3 v1 driver.

The transverse-width helper uses the established ``(coordinates, polygon)``
signature.  This shim keeps the versioned scientific driver immutable while
adapting its initial three-positional-argument call before execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import v3_true_mainline_area_equivalent_rectangular_preflight_v1 as v1


_polygon_transverse_widths_m = v1.polygon_transverse_widths_m


def _adapter(component_geometry, x_m, y_m):
    return _polygon_transverse_widths_m(
        np.column_stack([
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
        ]),
        component_geometry,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    v1.polygon_transverse_widths_m = _adapter
    print(json.dumps(
        v1.run(args.config), indent=2, ensure_ascii=True, allow_nan=False
    ))


if __name__ == "__main__":
    main()
