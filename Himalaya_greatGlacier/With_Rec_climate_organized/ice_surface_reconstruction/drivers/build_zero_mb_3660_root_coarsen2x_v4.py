"""Independent v4 with deterministic conservative junction remapping.

V3 showed that OGGM's generic nearest-tail reprojection can move a previously
registered root receiver by 150 m.  For a 2x nested grid, v4 instead maps every
old root index ``i`` to ``ceil(i / 2)``.  Even-index receivers are exact and
odd-index receivers move one half coarse cell downstream (50 m), preserving
flow direction without a per-branch choice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from oggm.core.centerlines import Centerline
from shapely.geometry import Point

from . import build_zero_mb_3660_root_coarsen2x_v2 as implementation


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v4"
)


def main() -> None:
    original_set_flows_to = Centerline.set_flows_to

    def nested_grid_set_flows_to(self, other, *args, **kwargs):
        old_index = self.__dict__.get("_lazy_flows_to_indice")
        if old_index is None or int(other.nx) != 342:
            self.__dict__.pop("_lazy_flows_to_indice", None)
            return original_set_flows_to(self, other, *args, **kwargs)
        new_index = int(math.ceil(int(old_index) / 2.0))
        self.flows_to = other
        self.flows_to_point = Point(other.line.coords[new_index])
        other.inflow_points.append(self.flows_to_point)
        other.inflows.append(self)
        self.__dict__.pop("_lazy_flows_to_indice", None)

    Centerline.set_flows_to = nested_grid_set_flows_to
    implementation.OUTPUT = OUTPUT
    implementation.main()
    summary_path = OUTPUT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    payload["experiment"] = "zero-mb-3660-conservative-root-coarsen2x-static-preflight-v4"
    payload["receiver_mapping_rule"] = "old root index i maps to ceil(i/2); maximum downstream shift 50 m"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
