"""Boundary-corrected v2 of the system-2005 branch-3 coarsening.

V1 paired the final active 50 m cell with the terminal dry cell and therefore
created ice area.  V2 keeps the dry terminal control explicitly dry.  The last
three active cells are represented by one 100 m area-equivalent control; all
earlier cells remain head-aligned pairs.  This preserves total model length,
active area, volume, and area-weighted surface/bed integrals exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from oggm.core.flowline import MixedBedFlowline
from shapely.geometry import LineString

from . import build_system2005_branch3_coarsen2x_static_v1 as v1


OUTPUT = (
    Path(__file__).resolve().parents[4]
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_system2005_branch3_coarsen2x_static_v2"
)


def build_branch(source):
    if int(source.nx) != 18 or abs(float(source.dx_meter) - 50.0) > 1e-12:
        raise RuntimeError(f"unexpected-branch-grid:{source.nx}:{source.dx_meter}")
    active = np.flatnonzero(np.asarray(source.thick) > v1.ACTIVE_TOLERANCE_M)
    if active.tolist() != list(range(17)):
        raise RuntimeError(f"unexpected-active-support:{active.tolist()}")
    if not np.all(np.asarray(source._lambdas) == 0.0):
        raise RuntimeError("branch3-is-not-rectangular")

    width = np.asarray(source.widths_m, dtype=float)
    section = np.asarray(source.section, dtype=float)
    surface = np.asarray(source.surface_h, dtype=float)
    bed = np.asarray(source.bed_h, dtype=float)
    coords = np.asarray(source.line.coords, dtype=float)
    a = np.arange(0, 14, 2)
    b = a + 1
    pair_width_sum = width[a] + width[b]
    coarse_width = list(0.5 * pair_width_sum)
    coarse_section = list(0.5 * (section[a] + section[b]))
    coarse_surface = list((width[a] * surface[a] + width[b] * surface[b]) / pair_width_sum)
    coarse_bed = list((width[a] * bed[a] + width[b] * bed[b]) / pair_width_sum)
    coarse_coords = list(0.5 * (coords[a] + coords[b]))

    tail_active = np.asarray([14, 15, 16], dtype=int)
    tail_width_sum = float(np.sum(width[tail_active]))
    coarse_width.append(0.5 * tail_width_sum)
    coarse_section.append(0.5 * float(np.sum(section[tail_active])))
    coarse_surface.append(float(np.sum(width[tail_active] * surface[tail_active]) / tail_width_sum))
    coarse_bed.append(float(np.sum(width[tail_active] * bed[tail_active]) / tail_width_sum))
    coarse_coords.append(np.sum(width[tail_active, None] * coords[tail_active], axis=0) / tail_width_sum)

    # Preserve the original dry terminal cell as an area-equivalent 100 m cell.
    coarse_width.append(0.5 * float(width[17]))
    coarse_section.append(0.0)
    coarse_surface.append(float(bed[17]))
    coarse_bed.append(float(bed[17]))
    coarse_coords.append(coords[17])

    coarse_width = np.asarray(coarse_width, dtype=float)
    coarse_section = np.asarray(coarse_section, dtype=float)
    coarse_surface = np.asarray(coarse_surface, dtype=float)
    coarse_bed = np.asarray(coarse_bed, dtype=float)
    coarse_coords = np.asarray(coarse_coords, dtype=float)
    thick = coarse_surface - coarse_bed
    if np.max(np.abs(coarse_section[:-1] - coarse_width[:-1] * thick[:-1])) > 1e-8:
        raise RuntimeError("rectangular-control-volume-closure-failed")
    if coarse_section[-1] != 0.0 or thick[-1] != 0.0:
        raise RuntimeError("terminal-dry-control-not-preserved")
    return MixedBedFlowline(
        line=LineString(coarse_coords),
        dx=float(source.dx) * 2.0,
        map_dx=float(source.map_dx),
        surface_h=coarse_surface,
        bed_h=coarse_bed,
        section=coarse_section,
        bed_shape=np.ones(len(coarse_section), dtype=float),
        is_trapezoid=np.ones(len(coarse_section), dtype=bool),
        lambdas=np.zeros(len(coarse_section), dtype=float),
        widths_m=coarse_width,
        rgi_id=source.rgi_id,
        water_level=source.water_level,
    )


def main() -> None:
    v1.OUTPUT = OUTPUT
    v1.build_branch = build_branch
    v1.main()
    summary_path = OUTPUT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["schema"] = "lia-191-system2005-branch3-coarsen2x-static-v2"
    payload["mechanism"] = "branch3-2x-coarsening-with-dry-terminal-and-three-cell-active-tail-control"
    payload["supersedes_static_v1_boundary_failure"] = True
    payload["boundary_rule"] = {
        "head_pairs": "nodes 0-13 paired consecutively",
        "last_active_control": "nodes 14-16 represented by one area-equivalent 100 m cell",
        "dry_terminal_control": "node 17 retained dry with half width on the 100 m grid",
        "runtime_support_floor": False,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
