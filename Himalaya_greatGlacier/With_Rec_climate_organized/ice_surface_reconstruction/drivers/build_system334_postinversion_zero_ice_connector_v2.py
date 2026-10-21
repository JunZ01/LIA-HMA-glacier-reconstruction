"""Zero-mass suffix compatibility wrapper for the system-334 v1 builder.

V1 correctly stopped when a reverse branch had no positive ice in the suffix
selected for removal.  Such a suffix needs no area/volume condensation: it is
already zero mass and can be removed exactly.  V2 handles only that case and
delegates every positive-mass case unchanged to v1.
"""

from __future__ import annotations

import copy
import json

import numpy as np
from shapely.geometry import LineString

from . import build_system334_postinversion_zero_ice_connector_v1 as v1


_original = v1.rebuild_flowline


def _rebuild_with_zero_mass_suffix(source, receiver_surface, receiver_point):
    from oggm.core.flowline import MixedBedFlowline

    if receiver_surface is None or receiver_surface <= float(source.surface_h[-1]) + v1.TOLERANCE_M:
        return _original(source, receiver_surface, receiver_point)
    surface = np.asarray(source.surface_h, dtype=np.float64)
    feasible = np.where(surface >= receiver_surface - v1.TOLERANCE_M)[0]
    if feasible.size == 0:
        return _original(source, receiver_surface, receiver_point)
    keep_last = int(feasible[-1])
    active_suffix = (
        np.asarray(source.thick, dtype=np.float64)[keep_last:] > v1.TOLERANCE_M
    )
    if np.any(active_suffix):
        return _original(source, receiver_surface, receiver_point)

    count = keep_last + 1
    base = MixedBedFlowline(
        line=LineString(list(source.line.coords)[:count]),
        dx=float(source.dx),
        map_dx=float(source.map_dx),
        surface_h=np.asarray(source.surface_h)[:count].copy(),
        bed_h=np.asarray(source.bed_h)[:count].copy(),
        section=np.asarray(source.section)[:count].copy(),
        bed_shape=np.asarray(source.bed_shape)[:count].copy(),
        is_trapezoid=np.asarray(source.is_trapezoid)[:count].copy(),
        lambdas=np.asarray(source._lambdas)[:count].copy(),
        widths_m=np.asarray(source.widths_m)[:count].copy(),
        rgi_id=source.rgi_id,
        water_level=getattr(source, "water_level", 0.0),
    )
    candidate, diagnostics = _original(base, receiver_surface, receiver_point)
    diagnostics.update(
        {
            "source_reverse": True,
            "source_node_count": int(source.nx),
            "removed_source_node_count": int(source.nx - count),
            "removed_active_node_count": 0,
            "extra_active_cut_m": 0.0,
            "zero_mass_suffix_only": True,
        }
    )
    return candidate, diagnostics


def main() -> None:
    v1.rebuild_flowline = _rebuild_with_zero_mass_suffix
    v1.main()
    summary_path = v1.OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "lia-191-system334-postinversion-zero-ice-connector-v2",
            "mechanism": (
                "postinversion-active-terminal-condensation-plus-zero-ice-"
                "connectors-v2-zero-mass-suffix-compatible"
            ),
            "v1_technical_stop": "reverse branch 12 suffix contained zero mass",
            "scientific_contract_changed": False,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
