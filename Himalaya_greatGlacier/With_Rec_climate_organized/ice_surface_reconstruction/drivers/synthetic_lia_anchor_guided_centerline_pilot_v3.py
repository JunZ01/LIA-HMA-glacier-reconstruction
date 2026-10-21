"""Registered-main-first centerline pilot built on the v2 surface contract."""

from __future__ import annotations

import copy
from itertools import groupby

import numpy as np
from shapely.geometry import LineString

from . import synthetic_lia_system_geometry_preflight_v1 as geometry_v1
from . import synthetic_lia_system_geometry_preflight_v3 as _geometry_v3  # noqa: F401
from . import synthetic_lia_system_geometry_preflight_v4 as _geometry_v4  # noqa: F401
from . import synthetic_lia_anchor_guided_centerline_pilot_v1 as pilot


pilot.run_native_geometry = geometry_v1.run


def _registered_main_first_filter(lines, heads, k, r):
    """Apply OGGM's native filter, selecting the appended anchor route first."""
    output_lines = []
    remaining = copy.copy(lines)
    last_line = None
    target_head = heads[-1]
    first = True
    while remaining:
        if output_lines:
            remove_area = last_line.buffer(k)
            kept = []
            for line in remaining:
                difference = line.difference(remove_area)
                if difference.geom_type == "MultiLineString":
                    parts = list(difference.geoms)
                    difference = None
                    for part in parts:
                        if any(part.intersects(head) for head in heads):
                            difference = part
                            break
                if difference is not None and difference.length > r:
                    difference = LineString(line.coords[0:2] + difference.coords[2:])
                    kept.append(difference)
            remaining = kept
        if not remaining:
            break
        if first:
            distances = np.asarray([line.distance(target_head) for line in remaining])
            selected = remaining[int(np.argmin(distances))]
            first = False
        else:
            selected = remaining[int(np.argmax([line.length for line in remaining]))]
        remaining.remove(selected)
        if output_lines:
            output_lines.append(LineString(np.asarray(selected.xy)[:, 0:-1].T))
        else:
            output_lines.append(selected)
        last_line = selected
    output_heads = []
    for line in output_lines:
        for head in heads:
            if line.intersects(head):
                output_heads.append(head)
                break
    return output_lines, output_heads


def main() -> None:
    import oggm.core.centerlines as centerlines

    original = centerlines._filter_lines
    centerlines._filter_lines = _registered_main_first_filter
    try:
        pilot.main()
    finally:
        centerlines._filter_lines = original


if __name__ == "__main__":
    main()
