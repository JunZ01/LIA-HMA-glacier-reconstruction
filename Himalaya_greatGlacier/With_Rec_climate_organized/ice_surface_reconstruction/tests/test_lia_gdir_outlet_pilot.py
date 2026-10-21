from types import SimpleNamespace

import numpy as np
from shapely.geometry import LineString, box

from ..drivers.lia_gdir_outlet_pilot import (
    _GeometryFlowline,
    _boundary_crossings,
    _centerline_network_payload,
    audit_outlet_geometry,
)


def test_outlet_audit_accepts_one_node_terminus_and_zero_buffer() -> None:
    grid = SimpleNamespace(x0=0.0, y0=0.0, dx=1.0, dy=-1.0, nx=16, ny=11)
    polygon = box(0.0, -5.0, 10.0, 5.0)
    interior = SimpleNamespace(
        line=LineString([(value, 0.0) for value in range(1, 10)]),
        flows_to=None,
    )
    model = _GeometryFlowline(
        line=LineString([(value, 0.0) for value in range(1, 16)]),
        section=np.asarray([1.0] * 9 + [0.0] * 6),
        dx_meter=1.0,
    )
    network_payload = {
        "main_branch_index": 0,
        "main_extension_length_m": 1.0,
        "main_extension_join_gap_m": 0.0,
        "branches": [
            {
                "branch_index": 0,
                "coords_projected_m": np.asarray(
                    [(value, 0.0) for value in range(1, 11)], dtype=float
                ),
            }
        ],
    }

    audit, spatial = audit_outlet_geometry(
        centerlines=[interior],
        model_flowlines=[model],
        network_payload=network_payload,
        component_geometry=polygon,
        grid=grid,
    )

    assert audit["all_gates_pass"]
    assert audit["outlet_candidate_count"] == 1
    assert audit["last_positive_to_outlet_distance_m"] == 1.0
    assert audit["outlet_to_model_endpoint_buffer_m"] == 5.0
    assert spatial["markers"]["outlet"].x == 10.0


def test_boundary_crossings_ignore_polygon_holes() -> None:
    from shapely.geometry import Polygon

    polygon = Polygon(
        [(0, -5), (10, -5), (10, 5), (0, 5)],
        holes=[[(4, -1), (6, -1), (6, 1), (4, 1)]],
    )
    line = LineString([(1, 0), (15, 0)])

    crossings = _boundary_crossings(line, polygon)

    assert len(crossings) == 1
    assert crossings[0].x == 10.0


def test_network_payload_extends_main_line_to_exact_outer_outlet() -> None:
    grid = SimpleNamespace(x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    polygon = box(0.0, -5.0, 10.0, 5.0)
    main = SimpleNamespace(
        line=LineString([(value, 0.0) for value in range(1, 9)]),
        surface_h=np.arange(108.0, 100.0, -1.0),
        flows_to=None,
        dx=1.0,
        dx_meter=1.0,
    )
    downstream = {
        "full_line": LineString([(value, 0.0) for value in range(1, 16)]),
        "surface_h": np.arange(100.0, 93.0, -1.0),
    }

    payload = _centerline_network_payload(
        [main], downstream, polygon, grid, "EPSG:32645"
    )
    branch = payload["branches"][0]

    assert np.allclose(branch["coords_projected_m"][-1], [10.0, 0.0])
    assert branch["coords_projected_m"].shape == (10, 2)
    assert branch["reference_surface_m"].shape == (10,)
    assert payload["main_extension_length_m"] == 2.0
    assert payload["main_extension_join_gap_m"] == 0.0


def test_network_payload_truncates_main_line_that_already_crossed_outlet() -> None:
    grid = SimpleNamespace(x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    polygon = box(0.0, -5.0, 10.0, 5.0)
    main = SimpleNamespace(
        line=LineString([(value, 0.0) for value in range(1, 13)]),
        surface_h=np.arange(112.0, 100.0, -1.0),
        flows_to=None,
        dx=1.0,
        dx_meter=1.0,
    )
    downstream = {
        "full_line": LineString([(value, 0.0) for value in range(1, 16)]),
        "surface_h": np.arange(100.0, 97.0, -1.0),
    }

    payload = _centerline_network_payload(
        [main], downstream, polygon, grid, "EPSG:32645"
    )
    branch = payload["branches"][0]

    assert np.allclose(branch["coords_projected_m"][-1], [10.0, 0.0])
    assert branch["coords_projected_m"].shape == (10, 2)
    assert branch["reference_surface_m"].shape == (10,)
    assert payload["main_extension_length_m"] == -2.0
