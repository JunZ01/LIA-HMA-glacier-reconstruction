from __future__ import annotations

import argparse
import json

import pytest

from ..drivers.lia_lhs_single_chain_spatial_downstream import (
    _load_spatial_observation_domain,
)


def test_spatial_domain_keeps_initial_and_observation_counts_distinct(tmp_path):
    system_dir = tmp_path / "2955"
    system_dir.mkdir()
    (system_dir / "elevation_band_audit.json").write_text(
        json.dumps(
            {
                "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
                "initial_lia_ice_node_count": 217,
                "ice_node_count": 217,
                "observation_domain_node_count": 351,
                "total_node_count_with_buffer": 451,
                "observation_domain_definition": "spatial DEM route",
            }
        ),
        encoding="utf-8",
    )
    domain = _load_spatial_observation_domain(
        argparse.Namespace(elevation_band_root=tmp_path, system_id=2955)
    )
    assert domain["initial_lia_node_count"] == 217
    assert domain["node_count"] == 351


def test_spatial_domain_rejects_non_v2_geometry(tmp_path):
    system_dir = tmp_path / "2955"
    system_dir.mkdir()
    (system_dir / "elevation_band_audit.json").write_text(
        json.dumps(
            {
                "geometry_variant": "core-lia-plus-spatial-dem-downstream-v1",
                "initial_lia_ice_node_count": 217,
                "observation_domain_node_count": 351,
                "total_node_count_with_buffer": 451,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v2-geometry-required"):
        _load_spatial_observation_domain(
            argparse.Namespace(elevation_band_root=tmp_path, system_id=2955)
        )
