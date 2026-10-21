"""Run one chain with the spatial DEM downstream observation domain."""

from __future__ import annotations

import json
from pathlib import Path

from . import lia_lhs_single_chain as _base
from . import lia_lhs_single_chain_legacy_20260813 as _legacy


def _load_spatial_observation_domain(args) -> dict:
    audit_path = (
        Path(args.elevation_band_root)
        / str(args.system_id)
        / "elevation_band_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("geometry_variant") != "core-lia-plus-spatial-dem-downstream-v2":
        raise ValueError("spatial-downstream-v2-geometry-required")
    node_count = int(audit["observation_domain_node_count"])
    initial_lia_nodes = int(audit["initial_lia_ice_node_count"])
    total_nodes = int(audit["total_node_count_with_buffer"])
    if not 0 < initial_lia_nodes < node_count < total_nodes:
        raise ValueError("invalid-spatial-observation-domain-node-counts")
    return {
        "source": str(audit_path.resolve()),
        "node_count": node_count,
        "initial_lia_node_count": initial_lia_nodes,
        "definition": audit["observation_domain_definition"],
    }


def run(args):
    previous_loader = _base._load_observation_domain
    _base._load_observation_domain = _load_spatial_observation_domain
    try:
        return _base.run(args)
    finally:
        _base._load_observation_domain = previous_loader


def main() -> int:
    previous_run = _legacy.run
    _legacy.run = run
    try:
        return _legacy.main()
    finally:
        _legacy.run = previous_run


if __name__ == "__main__":
    raise SystemExit(main())
