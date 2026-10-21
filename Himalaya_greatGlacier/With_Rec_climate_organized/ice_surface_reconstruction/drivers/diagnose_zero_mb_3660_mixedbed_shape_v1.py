"""Static analytic feasibility of a local MixedBed re-representation for 3660."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = (
    PROJECT
    / "w382inv3660v1/synthetic_gdir/RGI60-14/RGI60-14.93"
    / "RGI60-14.93660/model_flowlines.pkl"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v1"
)
ROOT_ID = 32
NODES = np.arange(247, 252, dtype=int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape_name(flowline, node: int) -> str:
    if not bool(flowline.is_trapezoid[node]):
        return "parabolic"
    return "rectangular" if float(flowline._lambdas[node]) == 0.0 else "trapezoid"


def _active_reverse_edges(flowlines) -> list[dict]:
    rows = []
    for flowline_id, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        for node in np.flatnonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9)):
            rows.append({
                "flowline": int(flowline_id),
                "node": int(node),
                "surface_step_m": float(surface[node + 1] - surface[node]),
            })
    return rows


def _apply_rectangular(flowline, nodes: np.ndarray) -> None:
    surface = np.asarray(flowline.surface_h, dtype=float).copy()
    section = np.asarray(flowline.section, dtype=float).copy()
    widths = np.asarray(flowline.widths_m, dtype=float).copy()
    target_thick = section[nodes] / widths[nodes]

    flowline.is_trapezoid[nodes] = True
    flowline._lambdas[nodes] = 0.0
    flowline._w0_m[nodes] = widths[nodes]
    flowline.bed_shape[nodes] = np.nan
    flowline._ptrap = np.where(flowline.is_trapezoid)[0]
    flowline._do_trapeze = bool(np.any(flowline.is_trapezoid))
    flowline._prec = flowline.is_trapezoid & (flowline._lambdas == 0.0)
    flowline._sqrt_bed = np.sqrt(flowline.bed_shape)
    flowline.bed_h[nodes] = surface[nodes] - target_thick
    flowline.thick = np.where(
        np.isin(np.arange(flowline.nx), nodes),
        np.where(np.isin(np.arange(flowline.nx), nodes), 0.0, 0.0),
        flowline.thick,
    )
    # Set the target nodes separately; the awkward-looking assignment above
    # deliberately keeps all non-target values bit-for-bit unchanged.
    thick = np.asarray(flowline.thick, dtype=float).copy()
    thick[nodes] = target_thick
    flowline.thick = thick


def _staggered_velocity(flowline, staggered_bin: int, glen_a: float, fs: float) -> float:
    from oggm import cfg
    from oggm.cfg import G, SEC_IN_YEAR

    surface = np.asarray(flowline.surface_h, dtype=float)
    thick = np.asarray(flowline.thick, dtype=float)
    slope = (surface[staggered_bin - 1] - surface[staggered_bin]) / float(flowline.dx_meter)
    thick_stag = 0.5 * (thick[staggered_bin - 1] + thick[staggered_bin])
    n = float(cfg.PARAMS["glen_n"])
    rho = float(cfg.PARAMS["ice_density"])
    fd = 2.0 / (n + 2.0) * glen_a
    rhogh = (rho * G * slope) ** n
    velocity = thick_stag ** (n + 1.0) * fd * rhogh + thick_stag ** (n - 1.0) * fs * rhogh
    return float(velocity * SEC_IN_YEAR)


def main() -> None:
    from oggm import cfg

    OUTPUT.mkdir(parents=True, exist_ok=False)
    cfg.initialize(logging_level="WARNING")
    glen_a = 1.4759277199273846e-23
    fs = 0.0
    with gzip.open(SOURCE, "rb") as stream:
        original = pickle.load(stream)
    candidate = copy.deepcopy(original)

    root0 = original[ROOT_ID]
    root1 = candidate[ROOT_ID]
    surface0 = np.asarray(root0.surface_h, dtype=float).copy()
    section0 = np.asarray(root0.section, dtype=float).copy()
    widths0 = np.asarray(root0.widths_m, dtype=float).copy()
    thick0 = np.asarray(root0.thick, dtype=float).copy()
    bed0 = np.asarray(root0.bed_h, dtype=float).copy()
    area0 = float(sum(flowline.area_m2 for flowline in original))
    volume0 = float(sum(flowline.volume_m3 for flowline in original))
    length0 = float(root0.length_m)
    terminal0 = float(root0.thick[-1])
    reverse0 = _active_reverse_edges(original)

    rows = []
    for node in NODES:
        rectangular_thick = float(section0[node] / widths0[node])
        rows.append({
            "node": int(node),
            "original_shape": _shape_name(root0, int(node)),
            "surface_m": float(surface0[node]),
            "original_bed_m": float(bed0[node]),
            "original_thickness_m": float(thick0[node]),
            "section_m2": float(section0[node]),
            "surface_width_m": float(widths0[node]),
            "rectangular_thickness_m": rectangular_thick,
            "rectangular_bed_m": float(surface0[node] - rectangular_thick),
            "thickness_change_m": float(rectangular_thick - thick0[node]),
            "bed_change_m": float(thick0[node] - rectangular_thick),
            "rectangular_w0_m": float(widths0[node]),
            "rectangular_lambda": 0.0,
        })

    _apply_rectangular(root1, NODES)
    surface1 = np.asarray(root1.surface_h, dtype=float)
    section1 = np.asarray(root1.section, dtype=float)
    widths1 = np.asarray(root1.widths_m, dtype=float)
    area1 = float(sum(flowline.area_m2 for flowline in candidate))
    volume1 = float(sum(flowline.volume_m3 for flowline in candidate))
    length1 = float(root1.length_m)
    terminal1 = float(root1.thick[-1])
    reverse1 = _active_reverse_edges(candidate)

    initial_velocity = {}
    for staggered_bin in range(247, 253):
        old = _staggered_velocity(root0, staggered_bin, glen_a, fs)
        new = _staggered_velocity(root1, staggered_bin, glen_a, fs)
        initial_velocity[str(staggered_bin)] = {
            "original_m_per_year": old,
            "candidate_m_per_year": new,
            "candidate_to_original_ratio": abs(new / old) if old != 0 else None,
        }

    tolerances = {"absolute_geometry": 1e-8, "relative_area_volume": 1e-12}
    gates = {
        "all_target_nodes_legal_rectangular": bool(
            np.all(root1.is_trapezoid[NODES])
            and np.all(root1._lambdas[NODES] == 0.0)
            and np.all(root1._w0_m[NODES] > 0.0)
        ),
        "target_sections_unchanged": bool(np.max(np.abs(section1[NODES] - section0[NODES])) <= 1e-8),
        "target_surface_widths_unchanged": bool(np.max(np.abs(widths1[NODES] - widths0[NODES])) <= 1e-8),
        "maximum_surface_change_zero": bool(np.max(np.abs(surface1 - surface0)) <= 1e-8),
        "total_area_unchanged": bool(abs(area1 - area0) <= max(1e-8, area0 * 1e-12)),
        "total_volume_unchanged": bool(abs(volume1 - volume0) <= max(1e-8, volume0 * 1e-12)),
        "root_length_unchanged": bool(abs(length1 - length0) <= 1e-8),
        "terminal_thickness_unchanged": bool(abs(terminal1 - terminal0) <= 1e-8),
        "active_reverse_edges_unchanged": reverse1 == reverse0,
        "hypsometry_unchanged": bool(
            np.max(np.abs(surface1 - surface0)) <= 1e-8
            and np.max(np.abs(widths1 - widths0)) <= 1e-8
        ),
        "initial_bin249_velocity_reduced": bool(
            initial_velocity["249"]["candidate_to_original_ratio"] < 1.0
        ),
    }
    payload = {
        "experiment": "zero-mb-3660-local-mixedbed-shape-static-preflight-v1",
        "system_id": 3660,
        "read_only": True,
        "source_model_flowlines": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "root_flowline": ROOT_ID,
        "target_nodes": NODES.tolist(),
        "candidate": "per-node rectangular limit preserving section, surface width and surface elevation",
        "analytic_relation": {
            "section": "S = W * H for lambda=0 rectangular",
            "target_thickness": "H_new = S_old / W_surface_old",
            "target_bed": "bed_new = surface_old - H_new",
            "preserved": "S, W_surface, surface, dx; therefore node V, node A and hypsometry",
        },
        "node_diagnostics": rows,
        "initial_staggered_velocity": initial_velocity,
        "metrics": {
            "area_before_m2": area0,
            "area_after_m2": area1,
            "area_difference_m2": area1 - area0,
            "volume_before_m3": volume0,
            "volume_after_m3": volume1,
            "volume_difference_m3": volume1 - volume0,
            "root_length_before_m": length0,
            "root_length_after_m": length1,
            "terminal_before_m": terminal0,
            "terminal_after_m": terminal1,
            "maximum_surface_change_m": float(np.max(np.abs(surface1 - surface0))),
            "maximum_target_section_change_m2": float(np.max(np.abs(section1[NODES] - section0[NODES]))),
            "maximum_target_width_change_m": float(np.max(np.abs(widths1[NODES] - widths0[NODES]))),
            "maximum_absolute_bed_change_m": float(np.max(np.abs(root1.bed_h[NODES] - bed0[NODES]))),
            "active_reverse_before": len(reverse0),
            "active_reverse_after": len(reverse1),
        },
        "gates": gates,
        "static_pass": all(gates.values()),
        "next_action": "one-versioned-30s-5y-zeroMB" if all(gates.values()) else "close-mechanism-before-dynamics",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
