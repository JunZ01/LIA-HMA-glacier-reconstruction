"""Build one analytic local MixedBed shape candidate for system 3660.

Nodes 247--251 on root 32 retain their section, surface elevation and dx.
Their rectangular thicknesses follow the straight line between untouched
nodes 246 and 252.  One closed-form scale factor preserves the sum of surface
widths, hence local area and the 100 m hypsometry bin exactly.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from collections import defaultdict
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
    / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v2"
)
ROOT_ID = 32
NODES = np.arange(247, 252, dtype=int)
LEFT_ANCHOR = 246
RIGHT_ANCHOR = 252


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_reverse_edges(flowlines) -> list[tuple[int, int, float]]:
    rows = []
    for flowline_id, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        for node in np.flatnonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9)):
            rows.append((int(flowline_id), int(node), float(surface[node + 1] - surface[node])))
    return rows


def _hypsometry_100m(flowlines) -> dict[int, float]:
    values = defaultdict(float)
    for flowline in flowlines:
        surface = np.asarray(flowline.surface_h, dtype=float)
        width = np.asarray(flowline.widths_m, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        for elevation, node_width in zip(surface[active], width[active]):
            values[int(np.floor(elevation / 100.0) * 100)] += float(node_width * flowline.dx_meter)
    return dict(values)


def _apply_candidate(flowline) -> dict:
    surface = np.asarray(flowline.surface_h, dtype=float).copy()
    section = np.asarray(flowline.section, dtype=float).copy()
    width = np.asarray(flowline.widths_m, dtype=float).copy()
    thick = np.asarray(flowline.thick, dtype=float).copy()

    fraction = (NODES - LEFT_ANCHOR) / float(RIGHT_ANCHOR - LEFT_ANCHOR)
    base_thick = thick[LEFT_ANCHOR] + fraction * (thick[RIGHT_ANCHOR] - thick[LEFT_ANCHOR])
    # sum_i S_i / (c * Hbase_i) = sum_i Wold_i
    scale = float(np.sum(section[NODES] / base_thick) / np.sum(width[NODES]))
    target_thick = scale * base_thick
    target_width = section[NODES] / target_thick

    flowline.is_trapezoid[NODES] = True
    flowline._lambdas[NODES] = 0.0
    flowline._w0_m[NODES] = target_width
    flowline.bed_shape[NODES] = np.nan
    flowline._ptrap = np.where(flowline.is_trapezoid)[0]
    flowline._do_trapeze = bool(np.any(flowline.is_trapezoid))
    flowline._prec = flowline.is_trapezoid & (flowline._lambdas == 0.0)
    flowline._sqrt_bed = np.sqrt(flowline.bed_shape)
    flowline.bed_h[NODES] = surface[NODES] - target_thick
    new_thick = thick.copy()
    new_thick[NODES] = target_thick
    flowline.thick = new_thick

    return {
        "base_thickness_m": base_thick.tolist(),
        "closed_form_scale": scale,
        "target_thickness_m": target_thick.tolist(),
        "target_surface_width_m": target_width.tolist(),
    }


def _initial_velocity(flowline, staggered_bin: int, glen_a: float, fs: float) -> float:
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
    width0 = np.asarray(root0.widths_m, dtype=float).copy()
    thick0 = np.asarray(root0.thick, dtype=float).copy()
    bed0 = np.asarray(root0.bed_h, dtype=float).copy()
    area0 = float(sum(flowline.area_m2 for flowline in original))
    volume0 = float(sum(flowline.volume_m3 for flowline in original))
    length0 = float(root0.length_m)
    reverse0 = _active_reverse_edges(original)
    hyp0 = _hypsometry_100m(original)

    construction = _apply_candidate(root1)
    surface1 = np.asarray(root1.surface_h, dtype=float)
    section1 = np.asarray(root1.section, dtype=float)
    width1 = np.asarray(root1.widths_m, dtype=float)
    thick1 = np.asarray(root1.thick, dtype=float)
    bed1 = np.asarray(root1.bed_h, dtype=float)
    area1 = float(sum(flowline.area_m2 for flowline in candidate))
    volume1 = float(sum(flowline.volume_m3 for flowline in candidate))
    length1 = float(root1.length_m)
    reverse1 = _active_reverse_edges(candidate)
    hyp1 = _hypsometry_100m(candidate)
    hyp_keys = sorted(set(hyp0) | set(hyp1))
    hyp_max = max(abs(hyp1.get(key, 0.0) - hyp0.get(key, 0.0)) for key in hyp_keys)

    velocity = {}
    for staggered_bin in range(247, 253):
        old = _initial_velocity(root0, staggered_bin, glen_a, fs)
        new = _initial_velocity(root1, staggered_bin, glen_a, fs)
        velocity[str(staggered_bin)] = {
            "original_m_per_year": old,
            "candidate_m_per_year": new,
            "candidate_to_original_ratio": abs(new / old) if old != 0 else None,
        }

    node_rows = []
    for node in NODES:
        node_rows.append({
            "node": int(node),
            "surface_m": float(surface0[node]),
            "section_m2": float(section0[node]),
            "old_width_m": float(width0[node]),
            "new_width_m": float(width1[node]),
            "old_thickness_m": float(thick0[node]),
            "new_thickness_m": float(thick1[node]),
            "old_bed_m": float(bed0[node]),
            "new_bed_m": float(bed1[node]),
            "old_shape": "rectangular" if root0.is_rectangular[node] else str(root0.shape_str[node]),
            "new_shape": "rectangular",
        })

    old_internal = np.diff(thick0[LEFT_ANCHOR:RIGHT_ANCHOR + 1])
    new_internal = np.diff(thick1[LEFT_ANCHOR:RIGHT_ANCHOR + 1])
    gates = {
        "analytic_scale_positive": construction["closed_form_scale"] > 0.0,
        "all_target_rectangles_legal": bool(
            np.all(root1.is_rectangular[NODES]) and np.all(root1._w0_m[NODES] > 0.0)
        ),
        "each_target_section_unchanged": bool(np.max(np.abs(section1[NODES] - section0[NODES])) <= 1e-8),
        "each_target_surface_unchanged": bool(np.max(np.abs(surface1[NODES] - surface0[NODES])) <= 1e-8),
        "each_target_dx_unchanged": True,
        "local_width_sum_unchanged": bool(abs(np.sum(width1[NODES]) - np.sum(width0[NODES])) <= 1e-8),
        "total_area_unchanged": bool(abs(area1 - area0) <= max(1e-8, area0 * 1e-12)),
        "total_volume_unchanged": bool(abs(volume1 - volume0) <= max(1e-8, volume0 * 1e-12)),
        "root_length_unchanged": bool(abs(length1 - length0) <= 1e-8),
        "maximum_surface_change_zero": bool(np.max(np.abs(surface1 - surface0)) <= 1e-8),
        "terminal_unchanged": bool(root1.thick[-1] == root0.thick[-1]),
        "hypsometry_100m_unchanged": bool(hyp_max <= 1e-8),
        "active_reverse_edges_unchanged": reverse1 == reverse0,
        "internal_thickness_jump_reduced": bool(np.max(np.abs(new_internal)) < np.max(np.abs(old_internal))),
        "initial_bin249_velocity_reduced": bool(velocity["249"]["candidate_to_original_ratio"] < 1.0),
    }
    candidate_path = OUTPUT / "candidate_model_flowlines.pkl"
    static_pass = all(gates.values())
    if static_pass:
        with gzip.open(candidate_path, "wb") as stream:
            pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)

    payload = {
        "experiment": "zero-mb-3660-local-mixedbed-shape-static-preflight-v2",
        "system_id": 3660,
        "mechanism": "closed-form-linear-thickness-local-rectangular-representation",
        "source_model_flowlines": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "root_flowline": ROOT_ID,
        "target_nodes": NODES.tolist(),
        "untouched_thickness_anchors": [LEFT_ANCHOR, RIGHT_ANCHOR],
        "construction": construction,
        "node_diagnostics": node_rows,
        "initial_staggered_velocity": velocity,
        "metrics": {
            "area_difference_m2": area1 - area0,
            "volume_difference_m3": volume1 - volume0,
            "length_difference_m": length1 - length0,
            "maximum_surface_change_m": float(np.max(np.abs(surface1 - surface0))),
            "maximum_target_section_change_m2": float(np.max(np.abs(section1[NODES] - section0[NODES]))),
            "local_width_sum_difference_m": float(np.sum(width1[NODES]) - np.sum(width0[NODES])),
            "maximum_100m_hypsometry_area_difference_m2": float(hyp_max),
            "maximum_old_local_thickness_step_m": float(np.max(np.abs(old_internal))),
            "maximum_new_local_thickness_step_m": float(np.max(np.abs(new_internal))),
            "maximum_absolute_bed_change_m": float(np.max(np.abs(bed1[NODES] - bed0[NODES]))),
            "active_reverse_before": len(reverse0),
            "active_reverse_after": len(reverse1),
        },
        "gates": gates,
        "static_pass": static_pass,
        "candidate_model_flowlines": str(candidate_path) if static_pass else None,
        "candidate_sha256": _sha256(candidate_path) if static_pass else None,
        "next_action": "one-30s-5y-zeroMB" if static_pass else "close-before-dynamics",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
