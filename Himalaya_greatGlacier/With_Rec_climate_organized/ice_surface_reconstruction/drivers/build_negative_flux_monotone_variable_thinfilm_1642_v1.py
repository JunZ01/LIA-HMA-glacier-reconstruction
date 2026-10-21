"""Build the preregistered monotone variable thin-film state for system 1642."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from ..paths import resolve
from .build_negative_flux_conservative_thinfilm_state_v1 import (
    _area,
    _volume,
    _weighted_cdf_error,
    _write_json,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_runs(nodes: np.ndarray) -> list[np.ndarray]:
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(nodes) != 1) + 1
    return [part for part in np.split(nodes, cuts) if part.size]


def _apply_monotone_state(
    template,
    original_thickness,
    affected: dict[int, np.ndarray],
    scale: float,
    maximum_film_m: float,
    minimum_film_m: float,
    margin_m: float,
):
    out = copy.deepcopy(template)
    diagnostics = []
    for branch, fl in enumerate(out):
        thick = np.asarray(original_thickness[branch], dtype=float).copy()
        thick[thick > 0] *= float(scale)
        fl.thick = thick
        if branch not in affected:
            continue
        for run in _target_runs(affected[branch]):
            if int(run[0]) == 0:
                raise RuntimeError(f"target-run-has-no-upstream-anchor:{branch}")
            for node in run:
                node = int(node)
                upstream_surface = float(fl.surface_h[node - 1])
                bed = float(fl.bed_h[node])
                feasible_max = upstream_surface - bed - float(margin_m)
                film = min(float(maximum_film_m), feasible_max)
                if film < float(minimum_film_m):
                    raise RuntimeError(
                        f"positive-film-monotonic-infeasible:{branch}:{node}:{feasible_max}"
                    )
                candidate = np.asarray(fl.thick, dtype=float).copy()
                candidate[node] = film
                fl.thick = candidate
                diagnostics.append(
                    {
                        "branch": branch,
                        "node": node,
                        "maximum_feasible_thickness_m": feasible_max,
                        "assigned_thickness_m": film,
                        "surface_margin_to_upstream_m": float(
                            fl.surface_h[node - 1] - fl.surface_h[node]
                        ),
                    }
                )
    return out, diagnostics


def _reverse_edges(flowlines, tolerance: float):
    identity = {id(fl): index for index, fl in enumerate(flowlines)}
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        active = np.asarray(fl.thick, dtype=float) > 0
        surface = np.asarray(fl.surface_h, dtype=float)
        for node, rise in enumerate(np.diff(surface)):
            if active[node] and active[node + 1] and rise > tolerance:
                within.append({"branch": branch, "node": node, "rise_m": float(rise)})
        if fl.flows_to is not None and active[-1]:
            receiver = fl.flows_to
            receiver_node = int(fl.flows_to_indice)
            rise = float(receiver.surface_h[receiver_node] - surface[-1])
            if rise > tolerance:
                junction.append(
                    {
                        "branch": branch,
                        "receiver_branch": identity[id(receiver)],
                        "receiver_node": receiver_node,
                        "rise_m": rise,
                    }
                )
    return within, junction


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = resolve(config["source_inversion_root"])
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    output_root.mkdir(parents=True)
    _write_json(output_root / "config_snapshot.json", config)

    source_config_path = source_root / "config_snapshot.json"
    source_formal_path = source_root / "inversion_summary_formal.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_formal = json.loads(source_formal_path.read_text(encoding="utf-8"))
    if int(source_formal["system_id"]) != int(config["system_id"]):
        raise RuntimeError("source-system-id-mismatch")
    if source_config["stage2"] != config["stage2_frozen"]:
        raise RuntimeError("stage2-contract-drift")

    from oggm import cfg, utils

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    source_gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=source_root / "synthetic_gdir",
        reset=False,
    )
    source_model = source_gdir.read_pickle("model_flowlines")
    inversion = source_gdir.read_pickle("inversion_flowlines")
    original_thickness = [np.asarray(fl.thick, dtype=float).copy() for fl in source_model]
    original_bed = [np.asarray(fl.bed_h, dtype=float).copy() for fl in source_model]
    original_volume = _volume(source_model)

    affected: dict[int, np.ndarray] = {}
    for branch, (ifl, mfl) in enumerate(zip(inversion, source_model)):
        flux = np.asarray(ifl.flux, dtype=float)
        thick = np.asarray(mfl.thick, dtype=float)[: len(flux)]
        nodes = np.flatnonzero((flux < 0) & (thick <= 0))
        if nodes.size:
            affected[branch] = nodes
    expected = {int(key): int(value) for key, value in config["expected_negative_inactive_nodes"].items()}
    observed = {branch: int(nodes.size) for branch, nodes in affected.items()}
    if observed != expected:
        raise RuntimeError(f"affected-node-contract-drift:{observed}:{expected}")

    maximum_film = float(config["maximum_film_thickness_m"])
    minimum_film = float(config["minimum_film_thickness_m"])
    margin = float(config["surface_monotonic_margin_m"])

    def evaluate(scale: float):
        candidate, diagnostics = _apply_monotone_state(
            source_model,
            original_thickness,
            affected,
            scale,
            maximum_film,
            minimum_film,
            margin,
        )
        return _volume(candidate), candidate, diagnostics

    upper = 1.0
    upper_volume, _, _ = evaluate(upper)
    if upper_volume <= original_volume:
        raise RuntimeError("film-does-not-add-volume-at-unit-scale")
    lower = None
    for trial in np.linspace(0.9995, 0.90, 200):
        try:
            volume, _, _ = evaluate(float(trial))
        except RuntimeError:
            continue
        if volume < original_volume:
            lower = float(trial)
            break
    if lower is None:
        raise RuntimeError("volume-root-not-bracketed-in-feasible-interval")
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        volume, _, _ = evaluate(middle)
        if volume < original_volume:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    modified_volume, modified, film_rows = evaluate(scale)

    modified_area = _area(modified)
    outline_area = float(source_formal["outline_area_m2"])
    volume_relative = float((modified_volume - original_volume) / original_volume)
    area_error = float(abs(modified_area - outline_area) / outline_area)
    roots = [index for index, fl in enumerate(modified) if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root = modified[roots[0]]
    root_active = np.flatnonzero(np.asarray(root.thick, dtype=float) > 0)
    root_length = float((root_active[-1] + 1) * root.dx_meter)
    free_tail = float((root.nx - root_active[-1] - 1) * root.dx_meter)
    registered_length = float(config["registered_lia_length_m"])
    length_error = abs(root_length - registered_length) / registered_length

    reference_z, reference_w, candidate_z, candidate_w = [], [], [], []
    for ifl, mfl in zip(inversion, modified):
        n = len(np.asarray(ifl.flux, dtype=float))
        reference_z.extend(np.asarray(ifl.surface_h, dtype=float))
        reference_w.extend(np.asarray(ifl.widths_m, dtype=float) * ifl.dx_meter)
        active = np.asarray(mfl.thick, dtype=float)[:n] > 0
        candidate_z.extend(np.asarray(mfl.surface_h, dtype=float)[:n][active])
        candidate_w.extend(np.asarray(mfl.widths_m, dtype=float)[:n][active] * mfl.dx_meter)
    hypsometry_error = _weighted_cdf_error(
        np.asarray(reference_z), np.asarray(reference_w),
        np.asarray(candidate_z), np.asarray(candidate_w),
        float(config["gates"]["hypsometry_bin_m"]),
    )
    tolerance = float(config["gates"]["reverse_slope_tolerance_m"])
    reverse, junction_reverse = _reverse_edges(modified, tolerance)
    bed_difference = max(
        float(np.max(np.abs(np.asarray(fl.bed_h) - bed)))
        for fl, bed in zip(modified, original_bed)
    )

    flags = []
    gates = config["gates"]
    if abs(volume_relative) > float(gates["volume_conservation_fraction_max"]):
        flags.append("volume-not-conserved")
    if area_error > float(gates["model_area_relative_error_max"]):
        flags.append("model-outline-area-mismatch")
    if length_error > float(gates["root_length_relative_error_max"]):
        flags.append("model-root-length-mismatch")
    if hypsometry_error > float(gates["hypsometry_cdf_error_max"]):
        flags.append("outline-hypsometry-cdf-mismatch")
    if free_tail < float(gates["minimum_free_downstream_m"]):
        flags.append("root-free-tail-too-short")
    if float(root.thick[-1]) > 0:
        flags.append("root-safety-tail-has-ice")
    if reverse:
        flags.append("active-within-branch-reverse-slope")
    if junction_reverse:
        flags.append("active-junction-reverse-slope")
    if bed_difference != 0:
        flags.append("bed-changed")

    with (output_root / "assigned_films.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(film_rows[0]))
        writer.writeheader()
        writer.writerows(film_rows)

    summary = {
        "status": "initial-state-static-pass" if not flags else "initial-state-static-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "mechanism": {
            "name": "negative-flux-monotone-variable-thinfilm-global-active-volume-conservation-v1",
            "affected_node_rule": "inversion apparent flux < 0 and initialized thickness == 0",
            "affected_nodes": observed,
            "film_rule": "largest thickness <=5m preserving downstream non-increasing surface at every target node",
            "minimum_film_thickness_m": minimum_film,
            "maximum_film_thickness_m": maximum_film,
            "assigned_film_min_m": min(row["assigned_thickness_m"] for row in film_rows),
            "assigned_film_max_m": max(row["assigned_thickness_m"] for row in film_rows),
            "assigned_film_below_max_count": sum(row["assigned_thickness_m"] < maximum_film for row in film_rows),
            "donor_rule": "one common multiplicative scale on all previously active thicknesses",
            "active_thickness_scale": scale,
            "bed_changed": False,
            "topology_changed": False,
            "climate_changed": False,
            "stage2_changed": False,
        },
        "static": {
            "source_volume_m3": original_volume,
            "modified_volume_m3": modified_volume,
            "volume_relative_difference": volume_relative,
            "outline_area_m2": outline_area,
            "modified_active_area_m2": modified_area,
            "model_outline_area_relative_error": area_error,
            "registered_lia_length_m": registered_length,
            "model_root_length_m": root_length,
            "root_length_relative_error": length_error,
            "root_free_tail_m": free_tail,
            "root_terminal_thickness_m": float(root.thick[-1]),
            "outline_hypsometry_cdf_error_100m": hypsometry_error,
            "maximum_absolute_bed_difference_m": bed_difference,
            "active_within_branch_reverse_slope_count": len(reverse),
            "active_junction_reverse_slope_count": len(junction_reverse),
            "active_reverse_edges": reverse,
            "active_junction_reverse_edges": junction_reverse,
        },
        "source_hashes": {
            "config_snapshot": _sha256(source_config_path),
            "inversion_summary_formal": _sha256(source_formal_path),
            "model_flowlines": _sha256(source_gdir.get_filepath("model_flowlines")),
            "inversion_flowlines": _sha256(source_gdir.get_filepath("inversion_flowlines")),
        },
        "gates": gates,
        "review_flags": flags,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "decision": "release-1642-zero-mb-only" if not flags else "close-mechanism-at-static",
    }
    _write_json(output_root / "initial_state_summary.json", summary)

    if not flags:
        shutil.copytree(source_root / "synthetic_gdir", output_root / "synthetic_gdir")
        output_gdir = utils.GlacierDirectory(
            str(config["synthetic_rgi_id"]),
            base_dir=output_root / "synthetic_gdir",
            reset=False,
        )
        output_gdir.write_pickle(modified, "model_flowlines")
        formal = {
            **source_formal,
            "status": "inversion-pass",
            "experiment": config["experiment"],
            "review_flags": [],
            "model_area_m2": modified_area,
            "model_volume_m3": modified_volume,
            "model_outline_area_relative_error": area_error,
            "model_root_length_m": root_length,
            "model_free_tail_m": free_tail,
            "root_terminal_thickness_m": float(root.thick[-1]),
            "active_within_branch_reverse_slope_count": len(reverse),
            "active_junction_reverse_slope_count": len(junction_reverse),
            "active_reverse_edges": reverse,
            "active_junction_reverse_edges": junction_reverse,
            "initial_state_mechanism": summary["mechanism"],
            "initial_state_static_summary": str(output_root / "initial_state_summary.json"),
            "zero_mb_run": False,
        }
        _write_json(output_root / "inversion_summary_formal.json", formal)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
