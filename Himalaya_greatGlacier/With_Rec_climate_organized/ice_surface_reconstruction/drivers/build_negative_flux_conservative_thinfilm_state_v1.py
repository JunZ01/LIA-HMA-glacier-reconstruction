"""Build one preregistered conservative thin-film state for system 3131.

The mechanism is an initial-state diagnostic, not an inversion replacement:

* retain the immutable standard-inversion bed, shape parameters, topology,
  flowline widths parameters, climate, and Stage-2 physics;
* put a 0.1*dx finite ice film only on source-domain nodes that have negative
  apparent flux and were clipped to zero thickness by OGGM;
* multiply every previously active thickness by one common scalar found by
  bisection so total model volume is conserved to numerical precision.

No width multiplier is applied.  The output is eligible for zero-MB only when
all preregistered static area/volume/length/hypsometry and active-slope gates
pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from ..paths import resolve


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _volume(flowlines) -> float:
    return float(sum(np.sum(np.asarray(fl.section, dtype=float)) * fl.dx_meter for fl in flowlines))


def _area(flowlines) -> float:
    return float(
        sum(
            np.sum(np.asarray(fl.widths_m, dtype=float)[np.asarray(fl.thick, dtype=float) > 0])
            * fl.dx_meter
            for fl in flowlines
        )
    )


def _weighted_cdf_error(reference_z, reference_w, candidate_z, candidate_w, bin_m: float) -> float:
    lo = np.floor(min(np.min(reference_z), np.min(candidate_z)) / bin_m) * bin_m
    hi = np.ceil(max(np.max(reference_z), np.max(candidate_z)) / bin_m) * bin_m + bin_m
    edges = np.arange(lo, hi + 0.5 * bin_m, bin_m)
    ref = np.histogram(reference_z, bins=edges, weights=reference_w)[0].astype(float)
    cand = np.histogram(candidate_z, bins=edges, weights=candidate_w)[0].astype(float)
    if ref.sum() <= 0 or cand.sum() <= 0:
        return float("inf")
    return float(np.max(np.abs(np.cumsum(ref / ref.sum()) - np.cumsum(cand / cand.sum()))))


def _apply_state(template, original_thickness, affected, scale: float, film_m: float):
    out = copy.deepcopy(template)
    for branch, fl in enumerate(out):
        thick = np.asarray(original_thickness[branch], dtype=float).copy()
        was_active = thick > 0
        thick[was_active] *= scale
        if branch in affected:
            thick[np.asarray(affected[branch], dtype=int)] = film_m
        fl.thick = thick
    return out


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = resolve(config["source_inversion_root"])
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    output_root.mkdir(parents=True)
    _write_json(output_root / "config_snapshot.json", config)

    source_config = json.loads((source_root / "config_snapshot.json").read_text(encoding="utf-8"))
    source_formal = json.loads((source_root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    if int(source_formal["system_id"]) != int(config["system_id"]):
        raise RuntimeError("source-system-id-mismatch")
    if source_formal.get("review_flags") != ["model-outline-area-mismatch"]:
        raise RuntimeError(f"unexpected-source-failure:{source_formal.get('review_flags')}")
    if source_config["stage2"] != config["stage2_frozen"]:
        raise RuntimeError("stage2-contract-drift")

    from oggm import cfg, utils

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    source_gdir = utils.GlacierDirectory(
        str(source_config["synthetic_rgi_id"]),
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
        n_source = len(np.asarray(ifl.flux, dtype=float))
        flux = np.asarray(ifl.flux, dtype=float)
        thick = np.asarray(mfl.thick, dtype=float)[:n_source]
        nodes = np.flatnonzero((flux < 0) & (thick <= 0))
        if nodes.size:
            affected[branch] = nodes
    expected = {int(k): int(v) for k, v in config["expected_negative_inactive_nodes"].items()}
    observed = {branch: int(nodes.size) for branch, nodes in affected.items()}
    if observed != expected:
        raise RuntimeError(f"affected-node-contract-drift:observed={observed}:expected={expected}")

    dx_values = {float(source_model[branch].dx_meter) for branch in affected}
    if len(dx_values) != 1:
        raise RuntimeError(f"affected-dx-not-unique:{sorted(dx_values)}")
    film_m = float(config["film_thickness_fraction_of_dx"]) * dx_values.pop()

    lower, upper = 0.0, 1.0
    lower_volume = _volume(_apply_state(source_model, original_thickness, affected, lower, film_m))
    upper_volume = _volume(_apply_state(source_model, original_thickness, affected, upper, film_m))
    if not (lower_volume < original_volume < upper_volume):
        raise RuntimeError(
            f"volume-root-not-bracketed:{lower_volume}:{original_volume}:{upper_volume}"
        )
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        trial_volume = _volume(
            _apply_state(source_model, original_thickness, affected, middle, film_m)
        )
        if trial_volume < original_volume:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    modified = _apply_state(source_model, original_thickness, affected, scale, film_m)

    modified_volume = _volume(modified)
    modified_area = _area(modified)
    outline_area = float(source_formal["outline_area_m2"])
    volume_relative = float((modified_volume - original_volume) / original_volume)
    area_error = float(abs(modified_area - outline_area) / outline_area)

    roots = [index for index, fl in enumerate(modified) if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root = modified[roots[0]]
    root_active = np.flatnonzero(np.asarray(root.thick, dtype=float) > 0)
    root_length = float((root_active[-1] + 1) * root.dx_meter) if root_active.size else 0.0
    free_tail = float((root.nx - (root_active[-1] + 1)) * root.dx_meter) if root_active.size else float(root.nx * root.dx_meter)
    length_error = float(
        abs(root_length - float(config["registered_lia_length_m"]))
        / float(config["registered_lia_length_m"])
    )

    reference_z, reference_w = [], []
    candidate_z, candidate_w = [], []
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

    identity = {id(fl): index for index, fl in enumerate(modified)}
    reverse = []
    junction_reverse = []
    tolerance = float(config["gates"]["reverse_slope_tolerance_m"])
    for branch, fl in enumerate(modified):
        active = np.asarray(fl.thick, dtype=float) > 0
        for node, rise in enumerate(np.diff(np.asarray(fl.surface_h, dtype=float))):
            if active[node] and active[node + 1] and rise > tolerance:
                reverse.append({"branch": branch, "node": node, "rise_m": float(rise)})
        if fl.flows_to is not None and active[-1]:
            receiver = fl.flows_to
            receiver_node = int(fl.flows_to_indice)
            rise = float(receiver.surface_h[receiver_node] - fl.surface_h[-1])
            if rise > tolerance:
                junction_reverse.append(
                    {
                        "branch": branch,
                        "receiver_branch": identity[id(receiver)],
                        "receiver_node": receiver_node,
                        "rise_m": rise,
                    }
                )

    bed_difference = float(
        max(np.max(np.abs(np.asarray(fl.bed_h) - bed)) for fl, bed in zip(modified, original_bed))
    )
    flags = []
    if abs(volume_relative) > float(config["gates"]["volume_conservation_fraction_max"]):
        flags.append("volume-not-conserved")
    if area_error > float(config["gates"]["model_area_relative_error_max"]):
        flags.append("model-outline-area-mismatch")
    if length_error > float(config["gates"]["root_length_relative_error_max"]):
        flags.append("model-root-length-mismatch")
    if hypsometry_error > float(config["gates"]["hypsometry_cdf_error_max"]):
        flags.append("outline-hypsometry-cdf-mismatch")
    if free_tail < float(config["gates"]["minimum_free_downstream_m"]):
        flags.append("root-free-tail-too-short")
    if float(root.thick[-1]) > 0:
        flags.append("root-safety-tail-has-ice")
    if reverse:
        flags.append("active-within-branch-reverse-slope")
    if junction_reverse:
        flags.append("active-junction-reverse-slope")
    if bed_difference != 0:
        flags.append("bed-changed")

    change_rows = []
    for branch, (before, after) in enumerate(zip(source_model, modified)):
        before_thick = np.asarray(before.thick, dtype=float)
        after_thick = np.asarray(after.thick, dtype=float)
        for node in np.flatnonzero(before_thick != after_thick):
            change_rows.append(
                {
                    "branch": branch,
                    "node": int(node),
                    "distance_m": float(node * after.dx_meter),
                    "negative_inactive_target": bool(branch in affected and node in affected[branch]),
                    "thickness_before_m": float(before_thick[node]),
                    "thickness_after_m": float(after_thick[node]),
                    "width_before_m": float(before.widths_m[node]),
                    "width_after_m": float(after.widths_m[node]),
                    "section_before_m2": float(before.section[node]),
                    "section_after_m2": float(after.section[node]),
                    "surface_before_m": float(before.surface_h[node]),
                    "surface_after_m": float(after.surface_h[node]),
                }
            )
    with (output_root / "initial_state_node_changes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(change_rows[0]))
        writer.writeheader()
        writer.writerows(change_rows)

    static_pass = not flags
    summary = {
        "status": "initial-state-static-pass" if static_pass else "initial-state-static-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "source_inversion_root": str(source_root),
        "mechanism": {
            "name": "negative-flux-local-thinfilm-global-active-thickness-conservation-v1",
            "affected_node_rule": "inversion apparent flux < 0 and initialized thickness == 0",
            "affected_nodes": observed,
            "film_thickness_fraction_of_dx": float(config["film_thickness_fraction_of_dx"]),
            "film_thickness_m": film_m,
            "donor_rule": "one common multiplicative scale on all previously active thicknesses",
            "active_thickness_scale": scale,
            "explicit_width_multiplier": 1.0,
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
            "registered_lia_length_m": float(config["registered_lia_length_m"]),
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
        "gates": config["gates"],
        "review_flags": flags,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "decision": "release-3131-zero-mb-only" if static_pass else "close-mechanism-no-1642",
    }
    _write_json(output_root / "initial_state_summary.json", summary)

    if static_pass:
        shutil.copytree(source_root / "synthetic_gdir", output_root / "synthetic_gdir")
        output_gdir = utils.GlacierDirectory(
            str(source_config["synthetic_rgi_id"]),
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
