"""Run the frozen 24-system fixed-bed geometry and MB-width audit."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_dynamic_matrix_pilot import (
    _configure_oggm,
    load_source_state,
    run_case,
    write_json,
)
from .lia_system_grid import normalize_lia_components


def _period_years(period: str) -> list[int]:
    start, end = period.split("_")
    return list(range(int(start[:4]), int(end[:4])))


def _annual_mb_mwea(model, heights: np.ndarray, years: list[int]) -> np.ndarray:
    from oggm import cfg

    values = [
        model.get_annual_mb(heights, year=year)
        * cfg.SEC_IN_YEAR
        * cfg.PARAMS["ice_density"]
        / 1000.0
        for year in years
    ]
    return np.mean(np.asarray(values, dtype=float), axis=0)


def _gate(abs_bias: float, sigma: float) -> str:
    if not np.isfinite(abs_bias) or not np.isfinite(sigma) or sigma <= 0:
        return "unclassified-missing-threshold"
    if abs_bias <= sigma:
        return "green"
    if abs_bias <= 2.0 * sigma:
        return "yellow"
    return "red"


def _mb_width_rows(
    nodes: pd.DataFrame,
    *,
    main_rgi_id: str,
    modern_gdir_root: Path,
    years: list[int],
    ela_offsets_m: list[float],
    geodetic_sigma_mwea: float,
) -> list[dict]:
    from oggm import GlacierDirectory
    from oggm.core.massbalance import MonthlyTIModel

    gdir = GlacierDirectory(main_rgi_id, base_dir=str(modern_gdir_root))
    calibration = gdir.read_json("mb_calib")
    baseline_temp_bias = float(calibration["temp_bias"])
    gradient = float(
        calibration["mb_global_params"].get("temp_default_gradient", -0.0065)
    )
    heights = nodes["surface_h_m"].to_numpy(dtype=float)
    observed_width = nodes["observed_width_m"].to_numpy(dtype=float)
    effective_width = nodes["effective_width_m"].to_numpy(dtype=float)
    dx = nodes["dx_meter"].to_numpy(dtype=float)
    observed_area = float(np.sum(observed_width * dx))
    effective_area = float(np.sum(effective_width * dx))
    rows = []
    for offset in ela_offsets_m:
        temperature_delta = -gradient * float(offset)
        model = MonthlyTIModel(
            gdir,
            temp_bias=baseline_temp_bias + temperature_delta,
            check_calib_params=False,
        )
        mb = _annual_mb_mwea(model, heights, years)
        observed_specific = float(
            np.sum(mb * observed_width * dx) / observed_area
        )
        effective_specific = float(
            np.sum(mb * effective_width * dx) / effective_area
        )
        net_bias = effective_specific - observed_specific
        absolute_flux_bias = float(
            np.sum(np.abs(mb * (effective_width - observed_width)) * dx)
            / observed_area
        )
        rows.append(
            {
                "ela_offset_m": float(offset),
                "temperature_bias_delta_c": float(temperature_delta),
                "observed_specific_mb_mwea": observed_specific,
                "effective_specific_mb_mwea": effective_specific,
                "width_induced_net_mb_bias_mwea": net_bias,
                "width_induced_absolute_flux_bias_mwea": absolute_flux_bias,
                "geodetic_sigma_mwea": float(geodetic_sigma_mwea),
                "gate": _gate(abs(net_bias), geodetic_sigma_mwea),
            }
        )
    return rows


def _system_summary(
    *,
    sample_row,
    audits: list[dict],
    node_frame: pd.DataFrame,
    mb_rows: list[dict],
) -> dict:
    source_area = float(sum(a["cross_section_mapping"]["source_area_m2"] for a in audits))
    source_volume = float(
        sum(a["cross_section_mapping"]["source_volume_m3"] for a in audits)
    )
    strip_area = float(
        np.sum(node_frame["observed_width_m"] * node_frame["dx_meter"])
    )
    effective_area = float(
        np.sum(node_frame["effective_width_m"] * node_frame["dx_meter"])
    )
    section_volume = float(
        sum(a["cross_section_mapping"]["raw_flowline_volume_m3"] for a in audits)
    )
    flowline_volume = float(sum(a["flowline_volume_m3"] for a in audits))
    projection_rows = [
        row
        for audit in audits
        for row in audit["cross_section_mapping"]["trapezoid_projection"]
    ]
    worst_mb = max(mb_rows, key=lambda row: abs(row["width_induced_net_mb_bias_mwea"]))
    gate_order = {
        "green": 0,
        "yellow": 1,
        "red": 2,
        "unclassified-missing-threshold": 3,
    }
    overall_gate = max((row["gate"] for row in mb_rows), key=gate_order.get)
    observed_mean_z = float(
        np.sum(
            node_frame["surface_h_m"]
            * node_frame["observed_width_m"]
            * node_frame["dx_meter"]
        )
        / strip_area
    )
    effective_mean_z = float(
        np.sum(
            node_frame["surface_h_m"]
            * node_frame["effective_width_m"]
            * node_frame["dx_meter"]
        )
        / effective_area
    )
    return {
        "lia_unique_id": int(sample_row.lia_unique_id),
        "main_rgi_id": str(sample_row.MainRGI),
        "rgi_region": int(sample_row.rgi_region),
        "lia_area_km2_sample_manifest": float(sample_row.lia_area_km2),
        "component_count": int(len(audits)),
        "flowline_count": int(
            sum(a["flowline_build"]["network"]["flowline_count"] for a in audits)
        ),
        "node_count": int(len(node_frame)),
        "source_area_m2": source_area,
        "observed_strip_area_m2": strip_area,
        "effective_contract_area_m2": effective_area,
        "sampling_area_error_fraction": (strip_area - source_area) / source_area,
        "contract_area_error_fraction": (effective_area - strip_area) / strip_area,
        "total_effective_area_error_fraction": (effective_area - source_area) / source_area,
        "source_volume_m3": source_volume,
        "measured_section_volume_m3": section_volume,
        "flowline_volume_m3": flowline_volume,
        "sampling_volume_error_fraction": (
            (section_volume - source_volume) / source_volume
        ),
        "contract_section_preservation_error_fraction": (
            (flowline_volume - section_volume) / section_volume
        ),
        "maximum_absolute_bed_adjustment_m": float(
            max(
                row["equivalent_bed_adjustment_maximum_absolute_m"]
                for row in projection_rows
            )
        ),
        "maximum_absolute_thickness_adjustment_m": float(
            max(
                row["thickness_adjustment_maximum_absolute_m"]
                for row in projection_rows
            )
        ),
        "shape_factor_below_or_equal_0p5_fraction": float(
            np.mean(node_frame["regime_code"].eq(1))
        ),
        "shape_factor_above_1_fraction": float(
            np.mean(node_frame["regime_code"].eq(3))
        ),
        "shape_factor_above_1p5_fraction": float(
            np.mean(node_frame["shape_factor"].gt(1.5))
        ),
        "maximum_shape_factor": float(node_frame["shape_factor"].max()),
        "maximum_lambda": float(node_frame["lambda"].max()),
        "maximum_absolute_width_error_fraction": float(
            node_frame["width_error_fraction"].abs().max()
        ),
        "width_weighted_mean_surface_elevation_shift_m": (
            effective_mean_z - observed_mean_z
        ),
        "geodetic_sigma_mwea": float(sample_row.geodetic_sigma_mwea),
        "worst_ela_offset_m": float(worst_mb["ela_offset_m"]),
        "maximum_absolute_width_induced_net_mb_bias_mwea": float(
            abs(worst_mb["width_induced_net_mb_bias_mwea"])
        ),
        "maximum_width_induced_absolute_flux_bias_mwea": float(
            max(row["width_induced_absolute_flux_bias_mwea"] for row in mb_rows)
        ),
        "mb_gate": overall_gate,
    }


def run(
    config: dict, *, system_ids: set[int] | None = None
) -> tuple[pd.DataFrame, dict]:
    _configure_oggm(config)
    audit_config = config["lia_dynamic_fixed_bed_audit"]
    output_dir = resolve(config["output_dir"]) / str(audit_config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_bed_geometry_failures_v03.csv").unlink(missing_ok=True)
    sample_dir = resolve(config["output_dir"]) / str(
        audit_config.get("sample_input_subdir", audit_config["output_subdir"])
    )
    sample = pd.read_csv(sample_dir / "fixed_bed_audit_sample_v03.csv")
    if system_ids is not None:
        sample = sample[sample["lia_unique_id"].isin(system_ids)].copy()
        missing = system_ids - set(sample["lia_unique_id"].astype(int))
        if missing:
            raise ValueError(
                "requested-fixed-bed-system-not-in-frozen-sample:"
                + ",".join(map(str, sorted(missing)))
            )
    source_dir = resolve(config["output_dir"]) / str(
        config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
    )
    outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
    outlines_by_id = outlines.set_index("UniqueID")
    years = _period_years(audit_config["geodetic_period"])
    modern_root = resolve(audit_config["modern_mb_gdir_root"])
    summaries = []
    failures = []
    all_mb_rows = []
    for sample_row in sample.itertuples(index=False):
        unique_id = int(sample_row.lia_unique_id)
        try:
            candidates = sorted(
                source_dir.glob(f"lia_{unique_id}_*_system_grid_v1.nc")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"fixed-bed-system-grid-file-count:{unique_id}:{len(candidates)}"
                )
            source = load_source_state(candidates[0], audit_config["bed_scenario"])
            geometry = gpd.GeoSeries(
                [outlines_by_id.loc[unique_id].geometry], crs=outlines.crs
            ).to_crs(source.crs).iloc[0]
            component_audits = []
            node_frames = []
            for component_index, component in enumerate(
                normalize_lia_components(geometry)
            ):
                case_audit = run_case(
                    source=source,
                    component_geometry=component,
                    component_index=component_index,
                    map_dx_m=float(audit_config["map_dx_m"]),
                    flowline_dx=float(audit_config["flowline_dx"]),
                    routing_surface=str(audit_config["routing_surface"]),
                    dem_path=resolve(config["surface_dem_path"]),
                    buffer_m=float(audit_config["buffer_m"]),
                    output_dir=output_dir,
                    checkpoints=[],
                    glen_a=float(config["lia_dynamic_pilot"]["glen_a"]),
                    fs=float(config["lia_dynamic_pilot"]["fs"]),
                    smooth_trib_influx=bool(
                        config["lia_dynamic_pilot"]["smooth_trib_influx"]
                    ),
                    run_dynamics=False,
                    cross_section_contract=str(
                        audit_config["cross_section_contract"]
                    ),
                )
                component_audits.append(case_audit)
                nodes = pd.DataFrame(
                    case_audit["cross_section_mapping"]["fixed_bed_node_rows"]
                )
                nodes.insert(0, "component_index", component_index)
                node_frames.append(nodes)
            node_frame = pd.concat(node_frames, ignore_index=True)
            node_frame.insert(0, "lia_unique_id", unique_id)
            node_frame.to_csv(
                output_dir / f"fixed_bed_nodes_{unique_id}_v03.csv", index=False
            )
            mb_rows = _mb_width_rows(
                node_frame,
                main_rgi_id=str(sample_row.MainRGI),
                modern_gdir_root=modern_root,
                years=years,
                ela_offsets_m=[
                    float(value) for value in audit_config["ela_offsets_m"]
                ],
                geodetic_sigma_mwea=float(sample_row.geodetic_sigma_mwea),
            )
            for row in mb_rows:
                row.update(
                    {
                        "lia_unique_id": unique_id,
                        "main_rgi_id": str(sample_row.MainRGI),
                    }
                )
            all_mb_rows.extend(mb_rows)
            summaries.append(
                _system_summary(
                    sample_row=sample_row,
                    audits=component_audits,
                    node_frame=node_frame,
                    mb_rows=mb_rows,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "lia_unique_id": unique_id,
                    "main_rgi_id": str(sample_row.MainRGI),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=15),
                }
            )
        pd.DataFrame(summaries).to_csv(
            output_dir / "fixed_bed_geometry_systems_v03.csv", index=False
        )
        if failures:
            pd.DataFrame(failures).to_csv(
                output_dir / "fixed_bed_geometry_failures_v03.csv", index=False
            )
        if all_mb_rows:
            pd.DataFrame(all_mb_rows).to_csv(
                output_dir / "fixed_bed_mb_sensitivity_v03.csv", index=False
            )
    frame = pd.DataFrame(summaries)
    gate_counts = frame["mb_gate"].value_counts().to_dict() if not frame.empty else {}
    invariant_pass = bool(
        not frame.empty
        and np.all(frame["maximum_absolute_bed_adjustment_m"] <= 1e-9)
        and np.all(frame["maximum_absolute_thickness_adjustment_m"] <= 1e-9)
        and np.all(
            np.abs(frame["contract_section_preservation_error_fraction"]) <= 1e-12
        )
    )
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "sample_count": int(len(sample)),
        "success_count": int(len(frame)),
        "failure_count": int(len(failures)),
        "mb_gate_counts": {str(key): int(value) for key, value in gate_counts.items()},
        "fixed_bed_invariants_pass": invariant_pass,
        "dynamic_pilot_gate": (
            "pass-to-junction-audit"
            if invariant_pass and not failures and gate_counts.get("red", 0) == 0
            else "stop-before-dynamics"
        ),
        "modern_width_closure_status": "pending-common-coverage-audit",
        "source_products_mutated": False,
    }
    write_json(output_dir / "fixed_bed_geometry_summary_v03.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--output-subdir")
    parser.add_argument(
        "--routing-surface", choices=["modern-dem", "lia-surface", "lia-bed"]
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_subdir:
        audit = config["lia_dynamic_fixed_bed_audit"]
        audit["sample_input_subdir"] = audit["output_subdir"]
        audit["output_subdir"] = str(args.output_subdir)
    if args.routing_surface:
        config["lia_dynamic_fixed_bed_audit"]["routing_surface"] = str(
            args.routing_surface
        )
    system_ids = set(args.system_id) if args.system_id else None
    _, summary = run(config, system_ids=system_ids)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
