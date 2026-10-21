"""Read-only Cycle-1 Q4 preflight audit.

This script never imports or advances an OGGM model.  It consolidates three
already-completed evidence sets:

* the named system-1044 B-window candidate under the current formal and
  observation-domain gates;
* the ten-point delta-T spin-up histories for systems 2711 and 3053; and
* the saved five-year zero-MB geometry tests for systems 2387 and 5508.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_observation_domain import saved_series_observation_domain_assessment


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
CORE_ROOT = RESULT_ROOT / "lia_elevation_band_core191_length_corrected_v1"
CONTROLLER_ROOT = RESULT_ROOT / "lia_191_controller_v1"
SOURCE_1044 = (
    RESULT_ROOT
    / "layered_coordinate_search_v1"
    / "layer_B_melt_delta_boundary_refine_1044_round2_v1"
    / "system_1044"
    / "mf_4p666_dt_p1p192"
)
DEFAULT_OUTPUT = (
    RESULT_ROOT
    / "lia_191_continuous_class_controller_v1"
    / "q4_preflight_audit_v1"
)

FORMAL_SEPARATION_TOLERANCE = 0.20
TOTAL_LENGTH_TOLERANCE = 0.10
SPINUP_STABILITY_TOLERANCE = 0.02
ZERO_MB_VOLUME_TOLERANCE = 0.001
ZERO_MB_TERMINAL_THICKNESS_TOLERANCE_M = 10.0
EXPECTED_DELTA_GRID = np.arange(-3.0, 1.5001, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _audit_1044() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = SOURCE_1044 / "summary.json"
    series_path = SOURCE_1044 / "timeseries.csv"
    summary = _read_json(summary_path)
    series = pd.read_csv(series_path)
    audit = _read_json(CORE_ROOT / "1044/elevation_band_audit.json")
    node_count = int(audit["ice_node_count"])
    dx_m = float(audit["node_dx_m"])
    assessment = saved_series_observation_domain_assessment(
        series, node_count, node_dx_m=dx_m
    )

    forward = series.loc[series["phase"].astype(str).eq("forward")].copy()
    forward["length_m"] = pd.to_numeric(forward["length_m"], errors="coerce")
    peak_m = float(forward["length_m"].max())
    endpoint_m = float(
        forward.loc[pd.to_numeric(forward["year"]).eq(2000), "length_m"].iloc[-1]
    )
    lia_m = float(summary["targets"]["lia_length_m"])
    rgi_m = float(summary["targets"]["rgi_length_m"])
    separation_m = abs(lia_m - rgi_m)
    raw_e_lia_sep = abs(peak_m - lia_m) / separation_m
    raw_e_rgi_sep = abs(endpoint_m - rgi_m) / separation_m
    e_lia_total = abs(peak_m - lia_m) / lia_m
    e_rgi_total = abs(endpoint_m - rgi_m) / rgi_m
    stability = float(summary["spinup_control"]["final_stability_fraction"])
    hard_numeric_gate = bool(
        summary.get("status") == "ok"
        and summary["spinup_control"].get("converged")
        and stability <= SPINUP_STABILITY_TOLERANCE
    )
    raw_endpoint_gate = bool(
        raw_e_lia_sep <= FORMAL_SEPARATION_TOLERANCE
        and raw_e_rgi_sep <= FORMAL_SEPARATION_TOLERANCE
        and e_lia_total <= TOTAL_LENGTH_TOLERANCE
        and e_rgi_total <= TOTAL_LENGTH_TOLERANCE
    )

    affected = (
        assessment["overflow_rows"]
        if assessment["overflow"]
        else assessment["unresolved_rows"]
    )
    first_phase = None
    first_year = math.nan
    first_length_m = math.nan
    first_length_gt_10m = math.nan
    first_terminal_thickness_m = math.nan
    if bool(affected.any()):
        first = series.loc[affected[affected].index[0]]
        first_phase = str(first["phase"])
        first_year = int(first["year"])
        first_length_m = float(first["length_m"])
        first_length_gt_10m = float(first.get("length_gt_10m", math.nan))
        first_terminal_thickness_m = float(
            first.get("terminal_thickness_m", math.nan)
        )

    current_accepted = bool(
        hard_numeric_gate and raw_endpoint_gate and assessment["valid"]
    )
    state = "accepted" if current_accepted else assessment["state"]
    if state == "observation-domain-valid" and not current_accepted:
        state = "not-accepted"

    row = {
        "lia_unique_id": 1044,
        "candidate_name": "mf_4p666_dt_p1p192",
        "source_dir": str(SOURCE_1044.resolve()),
        "source_registered_for_q4_audit": True,
        "source_registered_for_production": False,
        "summary_status": summary.get("status"),
        "spinup_converged": bool(summary["spinup_control"]["converged"]),
        "spinup_stability_fraction": stability,
        "lia_target_m": lia_m,
        "rgi_target_m": rgi_m,
        "separation_m": separation_m,
        "raw_forward_peak_m": peak_m,
        "raw_forward_2000_m": endpoint_m,
        "raw_e_lia_sep": raw_e_lia_sep,
        "raw_e_rgi_sep": raw_e_rgi_sep,
        "e_lia_total": e_lia_total,
        "e_rgi_total": e_rgi_total,
        "raw_endpoint_gates_pass": raw_endpoint_gate,
        "observation_domain_node_count": node_count,
        "observation_domain_length_m": node_count * dx_m,
        "observation_domain_state": assessment["state"],
        "observation_domain_valid": assessment["valid"],
        "observation_domain_overflow_rows": int(
            assessment["overflow_rows"].sum()
        ),
        "observation_domain_unresolved_rows": int(
            assessment["unresolved_rows"].sum()
        ),
        "first_affected_phase": first_phase,
        "first_affected_year": first_year,
        "first_affected_length_m": first_length_m,
        "first_affected_length_gt_10m": first_length_gt_10m,
        "first_affected_terminal_thickness_m": first_terminal_thickness_m,
        "formal_accepted_current_gate": current_accepted,
        "acceptance_state": state,
        "release_decision": (
            "stop_current_candidate_observation_domain_overflow"
            if assessment["overflow"]
            else (
                "stop_current_candidate_observation_domain_unresolved"
                if assessment["unresolved"] or not assessment["assessed"]
                else (
                    "release_current_candidate"
                    if current_accepted
                    else "stop_current_candidate_endpoint_gate"
                )
            )
        ),
    }

    manifest_rows = []
    for name in (
        "summary.json",
        "timeseries.csv",
        "spinup_state.pkl.gz",
        "state_2000.pkl.gz",
    ):
        path = SOURCE_1044 / name
        manifest_rows.append(
            {
                "lia_unique_id": 1044,
                "candidate_name": "mf_4p666_dt_p1p192",
                "file_role": name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame([row]), pd.DataFrame(manifest_rows)


def _window_fraction(volumes: dict[int, float], start: int, end: int) -> float:
    if start not in volumes or end not in volumes or volumes[start] == 0:
        return math.nan
    return (volumes[end] - volumes[start]) / volumes[start]


def _audit_delta_convergence() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for system_id in (2711, 3053):
        root = CONTROLLER_ROOT / f"runs/system_{system_id}/B_delta_t_profile_stage2_physics"
        for summary_path in sorted(root.glob("*/summary.json")):
            summary = _read_json(summary_path)
            series_path = summary_path.parent / "timeseries.csv"
            series = pd.read_csv(series_path)
            spin = series.loc[series["phase"].astype(str).eq("spinup")].copy()
            volumes = {
                int(row.year): float(row.volume_m3)
                for row in spin.itertuples(index=False)
            }
            lengths = {
                int(row.year): float(row.length_m)
                for row in spin.itertuples(index=False)
            }
            dt = float(summary["parameters"]["delta_t_c"])
            rows.append(
                {
                    "lia_unique_id": system_id,
                    "delta_t_c": dt,
                    "source_dir": str(summary_path.parent.resolve()),
                    "summary_status": summary.get("status"),
                    "spinup_actual_years": int(
                        summary["spinup_control"]["actual_years"]
                    ),
                    "spinup_converged": bool(
                        summary["spinup_control"]["converged"]
                    ),
                    "final_stability_fraction": float(
                        summary["spinup_control"]["final_stability_fraction"]
                    ),
                    "stability_ratio_to_2pct_gate": float(
                        summary["spinup_control"]["final_stability_fraction"]
                    )
                    / SPINUP_STABILITY_TOLERANCE,
                    "spinup_length_1500_m": lengths.get(1500, math.nan),
                    "spinup_volume_1500_m3": volumes.get(1500, math.nan),
                    "volume_fraction_1000_1200": _window_fraction(
                        volumes, 1000, 1200
                    ),
                    "volume_fraction_1100_1300": _window_fraction(
                        volumes, 1100, 1300
                    ),
                    "volume_fraction_1200_1400": _window_fraction(
                        volumes, 1200, 1400
                    ),
                    "volume_fraction_1300_1500": _window_fraction(
                        volumes, 1300, 1500
                    ),
                    "late_fraction_declining_with_time": bool(
                        _window_fraction(volumes, 1000, 1200)
                        > _window_fraction(volumes, 1100, 1300)
                        > _window_fraction(volumes, 1200, 1400)
                        > _window_fraction(volumes, 1300, 1500)
                        > 0
                    ),
                }
            )
    history = pd.DataFrame(rows).sort_values(["lia_unique_id", "delta_t_c"])

    summaries = []
    for system_id, group in history.groupby("lia_unique_id", sort=True):
        group = group.sort_values("delta_t_c")
        delta = group["delta_t_c"].to_numpy(float)
        stability = group["final_stability_fraction"].to_numpy(float)
        lengths = group["spinup_length_1500_m"].to_numpy(float)
        expected_grid_complete = bool(
            len(delta) == len(EXPECTED_DELTA_GRID)
            and np.allclose(delta, EXPECTED_DELTA_GRID)
        )
        slope_stability = float(np.polyfit(delta, stability, 1)[0])
        slope_length = float(np.polyfit(delta, lengths, 1)[0])
        all_positive_growth = bool(
            np.all(group["volume_fraction_1300_1500"].to_numpy(float) > 0)
        )
        all_nonconverged = bool(~group["spinup_converged"].astype(bool).any())
        all_out_of_range = bool(group["summary_status"].eq("out_of_range").all())
        best = group.loc[group["final_stability_fraction"].idxmin()]
        summaries.append(
            {
                "lia_unique_id": int(system_id),
                "chain_count": int(len(group)),
                "expected_ten_point_grid_complete": expected_grid_complete,
                "all_nonconverged": all_nonconverged,
                "all_out_of_range": all_out_of_range,
                "all_final_200y_volume_changes_positive": all_positive_growth,
                "all_late_fractions_decline_with_time": bool(
                    group["late_fraction_declining_with_time"].all()
                ),
                "minimum_final_stability_fraction": float(
                    group["final_stability_fraction"].min()
                ),
                "maximum_final_stability_fraction": float(
                    group["final_stability_fraction"].max()
                ),
                "best_stability_delta_t_c": float(best["delta_t_c"]),
                "best_stability_length_1500_m": float(
                    best["spinup_length_1500_m"]
                ),
                "minimum_stability_ratio_to_2pct_gate": float(
                    group["stability_ratio_to_2pct_gate"].min()
                ),
                "stability_slope_per_k": slope_stability,
                "length_slope_m_per_k": slope_length,
                "warming_shortens_level": bool(slope_length < 0),
                "warming_worsens_fractional_stability": bool(
                    slope_stability > 0
                ),
                "convergence_direction_class": (
                    "uniform_long_growth_nonconverged_no_delta_bracket"
                    if expected_grid_complete
                    and all_nonconverged
                    and all_out_of_range
                    and all_positive_growth
                    else "mixed_requires_manual_review"
                ),
                "release_decision": (
                    "stop_delta_only_and_simple_extension_route_structural_state_diagnosis"
                ),
            }
        )
    return history, pd.DataFrame(summaries)


def _audit_zero_mb() -> pd.DataFrame:
    core_audit = pd.read_csv(CORE_ROOT / "audit/core191_elevation_band_audit.csv")
    rows = []
    for system_id in (2387, 5508):
        row = core_audit.loc[core_audit["lia_unique_id"].eq(system_id)].iloc[0]
        detailed = _read_json(CORE_ROOT / f"{system_id}/elevation_band_audit.json")
        checkpoint = detailed["zero_mb_checkpoints"][-1]
        steps = pd.read_csv(CORE_ROOT / f"{system_id}/semiimplicit_zero_mb_steps.csv")
        delta = pd.to_numeric(steps["actual_volume_change_m3"], errors="raise")
        initial_length_m = float(detailed["pseudo_flowline_length_m"])
        final_length_m = float(checkpoint["length_m"])
        volume_change = float(checkpoint["volume_change_fraction_from_initial"])
        terminal = float(checkpoint["terminal_thickness_m"])
        free_distance = float(checkpoint["downstream_free_distance_m"])
        zero_mb_pass = bool(
            abs(volume_change) <= ZERO_MB_VOLUME_TOLERANCE
            and terminal <= ZERO_MB_TERMINAL_THICKNESS_TOLERANCE_M
        )
        boundary_excluded = bool(
            terminal <= ZERO_MB_TERMINAL_THICKNESS_TOLERANCE_M
            and free_distance > 0
        )
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": detailed["rgi_id"],
                "source_dir": str((CORE_ROOT / str(system_id)).resolve()),
                "length_contract_pass": bool(row["length_contract_pass"]),
                "static_geometry_pass": bool(row["static_geometry_pass"]),
                "zero_mb_pass_recomputed": zero_mb_pass,
                "initial_area_error_fraction": float(row["area_error_fraction"]),
                "initial_volume_error_fraction": float(row["volume_error_fraction"]),
                "hypsometry_maximum_cumulative_area_error_fraction": float(
                    row["hypsometry_maximum_cumulative_area_error_fraction"]
                ),
                "longitudinal_scale_factor": float(
                    detailed["longitudinal_scale_factor"]
                ),
                "maximum_initial_thickness_m": float(
                    detailed["maximum_initial_thickness_m"]
                ),
                "zero_mb_years": float(checkpoint["year"]),
                "initial_length_m": initial_length_m,
                "final_length_m": final_length_m,
                "length_change_m": final_length_m - initial_length_m,
                "length_change_fraction": (
                    final_length_m - initial_length_m
                )
                / initial_length_m,
                "zero_mb_volume_change_fraction": volume_change,
                "zero_mb_area_change_fraction": float(
                    checkpoint["area_change_fraction_from_initial"]
                ),
                "zero_mb_volume_tolerance_fraction": ZERO_MB_VOLUME_TOLERANCE,
                "step_count": int(checkpoint["step_count"]),
                "positive_volume_change_steps": int((delta > 0).sum()),
                "negative_volume_change_steps": int((delta < 0).sum()),
                "zero_volume_change_steps": int((delta == 0).sum()),
                "summed_step_volume_change_m3": float(delta.sum()),
                "minimum_timestep_seconds": float(
                    checkpoint["minimum_timestep_seconds"]
                ),
                "downstream_free_distance_m": free_distance,
                "terminal_thickness_m": terminal,
                "boundary_flux_excluded_by_saved_evidence": boundary_excluded,
                "drift_direction": (
                    "systematic_positive_volume_drift"
                    if volume_change > ZERO_MB_VOLUME_TOLERANCE
                    else (
                        "systematic_negative_volume_drift"
                        if volume_change < -ZERO_MB_VOLUME_TOLERANCE
                        else "within_tolerance"
                    )
                ),
                "diagnostic_class": (
                    "severe_internal_geometry_dynamic_drift"
                    if abs(volume_change) >= 0.01
                    else "mild_but_formal_internal_geometry_dynamic_drift"
                ),
                "release_decision": "stop_before_spinup_repair_geometry_then_repeat_zero_mb",
            }
        )
    return pd.DataFrame(rows)


def build(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rescore_1044, source_manifest = _audit_1044()
    delta_history, delta_summary = _audit_delta_convergence()
    zero_mb = _audit_zero_mb()

    decisions = pd.concat(
        [
            rescore_1044[["lia_unique_id", "acceptance_state", "release_decision"]]
            .rename(columns={"acceptance_state": "diagnostic_class"}),
            delta_summary[
                ["lia_unique_id", "convergence_direction_class", "release_decision"]
            ].rename(
                columns={"convergence_direction_class": "diagnostic_class"}
            ),
            zero_mb[["lia_unique_id", "diagnostic_class", "release_decision"]],
        ],
        ignore_index=True,
    ).sort_values("lia_unique_id")

    _atomic_csv(output_dir / "q4_1044_current_gate_rescore.csv", rescore_1044)
    _atomic_csv(output_dir / "q4_1044_source_manifest.csv", source_manifest)
    _atomic_csv(output_dir / "q4_delta_convergence_history.csv", delta_history)
    _atomic_csv(
        output_dir / "q4_delta_convergence_system_summary.csv", delta_summary
    )
    _atomic_csv(output_dir / "q4_zero_mb_geometry_drift_audit.csv", zero_mb)
    _atomic_csv(output_dir / "q4_system_decisions.csv", decisions)

    summary = {
        "schema_version": "q4-preflight-audit-v1",
        "read_only": True,
        "oggm_advanced": False,
        "facts_file_read_required_before_execution": True,
        "system_count": 5,
        "systems": [1044, 2387, 2711, 3053, 5508],
        "system_1044": {
            "source_registered_for_audit": True,
            "source_registered_for_production": False,
            "raw_endpoint_gates_pass": bool(
                rescore_1044.iloc[0]["raw_endpoint_gates_pass"]
            ),
            "observation_domain_state": str(
                rescore_1044.iloc[0]["observation_domain_state"]
            ),
            "formal_accepted_current_gate": bool(
                rescore_1044.iloc[0]["formal_accepted_current_gate"]
            ),
            "decision": str(rescore_1044.iloc[0]["release_decision"]),
        },
        "systems_2711_3053": {
            "ten_point_history_complete_count": int(
                delta_summary["expected_ten_point_grid_complete"].sum()
            ),
            "all_nonconverged_count": int(delta_summary["all_nonconverged"].sum()),
            "classification": sorted(
                delta_summary["convergence_direction_class"].unique().tolist()
            ),
            "decision": "stop_delta_only_and_simple_extension_route_structural_state_diagnosis",
        },
        "systems_2387_5508": {
            "static_geometry_pass_count": int(
                zero_mb["static_geometry_pass"].sum()
            ),
            "zero_mb_pass_count": int(zero_mb["zero_mb_pass_recomputed"].sum()),
            "boundary_flux_excluded_count": int(
                zero_mb["boundary_flux_excluded_by_saved_evidence"].sum()
            ),
            "decision": "stop_both_before_spinup_repair_geometry_then_repeat_zero_mb",
        },
        "q4_release_count": int(
            decisions["release_decision"].astype(str).str.startswith("release").sum()
        ),
        "q4_stop_count": int(
            ~decisions["release_decision"].astype(str).str.startswith("release")
        ),
        "overall_decision": "q4_preflight_complete_no_system_released",
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
