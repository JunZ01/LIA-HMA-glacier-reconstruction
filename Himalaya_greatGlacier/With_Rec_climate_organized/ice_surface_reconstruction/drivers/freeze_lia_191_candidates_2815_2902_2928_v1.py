"""Freeze the user-reviewed batch-1 formal dual-pass curves."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


SYSTEMS = (2815, 2902, 2928)
EXPECTED_TASKS = {
    2815: "2815-B_formal_delta_bisection_spatial_rerun_v1-11230d542cf5171c",
    2902: "2902-B_formal_delta_bisection_spatial_rerun_v1-ec24fcef6bc9d348",
    2928: "2928-B_formal_delta_bisection_2928_v1-87fa80a56981cec1",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(frame: pd.DataFrame, column: str, value: object) -> pd.Series:
    selected = frame[frame[column].astype(str).eq(str(value))]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {column}={value}, found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    root = project_root()
    experiment_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_formal_delta_bisection_batch1_v1"
    )
    best = pd.read_csv(experiment_root / "best_per_system_formal.csv")
    best["lia_unique_id"] = pd.to_numeric(best["lia_unique_id"]).astype(int)
    registry_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_frozen_curve_registry_v1"
    )
    registry_path = registry_root / "frozen_candidates.csv"
    existing = pd.read_csv(registry_path)
    new_rows: list[dict[str, object]] = []

    for system_id in SYSTEMS:
        selected = _one(best, "lia_unique_id", system_id)
        if str(selected["task_id"]) != EXPECTED_TASKS[system_id]:
            raise ValueError(f"system {system_id} selected task changed")
        if not _as_bool(selected["formal_dual_pass"]):
            raise ValueError(f"system {system_id} is not a formal dual pass")
        if not _as_bool(selected["stable_valid"]):
            raise ValueError(f"system {system_id} is not stable and observation-valid")
        if _as_bool(selected["observation_domain_overflow"]):
            raise ValueError(f"system {system_id} overflows the observation domain")

        source_run = Path(str(selected["output_dir"]))
        summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
        task = json.loads((source_run / "task.json").read_text(encoding="utf-8"))
        timeseries = pd.read_csv(source_run / "timeseries.csv")
        if summary.get("status") != "ok" or int(task["lia_unique_id"]) != system_id:
            raise ValueError(f"system {system_id} source metadata mismatch")
        if str(task["task_id"]) != EXPECTED_TASKS[system_id]:
            raise ValueError(f"system {system_id} task metadata mismatch")
        observation = summary.get("observation_domain") or {}
        if _as_bool(observation.get("overflow", True)):
            raise ValueError(f"system {system_id} summary reports observation overflow")
        spinup = summary["spinup_control"]
        stability = float(spinup["final_stability_fraction"])
        if not _as_bool(spinup["converged"]) or stability > 0.02:
            raise ValueError(f"system {system_id} fails adaptive spin-up stability")

        for column in ("year", "area_m2", "volume_m3", "observation_length_m"):
            timeseries[column] = pd.to_numeric(timeseries[column], errors="raise")
        timeseries["raw_model_length_m"] = pd.to_numeric(
            timeseries.get("raw_model_length_m", timeseries["length_m"]), errors="raise"
        )
        forward = timeseries[timeseries["phase"].eq("forward")].copy()
        spinup_rows = timeseries[timeseries["phase"].eq("spinup")].copy()
        if forward["year"].min() != 1470 or forward["year"].max() != 2000:
            raise ValueError(f"system {system_id} does not cover 1470-2000")
        if forward["year"].nunique() != 54:
            raise ValueError(f"system {system_id} does not have 54 forward checkpoints")

        peak_row = forward.loc[forward["observation_length_m"].idxmax()]
        minimum_row = forward.loc[forward["observation_length_m"].idxmin()]
        endpoint_row = _one(forward, "year", 2000)
        spinup_end = spinup_rows.sort_values("year").iloc[-1]
        lia_target = float(summary["targets"]["lia_length_m"])
        rgi_target = float(summary["targets"]["rgi_length_m"])
        separation = abs(lia_target - rgi_target)
        peak_length = float(peak_row["observation_length_m"])
        endpoint_length = float(endpoint_row["observation_length_m"])
        e_lia_total = abs(peak_length - lia_target) / lia_target
        e_rgi_total = abs(endpoint_length - rgi_target) / rgi_target
        e_lia_sep = abs(peak_length - lia_target) / separation
        e_rgi_sep = abs(endpoint_length - rgi_target) / separation
        if max(e_lia_total, e_rgi_total) > 0.10:
            raise ValueError(f"system {system_id} fails the 10% total-length gate")
        if max(e_lia_sep, e_rgi_sep) > 0.20:
            raise ValueError(f"system {system_id} fails the 20% separation gate")
        for calculated, stored, label in (
            (peak_length, float(selected["forward_peak_m"]), "peak"),
            (endpoint_length, float(selected["forward_2000_m"]), "2000"),
            (e_lia_sep, float(selected["e_lia_sep"]), "LIA separation"),
            (e_rgi_sep, float(selected["e_rgi_sep"]), "RGI separation"),
        ):
            if not np.isclose(calculated, stored, rtol=0.0, atol=1e-10):
                raise ValueError(f"system {system_id} {label} does not reproduce score table")

        corridor_low = min(lia_target, rgi_target) - 0.20 * separation
        corridor_high = max(lia_target, rgi_target) + 0.20 * separation
        corridor_exceedance = np.maximum(
            np.maximum(corridor_low - forward["observation_length_m"], 0.0),
            np.maximum(forward["observation_length_m"] - corridor_high, 0.0),
        )
        output_root = registry_root / f"system_{system_id}"
        if output_root.exists() and any(output_root.iterdir()):
            existing_record = json.loads(
                (output_root / "candidate_record.json").read_text(encoding="utf-8")
            )
            if existing_record["source"]["task_id"] != EXPECTED_TASKS[system_id]:
                raise ValueError(f"system {system_id} freeze directory contains another task")
        output_root.mkdir(parents=True, exist_ok=True)
        copied_names = (
            "timeseries.csv",
            "summary.json",
            "task.json",
            "task_result.json",
            "spinup_state.pkl.gz",
            "state_2000.pkl.gz",
            "run.log",
        )
        copied_paths: list[Path] = []
        for name in copied_names:
            source = source_run / name
            if not source.exists():
                raise FileNotFoundError(source)
            destination = output_root / name
            shutil.copy2(source, destination)
            copied_paths.append(destination)
        source_plot = experiment_root / "plots" / f"system_{system_id}_best_spinup_forward_length.png"
        if not source_plot.exists():
            raise FileNotFoundError(source_plot)
        destination_plot = output_root / source_plot.name
        shutil.copy2(source_plot, destination_plot)
        copied_paths.append(destination_plot)

        compact = timeseries[
            [
                "phase",
                "year",
                "observation_length_m",
                "raw_model_length_m",
                "area_m2",
                "volume_m3",
                "observation_domain_overflow",
            ]
        ].copy()
        compact_path = output_root / "trajectory_length_area_volume.csv"
        compact.to_csv(compact_path, index=False)
        copied_paths.append(compact_path)
        parameters = summary["parameters"]
        record = {
            "schema_version": 1,
            "frozen_at": "2026-08-15",
            "lia_unique_id": system_id,
            "rgi_id": str(summary["rgi_id"]),
            "freeze_state": "user-reviewed-provisional-historical-curve",
            "user_review": (
                "The separate complete spin-up and 1470-2000 length plot was shown "
                "on 2026-08-15; the user replied 'ok' to freeze and continue."
            ),
            "source": {
                "experiment": "lia_191_formal_delta_bisection_batch1_v1",
                "geometry_group": str(selected["geometry_group"]),
                "task_id": EXPECTED_TASKS[system_id],
                "fingerprint": str(selected["fingerprint"]),
                "source_run": str(source_run),
            },
            "parameters": {
                "delta_t_k": float(parameters["delta_t_c"]),
                "melt_f": float(parameters["melt_f"]),
                "melt_f_stage2_multiplier": float(task["melt_f_stage2_multiplier"]),
                "prcp_fac": float(parameters["prcp_fac"]),
                "prcp_fac_multiplier": float(parameters["prcp_fac_multiplier"]),
                "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
                "glen_a_stage2_multiplier": float(task["glen_a_stage2_multiplier"]),
                "temp_melt_c": float(parameters["temp_melt_c"]),
                "precipitation_anomaly_mode": "off",
            },
            "forcing_and_geometry": {
                "spinup_window": [1441, 1470],
                "forward_period": [1470, 2000],
                "monthly_climatology": "GSWP3-W5E5 1961-1990",
                "temperature_reconstruction": "ensemble_rec_ref annual anomaly",
                "temperature_offset_policy": "constant through spin-up and forward",
                "geometry": "core-lia-plus-spatial-dem-downstream-v2",
                "geometry_audit": str(observation.get("source")),
                "observation_domain_node_count": int(observation["node_count"]),
                "observation_domain_overflow": False,
            },
            "hard_gates": {
                "technical_clean": True,
                "spinup_converged": True,
                "spinup_actual_years": int(spinup["actual_years"]),
                "spinup_stability_fraction": stability,
                "spinup_stability_limit": 0.02,
                "length_observation_valid": True,
            },
            "endpoint_scores": {
                "lia_target_m": lia_target,
                "rgi_target_m": rgi_target,
                "forward_peak_year": int(peak_row["year"]),
                "forward_peak_m": peak_length,
                "forward_2000_m": endpoint_length,
                "e_lia_total": e_lia_total,
                "e_rgi_total": e_rgi_total,
                "e_lia_sep": e_lia_sep,
                "e_rgi_sep": e_rgi_sep,
                "passes_total_length_10pct": True,
                "passes_separation_20pct": True,
            },
            "trajectory_diagnostics": {
                "forward_minimum_year": int(minimum_row["year"]),
                "forward_minimum_m": float(minimum_row["observation_length_m"]),
                "minimum_at_2000": bool(int(minimum_row["year"]) == 2000),
                "peak_to_2000_amplitude_m": peak_length - endpoint_length,
                "amplitude_ratio_to_target": (peak_length - endpoint_length) / separation,
                "corridor_low_m": corridor_low,
                "corridor_high_m": corridor_high,
                "maximum_corridor_exceedance_m": float(corridor_exceedance.max()),
            },
            "saved_states": {
                "spinup_end": {
                    "year": int(spinup_end["year"]),
                    "length_m": float(spinup_end["observation_length_m"]),
                    "area_m2": float(spinup_end["area_m2"]),
                    "volume_m3": float(spinup_end["volume_m3"]),
                },
                "forward_peak": {
                    "year": int(peak_row["year"]),
                    "length_m": peak_length,
                    "area_m2": float(peak_row["area_m2"]),
                    "volume_m3": float(peak_row["volume_m3"]),
                },
                "year_2000": {
                    "length_m": endpoint_length,
                    "area_m2": float(endpoint_row["area_m2"]),
                    "volume_m3": float(endpoint_row["volume_m3"]),
                },
            },
            "auxiliary_status": {
                "minimum_at_2000": bool(int(minimum_row["year"]) == 2000),
                "modern_mb_continuation": "not-run-for-this-candidate",
                "final_parameter_estimate": False,
            },
        }
        record_path = output_root / "candidate_record.json"
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        hashes = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in copied_paths
        }
        (output_root / "file_hashes.json").write_text(
            json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        new_rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": summary["rgi_id"],
                "freeze_state": record["freeze_state"],
                "source_experiment": record["source"]["experiment"],
                "selected_task_id": EXPECTED_TASKS[system_id],
                "delta_t_k": record["parameters"]["delta_t_k"],
                "melt_f_stage2_multiplier": record["parameters"]["melt_f_stage2_multiplier"],
                "glen_a_stage2_multiplier": record["parameters"]["glen_a_stage2_multiplier"],
                "e_lia_total": e_lia_total,
                "e_rgi_total": e_rgi_total,
                "e_lia_sep": e_lia_sep,
                "e_rgi_sep": e_rgi_sep,
                "spinup_stability_fraction": stability,
                "candidate_record": str(record_path),
            }
        )

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    combined["lia_unique_id"] = pd.to_numeric(combined["lia_unique_id"]).astype(int)
    combined = combined.sort_values("lia_unique_id").drop_duplicates("lia_unique_id", keep="last")
    expected_systems = {991, 1077, 1616, 2815, 2902, 2928}
    if set(combined["lia_unique_id"]) != expected_systems:
        raise ValueError(f"frozen registry systems changed: {set(combined['lia_unique_id'])}")
    combined.to_csv(registry_path, index=False)
    registry_summary = {
        "schema_version": 1,
        "scope": (
            "curves explicitly reviewed and frozen under both the 10% total-length "
            "and 20% separation endpoint gates"
        ),
        "candidate_count": len(expected_systems),
        "systems": sorted(expected_systems),
        "launched_oggm": False,
    }
    (registry_root / "registry_summary.json").write_text(
        json.dumps(registry_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(registry_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
