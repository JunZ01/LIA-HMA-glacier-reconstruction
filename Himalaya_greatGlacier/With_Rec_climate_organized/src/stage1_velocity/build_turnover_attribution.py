"""Build a parallel, read-only imbalance/debris attribution preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("turnover_attribution")
SECONDS_PER_YEAR_DEFAULT = 365.25 * 24 * 60 * 60
DEBRIS_EXPLANATION_THRESHOLD = 0.10


def csv_bool(value: object) -> bool:
    """Parse CSV booleans without treating the string ``False`` as truthy."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            output_dir / "attribution.log", mode="w", encoding="utf-8"
        ),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def system_worker(task: tuple) -> tuple[str, int, list[dict], dict, dict | None]:
    """Compute one scenario/system group without touching project files."""
    scenario, system_id, records, params, settings = task
    try:
        density = float(settings["ice_density_kg_m3"])
        seconds = float(settings["seconds_per_year"])
        balance_factor = float(settings["balance_surface_factor"])
        sensitivity_factor = float(settings["sliding_sensitivity_surface_factor"])
        area_m2 = finite_float(params["modern_rgi_area_sum_km2"]) * 1e6
        target_mb = finite_float(params["hugonnet_mb_kg_m2_yr"])
        mb_error = (
            abs(finite_float(params["hugonnet_err_fully_correlated_mwea"])) * 1000
        )
        nominal_loss = max(0.0, -target_mb) * area_m2 / density / seconds
        conservative_loss = (
            max(0.0, -(target_mb - mb_error)) * area_m2 / density / seconds
        )
        output = []
        for row in records:
            section = finite_float(row["fixed_section_m2"])
            flux = finite_float(row["candidate_flux_m3_s"])
            observed = finite_float(row["observed_speed_median_m_per_year"])
            observed_low = finite_float(row["observed_speed_low_m_per_year"])
            observed_high = finite_float(row["observed_speed_high_m_per_year"])
            if section <= 0 or flux < 0:
                raise ValueError("non-positive section or negative candidate flux")
            upper_speed = flux / section * seconds * balance_factor
            lower_nominal = (
                max(0.0, flux - nominal_loss) / section * seconds * balance_factor
            )
            lower_extreme = (
                max(0.0, flux - conservative_loss)
                / section
                * seconds
                * balance_factor
            )
            upper_sensitivity = flux / section * seconds * sensitivity_factor
            lower_sensitivity = (
                max(0.0, flux - conservative_loss)
                / section
                * seconds
                * sensitivity_factor
            )
            elevation_offset = float(row["gate_surface_h_m"]) - float(
                row["zmed_m"]
            )
            position = float(row["gate_flowline_fraction"])
            if position < 1 / 3:
                position_bin = "upstream-third"
            elif position <= 2 / 3:
                position_bin = "middle-third"
            else:
                position_bin = "downstream-third"
            output.append(
                {
                    "scenario": scenario,
                    "lia_unique_id": int(system_id),
                    "rgi_id": row["rgi_id"],
                    "candidate_id": row["candidate_id"],
                    "qc_status": row["qc_status"],
                    "observed_speed_m_per_year": observed,
                    "observed_low_m_per_year": observed_low,
                    "observed_high_m_per_year": observed_high,
                    "balance_flux_m3_s": flux,
                    "fixed_section_m2": section,
                    "balance_modeled_speed_m_per_year": upper_speed,
                    "nominal_imbalance_lower_speed_m_per_year": lower_nominal,
                    "extreme_imbalance_lower_speed_m_per_year": lower_extreme,
                    "sliding_sensitivity_upper_speed_m_per_year": upper_sensitivity,
                    "sliding_sensitivity_lower_speed_m_per_year": lower_sensitivity,
                    "nominal_imbalance_loss_m3_s": nominal_loss,
                    "conservative_imbalance_loss_m3_s": conservative_loss,
                    "current_observed_to_modeled_ratio": finite_float(
                        row["observed_to_modeled_surface_ratio"]
                    ),
                    "inside_current_interval": csv_bool(
                        row["inside_observation_interval"]
                    ),
                    "inside_extreme_imbalance_envelope": bool(
                        lower_extreme <= observed <= upper_speed
                    ),
                    "inside_nominal_imbalance_envelope": bool(
                        lower_nominal <= observed <= upper_speed
                    ),
                    "inside_sliding_sensitivity_envelope": bool(
                        lower_sensitivity <= observed <= upper_sensitivity
                    ),
                    "inside_sliding_balance_observation_interval": bool(
                        observed_low <= upper_sensitivity <= observed_high
                    ),
                    "gate_flowline_fraction": position,
                    "gate_position_bin": position_bin,
                    "gate_surface_h_m": finite_float(row["gate_surface_h_m"]),
                    "zmed_m": finite_float(row["zmed_m"]),
                    "gate_elevation_offset_from_zmed_m": elevation_offset,
                    "main_flowline_length_m": finite_float(
                        row["main_flowline_length_m"]
                    ),
                    "debris_ratio_fraction": finite_float(
                        row["ratio_dc_fraction"]
                    ),
                    "debris_ratio_class": row["ratio_fraction_class"],
                    "debris_binary_status": row["debris_binary_status"],
                    "debris_ratio_ctsmean": finite_float(row["ratio_ctsmean"]),
                    "debris_observation_scope": "member-outline-proxy",
                    "region_label": row["region_label"],
                    "precip_prior_class": row["precip_prior_class"],
                    "hugonnet_mb_kg_m2_yr": target_mb,
                    "hugonnet_mb_error_kg_m2_yr": mb_error,
                    "modern_rgi_area_sum_km2": finite_float(
                        row["modern_rgi_area_sum_km2"]
                    ),
                    "turnover_diagnostic": row["turnover_diagnostic"],
                }
            )
        gate_frame = pd.DataFrame(output)
        pressure_list = params["candidate_edge_class"] == "low-turnover-edge"
        if gate_frame.empty:
            summary = {
                "scenario": scenario,
                "lia_unique_id": int(system_id),
                "gate_count": 0,
                "strong_gate_count": int(params["evaluable_strong_gate_count"]),
                "envelope_explains_count": 0,
                "nominal_envelope_explains_count": 0,
                "sliding_sensitivity_explains_count": 0,
                "attribution_class": "no-evaluable-strong-gate",
                "pressure_list": pressure_list,
                "three_way_attribution_class": "no-evaluable-strong-gate",
                "debris_ratio_fraction": params["debris_ratio_fraction"],
                "debris_ratio_class": params["debris_ratio_class"],
                "debris_observation_scope": "member-outline-proxy",
                "turnover_diagnostic": params["turnover_diagnostic"],
            }
        else:
            explains = int(gate_frame["inside_extreme_imbalance_envelope"].sum())
            nominal = int(gate_frame["inside_nominal_imbalance_envelope"].sum())
            sensitivity = int(
                gate_frame["inside_sliding_sensitivity_envelope"].sum()
            )
            count = len(gate_frame)
            if explains == count:
                attribution_class = "imbalance-envelope-explains-all"
            elif explains > 0:
                attribution_class = "partial-imbalance-explanation"
            else:
                attribution_class = "residual-outside-imbalance-envelope"
            if explains == count:
                three_way = "imbalance-envelope-explains"
            elif (
                pressure_list
                and params["debris_ratio_fraction"]
                >= float(settings["debris_explanation_threshold"])
            ):
                three_way = "debris-stratified-residual"
            elif pressure_list:
                three_way = "residual-outside-envelope"
            else:
                three_way = "not-pressure-list"
            lower_to_upper = (
                gate_frame["extreme_imbalance_lower_speed_m_per_year"]
                / gate_frame["balance_modeled_speed_m_per_year"].replace(0, np.nan)
            )
            summary = {
                "scenario": scenario,
                "lia_unique_id": int(system_id),
                "gate_count": count,
                "strong_gate_count": int(params["evaluable_strong_gate_count"]),
                "envelope_explains_count": explains,
                "nominal_envelope_explains_count": nominal,
                "sliding_sensitivity_explains_count": sensitivity,
                "attribution_class": attribution_class,
                "pressure_list": pressure_list,
                "three_way_attribution_class": three_way,
                "debris_ratio_fraction": params["debris_ratio_fraction"],
                "debris_ratio_class": params["debris_ratio_class"],
                "median_current_observed_to_modeled_ratio": finite_float(
                    gate_frame["current_observed_to_modeled_ratio"].median()
                ),
                "median_extreme_lower_to_upper_speed_ratio": finite_float(
                    lower_to_upper.median()
                ),
                "debris_class_mode": gate_frame["debris_ratio_class"].mode().iloc[0],
                "gate_position_mode": gate_frame["gate_position_bin"].mode().iloc[0],
                "debris_observation_scope": "member-outline-proxy",
                "turnover_diagnostic": params["turnover_diagnostic"],
            }
        return scenario, int(system_id), output, summary, None
    except Exception as exc:
        return scenario, int(system_id), [], None, {
            "scenario": scenario,
            "lia_unique_id": int(system_id),
            "stage": "attribution-worker",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def load_scenario_inputs(
    scenario: str,
    scan_dir: Path,
    manifest: pd.DataFrame,
    gate_stats: pd.DataFrame,
    debris: pd.DataFrame,
) -> tuple[list[tuple], int]:
    diagnostics = pd.read_csv(scan_dir / "system_diagnostics.csv")
    candidates = pd.read_csv(scan_dir / "gate_residuals.csv")
    selected = diagnostics[["lia_unique_id", "best_candidate_id"]].dropna()
    selected = selected.rename(columns={"best_candidate_id": "candidate_id"})
    gates = candidates.merge(
        selected, on=["lia_unique_id", "candidate_id"], validate="many_to_one"
    )
    gate_stats_fields = [
        "rgi_id",
        "gate_flowline_fraction",
        "gate_surface_h_m",
        "zmed_m",
        "main_flowline_length_m",
    ]
    gates = gates.merge(
        gate_stats[gate_stats_fields], on="rgi_id", validate="many_to_one"
    )
    debris_fields = [
        "rgi_id",
        "ratio_dc_fraction",
        "ratio_fraction_class",
        "debris_binary_status",
        "ratio_ctsmean",
    ]
    gates = gates.merge(
        debris[debris_fields], on="rgi_id", how="left", validate="many_to_one"
    )
    gates["ratio_fraction_class"] = gates["ratio_fraction_class"].fillna(
        "unknown"
    )
    gates["debris_binary_status"] = gates["debris_binary_status"].fillna(
        "unknown"
    )
    for field in ("ratio_dc_fraction", "ratio_ctsmean"):
        gates[field] = pd.to_numeric(gates[field], errors="coerce").fillna(0.0)
    system_fields = [
        "lia_unique_id",
        "region_label",
        "precip_prior_class",
        "hugonnet_mb_kg_m2_yr",
        "hugonnet_err_fully_correlated_mwea",
        "modern_rgi_area_sum_km2",
    ]
    gates = gates.merge(
        manifest.reset_index(drop=True)[system_fields],
        on="lia_unique_id",
        validate="many_to_one",
    )
    diag_fields = [
        "lia_unique_id",
        "evaluable_strong_gate_count",
        "turnover_diagnostic",
        "candidate_edge_class",
    ]
    diag = diagnostics[diag_fields].copy()
    gates = gates.merge(diag, on="lia_unique_id", validate="many_to_one")
    tasks = []
    groups = gates.groupby("lia_unique_id", sort=True)
    for row in diagnostics.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        if system_id in groups.groups:
            records = groups.get_group(system_id).to_dict("records")
        else:
            records = []
        params = {
            "modern_rgi_area_sum_km2": float(
                manifest.loc[system_id, "modern_rgi_area_sum_km2"]
            ),
            "hugonnet_mb_kg_m2_yr": float(
                manifest.loc[system_id, "hugonnet_mb_kg_m2_yr"]
            ),
            "hugonnet_err_fully_correlated_mwea": float(
                manifest.loc[
                    system_id, "hugonnet_err_fully_correlated_mwea"
                ]
            ),
            "evaluable_strong_gate_count": int(
                row.evaluable_strong_gate_count
            ),
            "turnover_diagnostic": row.turnover_diagnostic,
            "candidate_edge_class": row.candidate_edge_class,
            "debris_ratio_fraction": finite_float(row.debris_ratio_fraction),
            "debris_ratio_class": str(row.debris_ratio_class),
        }
        tasks.append((scenario, system_id, records, params, None))
    return tasks, len(gates)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    configure_logging(output_dir)
    started = time.time()
    workers = int(args.workers or config["workers"])
    settings = {
        "ice_density_kg_m3": float(config["ice_density_kg_m3"]),
        "seconds_per_year": float(
            config.get("seconds_per_year", SECONDS_PER_YEAR_DEFAULT)
        ),
        "balance_surface_factor": float(config["balance_surface_factor"]),
        "sliding_sensitivity_surface_factor": float(
            config["sliding_sensitivity_surface_factor"]
        ),
        "debris_explanation_threshold": float(
            config.get(
                "debris_explanation_threshold", DEBRIS_EXPLANATION_THRESHOLD
            )
        ),
    }
    manifest_path = (REPO_ROOT / config["manifest_path"]).resolve()
    manifest = pd.read_csv(manifest_path).set_index("lia_unique_id", drop=False)
    gate_path = (REPO_ROOT / config["velocity_gate_path"]).resolve()
    gate_stats = pd.read_csv(gate_path)
    debris_path = (REPO_ROOT / config["debris_registry_path"]).resolve()
    debris = pd.read_csv(debris_path)
    scenario_dirs = {
        "farinotti": (REPO_ROOT / config["farinotti_scan_dir"]).resolve(),
        "millan": (REPO_ROOT / config["millan_scan_dir"]).resolve(),
    }
    tasks = []
    input_rows = {}
    for scenario, scan_dir in scenario_dirs.items():
        scenario_tasks, row_count = load_scenario_inputs(
            scenario, scan_dir, manifest, gate_stats, debris
        )
        tasks.extend(
            (scenario, system_id, records, params, settings)
            for scenario, system_id, records, params, _ in scenario_tasks
        )
        input_rows[scenario] = row_count
    write_json_atomic(
        output_dir / "run_manifest.json",
        {
            "run_id": config["run_id"],
            "method_version": config["method_version"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(config_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "input_hashes": {
                "manifest": sha256(manifest_path),
                "velocity_gates": sha256(gate_path),
                "debris_registry": sha256(debris_path),
                "farinotti_diagnostics": sha256(
                    scenario_dirs["farinotti"] / "system_diagnostics.csv"
                ),
                "millan_diagnostics": sha256(
                    scenario_dirs["millan"] / "system_diagnostics.csv"
                ),
            },
            "worker_count": workers,
            "worker_write_policy": "workers-return-rows-parent-writes",
            "debris_explanation_threshold": settings[
                "debris_explanation_threshold"
            ],
            "input_gate_rows": input_rows,
            "output_dir": str(output_dir),
        },
    )

    gate_rows: list[dict] = []
    system_rows: list[dict] = []
    failures: list[dict] = []
    total_tasks = len(tasks)
    completed = 0
    write_json_atomic(
        output_dir / "progress.json",
        {
            "run_id": config["run_id"],
            "phase": "attribution-preflight",
            "state": "running",
            "completed_task_count": 0,
            "total_task_count": total_tasks,
            "completed_system_scenario_count": 0,
            "total_system_scenario_count": total_tasks,
            "failure_count": 0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(system_worker, task): (task[0], task[1]) for task in tasks
        }
        for future in as_completed(futures):
            task_scenario, task_system_id = futures[future]
            try:
                scenario, system_id, rows, summary, failure = future.result()
            except Exception as exc:
                scenario, system_id, rows, summary = (
                    task_scenario,
                    task_system_id,
                    [],
                    None,
                )
                failure = {
                    "scenario": scenario,
                    "lia_unique_id": system_id,
                    "stage": "process-pool",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            gate_rows.extend(rows)
            if summary is not None:
                system_rows.append(summary)
            if failure is not None:
                failures.append(failure)
            completed += 1
            write_json_atomic(
                output_dir / "progress.json",
                {
                    "run_id": config["run_id"],
                    "phase": "attribution-preflight",
                    "state": "running" if completed < total_tasks else "complete",
                    "completed_task_count": completed,
                    "total_task_count": total_tasks,
                    "completed_system_scenario_count": completed,
                    "total_system_scenario_count": total_tasks,
                    "last_scenario": scenario,
                    "last_system_id": system_id,
                    "failure_count": len(failures),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if completed % 20 == 0 or completed == total_tasks:
                LOGGER.info(
                    "Progress %d/%d (last=%s/%d)",
                    completed,
                    total_tasks,
                    scenario,
                    system_id,
                )

    gate_frame = pd.DataFrame(gate_rows)
    system_frame = pd.DataFrame(system_rows)
    failure_frame = pd.DataFrame(
        failures,
        columns=["scenario", "lia_unique_id", "stage", "error_type", "error"],
    )
    gate_frame.to_csv(output_dir / "gate_attribution.csv", index=False)
    system_frame.to_csv(output_dir / "system_attribution.csv", index=False)
    failure_frame.to_csv(output_dir / "failures.csv", index=False)
    pressure_frame = system_frame.loc[system_frame["pressure_list"]].copy()
    pressure_frame.to_csv(
        output_dir / "pressure_list_attribution.csv", index=False
    )
    by_scenario = {}
    for scenario, frame in system_frame.groupby("scenario"):
        by_scenario[scenario] = frame["attribution_class"].value_counts().to_dict()
    three_way_by_scenario = {}
    for scenario, frame in pressure_frame.groupby("scenario"):
        three_way_by_scenario[scenario] = (
            frame["three_way_attribution_class"].value_counts().to_dict()
        )
    by_debris = (
        gate_frame.groupby(["scenario", "debris_ratio_class"])
        .agg(
            gate_count=("rgi_id", "size"),
            envelope_explains=("inside_extreme_imbalance_envelope", "sum"),
            median_ratio=("current_observed_to_modeled_ratio", "median"),
        )
        .reset_index()
    )
    by_debris.to_csv(output_dir / "attribution_by_debris_class.csv", index=False)
    by_position = (
        gate_frame.groupby(["scenario", "gate_position_bin"])
        .agg(
            gate_count=("rgi_id", "size"),
            envelope_explains=("inside_extreme_imbalance_envelope", "sum"),
            median_ratio=("current_observed_to_modeled_ratio", "median"),
        )
        .reset_index()
    )
    by_position.to_csv(output_dir / "attribution_by_gate_position.csv", index=False)
    surface_factor_sensitivity = {}
    for scenario, frame in gate_frame.groupby("scenario"):
        scenario_systems = system_frame.loc[
            (system_frame["scenario"] == scenario)
            & (system_frame["gate_count"] > 0)
        ]
        current_systems = int(
            scenario_systems["envelope_explains_count"]
            .eq(scenario_systems["gate_count"])
            .sum()
        )
        sensitivity_systems = int(
            scenario_systems["sliding_sensitivity_explains_count"]
            .eq(scenario_systems["gate_count"])
            .sum()
        )
        sensitivity_ratio = (
            frame["observed_speed_m_per_year"]
            / frame["sliding_sensitivity_upper_speed_m_per_year"].replace(
                0, np.nan
            )
        )
        surface_factor_sensitivity[scenario] = {
            "gate_count": int(len(frame)),
            "balance_observation_interval_gate_count_factor_1p25": int(
                frame["inside_current_interval"].sum()
            ),
            "balance_observation_interval_gate_count_factor_1p00": int(
                frame["inside_sliding_balance_observation_interval"].sum()
            ),
            "extreme_envelope_gate_count_factor_1p25": int(
                frame["inside_extreme_imbalance_envelope"].sum()
            ),
            "extreme_envelope_gate_count_factor_1p00": int(
                frame["inside_sliding_sensitivity_envelope"].sum()
            ),
            "extreme_envelope_all_gate_system_count_factor_1p25": current_systems,
            "extreme_envelope_all_gate_system_count_factor_1p00": sensitivity_systems,
            "median_observed_to_balance_speed_ratio_factor_1p25": finite_float(
                frame["current_observed_to_modeled_ratio"].median()
            ),
            "median_observed_to_balance_speed_ratio_factor_1p00": finite_float(
                sensitivity_ratio.median()
            ),
        }
    summary = {
        "run_id": config["run_id"],
        "status": "completed-attribution-preflight-not-final-correction",
        "task_count": total_tasks,
        "worker_count": workers,
        "gate_row_count": len(gate_frame),
        "system_scenario_row_count": len(system_frame),
        "failure_count": len(failure_frame),
        "failures_by_type": failure_frame["error_type"].value_counts().to_dict()
        if not failure_frame.empty
        else {},
        "attribution_class_by_scenario": by_scenario,
        "pressure_list_system_scenario_count": len(pressure_frame),
        "three_way_pressure_list_by_scenario": three_way_by_scenario,
        "surface_factor_sensitivity_by_scenario": surface_factor_sensitivity,
        "debris_explanation_threshold": settings[
            "debris_explanation_threshold"
        ],
        "input_gate_rows": input_rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "scope_note": "debris is a member-outline proxy; imbalance is an extreme diagnostic envelope",
    }
    write_json_atomic(output_dir / "attribution_summary.json", summary)
    LOGGER.info("Completed %s", json.dumps(summary, ensure_ascii=True))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
