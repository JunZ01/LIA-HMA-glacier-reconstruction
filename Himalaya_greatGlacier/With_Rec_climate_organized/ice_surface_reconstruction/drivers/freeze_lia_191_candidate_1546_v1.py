"""Freeze the auto-approved C2 class-recenter curve for system 1546."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root
from .freeze_lia_191_candidate_2337_v1 import _as_bool, _sha256


SYSTEM_ID = 1546
TASK_ID = "1546-B_cls_C2_dt_rec_v1-614218f49edc6746"
EXPERIMENT = "lia_191_class_C2_delta_recenter_v1"


def _plot_curve(timeseries: pd.DataFrame, lia: float, rgi: float, output: Path) -> None:
    spin = timeseries.loc[timeseries["phase"].eq("spinup")].copy()
    forward = timeseries.loc[timeseries["phase"].eq("forward")].copy()
    separation = abs(lia - rgi)
    low = min(lia, rgi) - 0.20 * separation
    high = max(lia, rgi) + 0.20 * separation
    y_values = pd.concat(
        [spin["observation_length_m"], forward["observation_length_m"]]
    ).dropna()
    y_min = min(float(y_values.min()), low) - 0.08 * separation
    y_max = max(float(y_values.max()), high) + 0.08 * separation
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for axis, frame, title in (
        (axes[0], spin, "Adaptive spin-up"),
        (axes[1], forward, "Forward 1470-2000"),
    ):
        axis.axhspan(low / 1000, high / 1000, color="#d9ead3", alpha=0.45)
        axis.axhline(lia / 1000, color="#8b1a1a", linestyle="--", linewidth=1.4, label="LIA target")
        axis.axhline(rgi / 1000, color="#1f4e79", linestyle="--", linewidth=1.4, label="RGI target")
        axis.plot(
            frame["year"],
            frame["observation_length_m"] / 1000,
            color="#222222",
            linewidth=1.8,
        )
        axis.set_title(title)
        axis.set_xlabel("Model year" if title.startswith("Adaptive") else "Calendar year")
        axis.set_ylim(y_min / 1000, y_max / 1000)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observation-domain length (km)")
    axes[1].legend(loc="best", frameon=False)
    figure.suptitle("LIA system 1546 | C2 glen_a x0.75 + delta-T recenter")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    root = project_root()
    experiment_root = (
        root / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT
    )
    scores = pd.read_csv(experiment_root / "scores.csv")
    selected_rows = scores.loc[scores["task_id"].astype(str).eq(TASK_ID)]
    if len(selected_rows) != 1:
        raise ValueError(f"expected one selected task, found {len(selected_rows)}")
    selected = selected_rows.iloc[0]
    if int(selected["lia_unique_id"]) != SYSTEM_ID or not _as_bool(selected["accepted"]):
        raise ValueError("selected task is not the accepted system-1546 chain")
    if _as_bool(selected["observation_domain_overflow"]):
        raise ValueError("selected task overflows the observation domain")

    source_run = Path(str(selected["output_dir"]))
    summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    task = json.loads((source_run / "task.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(source_run / "timeseries.csv")
    if summary.get("status") != "ok" or str(task["task_id"]) != TASK_ID:
        raise ValueError("source metadata mismatch")
    observation = summary.get("observation_domain") or {}
    spinup = summary["spinup_control"]
    stability = float(spinup["final_stability_fraction"])
    if _as_bool(observation.get("overflow", True)):
        raise ValueError("source summary reports observation overflow")
    if not _as_bool(spinup["converged"]) or stability > 0.02:
        raise ValueError("source summary fails spin-up stability")

    for column in ("year", "area_m2", "volume_m3", "observation_length_m"):
        timeseries[column] = pd.to_numeric(timeseries[column], errors="raise")
    timeseries["raw_model_length_m"] = pd.to_numeric(
        timeseries.get("raw_model_length_m", timeseries["length_m"]), errors="raise"
    )
    forward = timeseries.loc[timeseries["phase"].eq("forward")].copy()
    spin_rows = timeseries.loc[timeseries["phase"].eq("spinup")].copy()
    if forward["year"].min() != 1470 or forward["year"].max() != 2000:
        raise ValueError("forward coverage changed")
    peak = forward.loc[forward["observation_length_m"].idxmax()]
    endpoint = forward.loc[forward["year"].eq(2000)].iloc[-1]
    minimum = forward.loc[forward["observation_length_m"].idxmin()]
    spin_end = spin_rows.sort_values("year").iloc[-1]
    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    separation = abs(lia - rgi)
    peak_m = float(peak["observation_length_m"])
    endpoint_m = float(endpoint["observation_length_m"])
    e_lia_total = abs(peak_m - lia) / lia
    e_rgi_total = abs(endpoint_m - rgi) / rgi
    e_lia_sep = abs(peak_m - lia) / separation
    e_rgi_sep = abs(endpoint_m - rgi) / separation
    if max(e_lia_total, e_rgi_total) > 0.10 or max(e_lia_sep, e_rgi_sep) > 0.20:
        raise ValueError("formal gates do not reproduce")

    registry_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_frozen_curve_registry_v1"
    )
    output_root = registry_root / f"system_{SYSTEM_ID}"
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("system 1546 already has a frozen package")
    output_root.mkdir(parents=True, exist_ok=True)

    plot_path = experiment_root / "plots" / "system_1546_spinup_forward_length.png"
    _plot_curve(timeseries, lia, rgi, plot_path)
    copied_paths: list[Path] = []
    for name in (
        "timeseries.csv",
        "summary.json",
        "task.json",
        "task_result.json",
        "spinup_state.pkl.gz",
        "state_2000.pkl.gz",
        "run.log",
    ):
        source = source_run / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output_root / name
        shutil.copy2(source, destination)
        copied_paths.append(destination)
    destination_plot = output_root / plot_path.name
    shutil.copy2(plot_path, destination_plot)
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

    corridor_low = min(lia, rgi) - 0.20 * separation
    corridor_high = max(lia, rgi) + 0.20 * separation
    exceedance = np.maximum(
        np.maximum(corridor_low - forward["observation_length_m"], 0.0),
        np.maximum(forward["observation_length_m"] - corridor_high, 0.0),
    )
    parameters = summary["parameters"]
    record = {
        "schema_version": 1,
        "frozen_at": "2026-08-15",
        "lia_unique_id": SYSTEM_ID,
        "rgi_id": str(summary["rgi_id"]),
        "freeze_state": "auto-approved-provisional-historical-curve",
        "user_review": (
            "The user enabled class-based continuous execution and automatic "
            "freezing of formal passes on 2026-08-15."
        ),
        "source": {
            "experiment": EXPERIMENT,
            "geometry_group": "core191-length-corrected",
            "task_id": TASK_ID,
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
            "geometry": "core191-length-corrected",
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
            "lia_target_m": lia,
            "rgi_target_m": rgi,
            "forward_peak_year": int(peak["year"]),
            "forward_peak_m": peak_m,
            "forward_2000_m": endpoint_m,
            "e_lia_total": e_lia_total,
            "e_rgi_total": e_rgi_total,
            "e_lia_sep": e_lia_sep,
            "e_rgi_sep": e_rgi_sep,
            "passes_total_length_10pct": True,
            "passes_separation_20pct": True,
        },
        "trajectory_diagnostics": {
            "forward_minimum_year": int(minimum["year"]),
            "forward_minimum_m": float(minimum["observation_length_m"]),
            "minimum_at_2000": bool(int(minimum["year"]) == 2000),
            "peak_to_2000_amplitude_m": peak_m - endpoint_m,
            "amplitude_ratio_to_target": (peak_m - endpoint_m) / separation,
            "corridor_low_m": corridor_low,
            "corridor_high_m": corridor_high,
            "maximum_corridor_exceedance_m": float(exceedance.max()),
        },
        "saved_states": {
            "spinup_end": {
                "year": int(spin_end["year"]),
                "length_m": float(spin_end["observation_length_m"]),
                "area_m2": float(spin_end["area_m2"]),
                "volume_m3": float(spin_end["volume_m3"]),
            },
            "forward_peak": {
                "year": int(peak["year"]),
                "length_m": peak_m,
                "area_m2": float(peak["area_m2"]),
                "volume_m3": float(peak["volume_m3"]),
            },
            "year_2000": {
                "length_m": endpoint_m,
                "area_m2": float(endpoint["area_m2"]),
                "volume_m3": float(endpoint["volume_m3"]),
            },
        },
        "auxiliary_status": {
            "minimum_at_2000": bool(int(minimum["year"]) == 2000),
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

    registry_path = registry_root / "frozen_candidates.csv"
    registry = pd.read_csv(registry_path)
    new_row = {
        "lia_unique_id": SYSTEM_ID,
        "rgi_id": summary["rgi_id"],
        "freeze_state": record["freeze_state"],
        "source_experiment": EXPERIMENT,
        "selected_task_id": TASK_ID,
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
    registry = pd.concat([registry, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    registry["lia_unique_id"] = pd.to_numeric(registry["lia_unique_id"]).astype(int)
    registry = registry.sort_values("lia_unique_id").drop_duplicates("lia_unique_id", keep="last")
    expected = {991, 1077, 1546, 1616, 2266, 2337, 2815, 2902, 2928, 2939, 4319}
    if set(registry["lia_unique_id"]) != expected:
        raise ValueError(f"frozen registry systems changed: {set(registry['lia_unique_id'])}")
    registry.to_csv(registry_path, index=False)
    registry_summary = {
        "schema_version": 1,
        "scope": (
            "reviewed or auto-approved provisional curves passing both the 10% "
            "total-length and 20% separation endpoint gates"
        ),
        "candidate_count": len(expected),
        "systems": sorted(expected),
        "launched_oggm": False,
    }
    (registry_root / "registry_summary.json").write_text(
        json.dumps(registry_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
