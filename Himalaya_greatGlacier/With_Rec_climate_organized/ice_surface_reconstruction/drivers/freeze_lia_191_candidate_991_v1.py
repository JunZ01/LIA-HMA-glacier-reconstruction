"""Freeze the user-reviewed provisional historical curve for LIA system 991.

This is a read-only validation of the completed OGGM chain followed by a
small, versioned copy of the selected states and trajectory.  It launches no
glacier model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


SOURCE_NAME = "lia_191_finish_refine_991_v2"
SELECTED_TASK_ID = "991-B_finish_refine_991_v2-703e4cca9afaba39"
REGISTRY_NAME = "lia_191_frozen_curve_registry_v1"
SYSTEM_ID = 991


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


def _one(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame[frame[column].astype(str).eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {column}={value}, found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    root = project_root()
    source_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / SOURCE_NAME
    )
    registry_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / REGISTRY_NAME
    )
    output_root = registry_root / f"system_{SYSTEM_ID}"

    selection = json.loads((source_root / "selection_summary.json").read_text(encoding="utf-8"))
    if selection["selected_task_id"] != SELECTED_TASK_ID:
        raise ValueError("selection_summary.json points to a different task")

    manifest_row = _one(
        pd.read_csv(source_root / "task_manifest.csv"), "task_id", SELECTED_TASK_ID
    )
    score_row = _one(pd.read_csv(source_root / "scores.csv"), "task_id", SELECTED_TASK_ID)
    source_run = Path(str(manifest_row["output_dir"]))
    summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(source_run / "timeseries.csv")

    if summary.get("status") != "ok" or _as_bool(score_row["technical_failure"]):
        raise ValueError("selected chain is not technically clean")
    spinup = summary["spinup_control"]
    if not _as_bool(spinup["converged"]):
        raise ValueError("selected chain did not converge")
    stability = float(spinup["final_stability_fraction"])
    if stability > 0.02:
        raise ValueError(f"spin-up stability gate failed: {stability}")
    if not _as_bool(score_row["length_observation_valid"]):
        raise ValueError("selected chain does not have a valid observation length")
    if _as_bool(score_row["observation_domain_overflow"]) or _as_bool(
        score_row["observation_domain_unresolved"]
    ):
        raise ValueError("selected chain fails the observation-domain gate")

    for column in ("year", "area_m2", "volume_m3", "observation_length_m"):
        timeseries[column] = pd.to_numeric(timeseries[column], errors="raise")
    forward = timeseries[timeseries["phase"].eq("forward")].copy()
    spinup_rows = timeseries[timeseries["phase"].eq("spinup")].copy()
    if forward["year"].min() != 1470 or forward["year"].max() != 2000:
        raise ValueError("forward trajectory does not cover 1470-2000")
    if forward["year"].nunique() != 54:
        raise ValueError("expected 54 decadal forward checkpoints")

    peak_row = forward.loc[forward["observation_length_m"].idxmax()]
    minimum_row = forward.loc[forward["observation_length_m"].idxmin()]
    endpoint_row = _one(forward, "year", "2000")
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
    if e_lia_total > 0.10 or e_rgi_total > 0.10:
        raise ValueError("selected chain fails the active 10% total-length gate")
    if e_lia_sep > 0.20 or e_rgi_sep > 0.20:
        raise ValueError("selected chain fails the formal 20% separation gate")

    corridor_low = min(lia_target, rgi_target) - 0.20 * separation
    corridor_high = max(lia_target, rgi_target) + 0.20 * separation
    corridor_exceedance = np.maximum(
        np.maximum(corridor_low - forward["observation_length_m"], 0.0),
        np.maximum(forward["observation_length_m"] - corridor_high, 0.0),
    )

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
    compact.to_csv(output_root / "trajectory_length_area_volume.csv", index=False)
    copied_paths.append(output_root / "trajectory_length_area_volume.csv")

    record = {
        "schema_version": 1,
        "frozen_at": "2026-08-14",
        "lia_unique_id": SYSTEM_ID,
        "rgi_id": str(summary["rgi_id"]),
        "freeze_state": "user-reviewed-provisional-historical-curve",
        "user_review": "The displayed complete spin-up and 1470-2000 length curve was explicitly accepted for continuation on 2026-08-14.",
        "source": {
            "experiment": SOURCE_NAME,
            "task_id": SELECTED_TASK_ID,
            "fingerprint": str(manifest_row["fingerprint"]),
            "source_run": str(source_run),
        },
        "parameters": {
            "delta_t_k": float(summary["parameters"]["delta_t_c"]),
            "melt_f": float(summary["parameters"]["melt_f"]),
            "melt_f_stage2_multiplier": float(manifest_row["melt_f_stage2_multiplier"]),
            "prcp_fac": float(summary["parameters"]["prcp_fac"]),
            "prcp_fac_multiplier": float(summary["parameters"]["prcp_fac_multiplier"]),
            "glen_a_pa_3_s_1": float(summary["parameters"]["glen_a_pa_3_s_1"]),
            "glen_a_stage2_multiplier": float(manifest_row["glen_a_stage2_multiplier"]),
            "temp_melt_c": float(summary["parameters"]["temp_melt_c"]),
            "precipitation_anomaly_mode": "off",
        },
        "forcing_and_geometry": {
            "spinup_window": [1441, 1470],
            "forward_period": [1470, 2000],
            "monthly_climatology": "GSWP3-W5E5 1961-1990",
            "temperature_reconstruction": "ensemble_rec_ref annual anomaly",
            "temperature_offset_policy": "constant through spin-up and forward",
            "geometry": "corrected Core-191 elevation-band input",
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
            "minimum_at_2000": False,
            "modern_mb_continuation": "not-run-for-this-candidate",
            "final_parameter_estimate": False,
        },
    }
    (output_root / "candidate_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    hashes = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in copied_paths
    }
    (output_root / "file_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    registry = pd.DataFrame(
        [
            {
                "lia_unique_id": SYSTEM_ID,
                "rgi_id": summary["rgi_id"],
                "freeze_state": record["freeze_state"],
                "source_experiment": SOURCE_NAME,
                "selected_task_id": SELECTED_TASK_ID,
                "delta_t_k": record["parameters"]["delta_t_k"],
                "melt_f_stage2_multiplier": record["parameters"]["melt_f_stage2_multiplier"],
                "glen_a_stage2_multiplier": record["parameters"]["glen_a_stage2_multiplier"],
                "e_lia_total": e_lia_total,
                "e_rgi_total": e_rgi_total,
                "e_lia_sep": e_lia_sep,
                "e_rgi_sep": e_rgi_sep,
                "spinup_stability_fraction": stability,
                "candidate_record": str(output_root / "candidate_record.json"),
            }
        ]
    )
    registry.to_csv(registry_root / "frozen_candidates.csv", index=False)
    registry_summary = {
        "schema_version": 1,
        "scope": "new curves explicitly reviewed and frozen under the current dual endpoint diagnostics; this is not a registry of every older provisional candidate",
        "candidate_count": 1,
        "systems": [SYSTEM_ID],
        "launched_oggm": False,
    }
    (registry_root / "registry_summary.json").write_text(
        json.dumps(registry_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
