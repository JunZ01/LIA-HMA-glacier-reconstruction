"""Freeze formal split-forward-delta passes for systems 2569 and 2622."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


SELECTIONS = {
    2569: {"forward_delta_t_c": -0.515, "run_tag": "forward_dt_m0p515"},
    2622: {"forward_delta_t_c": -0.010, "run_tag": "forward_dt_m0p010"},
}
EXPERIMENT = "lia_191_split_forward_delta_pilot_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _source_multiplier(source_task: dict, name: str, default: float = 1.0) -> float:
    value = source_task.get(name, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float(default)
    return float(value)


def _plot(combined: pd.DataFrame, summary: dict, path: Path) -> None:
    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    spin = combined[combined["phase"].astype(str).eq("spinup")]
    forward = combined[combined["phase"].astype(str).eq("forward")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(spin["year"], spin["observation_length_m"] / 1000, lw=1.8, color="#2563eb")
    axes[0].axhline(lia / 1000, ls="--", color="#b91c1c", lw=1.2, label="LIA target")
    axes[0].set(title="Adaptive spin-up", xlabel="Model year", ylabel="Length (km)")
    axes[0].legend(frameon=False)
    axes[1].plot(forward["year"], forward["observation_length_m"] / 1000, lw=1.8, color="#2563eb")
    axes[1].axhline(lia / 1000, ls="--", color="#b91c1c", lw=1.2, label="LIA target")
    axes[1].axhline(rgi / 1000, ls=":", color="#111827", lw=1.3, label="RGI target")
    axes[1].set(title="Forward 1470-2000", xlabel="Year", ylabel="Length (km)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    params = summary["parameters"]
    fig.suptitle(
        f"LIA system {summary['lia_unique_id']} | spin-up ΔT={params['spinup_delta_t_c']:+.3f} K, "
        f"forward ΔT={params['forward_delta_t_c']:+.3f} K"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _freeze_one(root: Path, registry_root: Path, scores: pd.DataFrame, system_id: int, selection: dict) -> dict:
    selected_rows = scores[
        scores["lia_unique_id"].astype(int).eq(system_id)
        & np.isclose(
            pd.to_numeric(scores["forward_delta_t_c"], errors="coerce"),
            float(selection["forward_delta_t_c"]),
            atol=1e-12,
        )
    ]
    if len(selected_rows) != 1:
        raise ValueError(f"selected-score-row-count:{system_id}:{len(selected_rows)}")
    score = selected_rows.iloc[0]
    if not _as_bool(score["formal_dual_pass"]) or _as_bool(score["observation_domain_overflow"]):
        raise ValueError(f"selected-chain-fails-formal-gate:{system_id}")

    run_dir = Path(str(score["output_dir"]))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source_dir = Path(summary["source_dir"])
    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_task = json.loads((source_dir / "task.json").read_text(encoding="utf-8"))
    if summary["status"] != "ok" or not bool(summary["metrics"]["formal_dual_pass"]):
        raise ValueError(f"summary-formal-gate-fails:{system_id}")
    if bool(summary["observation_domain"]["overflow"]):
        raise ValueError(f"summary-observation-overflow:{system_id}")
    stability = float(summary["source_spinup_stability_fraction"])
    if stability > 0.02:
        raise ValueError(f"source-spinup-unstable:{system_id}")

    source_series = pd.read_csv(source_dir / "timeseries.csv")
    spin = source_series[source_series["phase"].astype(str).eq("spinup")].copy()
    if spin.empty:
        raise ValueError(f"missing-source-spinup-series:{system_id}")
    if "observation_length_m" not in spin.columns:
        spin["observation_length_m"] = pd.to_numeric(spin["length_m"], errors="raise")
    forward = pd.read_csv(run_dir / "timeseries.csv")
    combined = pd.concat([spin, forward], ignore_index=True, sort=False)
    combined["year"] = pd.to_numeric(combined["year"], errors="raise")
    combined["observation_length_m"] = pd.to_numeric(
        combined["observation_length_m"], errors="raise"
    )
    forward = combined[combined["phase"].astype(str).eq("forward")].copy()
    peak = forward.loc[forward["observation_length_m"].idxmax()]
    endpoint = forward[forward["year"].eq(2000)].iloc[-1]
    minimum = forward.loc[forward["observation_length_m"].idxmin()]
    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    separation = abs(lia - rgi)
    peak_m = float(peak["observation_length_m"])
    endpoint_m = float(endpoint["observation_length_m"])
    scores_recomputed = {
        "e_lia_total": abs(peak_m - lia) / lia,
        "e_rgi_total": abs(endpoint_m - rgi) / rgi,
        "e_lia_sep": abs(peak_m - lia) / separation,
        "e_rgi_sep": abs(endpoint_m - rgi) / separation,
    }
    if max(scores_recomputed["e_lia_total"], scores_recomputed["e_rgi_total"]) > 0.10:
        raise ValueError(f"total-length-gate-does-not-reproduce:{system_id}")
    if max(scores_recomputed["e_lia_sep"], scores_recomputed["e_rgi_sep"]) > 0.20:
        raise ValueError(f"separation-gate-does-not-reproduce:{system_id}")

    output = registry_root / f"system_{system_id}"
    if output.exists() and any(output.iterdir()):
        current = json.loads((output / "candidate_record.json").read_text(encoding="utf-8"))
        if float(current["parameters"].get("forward_delta_t_k", math.nan)) != float(selection["forward_delta_t_c"]):
            raise ValueError(f"freeze-directory-contains-different-candidate:{system_id}")
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "timeseries.csv", index=False)
    compact_columns = [
        "phase", "year", "observation_length_m", "raw_model_length_m",
        "area_m2", "volume_m3", "observation_domain_overflow",
    ]
    combined[compact_columns].to_csv(output / "trajectory_length_area_volume.csv", index=False)
    shutil.copy2(source_dir / "spinup_state.pkl.gz", output / "spinup_state.pkl.gz")
    shutil.copy2(run_dir / "state_2000.pkl.gz", output / "state_2000.pkl.gz")
    shutil.copy2(run_dir / "summary.json", output / "summary.json")
    shutil.copy2(source_dir / "summary.json", output / "source_spinup_summary.json")
    shutil.copy2(source_dir / "task.json", output / "source_spinup_task.json")
    task_id = (
        f"{system_id}-B_split_forward_delta_v1-spin_"
        f"{float(summary['parameters']['spinup_delta_t_c']):+.3f}-fwd_"
        f"{float(summary['parameters']['forward_delta_t_c']):+.3f}"
    )
    task_payload = {
        "task_id": task_id,
        "lia_unique_id": system_id,
        "experiment": EXPERIMENT,
        "source_spinup_task_id": summary["source_task_id"],
        "source_spinup_dir": str(source_dir),
        "forward_run_dir": str(run_dir),
        "spinup_delta_t_c": float(summary["parameters"]["spinup_delta_t_c"]),
        "forward_delta_t_c": float(summary["parameters"]["forward_delta_t_c"]),
        "spinup_recomputed": False,
        "physical_parameters_changed": False,
    }
    (output / "task.json").write_text(json.dumps(task_payload, indent=2), encoding="utf-8")
    plot_path = output / f"system_{system_id}_spinup_forward_length.png"
    _plot(combined, summary, plot_path)

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
        "lia_unique_id": system_id,
        "rgi_id": summary["rgi_id"],
        "freeze_state": "auto-approved-provisional-historical-curve",
        "user_review": (
            "The user enabled continuous batched execution and automatic freezing "
            "of formal passes on 2026-08-15."
        ),
        "source": {
            "experiment": EXPERIMENT,
            "task_id": task_id,
            "source_spinup_task_id": summary["source_task_id"],
            "source_spinup_run": str(source_dir),
            "forward_run": str(run_dir),
        },
        "parameters": {
            "delta_t_k": float(parameters["forward_delta_t_c"]),
            "spinup_delta_t_k": float(parameters["spinup_delta_t_c"]),
            "forward_delta_t_k": float(parameters["forward_delta_t_c"]),
            "temperature_offset_policy": "independent constant spin-up and forward offsets",
            "melt_f": float(parameters["melt_f"]),
            "melt_f_stage2_multiplier": _source_multiplier(source_task, "melt_f_stage2_multiplier"),
            "prcp_fac": float(parameters["prcp_fac"]),
            "prcp_fac_multiplier": _source_multiplier(source_task, "prcp_fac_multiplier"),
            "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
            "glen_a_stage2_multiplier": _source_multiplier(source_task, "glen_a_stage2_multiplier"),
            "temp_melt_c": float(parameters["temp_melt_c"]),
            "precipitation_anomaly_mode": "off",
        },
        "forcing_and_geometry": {
            "spinup_window": source_summary["spinup_window"],
            "forward_period": [1470, 2000],
            "monthly_climatology": "GSWP3-W5E5 1961-1990",
            "temperature_reconstruction": "ensemble_rec_ref annual anomaly",
            "observation_domain_source": summary["observation_domain"]["source"],
            "observation_domain_node_count": int(summary["observation_domain"]["node_count"]),
            "observation_domain_overflow": False,
        },
        "hard_gates": {
            "technical_clean": True,
            "spinup_converged": True,
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
            **scores_recomputed,
            "passes_total_length_10pct": True,
            "passes_separation_20pct": True,
        },
        "trajectory_diagnostics": {
            "forward_minimum_year": int(minimum["year"]),
            "forward_minimum_m": float(minimum["observation_length_m"]),
            "peak_to_2000_amplitude_m": peak_m - endpoint_m,
            "amplitude_ratio_to_target": (peak_m - endpoint_m) / separation,
            "corridor_low_m": corridor_low,
            "corridor_high_m": corridor_high,
            "maximum_corridor_exceedance_m": float(exceedance.max()),
        },
        "auxiliary_status": {
            "modern_mb_continuation": "not-run-for-this-candidate",
            "final_parameter_estimate": False,
        },
    }
    (output / "candidate_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    hashed = [
        output / name
        for name in (
            "timeseries.csv", "trajectory_length_area_volume.csv", "spinup_state.pkl.gz",
            "state_2000.pkl.gz", "summary.json", "source_spinup_summary.json",
            "source_spinup_task.json", "task.json", plot_path.name,
        )
    ]
    hashes = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in hashed
    }
    (output / "file_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "lia_unique_id": system_id,
        "rgi_id": summary["rgi_id"],
        "freeze_state": record["freeze_state"],
        "source_experiment": EXPERIMENT,
        "selected_task_id": task_id,
        "delta_t_k": float(parameters["forward_delta_t_c"]),
        "spinup_delta_t_k": float(parameters["spinup_delta_t_c"]),
        "forward_delta_t_k": float(parameters["forward_delta_t_c"]),
        "melt_f_stage2_multiplier": record["parameters"]["melt_f_stage2_multiplier"],
        "glen_a_stage2_multiplier": record["parameters"]["glen_a_stage2_multiplier"],
        **scores_recomputed,
        "spinup_stability_fraction": stability,
        "candidate_record": str((output / "candidate_record.json").resolve()),
    }


def main() -> None:
    root = project_root()
    experiment_root = root / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT
    scores = pd.read_csv(experiment_root / "scores.csv")
    registry_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1"
    new_rows = [
        _freeze_one(root, registry_root, scores, system_id, selection)
        for system_id, selection in SELECTIONS.items()
    ]
    registry_path = registry_root / "frozen_candidates.csv"
    registry = pd.read_csv(registry_path)
    registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    registry["lia_unique_id"] = pd.to_numeric(registry["lia_unique_id"]).astype(int)
    registry = registry.sort_values("lia_unique_id").drop_duplicates("lia_unique_id", keep="last")
    expected = {991, 1077, 1546, 1616, 2266, 2337, 2569, 2622, 2815, 2902, 2928, 2939, 4319}
    if set(registry["lia_unique_id"]) != expected:
        raise ValueError(f"frozen-registry-membership-changed:{set(registry['lia_unique_id'])}")
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
        json.dumps(registry_summary, indent=2) + "\n", encoding="utf-8"
    )

    mismatches = []
    for system_id in sorted(expected):
        system_root = registry_root / f"system_{system_id}"
        hash_path = system_root / "file_hashes.json"
        hashes = json.loads(hash_path.read_text(encoding="utf-8"))
        for name, metadata in hashes.items():
            path = system_root / name
            if (
                not path.exists()
                or path.stat().st_size != int(metadata["bytes"])
                or _sha256(path) != str(metadata["sha256"])
            ):
                mismatches.append(f"{system_id}:{name}")
    if mismatches:
        raise ValueError(f"registry-hash-validation-failed:{mismatches}")
    print(json.dumps({**registry_summary, "hash_mismatches": 0}, indent=2))


if __name__ == "__main__":
    main()
