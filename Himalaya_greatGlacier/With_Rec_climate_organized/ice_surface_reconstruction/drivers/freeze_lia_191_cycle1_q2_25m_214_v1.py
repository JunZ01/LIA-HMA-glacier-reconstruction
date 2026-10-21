"""Freeze the formally accepted Cycle-1 Q2 25 m curve for system 214."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
SOURCE = (
    BASE
    / "lia_191_continuous_class_controller_v1"
    / "cycle1"
    / "Q2_25m_promotion35_v1"
    / "system_214"
)
BASE_RUN = SOURCE / "base_spinup_and_forward"
FORWARD_RUN = SOURCE / "registered_forward"
REGISTRY = BASE / "lia_191_frozen_curve_registry_v1"
DEST = REGISTRY / "system_214"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    registry_path = REGISTRY / "frozen_candidates.csv"
    registry = pd.read_csv(registry_path)
    frozen_ids = set(registry["lia_unique_id"].astype(int))
    if len(frozen_ids) != 21 or 214 in frozen_ids:
        raise RuntimeError(f"unexpected-frozen-registry-before-214:{sorted(frozen_ids)}")
    if DEST.exists():
        raise RuntimeError(f"freeze-destination-already-exists:{DEST}")

    base = json.loads((BASE_RUN / "summary.json").read_text(encoding="utf-8"))
    forward = json.loads((FORWARD_RUN / "summary.json").read_text(encoding="utf-8"))
    metrics = forward["metrics"]
    if not (
        base.get("status") == "ok"
        and bool(base["spinup_control"]["converged"])
        and float(base["spinup_control"]["final_stability_fraction"]) <= 0.02
        and not bool(base["observation_domain"]["overflow"])
        and forward.get("status") == "ok"
        and not bool(forward["observation_domain"]["overflow"])
        and bool(metrics["total_length_pass"])
        and bool(metrics["formal_separation_pass"])
        and bool(metrics["formal_dual_pass"])
    ):
        raise RuntimeError("system-214-formal-gates-not-satisfied")

    DEST.mkdir(parents=True)
    copies = {
        BASE_RUN / "spinup_state.pkl.gz": DEST / "spinup_state.pkl.gz",
        FORWARD_RUN / "state_2000.pkl.gz": DEST / "state_2000.pkl.gz",
        FORWARD_RUN / "timeseries.csv": DEST / "timeseries.csv",
        FORWARD_RUN / "summary.json": DEST / "summary.json",
        BASE_RUN / "summary.json": DEST / "source_spinup_summary.json",
        BASE_RUN / "task.json": DEST / "source_spinup_task.json",
    }
    for source, target in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

    task = {
        "schema_version": 1,
        "experiment": "cycle1-q2-25m-historical-v1",
        "lia_unique_id": 214,
        "source_spinup_run": str(BASE_RUN),
        "forward_run": str(FORWARD_RUN),
        "spinup_delta_t_c": forward["parameters"]["spinup_delta_t_c"],
        "forward_delta_t_c": forward["parameters"]["forward_delta_t_c"],
        "geometry_resolution_m": 25.0,
    }
    (DEST / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    timeseries = pd.read_csv(DEST / "timeseries.csv")
    timeseries.to_csv(DEST / "trajectory_length_area_volume.csv", index=False)
    target = forward["targets"]
    peak = float(metrics["forward_peak_m"])
    end = float(metrics["forward_2000_m"])
    separation = float(target["separation_m"])
    record = {
        "schema_version": 1,
        "frozen_at": str(date.today()),
        "lia_unique_id": 214,
        "rgi_id": forward["rgi_id"],
        "freeze_state": "auto-approved-provisional-historical-curve",
        "user_review": "Continuous class execution authorized automatic freezing of formal passes.",
        "source": {
            "experiment": "cycle1-q2-25m-historical-v1",
            "source_spinup_run": str(BASE_RUN),
            "forward_run": str(FORWARD_RUN),
        },
        "parameters": {
            **forward["parameters"],
            "temperature_offset_policy": "independent constant spin-up and forward offsets",
        },
        "forcing_and_geometry": {
            "spinup_window": base["spinup_window"],
            "forward_period": [1470, 2000],
            "monthly_climatology": "GSWP3-W5E5 1961-1990",
            "temperature_reconstruction": "ensemble_rec_ref annual anomaly",
            "node_spacing_m": 25.0,
            "observation_domain_source": forward["observation_domain"]["source"],
            "observation_domain_node_count": forward["observation_domain"]["node_count"],
            "observation_domain_overflow": False,
        },
        "hard_gates": {
            "technical_clean": True,
            "spinup_converged": True,
            "spinup_stability_fraction": base["spinup_control"]["final_stability_fraction"],
            "spinup_stability_limit": 0.02,
            "length_observation_valid": True,
        },
        "endpoint_scores": {
            "lia_target_m": target["lia_length_m"],
            "rgi_target_m": target["rgi_length_m"],
            "forward_peak_year": metrics["forward_peak_year"],
            "forward_peak_m": peak,
            "forward_2000_m": end,
            "e_lia_total": metrics["e_lia_total"],
            "e_rgi_total": metrics["e_rgi_total"],
            "e_lia_sep": metrics["e_lia_sep"],
            "e_rgi_sep": metrics["e_rgi_sep"],
            "passes_total_length_10pct": True,
            "passes_separation_20pct": True,
        },
        "trajectory_diagnostics": {
            "peak_to_2000_amplitude_m": peak - end,
            "amplitude_ratio_to_target": (peak - end) / separation,
        },
        "auxiliary_status": {
            "modern_mb_continuation": "not-run-for-this-candidate",
            "final_parameter_estimate": False,
        },
    }
    (DEST / "candidate_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    hashed_names = [
        "candidate_record.json",
        "source_spinup_summary.json",
        "source_spinup_task.json",
        "spinup_state.pkl.gz",
        "state_2000.pkl.gz",
        "summary.json",
        "task.json",
        "timeseries.csv",
        "trajectory_length_area_volume.csv",
    ]
    hashes = {
        name: {"bytes": (DEST / name).stat().st_size, "sha256": _sha256(DEST / name)}
        for name in hashed_names
    }
    (DEST / "file_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )

    row = {
        "lia_unique_id": 214,
        "rgi_id": forward["rgi_id"],
        "freeze_state": "auto-approved-provisional-historical-curve",
        "source_experiment": "cycle1-q2-25m-historical-v1",
        "selected_task_id": "214-cycle1-q2-25m-spin-m0p625-fwd-m0p635",
        "delta_t_k": forward["parameters"]["forward_delta_t_c"],
        "melt_f_stage2_multiplier": 1.0,
        "glen_a_stage2_multiplier": 1.0,
        "e_lia_total": metrics["e_lia_total"],
        "e_rgi_total": metrics["e_rgi_total"],
        "e_lia_sep": metrics["e_lia_sep"],
        "e_rgi_sep": metrics["e_rgi_sep"],
        "spinup_stability_fraction": base["spinup_control"]["final_stability_fraction"],
        "candidate_record": str(DEST / "candidate_record.json"),
        "spinup_delta_t_k": forward["parameters"]["spinup_delta_t_c"],
        "forward_delta_t_k": forward["parameters"]["forward_delta_t_c"],
    }
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True, sort=False)
    registry["lia_unique_id"] = registry["lia_unique_id"].astype(int)
    registry = registry.sort_values("lia_unique_id").drop_duplicates("lia_unique_id", keep="last")
    if len(registry) != 22 or 214 not in set(registry["lia_unique_id"]):
        raise RuntimeError("frozen-registry-214-append-failed")
    registry.to_csv(registry_path, index=False)
    registry_summary = {
        "schema_version": 1,
        "candidate_count": 22,
        "systems": sorted(registry["lia_unique_id"].astype(int).tolist()),
        "launched_oggm": False,
    }
    (REGISTRY / "registry_summary.json").write_text(
        json.dumps(registry_summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, metadata in hashes.items():
        path = DEST / name
        if path.stat().st_size != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"hash-validation-failed:{name}")
    print(json.dumps({**registry_summary, "newly_frozen": [214], "hash_mismatches": 0}, indent=2))


if __name__ == "__main__":
    main()
