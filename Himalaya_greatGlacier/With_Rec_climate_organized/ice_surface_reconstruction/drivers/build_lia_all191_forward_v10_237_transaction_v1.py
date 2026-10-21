"""Freeze the authorized 530-cell v10.237 one-dimensional forward transaction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
CANDIDATE_DIR = BASE / "lia_191_year900_forward_candidate_manifest_v3"
SOURCE_TX = BASE / "lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
CURVE_STANDARD = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
OUT = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1"
OUTPUT_ROOT = ROOT / "w237f"
WORKER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_v10_237.py"
CONTROLLER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_cpu12_controller_v10_237.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_hash(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUT.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError("v10.237 transaction or output root already exists")
    candidates = pd.read_csv(CANDIDATE_DIR / "forward_candidate_manifest.csv")
    if len(candidates) != 530 or candidates["lia_unique_id"].nunique() != 187:
        raise RuntimeError("candidate manifest is not exactly 530 cells / 187 systems")
    source = read_json(SOURCE_TX / "transaction_manifest.json")
    input_binding = Path(source["inputs"]["input_binding"])
    curve_standard_hash = sha256(CURVE_STANDARD)
    candidate_hash = sha256(CANDIDATE_DIR / "forward_candidate_manifest.csv")

    topology_sizes = []
    for row in candidates.itertuples(index=False):
        checkpoint = read_json(Path(row.confirmation_checkpoint_manifest_path))
        topology_sizes.append(sum(int(item["nx"]) for item in checkpoint["topology"]["flowlines"]))
    candidates["topology_total_nodes"] = topology_sizes
    order = candidates.sort_values(["topology_total_nodes", "lia_unique_id", "delta_t_eq_degC"]).reset_index(drop=True)
    representative_indices = [0, len(order) // 2, len(order) - 1]
    representative_ordinals = set(int(order.iloc[index].forward_candidate_ordinal) for index in representative_indices)

    rows = []
    for index, row in candidates.sort_values("forward_candidate_ordinal").iterrows():
        ordinal = int(row.forward_candidate_ordinal)
        payload = {
            "schema": "lia-all191-flowline-forward-fingerprint-v10.237",
            "candidate_manifest_sha256": candidate_hash,
            "source_checkpoint_state_sha256": row.confirmation_checkpoint_state_sha256,
            "source_checkpoint_manifest_sha256": row.confirmation_checkpoint_manifest_sha256,
            "lia_unique_id": int(row.lia_unique_id), "rgi_id": row.rgi_id,
            "parameters": {
                "delta_t_eq_degC": float(row.delta_t_eq_degC),
                "base_temp_bias_degC": float(row.base_temp_bias_degC),
                "effective_temp_bias_degC": float(row.effective_temp_bias_degC),
                "melt_f": float(row.melt_f), "prcp_fac": float(row.prcp_fac),
                "glen_a_pa_3_s_1": float(row.glen_a_pa_3_s_1), "fs": float(row.fs),
                "temp_melt_degC": float(row.temp_melt_degC),
            },
            "model": "OGGM-SemiImplicitModel-one-dimensional",
            "calendar_years": [1470, 2000], "fixed_timestep_days": 1.25,
            "forcing": "W5E5-1961-1990-plus-immutable-reconstruction-temperature-anomaly",
            "precipitation_anomaly_mode": "off", "extra_endpoint_ramp": None,
            "checkpoint_interval_years": 1, "curve_standard_sha256": curve_standard_hash,
        }
        item = row.to_dict()
        item["forward_fingerprint"] = canonical_hash(payload)
        item["production_member_key"] = f"f{ordinal - 1:04d}"
        item["preflight_member_key"] = (
            f"p{sorted(representative_ordinals).index(ordinal):04d}" if ordinal in representative_ordinals else ""
        )
        item["forward_start_year"] = 1470; item["forward_end_year"] = 2000
        item["annual_checkpoint_interval_years"] = 1
        item["forward_launch_authorized"] = True
        rows.append(item)
    matrix = pd.DataFrame(rows)
    if matrix["forward_fingerprint"].nunique() != 530:
        raise RuntimeError("forward fingerprint collision")
    preflight = matrix[matrix.forward_candidate_ordinal.isin(representative_ordinals)].sort_values("topology_total_nodes")

    projected_paths = []
    for row in matrix.itertuples(index=False):
        for year in (1470, 1770, 2000):
            projected_paths.append(OUTPUT_ROOT / "m" / row.production_member_key / "c" / "checkpoints" / f"year-{year:04d}" / "checkpoint.json")
    maximum_path_length = max(len(str(path.resolve())) for path in projected_paths)
    if maximum_path_length > 220:
        raise RuntimeError(f"insufficient Windows path margin: {maximum_path_length}")

    OUT.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(OUT / "forward_matrix.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    preflight.to_csv(OUT / "preflight_members.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-forward-transaction-v10.237",
        "status": "authorized-preflight-then-production",
        "launch_authorized": True,
        "authorization_text": "ok 直接开始吧",
        "population_systems": 187, "forward_cells": 530,
        "spinup_recomputation_authorized": False,
        "model": {"family": "OGGM-one-dimensional-SemiImplicitModel", "fixed_timestep_days": 1.25},
        "forward": {
            "start_year": 1470, "end_year": 2000, "annual_checkpoint_interval_years": 1,
            "forcing_rule": "immutable reconstruction anomaly plus same spinup-centered offset",
            "precipitation_anomaly_mode": "off", "extra_endpoint_ramp_or_fit": None,
            "glacier_specific_manual_tuning": False,
        },
        "inputs": {
            "candidate_manifest": str((CANDIDATE_DIR / "forward_candidate_manifest.csv").resolve()),
            "candidate_manifest_sha256": candidate_hash,
            "source_gdir_root": source["inputs"]["source_gdir_root"],
            "climate_pixels": source["inputs"]["climate_pixels"],
            "reconstruction_climate": source["inputs"]["reconstruction_climate"],
            "source_input_binding": str(input_binding.resolve()),
            "source_input_binding_sha256": sha256(input_binding),
            "curve_standard": str(CURVE_STANDARD.resolve()),
            "curve_standard_sha256": curve_standard_hash,
            "curve_standard_role": "post-run-provisional-screen; independent reproduction required before acceptance",
        },
        "runtime": {
            "python_executable": source["runtime"]["python_executable"],
            "unified_short_root": str(OUTPUT_ROOT.resolve()),
            "workers": 12, "thread_environment": source["runtime"]["thread_environment"],
            "worker_path": str(WORKER.resolve()), "worker_sha256": sha256(WORKER),
            "controller_path": str(CONTROLLER.resolve()), "controller_sha256": sha256(CONTROLLER),
            "maximum_projected_checkpoint_path_length": maximum_path_length,
            "windows_path_margin_to_260": 260 - maximum_path_length,
            "memory_pause_fraction": 0.80,
        },
        "preflight": {
            "cell_count": 3, "selection": "minimum-median-maximum-total-flowline-nodes",
            "fingerprints": preflight.forward_fingerprint.tolist(),
            "member_keys": preflight.preflight_member_key.tolist(),
            "run_years": [1470, 1770], "resume_comparison_at_year": 1770,
            "production_auto_authorized_only_if_all_three_pass": True,
        },
        "production": {
            "cell_count": 530, "fingerprints": matrix.forward_fingerprint.tolist(),
            "member_keys": matrix.production_member_key.tolist(),
            "authorized_after_preflight_pass": True,
        },
        "failure_policy": {
            "single_member_failure_isolated": True,
            "automatic_whole_batch_termination": False,
            "extreme_memory_condition": "pause-new-dispatch-and-await-user-decision",
        },
    }
    (OUT / "transaction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "lia-all191-forward-v10.237-transaction-summary-v1",
        "status": "authorized-not-yet-launched", "forward_cells": 530,
        "systems": 187, "preflight_cells": 3, "workers": 12,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "maximum_projected_path_length": maximum_path_length,
        "spinup_recomputation": False, "launch_authorized": True,
        "next_action": "validator-pass then run three-cell preflight; on pass launch CPU12 production automatically",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
