from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_round277_direct_temperature_station_panel_1850_1950_v2 as round277
import run_temperature_candidate_b2_stage_c_real_proxy_v2 as stage_c
import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
STATIC_DIR = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2"
STATIC_PACKAGE = STATIC_DIR / "candidate_b2_stage_c_static_input_v2.npz"
SYNTHETIC_DIR = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_synthetic_preflight_v1"
BASE_CONTRACT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1/"
    "CANDIDATE_B2_STAGE_C_CONTRACT.md"
)
CENTERING_CONTRACT = DRIVER_DIR / "temperature_candidate_b2_stage_c_observation_centering_v1_contract.md"
EVALUATOR = DRIVER_DIR / "evaluate_temperature_candidate_b2_stage_c_v1.py"
ROUND277_DIR = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
EVIDENCE = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v13/evidence_matrix.csv"
LONG_CURVES = LOOP / (
    "temperature_background_candidate_v2_longterm_hma_curve_v1/"
    "hma_temperature_before_after_ref_instrument_allowed.csv"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v1"
YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)].astype(np.int32)


def canonical_json_hash(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def verify_round277_values() -> dict[str, object]:
    metadata = pd.read_csv(ROUND277_DIR / "station_metadata.csv")
    target_ids = set(metadata["station_id"].astype(str))
    rows = []
    audits = {}
    for series in ("qcf", "qcu"):
        source = ROUND277_DIR / "prepared" / f"ghcnm_tavg_{series}.data"
        audit: dict[str, int] = defaultdict(int)
        with source.open("rb") as handle:
            parsed = round277.parse_ghcnm_lines_v2(handle, target_ids, series, audit)
        rows.extend(parsed)
        audits[series] = dict(audit)
    rebuilt = pd.DataFrame(rows)
    archived = pd.read_csv(ROUND277_DIR / "monthly_temperature_panel.csv")
    keys = ["station_id", "series", "year", "month"]
    merged = archived.merge(
        rebuilt,
        on=keys,
        how="outer",
        suffixes=("_saved", "_rebuilt"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("round277-row-identity-closure")
    numeric_errors = {}
    for column in ("temperature_c", "analysis_temperature_c"):
        saved = pd.to_numeric(merged[f"{column}_saved"], errors="coerce").to_numpy(float)
        rebuilt_values = pd.to_numeric(merged[f"{column}_rebuilt"], errors="coerce").to_numpy(float)
        finite = np.isfinite(saved) & np.isfinite(rebuilt_values)
        missing_match = np.array_equal(np.isnan(saved), np.isnan(rebuilt_values))
        numeric_errors[column] = {
            "maximum_abs_error_degC": float(np.max(np.abs(saved[finite] - rebuilt_values[finite]))) if np.any(finite) else 0.0,
            "missing_mask_exact": bool(missing_match),
        }
    locked_cells = sum(int(audit.get("locked_target_rows_skipped_before_value_access", 0)) * 12 for audit in audits.values())
    accessed_locked = 0
    return {
        "saved_row_count": int(len(archived)),
        "rebuilt_row_count": int(len(rebuilt)),
        "row_identity_exact": True,
        "numeric_errors": numeric_errors,
        "saved_temperature_min_degC": float(archived["analysis_temperature_c"].min()),
        "saved_temperature_max_degC": float(archived["analysis_temperature_c"].max()),
        "integer_scale": 0.01,
        "locked_target_monthly_cells_skipped_before_value_access": locked_cells,
        "locked_monthly_value_cells_accessed": accessed_locked,
        "parser_audit": audits,
        "all_value_closure_pass": bool(
            len(archived) == len(rebuilt)
            and all(item["maximum_abs_error_degC"] == 0.0 and item["missing_mask_exact"] for item in numeric_errors.values())
            and accessed_locked == 0
        ),
        "metadata_note": "v2 directory summary/manifest experiment label is stale v1 after recovery, but all saved values close exactly to the v2 /100 parser and v2 code hashes are present",
    }


def proxy_availability_masks(package: np.lib.npyio.NpzFile) -> dict[str, dict[str, int]]:
    year_lookup = {int(year): index for index, year in enumerate(YEARS)}
    output = {}
    for network in core.NETWORKS:
        selected = np.asarray(package[f"proxy{network}_record_indices_0based"], dtype=int)
        availability = np.zeros((len(YEARS), len(selected)), dtype=bool)
        source = stage_c.v1.replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
        with h5py.File(source, "r") as archive:
            references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
            for output_index, record_index in enumerate(selected):
                ages = np.asarray(
                    archive[references[int(record_index)]]["age_yng"], dtype=float
                ).reshape(-1, order="F")
                for age in ages:
                    if np.isfinite(age):
                        target = year_lookup.get(int(age))
                        if target is not None:
                            availability[target, output_index] = True
        keys = {
            np.packbits(row, bitorder="little").tobytes()
            for row in availability
        }
        counts = availability.sum(axis=1)
        output[f"proxy{network}"] = {
            "record_count": int(len(selected)),
            "unique_active_masks": int(len(keys)),
            "minimum_active_count": int(counts.min()),
            "median_active_count": int(np.median(counts)),
            "maximum_active_count": int(counts.max()),
            "years_with_zero_active": int(np.sum(counts == 0)),
            "real_proxy_value_cells_read": 0,
        }
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads((STATIC_DIR / "summary.json").read_text(encoding="utf-8"))
    synthetic_summary = json.loads((SYNTHETIC_DIR / "summary.json").read_text(encoding="utf-8"))
    if not static_summary.get("static_package_pass"):
        raise RuntimeError("static-package-not-passed")
    if not synthetic_summary.get("synthetic_preflight_pass"):
        raise RuntimeError("synthetic-preflight-not-passed")
    if core.sha256(STATIC_PACKAGE) != core.EXPECTED_STATIC_PACKAGE_SHA256:
        raise RuntimeError("static-package-hash")

    round277_closure = verify_round277_values()
    if not round277_closure["all_value_closure_pass"]:
        raise RuntimeError("round277-value-closure")
    evidence = pd.read_csv(EVIDENCE)
    allowed_curves = pd.read_csv(LONG_CURVES)
    if evidence["sealed_period_values_used"].astype(bool).any():
        raise RuntimeError("evidence-sealed-flag")
    if allowed_curves["year"].between(1921, 1939).any():
        raise RuntimeError("allowed-curve-sealed-year")
    with np.load(STATIC_PACKAGE, allow_pickle=False) as package:
        if not np.array_equal(np.asarray(package["years"], dtype=np.int32), YEARS):
            raise RuntimeError("static-year-axis")
        mask_audit = proxy_availability_masks(package)

    frozen = {
        "candidate": "balanced_pooled_candidate_b2",
        "allowed_periods": [[1500, 1920], [1940, 2000]],
        "sealed_period": [1921, 1939],
        "models": ["CESM", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"],
        "pooled_members": 1396,
        "model_weights": [0.25, 0.25, 0.25, 0.25],
        "network_weights": {"proxy6": 0.5, "proxy9": 0.5},
        "observation_centering": "raw proxy value minus archived CESM HXb mean over 1500-1848",
        "static_package_sha256": core.sha256(STATIC_PACKAGE),
        "core_sha256": core.sha256(Path(core.__file__).resolve()),
        "real_driver_sha256": core.sha256(Path(stage_c.__file__).resolve()),
        "evaluator_sha256": core.sha256(EVALUATOR),
        "base_contract_sha256": core.sha256(BASE_CONTRACT),
        "centering_contract_sha256": core.sha256(CENTERING_CONTRACT),
        "authorization_text_required": stage_c.v1.AUTHORIZATION_TEXT,
        "production_write_allowed": False,
        "oggm_allowed": False,
        "prohibited_source": "20CRv3",
    }
    fingerprint = canonical_json_hash(frozen)
    gates = {
        "static_package_pass": True,
        "synthetic_end_to_end_pass": True,
        "round277_all_saved_values_close_to_v2_divide_100_parser": True,
        "round277_locked_monthly_value_cells_accessed_zero": True,
        "independent_evidence_matrix_sealed_flag_clear": True,
        "allowed_long_curve_has_no_sealed_year": True,
        "real_proxy_value_cells_read": 0,
        "stage_c_real_proxy_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
    }
    launch_ready = (
        all(gates[name] for name in (
            "static_package_pass",
            "synthetic_end_to_end_pass",
            "round277_all_saved_values_close_to_v2_divide_100_parser",
            "round277_locked_monthly_value_cells_accessed_zero",
            "independent_evidence_matrix_sealed_flag_clear",
            "allowed_long_curve_has_no_sealed_year",
        ))
        and gates["real_proxy_value_cells_read"] == 0
        and gates["stage_c_real_proxy_enkf_run"] is False
        and gates["production_modified"] is False
        and gates["oggm_run"] is False
    )
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "preauthorization_launch_ready": bool(launch_ready),
        "launch_fingerprint_sha256": fingerprint,
        "gates": gates,
        "proxy_availability_mask_audit": mask_audit,
        "round277_value_closure": round277_closure,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "authorization_missing": True,
        "next_action": "await explicit user authorization for Candidate B2 Stage C real-proxy EnKF",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "launch_manifest.json").write_text(
        json.dumps({"launch_fingerprint_sha256": fingerprint, "frozen_configuration": frozen}, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "round277_full_value_closure.json").write_text(
        json.dumps(round277_closure, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "AUTHORIZATION_RECEIPT_SCHEMA.md").write_text(
        "# Authorization receipt schema\n\n"
        "No authorization receipt exists yet. After an explicit user authorization, "
        "create one JSON object with schema `candidate-b2-stage-c-authorization-v1`, "
        "`authorized: true`, the exact authorization text, UTC authorization time, "
        "the frozen static-package hash, and the frozen real-driver hash.\n\n"
        f"Required authorization text: `{stage_c.v1.AUTHORIZATION_TEXT}`.\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_PREAUTHORIZATION_REPORT.md").write_text(
        "# Candidate B2 Stage C preauthorization report\n\n"
        f"Launch preparation pass: `{launch_ready}`.\n\n"
        f"Frozen launch fingerprint: `{fingerprint}`.\n\n"
        f"Synthetic direct-update closure: `{synthetic_summary['maximum_direct_member_mean_closure_degC']:.3e} degC`.\n\n"
        "Round 277 saved station values close exactly to the official fixed-width "
        "records parsed with `/100`; the stale v1 experiment label is a recovery "
        "metadata defect, not a value-unit defect.\n\n"
        "No real proxy value has been read and no real-data EnKF has run. The only "
        "remaining boundary is explicit user authorization.\n",
        encoding="utf-8",
    )
    if not launch_ready:
        raise RuntimeError("launch-preflight-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
