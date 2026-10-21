from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd


EXPECTED_NPL_RWL_SHA256 = "d7f9d588e1362745857ac2aea3ca7c0686cee8b6ecc410e20af23a97d81d7bcd"
PRODUCT_VARIABLES = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = loop / "round_264c_round265_real_data_interface_preflight_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "smoother_core": drivers / "temperature_increment_smoother_core_v1.py",
        "safe_rwl_parser": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "signal_free_driver": drivers / "run_round256_npl065_modern_signal_free_sensitivity_v1.py",
        "round262_summary": loop / "round_262_npl065_first_difference_anchor_psm_v1/summary.json",
        "round263_summary": loop / "round_263_temperature_difference_generalization_v1/summary.json",
        "round264_summary": loop / "round_264_temperature_frequency_assimilation_architecture_audit_v1/summary.json",
        "round264a_summary": loop / "round_264a_temperature_increment_smoother_core_preflight_v1/summary.json",
        "round264b_v2_summary": loop / "round_264b_temperature_prior_increment_stationarity_v2/summary.json",
        "round264b_covariance": loop / "round_264b_temperature_prior_increment_stationarity_v2/independent_subregion_full_increment_covariance.csv",
        "npl_rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "npl_modern_curves": loop / "round_258_npl065_precalibration_holdout_v2/precalibration_holdout_curves.csv",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "existing_oof_fields": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/heldout_oof_fields.npz",
        "existing_oof_summary": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-round265-input:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["npl_rwl"]["sha256"] != EXPECTED_NPL_RWL_SHA256:
        raise RuntimeError("npl-rwl-hash-changed")

    parser_text = paths["safe_rwl_parser"].read_text(encoding="utf-8")
    parser_tokens = (
        "year = start_year + offset",
        "if LOCKED_START <= year <= LOCKED_END:",
        "continue",
        "series[year] = float(int(token.decode(\"ascii\", errors=\"strict\")))",
        '"locked_numeric_values_decoded": False',
    )
    if not all(token in parser_text for token in parser_tokens):
        raise RuntimeError("safe-rwl-parser-contract-changed")
    year_position = parser_text.index("year = start_year + offset")
    locked_position = parser_text.index("if LOCKED_START <= year <= LOCKED_END:", year_position)
    conversion_position = parser_text.index(
        'series[year] = float(int(token.decode("ascii", errors="strict")))',
        locked_position,
    )
    if not year_position < locked_position < conversion_position:
        raise RuntimeError("rwl-year-first-order-failed")

    summaries = {
        name: json.loads(paths[name].read_text(encoding="ascii" if name != "existing_oof_summary" else "utf-8"))
        for name in (
            "round262_summary",
            "round263_summary",
            "round264_summary",
            "round264a_summary",
            "round264b_v2_summary",
            "existing_oof_summary",
        )
    }
    if summaries["round262_summary"]["admitted_methods"] != ["signal_free_constrained"]:
        raise RuntimeError("round262-admission-changed")
    if summaries["round263_summary"]["external_pass_count"] != 2:
        raise RuntimeError("round263-generalization-changed")
    if not summaries["round264a_summary"]["checks"]["normal_equation_closure_at_most_1e_10"]:
        raise RuntimeError("smoother-core-preflight-changed")
    if not summaries["round264b_v2_summary"]["passes_all_gates"]:
        raise RuntimeError("prior-increment-admission-changed")
    if summaries["existing_oof_summary"]["ordinary_enkf_member_count"] != 500:
        raise RuntimeError("existing-oof-comparator-changed")

    netcdf_rows = []
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        required_datasets = ["time", "lat", "lon", *PRODUCT_VARIABLES.values()]
        absent = [name for name in required_datasets if name not in handle]
        if absent:
            raise RuntimeError("instrumental-dataset-missing:" + ",".join(absent))
        time_shape = tuple(int(value) for value in handle["time"].shape)
        lat_shape = tuple(int(value) for value in handle["lat"].shape)
        lon_shape = tuple(int(value) for value in handle["lon"].shape)
        if len(time_shape) != 1 or len(lat_shape) != 1 or len(lon_shape) != 1:
            raise RuntimeError("instrumental-coordinate-rank-changed")
        expected_field_shape = (time_shape[0], lat_shape[0], lon_shape[0])
        for product, variable in PRODUCT_VARIABLES.items():
            dataset = handle[variable]
            shape = tuple(int(value) for value in dataset.shape)
            if shape != expected_field_shape:
                raise RuntimeError(f"instrumental-field-shape-changed:{product}:{shape}")
            netcdf_rows.append({
                "product": product,
                "variable": variable,
                "shape": "x".join(str(value) for value in shape),
                "dtype": str(dataset.dtype),
                "numeric_values_read": False,
            })
    pd.DataFrame(netcdf_rows).to_csv(output / "instrumental_product_metadata.csv", index=False)

    execution_contract = {
        "experiment_id": "round_265_npl065_regional_increment_smoother_oof_v1",
        "authorization_state": "prepared_not_executed",
        "years": [1940, 2000],
        "outer_blocks": [[1940, 1951], [1952, 1963], [1964, 1975], [1976, 1987], [1988, 2000]],
        "state": {
            "independent_regions": ["west_lt80", "central_80_90", "east_ge90"],
            "derived_diagnostic": "hma",
            "hma_area_shares": summaries["round264b_v2_summary"]["subregion_area_shares"],
            "increment_spatial_covariance": str(paths["round264b_covariance"]),
            "increment_ar1_phi": summaries["round264b_v2_summary"]["recommended_ar1_phi"],
            "prior_member_years": [1001, 1848],
            "paired_increment_count": 847,
        },
        "new_proxy": {
            "record": "NPL065",
            "site": [27.89, 86.37],
            "mapped_state": "central_80_90",
            "chronology": "constrained modern-only signal-free chronology",
            "chronology_years": [1941, 2013],
            "documented_target": "CRU TS4.01 DJF Tmin 27.5-28.5N 86-87E",
            "difference_fit_years": [1961, 2013],
            "outer_test_years_excluded_from_difference_fit": True,
            "locked_RWL_slots_skipped_before_numeric_conversion": True,
        },
        "level_anchor": {
            "products": list(PRODUCT_VARIABLES),
            "policy": "leave_evaluated_product_out_and_use_training_years_only",
            "test_block_anchor_values_used": False,
            "anchor_uncertainty": "training_interproduct_spread_with_0p10_degC_floor",
        },
        "branches": [
            "anchor_only",
            "anchor_plus_NPL065_documented_DJF_difference",
            "anchor_plus_NPL065_direct_central_annual_difference_sensitivity",
            "existing_fold_native_level_PSM_comparator",
        ],
        "primary_branch": "anchor_plus_NPL065_documented_DJF_difference",
        "primary_gates": {
            "median_central_level_RMSE_improvement_vs_anchor_only_minimum": 0.05,
            "worst_central_or_HMA_product_RMSE_regression_maximum": 0.05,
            "positive_central_block_product_fraction_minimum": 0.70,
            "median_central_heldout_difference_correlation_minimum": 0.30,
            "median_HMA_level_RMSE_improvement_vs_anchor_only_minimum": 0.0,
        },
        "interpretation_boundary": "pass_allows_only_a_nonproduction_gridded_followup_not_paleo_production_or_OGGM",
    }
    (output / "ROUND_265_EXECUTION_CONTRACT.json").write_text(
        json.dumps(execution_contract, indent=2), encoding="ascii"
    )
    (output / "ROUND_265_EXECUTION_CONTRACT.md").write_text(
        "# Round 265 execution contract\n\n"
        "Round 265 is a modern-only 1940-2000 blocked-OOF test of the newly validated NPL065 annual-difference role. West, central, and east are the three independent states; HMA is derived.\n\n"
        "For every outer block and evaluated temperature product, the evaluated product is removed from the anchor and all block years are removed from both the anchor and PSM fit. Baiyu and Huang remain external supporting evidence and are not assimilated.\n\n"
        "The primary branch uses the documented local DJF Tmin difference operator. A direct central-annual fit is reported only as a sensitivity branch and cannot replace a failed primary result.\n\n"
        "Execution remains unauthorized. A pass would permit only a non-production gridded follow-up.\n",
        encoding="ascii",
    )

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "npl_rwl_hash_matches": True,
        "year_first_locked_safe_parser_verified": True,
        "instrumental_product_count": len(PRODUCT_VARIABLES),
        "instrumental_field_shape": list(expected_field_shape),
        "round265_authorization_state": "prepared_not_executed",
        "decision": "round265_real_data_interface_and_gates_frozen_await_execution_authorization",
        "proxy_or_climate_numeric_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 265 real-data interface preflight\n\n"
        "- NPL065 raw-source hash matches the frozen source.\n"
        "- The RWL parser skips locked and pre-1941 slots before numeric conversion.\n"
        f"- All five instrumental products share field shape `{expected_field_shape}`.\n"
        "- The state, anchor leave-one-product-out policy, branches, and primary gates are frozen.\n\n"
        "Decision: `round265_real_data_interface_and_gates_frozen_await_execution_authorization`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
