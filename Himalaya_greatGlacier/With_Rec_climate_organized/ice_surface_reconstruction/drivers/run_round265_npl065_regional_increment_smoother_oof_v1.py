from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import temperature_increment_smoother_core_v1 as smoother


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
STATE_REGIONS = ("west_lt80", "central_80_90", "east_ge90")
DIAGNOSTIC_REGIONS = STATE_REGIONS + ("hma",)
PRODUCT_VARIABLES = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}
PRIMARY_BRANCH = "anchor_plus_npl_documented_djf_difference"


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 10 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError(f"invalid-regression-input:{len(x)}")
    return np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]


def build_npl_chronology(paths: dict[str, Path], loop: Path) -> tuple[pd.Series, dict[str, object]]:
    round246 = load_module("round265_round246", paths["round246_driver"])
    ads = load_module("round265_ads", paths["round252_driver"])
    round253 = load_module("round265_round253", paths["round253_driver"])
    round256 = load_module("round265_round256", paths["round256_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["npl_rwl"])
    transformed, _ = round256.power_transform_frame(raw, round253)
    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy
    from smoothingspline import spline

    result = round256.signal_free(transformed, False, dplpy, ads, spline)
    if not result["converged"]:
        raise RuntimeError("npl-signal-free-did-not-converge")
    chronology = pd.Series(
        np.asarray(result["final_chronology"], dtype=float),
        index=np.arange(1941, 2014, dtype=int),
        name="signal_free_constrained",
    )
    if chronology.isna().any():
        raise RuntimeError("npl-chronology-nonfinite")
    return chronology, {
        "parse_metadata": parse_metadata,
        "iterations": int(result["iterations"]),
        "final_mad": float(result["final_mad"]),
    }


def load_regional_products(
    path: Path,
    common,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(path, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        selected_years = source_years[indices]
        if not np.array_equal(selected_years, YEARS):
            raise RuntimeError("instrumental-year-selection-failed")
        if np.any((selected_years >= 1921) & (selected_years <= 1939)):
            raise RuntimeError("locked-instrumental-year-selected")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        state_fields = {}
        for product, variable in PRODUCT_VARIABLES.items():
            cube = common.source_grid(
                common.clean(np.asarray(handle[variable][indices, :, :])),
                source_lat,
                source_lon,
                grid_lat,
                grid_lon,
            )
            state_fields[product] = common.cube_to_state(cube)
    return state_fields, {"source_lat": source_lat, "source_lon": source_lon}, grid_lat, grid_lon, selected_years


def combine_conditioning(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    anchor_values: np.ndarray,
    anchor_operator: np.ndarray,
    anchor_variances: np.ndarray,
    proxy_values: np.ndarray | None = None,
    proxy_operator: np.ndarray | None = None,
    proxy_variance: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    values = anchor_values
    operator = anchor_operator
    variances = anchor_variances
    if proxy_values is not None:
        if proxy_operator is None or proxy_variance is None or proxy_variance <= 0:
            raise RuntimeError("invalid-proxy-conditioning")
        values = np.concatenate((values, proxy_values))
        operator = np.vstack((operator, proxy_operator))
        variances = np.concatenate((variances, np.full(len(proxy_values), proxy_variance)))
    posterior_mean, _, diagnostics = smoother.gaussian_condition(
        prior_mean,
        prior_covariance,
        values,
        operator,
        np.diag(variances),
    )
    return posterior_mean, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-real-data", action="store_true")
    args = parser.parse_args()
    if not args.execute_real_data:
        raise RuntimeError("explicit---execute-real-data-authorization-flag-required")

    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = loop / "round_265_npl065_regional_increment_smoother_oof_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "core": Path(smoother.__file__).resolve(),
        "execution_contract": loop / "round_264c_round265_real_data_interface_preflight_v1/ROUND_265_EXECUTION_CONTRACT.json",
        "interface_summary": loop / "round_264c_round265_real_data_interface_preflight_v1/summary.json",
        "prior_summary": loop / "round_264b_temperature_prior_increment_stationarity_v2/summary.json",
        "prior_levels": loop / "round_264b_temperature_prior_increment_stationarity_v2/regional_prior_levels_1001_1848.csv",
        "prior_covariance": loop / "round_264b_temperature_prior_increment_stationarity_v2/independent_subregion_full_increment_covariance.csv",
        "npl_rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
        "round258_curves": loop / "round_258_npl065_precalibration_holdout_v2/precalibration_holdout_curves.csv",
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "round252_driver": drivers / "run_round252_dplr_ads_python_port_validation_v3.py",
        "round253_driver": drivers / "run_round253_npl065_ads_power_method_decomposition_v1.py",
        "round256_driver": drivers / "run_round256_npl065_modern_signal_free_sensitivity_v1.py",
        "common_driver": drivers / "temperature_fold_native_oenkf_increment_cv_v1/run_experiment_v1.py",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": result_root / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "existing_oof_fields": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/heldout_oof_fields.npz",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    interface = json.loads(paths["interface_summary"].read_text(encoding="ascii"))
    if interface["round265_authorization_state"] != "prepared_not_executed":
        raise RuntimeError("interface-preflight-state-changed")
    contract = json.loads(paths["execution_contract"].read_text(encoding="ascii"))
    prior_summary = json.loads(paths["prior_summary"].read_text(encoding="ascii"))
    shares = np.asarray([prior_summary["subregion_area_shares"][name] for name in STATE_REGIONS])
    if abs(float(shares.sum()) - 1.0) > 1e-12:
        raise RuntimeError("hma-share-closure-failed")

    common = load_module("round265_common", paths["common_driver"])
    chronology, chronology_metadata = build_npl_chronology(paths, loop)
    product_fields, source_coordinates, grid_lat, grid_lon, selected_years = load_regional_products(
        paths["instrumental_netcdf"], common
    )
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(
        mask05[None],
        source_coordinates["source_lat"],
        source_coordinates["source_lon"],
        grid_lat,
        grid_lon,
    )[0] >= 0.5
    weights = common.region_weights(grid_lat, grid_lon, mask2)
    regional_products: dict[str, dict[str, np.ndarray]] = {}
    for product, field in product_fields.items():
        regional_products[product] = {
            region: common.weighted_series(field, weights[region])
            for region in DIAGNOSTIC_REGIONS
        }

    prior_levels_frame = pd.read_csv(paths["prior_levels"])
    prior_level_matrix = prior_levels_frame[[f"{name}_level_degC" for name in STATE_REGIONS]].to_numpy(float)
    initial_covariance = np.cov(prior_level_matrix, rowvar=False, ddof=1)
    increment_covariance = pd.read_csv(paths["prior_covariance"], index_col=0).loc[list(STATE_REGIONS), list(STATE_REGIONS)].to_numpy(float)
    temporal_correlation = smoother.ar1_correlation(len(YEARS) - 1, float(prior_summary["recommended_ar1_phi"]))
    prior_mean, prior_covariance = smoother.build_state_prior(
        np.zeros(len(STATE_REGIONS)),
        np.zeros((len(YEARS) - 1, len(STATE_REGIONS))),
        initial_covariance,
        increment_covariance,
        temporal_correlation,
    )
    level_transform = smoother.level_transform(len(YEARS), len(STATE_REGIONS))

    round258 = pd.read_csv(paths["round258_curves"]).set_index("year")
    local_target = round258["cru_box_djf_tmin"]
    chronology_difference = chronology.diff().dropna()
    local_difference = local_target.diff().dropna()
    proxy_years = np.arange(1942, 2001, dtype=int)
    proxy_difference = chronology_difference.loc[proxy_years].to_numpy(float)
    proxy_operator = smoother.difference_operator(
        len(YEARS),
        len(STATE_REGIONS),
        proxy_years - YEARS[0],
        np.tile(np.asarray([[0.0, 1.0, 0.0]]), (len(proxy_years), 1)),
    )

    with np.load(paths["existing_oof_fields"]) as existing:
        existing_candidate_field = np.asarray(existing["candidate"], dtype=float)
        existing_full_field = np.asarray(existing["full_increment"], dtype=float)
        if not np.array_equal(np.asarray(existing["years"], dtype=int), YEARS):
            raise RuntimeError("existing-comparator-year-axis-changed")
    existing_curves = {
        "existing_fold_native_level_psm": {
            region: common.weighted_series(existing_candidate_field, weights[region])
            for region in DIAGNOSTIC_REGIONS
        },
        "existing_full_level_psm": {
            region: common.weighted_series(existing_full_field, weights[region])
            for region in DIAGNOSTIC_REGIONS
        },
    }

    score_rows = []
    curve_rows = []
    fit_rows = []
    diagnostics_rows = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        for evaluated_product in PRODUCT_VARIABLES:
            centered = {}
            for product in PRODUCT_VARIABLES:
                centered[product] = {
                    region: regional_products[product][region] - float(np.mean(regional_products[product][region][train]))
                    for region in DIAGNOSTIC_REGIONS
                }
            other_products = [name for name in PRODUCT_VARIABLES if name != evaluated_product]
            anchor_state = np.column_stack([
                np.mean(np.vstack([centered[name][region] for name in other_products]), axis=0)
                for region in STATE_REGIONS
            ])
            anchor_year_indices = np.repeat(np.flatnonzero(train), len(STATE_REGIONS))
            anchor_weights = np.tile(np.eye(len(STATE_REGIONS)), (int(train.sum()), 1))
            anchor_operator = smoother.level_operator(
                len(YEARS), len(STATE_REGIONS), anchor_year_indices, anchor_weights
            )
            anchor_values = anchor_state[train].reshape(-1)
            anchor_region_variance = []
            for region_index, region in enumerate(STATE_REGIONS):
                product_values = np.vstack([centered[name][region][train] for name in other_products])
                consensus = anchor_state[train, region_index]
                spread = float(np.var(product_values - consensus[None, :], ddof=1))
                anchor_region_variance.append(max(spread, 0.10 ** 2))
            anchor_variances = np.tile(np.asarray(anchor_region_variance), int(train.sum()))

            anchor_state_mean, anchor_diagnostics = combine_conditioning(
                prior_mean, prior_covariance, anchor_values, anchor_operator, anchor_variances
            )
            local_fit_years = np.arange(1961, 2014, dtype=int)
            local_train_years = local_fit_years[
                ~((local_fit_years >= start) & (local_fit_years <= end))
            ]
            beta_local = fit_ols(
                chronology_difference.loc[local_train_years].to_numpy(float),
                local_difference.loc[local_train_years].to_numpy(float),
            )
            documented_proxy_values = beta_local[0] + beta_local[1] * proxy_difference
            local_residual = (
                local_difference.loc[local_train_years].to_numpy(float)
                - beta_local[0]
                - beta_local[1] * chronology_difference.loc[local_train_years].to_numpy(float)
            )

            anchor_central_difference = np.diff(anchor_state[:, 1])
            anchor_difference_years = YEARS[1:]
            transfer_years = np.asarray([
                year for year in range(1961, 2001)
                if not (start <= year <= end)
            ], dtype=int)
            documented_by_year = dict(zip(proxy_years, documented_proxy_values))
            anchor_delta_by_year = dict(zip(anchor_difference_years, anchor_central_difference))
            transfer_residual = np.asarray([
                documented_by_year[int(year)] - anchor_delta_by_year[int(year)]
                for year in transfer_years
            ])
            documented_variance = max(float(np.var(transfer_residual, ddof=1)), 0.15 ** 2)
            documented_state_mean, documented_diagnostics = combine_conditioning(
                prior_mean,
                prior_covariance,
                anchor_values,
                anchor_operator,
                anchor_variances,
                documented_proxy_values,
                proxy_operator,
                documented_variance,
            )

            annual_fit_years = np.asarray([
                year for year in range(1942, 2001)
                if not (start <= year <= end)
            ], dtype=int)
            annual_target = np.asarray([anchor_delta_by_year[int(year)] for year in annual_fit_years])
            annual_proxy = chronology_difference.loc[annual_fit_years].to_numpy(float)
            beta_annual = fit_ols(annual_proxy, annual_target)
            annual_proxy_values = beta_annual[0] + beta_annual[1] * proxy_difference
            annual_residual = annual_target - beta_annual[0] - beta_annual[1] * annual_proxy
            annual_variance = max(float(np.var(annual_residual, ddof=1)), 0.15 ** 2)
            annual_state_mean, annual_diagnostics = combine_conditioning(
                prior_mean,
                prior_covariance,
                anchor_values,
                anchor_operator,
                anchor_variances,
                annual_proxy_values,
                proxy_operator,
                annual_variance,
            )
            fit_rows.append({
                "fold": fold,
                "period": f"{start}-{end}",
                "evaluated_product": evaluated_product,
                "documented_fit_intercept": float(beta_local[0]),
                "documented_fit_slope": float(beta_local[1]),
                "documented_local_residual_variance": float(np.var(local_residual, ddof=1)),
                "documented_transfer_variance": documented_variance,
                "direct_annual_fit_intercept": float(beta_annual[0]),
                "direct_annual_fit_slope": float(beta_annual[1]),
                "direct_annual_residual_variance": annual_variance,
            })
            for branch, diagnostics in (
                ("anchor_only", anchor_diagnostics),
                (PRIMARY_BRANCH, documented_diagnostics),
                ("anchor_plus_npl_direct_central_annual_difference_sensitivity", annual_diagnostics),
            ):
                diagnostics_rows.append({
                    "fold": fold,
                    "evaluated_product": evaluated_product,
                    "branch": branch,
                    **diagnostics,
                })

            state_means = {
                "anchor_only": anchor_state_mean,
                PRIMARY_BRANCH: documented_state_mean,
                "anchor_plus_npl_direct_central_annual_difference_sensitivity": annual_state_mean,
            }
            branch_curves = {}
            for branch, state_mean in state_means.items():
                state_levels = (level_transform @ state_mean).reshape(len(YEARS), len(STATE_REGIONS))
                branch_curves[branch] = {
                    region: state_levels[:, index] for index, region in enumerate(STATE_REGIONS)
                }
                branch_curves[branch]["hma"] = state_levels @ shares
            branch_curves.update(existing_curves)

            for branch, regions in branch_curves.items():
                for region in DIAGNOSTIC_REGIONS:
                    predicted = np.asarray(regions[region], dtype=float)
                    observed = centered[evaluated_product][region]
                    anchor_predicted = branch_curves["anchor_only"][region]
                    current_rmse = rmse(predicted[test], observed[test])
                    anchor_rmse = rmse(anchor_predicted[test], observed[test])
                    internal_delta = (YEARS >= start + 1) & (YEARS <= end)
                    predicted_delta = np.diff(predicted)[internal_delta[1:]]
                    observed_delta = np.diff(observed)[internal_delta[1:]]
                    score_rows.append({
                        "fold": fold,
                        "period": f"{start}-{end}",
                        "evaluated_product": evaluated_product,
                        "branch": branch,
                        "region": region,
                        "level_rmse_degC": current_rmse,
                        "level_rmse_improvement_vs_anchor_only": 1.0 - current_rmse / anchor_rmse,
                        "level_correlation": correlation(predicted[test], observed[test]),
                        "heldout_difference_correlation": correlation(predicted_delta, observed_delta),
                        "heldout_difference_rmse_degC": rmse(predicted_delta, observed_delta),
                    })
                    for year_index in np.flatnonzero(test):
                        curve_rows.append({
                            "year": int(YEARS[year_index]),
                            "fold": fold,
                            "evaluated_product": evaluated_product,
                            "branch": branch,
                            "region": region,
                            "prediction_degC": float(predicted[year_index]),
                            "observed_degC": float(observed[year_index]),
                        })

    scores = pd.DataFrame(score_rows)
    curves = pd.DataFrame(curve_rows)
    fits = pd.DataFrame(fit_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    scores.to_csv(output / "heldout_region_product_scores.csv", index=False)
    curves.to_csv(output / "heldout_regional_curves.csv", index=False)
    fits.to_csv(output / "fold_proxy_operator_fits.csv", index=False)
    diagnostics_frame.to_csv(output / "smoother_numerical_diagnostics.csv", index=False)

    primary = scores[scores["branch"].eq(PRIMARY_BRANCH)]
    central = primary[primary["region"].eq("central_80_90")]
    hma = primary[primary["region"].eq("hma")]
    central_hma = primary[primary["region"].isin(("central_80_90", "hma"))]
    gates = {
        "median_central_level_RMSE_improvement_at_least_5pct": bool(np.median(central["level_rmse_improvement_vs_anchor_only"]) >= 0.05),
        "worst_central_or_HMA_RMSE_regression_at_most_5pct": bool(np.min(central_hma["level_rmse_improvement_vs_anchor_only"]) >= -0.05),
        "positive_central_block_product_fraction_at_least_70pct": bool(np.mean(central["level_rmse_improvement_vs_anchor_only"] > 0) >= 0.70),
        "median_central_heldout_difference_correlation_at_least_0p30": bool(np.nanmedian(central["heldout_difference_correlation"]) >= 0.30),
        "median_HMA_level_RMSE_improvement_nonnegative": bool(np.median(hma["level_rmse_improvement_vs_anchor_only"]) >= 0.0),
    }
    passes = all(gates.values())
    decision = (
        "advance_npl065_frequency_role_to_nonproduction_gridded_followup"
        if passes
        else "retain_npl065_as_validation_evidence_do_not_advance_regional_smoother"
    )

    figure, axes = plt.subplots(2, 1, figsize=(12.2, 8.4), sharex=True, constrained_layout=True)
    colors = {
        "anchor_only": "#555555",
        PRIMARY_BRANCH: "#D55E00",
        "anchor_plus_npl_direct_central_annual_difference_sensitivity": "#0072B2",
        "existing_fold_native_level_psm": "#009E73",
    }
    for axis, region in zip(axes, ("central_80_90", "hma")):
        region_curves = curves[curves["region"].eq(region)]
        observed = region_curves.groupby("year", as_index=False)["observed_degC"].mean()
        axis.plot(observed["year"], observed["observed_degC"], color="#111111", linewidth=1.7, label="Five-product heldout mean")
        for branch, color in colors.items():
            selected = region_curves[region_curves["branch"].eq(branch)].groupby("year", as_index=False)["prediction_degC"].mean()
            axis.plot(selected["year"], selected["prediction_degC"], color=color, linewidth=1.35, label=branch)
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#bbbbbb", linewidth=0.6, linestyle="--")
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_title("Central HMA" if region == "central_80_90" else "HMA derived mean")
        axis.set_ylabel("Fold-centered degC")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(output / "round265_npl065_regional_increment_smoother_oof.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "authorization_flag": "--execute-real-data",
        "years_read": [1940, 2000],
        "locked_1921_1939_values_accessed": False,
        "state_regions": list(STATE_REGIONS),
        "hma_is_derived": True,
        "chronology_metadata": chronology_metadata,
        "primary_branch": PRIMARY_BRANCH,
        "primary_metrics": {
            "median_central_level_RMSE_improvement": float(np.median(central["level_rmse_improvement_vs_anchor_only"])),
            "worst_central_or_HMA_level_RMSE_improvement": float(np.min(central_hma["level_rmse_improvement_vs_anchor_only"])),
            "positive_central_block_product_fraction": float(np.mean(central["level_rmse_improvement_vs_anchor_only"] > 0)),
            "median_central_heldout_difference_correlation": float(np.nanmedian(central["heldout_difference_correlation"])),
            "median_HMA_level_RMSE_improvement": float(np.median(hma["level_rmse_improvement_vs_anchor_only"])),
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "direct_annual_branch_role": "reported_sensitivity_cannot_replace_failed_primary_documented_DJF_branch",
        "baiyu_or_huang_assimilated": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
        "execution_contract": contract,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    gate_lines = "\n".join(f"- `{name}`: `{value}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "# Round 265 NPL065 regional increment smoother OOF\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Passes all primary gates: `{passes}`.\n"
        f"- Median central level RMSE improvement: `{summary['primary_metrics']['median_central_level_RMSE_improvement']:.2%}`.\n"
        f"- Median central heldout difference correlation: `{summary['primary_metrics']['median_central_heldout_difference_correlation']:.3f}`.\n"
        f"- Median HMA level RMSE improvement: `{summary['primary_metrics']['median_HMA_level_RMSE_improvement']:.2%}`.\n\n"
        "## Gates\n\n" + gate_lines + "\n\n"
        "No locked-period value, production reconstruction, EnKF, or OGGM was used.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
