from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1 as b1
from run_temperature_west_colocated_block_r_sensitivity_v1 import (
    raw_increment_full_r,
)


OUTPUT = b1.RESULTS / (
    "climate_reconstruction_temperature_west_colocated_block_r_modern_oof_v1"
)
AUDIT = b1.RESULTS / (
    "climate_reconstruction_temperature_colocated_proxy_residual_"
    "correlation_audit_v1"
)
RECORDS = AUDIT / "modern_proxy_psm_residuals_1940_2000.csv"
CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
YEARS = np.arange(1940, 2001)
SPLITS = {
    "early_train_late_test": ((1940, 1969), (1970, 2000)),
    "late_train_early_test": ((1970, 2000), (1940, 1969)),
}
PRODUCTS = {
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}


def read_proxy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    operators = []
    observations = []
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for reference in references:
            proxy = archive[reference]
            operators.append(
                np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            )
            ages = np.rint(
                np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            ).astype(int)
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {
                int(year): float(value)
                for year, value in zip(ages, values)
                if np.isfinite(value)
            }
            observations.append(
                [lookup.get(int(year), np.nan) for year in YEARS]
            )
    hxb = np.asarray(operators, dtype=float)
    obs = np.asarray(observations, dtype=float)
    if hxb.ndim != 2 or obs.shape != (hxb.shape[0], len(YEARS)):
        raise RuntimeError("Modern proxy read closure failed")
    return hxb, obs


def pair_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(valid)) < 20:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def build_split_r(
    error_variance: np.ndarray,
    network: int,
    records: pd.DataFrame,
    residual_matrix: np.ndarray,
    training_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    covariance = np.diag(error_variance.copy())
    network_records = records.loc[records["network"].eq(network)].copy()
    index_lookup = {
        int(row.record_index_1based): position
        for position, row in network_records.reset_index(drop=True).iterrows()
    }
    active_groups = []
    for site_key, group in network_records.groupby("site_key"):
        if len(group) < 2:
            continue
        record_indices = group["record_index_1based"].astype(int).tolist()
        correlations = []
        for left, right in combinations(record_indices, 2):
            correlations.append(
                pair_correlation(
                    residual_matrix[index_lookup[left], training_mask],
                    residual_matrix[index_lookup[right], training_mask],
                )
            )
        finite = np.asarray(correlations, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        median = float(np.median(finite))
        if median < 0.5:
            continue
        rho = float(np.clip(median, 0.5, 0.9))
        zero_based = np.asarray(record_indices, dtype=int) - 1
        correlation = np.full((len(zero_based), len(zero_based)), rho)
        np.fill_diagonal(correlation, 1.0)
        standard_deviation = np.sqrt(error_variance[zero_based])
        block = correlation * np.outer(standard_deviation, standard_deviation)
        covariance[np.ix_(zero_based, zero_based)] = block
        active_groups.append(
            {
                "site_key": site_key,
                "record_count": len(zero_based),
                "training_median_residual_correlation": median,
                "rho_used": rho,
                "minimum_block_eigenvalue": float(np.linalg.eigvalsh(block).min()),
            }
        )
    if float(np.max(np.abs(np.diag(covariance) - error_variance))) > 1e-12:
        raise RuntimeError("Archived R diagonal changed")
    return covariance, {
        "active_group_count": len(active_groups),
        "active_record_count": int(sum(item["record_count"] for item in active_groups)),
        "groups": active_groups,
    }


def weighted_series(field: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, np.newaxis] * mask
    valid = np.isfinite(field) & mask[np.newaxis, :, :]
    numerator = np.nansum(
        np.where(valid, field, 0.0) * weights[np.newaxis, :, :], axis=(1, 2)
    )
    denominator = np.sum(valid * weights[np.newaxis, :, :], axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.full(field.shape[0], np.nan),
        where=denominator > 0,
    )


def metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    training_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float]:
    candidate_anomaly = candidate - float(np.nanmean(candidate[training_mask]))
    reference_anomaly = reference - float(np.nanmean(reference[training_mask]))
    valid = test_mask & np.isfinite(candidate_anomaly) & np.isfinite(reference_anomaly)
    if int(np.count_nonzero(valid)) < 20:
        raise RuntimeError("Insufficient modern OOF validation years")
    annual_rmse = float(
        np.sqrt(np.mean((candidate_anomaly[valid] - reference_anomaly[valid]) ** 2))
    )
    annual_correlation = float(
        np.corrcoef(candidate_anomaly[valid], reference_anomaly[valid])[0, 1]
    )
    candidate_lowpass = pd.Series(candidate_anomaly).rolling(
        11, center=True, min_periods=6
    ).mean().to_numpy(float)
    reference_lowpass = pd.Series(reference_anomaly).rolling(
        11, center=True, min_periods=6
    ).mean().to_numpy(float)
    lowpass_valid = valid & np.isfinite(candidate_lowpass) & np.isfinite(reference_lowpass)
    return {
        "annual_rmse_degC": annual_rmse,
        "annual_correlation": annual_correlation,
        "lowpass11_rmse_degC": float(
            np.sqrt(
                np.mean(
                    (candidate_lowpass[lowpass_valid] - reference_lowpass[lowpass_valid])
                    ** 2
                )
            )
        ),
        "lowpass11_correlation": float(
            np.corrcoef(
                candidate_lowpass[lowpass_valid], reference_lowpass[lowpass_valid]
            )[0, 1]
        ),
    }


def main() -> None:
    output_files = [
        OUTPUT / "modern_oof_west_curves_1940_2000.csv",
        OUTPUT / "modern_oof_split_product_metrics.csv",
        OUTPUT / "modern_oof_block_group_diagnostics.json",
        OUTPUT / "temperature_west_colocated_block_r_modern_oof.png",
        OUTPUT / "summary.json",
        OUTPUT / "source_manifest.json",
        OUTPUT / "FINAL_RESULT.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    manifest27 = b1.LOOP / (
        "round_27_temperature_innovation_gain_sign_decomposition_v1/"
        "source_manifest.json"
    )
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "reference_grid": b1.LOOP
        / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": b1.RESULTS
        / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "records": RECORDS,
        "climate": CLIMATE,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: b1.sha256(path) for name, path in paths.items()}

    record_columns = ["network", "record_index_1based", "site_key"] + [
        f"residual_{year}" for year in YEARS
    ]
    records = pd.read_csv(RECORDS, usecols=record_columns)
    residuals_by_network = {
        network: records.loc[records["network"].eq(network), [f"residual_{year}" for year in YEARS]].to_numpy(float)
        for network in (6, 9)
    }
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        pools = np.asarray(archive["all_Xb_indices"], dtype=int) - 1
    target, lat, lon = b1.base.western_target(
        prior, paths["reference_grid"], paths["hma_mask"]
    )
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))

    proxy_inputs = {}
    for network in (6, 9):
        hxb, observations = read_proxy(paths[f"proxy{network}"])
        residual_fill = np.asarray(
            scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float
        )
        error_variance = np.nanvar(residual_fill, axis=1, ddof=1)
        proxy_inputs[network] = (hxb, observations, error_variance)

    diag_network = {}
    closures = []
    for network in (6, 9):
        hxb, observations, error_variance = proxy_inputs[network]
        curve, closure = raw_increment_full_r(
            hxb, observations, np.diag(error_variance), pools, target
        )
        diag_network[network] = curve
        closures.append(closure)
    diag_curve = background + np.mean(
        np.stack([diag_network[6], diag_network[9]]), axis=0
    )

    split_curves = {}
    group_diagnostics = {}
    for split_name, (training_period, _) in SPLITS.items():
        training_mask = (YEARS >= training_period[0]) & (YEARS <= training_period[1])
        network_curves = {}
        group_diagnostics[split_name] = {}
        for network in (6, 9):
            hxb, observations, error_variance = proxy_inputs[network]
            error_covariance, diagnostics = build_split_r(
                error_variance,
                network,
                records,
                residuals_by_network[network],
                training_mask,
            )
            curve, closure = raw_increment_full_r(
                hxb, observations, error_covariance, pools, target
            )
            closures.append(closure)
            network_curves[network] = curve
            group_diagnostics[split_name][str(network)] = diagnostics
        split_curves[split_name] = background + np.mean(
            np.stack([network_curves[6], network_curves[9]]), axis=0
        )
    if max(closures) > 1e-12:
        raise RuntimeError("Modern OOF EnKF closure failed")

    with h5py.File(CLIMATE, "r") as archive:
        all_years = np.asarray(archive["time"], dtype=int)
        climate_indices = np.flatnonzero((all_years >= YEARS[0]) & (all_years <= YEARS[-1]))
        if not np.array_equal(all_years[climate_indices], YEARS):
            raise RuntimeError("Modern reference year axis mismatch")
        climate_lat = np.asarray(archive["lat"], dtype=float)
        climate_lon = np.asarray(archive["lon"], dtype=float)
        if not (np.allclose(climate_lat, lat) and np.allclose(climate_lon, lon)):
            raise RuntimeError("Modern reference grid mismatch")
        hma = np.load(paths["hma_mask"]).astype(bool)
        west = hma & (climate_lon[np.newaxis, :] < 80.0)
        product_series = {}
        for product, variable in PRODUCTS.items():
            field = np.asarray(archive[variable][climate_indices], dtype=float)
            field[field <= -9000] = np.nan
            product_series[product] = weighted_series(field, west, climate_lat)

    curve_columns: dict[str, np.ndarray] = {
        "year": YEARS,
        "diag_r": diag_curve,
        **{f"block_{name}": curve for name, curve in split_curves.items()},
        **{f"reference_{name}": curve for name, curve in product_series.items()},
    }
    pd.DataFrame(curve_columns).to_csv(output_files[0], index=False)

    metric_rows = []
    for split_name, (training_period, test_period) in SPLITS.items():
        training_mask = (YEARS >= training_period[0]) & (YEARS <= training_period[1])
        test_mask = (YEARS >= test_period[0]) & (YEARS <= test_period[1])
        for product, reference in product_series.items():
            diagonal_metrics = metrics(diag_curve, reference, training_mask, test_mask)
            block_metrics = metrics(
                split_curves[split_name], reference, training_mask, test_mask
            )
            metric_rows.append(
                {
                    "split": split_name,
                    "training_period": f"{training_period[0]}-{training_period[1]}",
                    "test_period": f"{test_period[0]}-{test_period[1]}",
                    "product": product,
                    **{f"diag_{key}": value for key, value in diagonal_metrics.items()},
                    **{f"block_{key}": value for key, value in block_metrics.items()},
                    "annual_rmse_improvement_fraction": (
                        diagonal_metrics["annual_rmse_degC"]
                        - block_metrics["annual_rmse_degC"]
                    )
                    / diagonal_metrics["annual_rmse_degC"],
                    "annual_correlation_change": (
                        block_metrics["annual_correlation"]
                        - diagonal_metrics["annual_correlation"]
                    ),
                }
            )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(output_files[1], index=False)
    improvements = metric_table["annual_rmse_improvement_fraction"]
    correlation_changes = metric_table["annual_correlation_change"]
    gates = {
        "median_annual_rmse_improvement_at_least_5_percent": bool(
            improvements.median() >= 0.05
        ),
        "worst_annual_rmse_regression_no_worse_than_5_percent": bool(
            improvements.min() >= -0.05
        ),
        "at_least_6_of_8_paths_improve": bool(int((improvements > 0).sum()) >= 6),
        "median_annual_correlation_change_nonnegative": bool(
            correlation_changes.median() >= 0
        ),
    }
    passes = bool(all(gates.values()))

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), constrained_layout=True)
    product_mean = np.nanmean(np.stack(list(product_series.values())), axis=0)
    for axis, (split_name, (training_period, test_period)) in zip(axes, SPLITS.items()):
        training_mask = (YEARS >= training_period[0]) & (YEARS <= training_period[1])
        centered_reference = product_mean - np.nanmean(product_mean[training_mask])
        centered_diag = diag_curve - np.mean(diag_curve[training_mask])
        centered_block = split_curves[split_name] - np.mean(
            split_curves[split_name][training_mask]
        )
        axis.plot(YEARS, centered_reference, color="#111111", linewidth=2.0, label="4-product mean")
        axis.plot(YEARS, centered_diag, color="#C43E4F", linewidth=1.8, label="Diagonal R")
        axis.plot(YEARS, centered_block, color="#147D73", linewidth=2.3, label="OOF block R")
        axis.axvspan(test_period[0], test_period[1], color="#D9D9D9", alpha=0.35, linewidth=0)
        axis.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
        axis.set(
            ylabel="Temperature anomaly (degC)",
            title=f"Train {training_period[0]}-{training_period[1]}, test {test_period[0]}-{test_period[1]}",
        )
        axis.legend(frameon=False, ncol=3)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Year")
    figure.savefig(output_files[3], dpi=250)
    plt.close(figure)

    output_files[2].write_text(
        json.dumps(group_diagnostics, indent=2), encoding="ascii"
    )
    after = {name: b1.sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("An input changed during modern OOF validation")
    summary = {
        "status": "complete",
        "period": [1940, 2000],
        "splits": {
            name: [list(periods[0]), list(periods[1])]
            for name, periods in SPLITS.items()
        },
        "validation_products": list(PRODUCTS),
        "berkeley_earth_excluded_from_validation": True,
        "maximum_enkf_algebraic_closure_degC": max(closures),
        "median_annual_rmse_improvement_fraction": float(improvements.median()),
        "worst_annual_rmse_improvement_fraction": float(improvements.min()),
        "improving_path_count": int((improvements > 0).sum()),
        "path_count": int(len(improvements)),
        "median_annual_correlation_change": float(correlation_changes.median()),
        "gates": gates,
        "all_modern_oof_gates_pass": passes,
        "decision": (
            "retain_block_r_as_validated_mechanism_candidate"
            if passes
            else "close_block_r_route_after_modern_oof_failure"
        ),
        "production_authorized": False,
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
    }
    output_files[4].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[5].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    output_files[6].write_text(
        "\n".join(
            [
                "# Final result",
                "",
                f"Modern split-trained block-R OOF gates pass: `{str(passes).lower()}`.",
                "",
                f"- Median annual RMSE improvement: `{100 * improvements.median():.2f}%`.",
                f"- Worst path RMSE improvement: `{100 * improvements.min():.2f}%`.",
                f"- Improving paths: `{int((improvements > 0).sum())}/{len(improvements)}`.",
                f"- Median annual correlation change: `{correlation_changes.median():+.4f}`.",
                *[f"- {name}: `{value}`" for name, value in gates.items()],
                "",
                "Berkeley Earth was excluded from validation. No source or production file was modified.",
                "",
            ]
        ),
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
