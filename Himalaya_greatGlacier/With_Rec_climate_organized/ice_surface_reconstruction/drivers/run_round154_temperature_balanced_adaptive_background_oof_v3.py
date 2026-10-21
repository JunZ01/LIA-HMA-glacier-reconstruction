"""Blocked modern test of one spatially balanced strategy-C background selector."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve().parent
FOLD_TOOLS = DRIVER / "temperature_fold_native_oenkf_increment_cv_v1"
sys.path.insert(0, str(FOLD_TOOLS))
import run_experiment_v1 as common

import decompose_temperature_lia_regional_prior_update_v1 as prior_tools
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1959), (1960, 1979), (1980, 2000))
PRIMARY_PRODUCTS = ("CRU", "GISTEMP", "HadCRUT5", "NOAA")
ALL_PRODUCTS = ("Berkeley",) + PRIMARY_PRODUCTS
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
BANDS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path, include_hash: bool = True) -> dict[str, object]:
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def harmonized_longitudes(network: int, path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if network == 6:
        records = frame.proxy_index.to_numpy(dtype=int)
        longitude = frame.longitude.to_numpy(dtype=float)
    else:
        records = frame.proxy9_record_1based.to_numpy(dtype=int)
        longitude = frame.longitude.to_numpy(dtype=float)
    if not np.array_equal(records, np.arange(1, len(records) + 1)):
        raise RuntimeError(f"metadata-record-order:{network}")
    if not np.all(np.isfinite(longitude)):
        raise RuntimeError(f"metadata-longitude-nonfinite:{network}")
    return longitude


def load_operator(path: Path, predictor_path: Path, allowed_path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["b"], dtype=float).transpose(2, 1, 0)
        residual_variance = np.asarray(handle["residual_variance"], dtype=float).T
        model_type = np.asarray(handle["model_type"], dtype=int).reshape(-1)
        sample_indices = np.asarray(handle["sample_indices"], dtype=int).reshape(-1) - 1
        locked = bool(np.asarray(handle["locked_1921_1939_accessed"]).squeeze())
    predictors = scipy.io.loadmat(predictor_path, variable_names=["model_t_848", "model_p_848"])
    model_t = np.asarray(predictors["model_t_848"], dtype=float)
    model_p = np.asarray(predictors["model_p_848"], dtype=float)
    allowed = scipy.io.loadmat(
        allowed_path,
        variable_names=["observations", "years", "archived_hxb"],
    )
    observations = np.asarray(allowed["observations"], dtype=float).T
    years = np.asarray(allowed["years"], dtype=int).reshape(-1)
    archived_hxb = np.asarray(allowed["archived_hxb"], dtype=float)
    proxy_count = observations.shape[0]
    if locked or not np.array_equal(years, YEARS):
        raise RuntimeError(f"operator-year-or-lock-contract:{path.name}")
    if coefficients.shape != (proxy_count, 4, 3):
        raise RuntimeError(f"coefficient-shape:{path.name}:{coefficients.shape}")
    if residual_variance.shape != (proxy_count, 4):
        raise RuntimeError(f"variance-shape:{path.name}:{residual_variance.shape}")
    if model_t.shape != (proxy_count, 848) or model_p.shape != (proxy_count, 848):
        raise RuntimeError(f"predictor-shape:{path.name}:{model_t.shape}:{model_p.shape}")
    hxb = np.full((proxy_count, 4, 848), np.nan, dtype=float)
    for branch in range(4):
        b = coefficients[:, branch]
        valid = np.all(np.isfinite(b[:, :2]), axis=1)
        hxb[valid, branch] = b[valid, 0, None] + b[valid, 1, None] * model_t[valid]
        bivariate = valid & (model_type == 2) & np.isfinite(b[:, 2])
        hxb[bivariate, branch] += b[bivariate, 2, None] * model_p[bivariate]
    closure_mask = np.isfinite(hxb[:, 0]) & np.isfinite(archived_hxb)
    closure = float(np.max(np.abs(hxb[:, 0][closure_mask] - archived_hxb[closure_mask])))
    if closure > 1e-9:
        raise RuntimeError(f"full-hxb848-closure:{path.name}:{closure}")
    return {
        "observations": observations,
        "hxb": hxb,
        "residual_variance": residual_variance,
        "sample_indices": sample_indices,
        "hxb848_closure": closure,
    }


def pearson_scores(observed: np.ndarray, modeled: np.ndarray) -> np.ndarray:
    y = observed - np.mean(observed)
    x = modeled - np.mean(modeled, axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(y * y) * np.sum(x * x, axis=0))
    scores = np.full(modeled.shape[1], -np.inf, dtype=float)
    valid = denominator > 0
    scores[valid] = y @ x[:, valid] / denominator[valid]
    return scores


def selection_scores(
    observed: np.ndarray,
    modeled: np.ndarray,
    longitude: np.ndarray,
    balanced: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    if not balanced:
        return pearson_scores(observed, modeled), {"all": int(len(observed))}
    pieces = []
    counts: dict[str, int] = {}
    for name, (lower, upper) in BANDS.items():
        selected = (longitude >= lower) & (longitude < upper)
        counts[name] = int(selected.sum())
        if selected.sum() < 3 or np.std(observed[selected]) <= 0:
            continue
        pieces.append(pearson_scores(observed[selected], modeled[selected]))
    if not pieces:
        return np.full(modeled.shape[1], -np.inf), counts
    stack = np.stack(pieces)
    finite = np.isfinite(stack)
    numerator = np.sum(np.where(finite, stack, 0.0), axis=0)
    denominator = finite.sum(axis=0)
    scores = np.divide(
        numerator,
        denominator,
        out=np.full(modeled.shape[1], -np.inf),
        where=denominator > 0,
    )
    return scores, counts


def adaptive_background(
    prior: np.ndarray,
    pools: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    residual_variance: np.ndarray,
    longitude: np.ndarray,
    branch_by_year: np.ndarray,
    balanced: bool,
    years: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    if observations.shape[1] != len(years) or len(branch_by_year) != len(years):
        raise RuntimeError('adaptive-background-year-shape')
    result = np.full((len(years), prior.shape[0]), np.nan)
    rows: list[dict[str, object]] = []
    selected_indices = np.full((len(years), len(pools), 100), -1, dtype=int)
    for year_index, branch in enumerate(branch_by_year):
        valid_operator = (
            np.isfinite(residual_variance[:, branch])
            & (residual_variance[:, branch] > 0)
            & np.all(np.isfinite(hxb[:, branch]), axis=1)
        )
        active = np.isfinite(observations[:, year_index]) & valid_operator
        if active.sum() < 3:
            fields = np.stack([np.mean(prior[:, pool], axis=1) for pool in pools])
            result[year_index] = np.mean(fields, axis=0)
            rows.append({
                "year": int(years[year_index]),
                "branch": int(branch),
                "active_proxy_count": int(active.sum()),
                "active_west_count": 0,
                "active_central_count": 0,
                "active_east_count": 0,
                "valid_band_count": 0,
            })
            continue
        scores, counts = selection_scores(
            observations[active, year_index],
            hxb[active, branch],
            longitude[active],
            balanced,
        )
        fields = []
        for pool_index, pool in enumerate(pools):
            order = np.argsort(-scores[pool], kind="stable")[:100]
            selected = pool[order]
            selected_indices[year_index, pool_index] = selected
            fields.append(np.mean(prior[:, selected], axis=1))
        result[year_index] = np.mean(fields, axis=0)
        rows.append({
            "year": int(years[year_index]),
            "branch": int(branch),
            "active_proxy_count": int(active.sum()),
            "active_west_count": int(counts.get("west_lt80", 0)),
            "active_central_count": int(counts.get("central_80_90", 0)),
            "active_east_count": int(counts.get("east_ge90", 0)),
            "valid_band_count": int(sum(value >= 3 for value in counts.values())) if balanced else 1,
        })
    return result, rows, selected_indices


def region_weights(mask: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    lon2 = np.broadcast_to(longitudes[None], mask.shape)
    cosine = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], mask.shape)
    result = {}
    for name, (lower, upper) in REGIONS.items():
        selected = mask & (lon2 >= lower) & (lon2 < upper)
        if not selected.any():
            raise RuntimeError(f"empty-region:{name}")
        result[name] = cosine * selected
    return result


def weighted_cube_series(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return common.weighted_series(cube.reshape(cube.shape[0], -1).T, weights.reshape(-1))


def components(values: np.ndarray) -> dict[str, np.ndarray]:
    annual = values - np.nanmean(values)
    low = pd.Series(values).rolling(11, center=True, min_periods=11).mean().to_numpy()
    valid = np.isfinite(low)
    return {"annual": annual, "low11": low[valid] - np.mean(low[valid])}


def metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if valid.sum() < 3:
        return {"rmse_degC": np.nan, "correlation": np.nan, "beta": np.nan}
    left = predicted[valid]
    right = observed[valid]
    variance = float(np.var(right, ddof=1))
    return {
        "rmse_degC": common.rmse(left, right),
        "correlation": common.correlation(left, right),
        "beta": float(np.cov(left, right, ddof=1)[0, 1] / variance) if variance > 0 else np.nan,
    }


def score_branches(
    branch_curves: dict[str, dict[str, np.ndarray]],
    observed_curves: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        selected = (YEARS >= start) & (YEARS <= end)
        for region in REGIONS:
            for branch, curves in branch_curves.items():
                pred_components = components(curves[region][selected])
                for product, product_curves in observed_curves.items():
                    obs_components = components(product_curves[region][selected])
                    for frequency in ("annual", "low11"):
                        item = metrics(pred_components[frequency], obs_components[frequency])
                        rows.append({
                            "fold": fold,
                            "period": f"{start}-{end}",
                            "region": region,
                            "branch": branch,
                            "product": product,
                            "frequency": frequency,
                            **item,
                        })
    return pd.DataFrame(rows)


def comparisons(scores: pd.DataFrame, candidate: str, control: str, label: str) -> pd.DataFrame:
    keys = ["fold", "period", "region", "product", "frequency"]
    left = scores[scores.branch.eq(candidate)][keys + ["rmse_degC"]].rename(columns={"rmse_degC": "candidate_rmse_degC"})
    right = scores[scores.branch.eq(control)][keys + ["rmse_degC"]].rename(columns={"rmse_degC": "control_rmse_degC"})
    merged = left.merge(right, on=keys, validate="one_to_one")
    merged["comparison"] = label
    merged["rmse_improvement"] = 1.0 - merged.candidate_rmse_degC / merged.control_rmse_degC
    return merged


def main() -> int:
    root = DRIVER.parents[3]
    results = root / "validation_data/results/paleo_surface_validation_v0"
    loop = results / "climate_reconstruction_improvement_loop_v2"
    output = loop / "round_154_temperature_balanced_adaptive_background_oof_v3"
    if output.exists():
        raise FileExistsError(output)
    contract = DRIVER / "round154_temperature_balanced_adaptive_background_oof_v3_contract.md"
    fold_source = results / "temperature_fold_native_oenkf_amplitude_20year_cv_v1"
    allowed_root = results / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "prior": results / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6_operator": fold_source / "proxy6_20year_operators_500.mat",
        "proxy9_operator": fold_source / "proxy9_20year_operators_500.mat",
        "proxy6_predictors": fold_source / "proxy6_archived_model_predictors_848.mat",
        "proxy9_predictors": fold_source / "proxy9_archived_model_predictors_848.mat",
        "proxy6_allowed": allowed_root / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9_allowed": allowed_root / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "proxy6_metadata": results / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9_metadata": results / "climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "proxy6_source": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9_source": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "instrumental": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": results / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    hash_names = set(paths) - {"instrumental", "proxy6_source", "proxy9_source"}
    before = {name: file_state(path, name in hash_names) for name, path in paths.items()}
    output.mkdir(parents=True)
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError(f"prior-pool-shape:{prior.shape}:{pools.shape}")
    branch_by_year = np.full(len(YEARS), -1, dtype=int)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        branch_by_year[(YEARS >= start) & (YEARS <= end)] = fold
    if np.any(branch_by_year < 0) or np.any((YEARS >= 1921) & (YEARS <= 1939)):
        raise RuntimeError("year-contract-failed")

    operators = {}
    longitudes = {}
    for network in (6, 9):
        operators[network] = load_operator(
            paths[f"proxy{network}_operator"],
            paths[f"proxy{network}_predictors"],
            paths[f"proxy{network}_allowed"],
        )
        longitudes[network] = harmonized_longitudes(network, paths[f"proxy{network}_metadata"])
        if len(longitudes[network]) != operators[network]["observations"].shape[0]:
            raise RuntimeError(f"metadata-operator-count:{network}")
        if not np.array_equal(operators[network]["sample_indices"], pools[0]):
            raise RuntimeError(f"experiment1-index-closure:{network}")

    fixed_flat = np.mean(np.stack([np.mean(prior[:, pool], axis=1) for pool in pools]), axis=0)
    fixed = np.repeat(fixed_flat[None], len(YEARS), axis=0)
    branches_by_network: dict[int, dict[str, np.ndarray]] = {}
    active_rows = []
    selected_store = {}
    for network in (6, 9):
        operator = operators[network]
        branches_by_network[network] = {}
        for name, balanced, current_pools in (
            ("current_global", False, pools),
            ("balanced_equal_bands", True, pools),
            ("current_global_exp1", False, pools[:1]),
            ("balanced_equal_bands_exp1", True, pools[:1]),
        ):
            field, rows, selected = adaptive_background(
                prior,
                current_pools,
                np.asarray(operator["observations"]),
                np.asarray(operator["hxb"]),
                np.asarray(operator["residual_variance"]),
                longitudes[network],
                branch_by_year,
                balanced,
                YEARS,
            )
            branches_by_network[network][name] = field
            selected_store[f"proxy{network}_{name}"] = selected
            active_rows.extend({"network": network, "selector": name, **row} for row in rows)
    pd.DataFrame(active_rows).to_csv(output / "active_proxy_band_counts.csv", index=False)

    state_branches = {"fixed_background": fixed}
    for branch in ("current_global", "balanced_equal_bands", "current_global_exp1", "balanced_equal_bands_exp1"):
        state_branches[branch] = (
            branches_by_network[6][branch] + branches_by_network[9][branch] + fixed
        ) / 3.0

    native_lat, native_lon = common.state_coordinates()
    with h5py.File(paths["instrumental"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observations = {
            name: common.clean(np.asarray(handle[common.OBSERVATION_VARIABLES[name]][time_indices], dtype=float))
            for name in ALL_PRODUCTS
        }
    mask = np.load(paths["hma_mask"]).astype(bool)
    if mask.shape != observations["CRU"].shape[1:]:
        raise RuntimeError(f"instrument-mask-shape:{mask.shape}:{observations['CRU'].shape}")
    weights = region_weights(mask, source_lat, source_lon)
    field_cubes = {
        name: common.source_grid(
            common.state_to_cube(values.T, len(native_lat), len(native_lon)),
            native_lat,
            native_lon,
            source_lat,
            source_lon,
        )
        for name, values in state_branches.items()
    }
    branch_curves = {
        name: {region: weighted_cube_series(cube, weight) for region, weight in weights.items()}
        for name, cube in field_cubes.items()
    }
    observed_curves = {
        name: {region: weighted_cube_series(cube, weight) for region, weight in weights.items()}
        for name, cube in observations.items()
    }

    score_table = score_branches(branch_curves, observed_curves)
    score_table.to_csv(output / "blocked_oof_scores.csv", index=False)
    pair_tables = [
        comparisons(score_table, "balanced_equal_bands", "current_global", "balanced50_vs_current50"),
        comparisons(score_table, "balanced_equal_bands", "fixed_background", "balanced50_vs_fixed"),
        comparisons(score_table, "balanced_equal_bands_exp1", "current_global_exp1", "balanced_exp1_vs_current_exp1"),
    ]
    pairwise = pd.concat(pair_tables, ignore_index=True)
    pairwise.to_csv(output / "blocked_oof_pairwise_improvements.csv", index=False)

    primary = pairwise[
        pairwise.comparison.eq("balanced50_vs_current50")
        & pairwise.frequency.eq("low11")
        & pairwise.product.isin(PRIMARY_PRODUCTS)
    ].copy()
    exp1 = pairwise[
        pairwise.comparison.eq("balanced_exp1_vs_current_exp1")
        & pairwise.frequency.eq("low11")
        & pairwise.product.isin(PRIMARY_PRODUCTS)
    ]
    region_medians = primary.groupby("region").rmse_improvement.median()
    fold_medians = primary.groupby("fold").rmse_improvement.median()
    product_medians = primary.groupby("product").rmse_improvement.median()
    primary_median = float(primary.rmse_improvement.median())
    improving_fraction = float((primary.rmse_improvement > 0).mean())
    exp1_median = float(exp1.rmse_improvement.median())
    gates = {
        "median_low11_rmse_improvement_at_least_5pct": primary_median >= 0.05,
        "at_least_60pct_region_product_fold_pairs_improve": improving_fraction >= 0.60,
        "east_median_improvement_positive": float(region_medians["east_ge90"]) > 0,
        "no_region_median_regression_worse_than_5pct": float(region_medians.min()) >= -0.05,
        "at_least_two_of_three_folds_positive": int((fold_medians > 0).sum()) >= 2,
        "no_fold_median_regression_worse_than_5pct": float(fold_medians.min()) >= -0.05,
        "at_least_three_of_four_products_positive": int((product_medians > 0).sum()) >= 3,
        "experiment1_direction_positive": exp1_median > 0,
    }
    passes = all(gates.values())

    curve_rows = []
    for region in REGIONS:
        for name, curves in branch_curves.items():
            for year, value in zip(YEARS, curves[region]):
                curve_rows.append({"year": int(year), "region": region, "kind": "background", "series": name, "value_degC": float(value)})
        for name, curves in observed_curves.items():
            for year, value in zip(YEARS, curves[region]):
                curve_rows.append({"year": int(year), "region": region, "kind": "observation", "series": name, "value_degC": float(value)})
    pd.DataFrame(curve_rows).to_csv(output / "modern_regional_curves.csv", index=False)
    np.savez_compressed(
        output / "balanced_background_oof_fields.npz",
        years=YEARS,
        source_lat=source_lat,
        source_lon=source_lon,
        hma_mask=mask,
        **field_cubes,
    )

    paleo_summary = None
    if passes:
        paleo_years = np.arange(1901, 1921, dtype=int)
        paleo_network = {}
        for network in (6, 9):
            _, paleo_observations, locked = round25.read_proxy_allowed(paths[f"proxy{network}_source"], paleo_years)
            if locked:
                raise RuntimeError("locked-paleo-proxy-value-read")
            full_hxb = np.asarray(operators[network]["hxb"])[:, 0]
            full_variance = np.asarray(operators[network]["residual_variance"])[:, 0]
            full_branch = np.zeros(len(paleo_years), dtype=int)
            current, _, _ = adaptive_background(
                prior, pools, paleo_observations, full_hxb[:, None, :], full_variance[:, None],
                longitudes[network], full_branch, False, paleo_years,
            )
            balanced, _, _ = adaptive_background(
                prior, pools, paleo_observations, full_hxb[:, None, :], full_variance[:, None],
                longitudes[network], full_branch, True, paleo_years,
            )
            paleo_network[network] = {"current_global": current, "balanced_equal_bands": balanced}
        paleo_state = {
            name: (paleo_network[6][name] + paleo_network[9][name] + np.repeat(fixed_flat[None], 20, axis=0)) / 3.0
            for name in ("current_global", "balanced_equal_bands")
        }
        paleo_cubes = {
            name: common.source_grid(
                common.state_to_cube(values.T, len(native_lat), len(native_lon)),
                native_lat, native_lon, source_lat, source_lon,
            )
            for name, values in paleo_state.items()
        }
        paleo_rows = []
        paleo_trends = {}
        for name, cube in paleo_cubes.items():
            paleo_trends[name] = {}
            for region, weight in weights.items():
                values = weighted_cube_series(cube, weight)
                paleo_trends[name][region] = float(np.polyfit(paleo_years, values, 1)[0] * 100.0)
                for year, value in zip(paleo_years, values):
                    paleo_rows.append({"year": int(year), "region": region, "series": name, "value_degC": float(value)})
        pd.DataFrame(paleo_rows).to_csv(output / "allowed_1901_1920_candidate_curves.csv", index=False)
        paleo_summary = {
            "trends_degC_per_century": paleo_trends,
            "east_trend_change_degC_per_century": paleo_trends["balanced_equal_bands"]["east_ge90"] - paleo_trends["current_global"]["east_ge90"],
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, region in enumerate(("hma", "east_ge90")):
        axis = axes[0, column]
        axis.plot(YEARS, branch_curves["current_global"][region], color="#B2182B", lw=1.8, label="Current global selector")
        axis.plot(YEARS, branch_curves["balanced_equal_bands"][region], color="#2166AC", lw=2.1, label="Balanced selector")
        observed_mean = np.nanmean(np.stack([observed_curves[name][region] for name in PRIMARY_PRODUCTS]), axis=0)
        axis.plot(YEARS, observed_mean, color="#222222", lw=1.2, alpha=0.8, label="Four-product mean")
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#999999", lw=0.7, ls="--")
        axis.axhline(0, color="#999999", lw=0.6)
        axis.set_title("HMA" if region == "hma" else "East HMA")
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    heat = primary.pivot_table(index="region", columns="product", values="rmse_improvement", aggfunc="median").reindex(REGIONS).reindex(columns=PRIMARY_PRODUCTS)
    image = axes[1, 0].imshow(100.0 * heat.to_numpy(), cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    axes[1, 0].set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=25, ha="right")
    axes[1, 0].set_yticks(np.arange(len(heat.index)), heat.index)
    axes[1, 0].set_title("Median low-pass RMSE improvement (%)")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].bar([str(index) for index in fold_medians.index], 100.0 * fold_medians.to_numpy(), color=["#2166AC" if value > 0 else "#B2182B" for value in fold_medians])
    axes[1, 1].axhline(0, color="#222222", lw=0.8)
    axes[1, 1].axhline(5, color="#777777", lw=0.7, ls="--")
    axes[1, 1].set(title="Held-out block stability", xlabel="Fold", ylabel="Median RMSE improvement (%)")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "temperature_balanced_adaptive_background_oof.png", dpi=220)
    plt.close(fig)

    if paleo_summary is not None:
        paleo = pd.read_csv(output / "allowed_1901_1920_candidate_curves.csv")
        fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        for name, color in (("current_global", "#B2182B"), ("balanced_equal_bands", "#2166AC")):
            values = paleo[(paleo.region == "east_ge90") & (paleo.series == name)]
            axis.plot(values.year, values.value_degC, color=color, lw=2.2, label=name.replace("_", " "))
        axis.set(title="Allowed eastern-HMA background candidate, 1901-1920", xlabel="Year", ylabel="Temperature anomaly (degC)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        fig.savefig(output / "allowed_1901_1920_balanced_background_candidate.png", dpi=220)
        plt.close(fig)

    after = {name: file_state(path, False) for name, path in paths.items()}
    sources_unchanged = all(
        before[name]["size_bytes"] == after[name]["size_bytes"]
        and before[name]["mtime_ns"] == after[name]["mtime_ns"]
        for name in paths
    )
    if not sources_unchanged:
        raise RuntimeError("source-purity-failed")
    selection_overlap = {}
    for network in (6, 9):
        current = selected_store[f"proxy{network}_current_global"]
        balanced = selected_store[f"proxy{network}_balanced_equal_bands"]
        overlap = [len(np.intersect1d(current[y, pool], balanced[y, pool])) for y in range(len(YEARS)) for pool in range(len(pools))]
        selection_overlap[str(network)] = {
            "median_selected_member_overlap_of_100": float(np.median(overlap)),
            "minimum_selected_member_overlap_of_100": int(np.min(overlap)),
        }
    summary = {
        "experiment_id": "round154_temperature_balanced_adaptive_background_oof_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": "advance_balanced_selector_to_allowed_trajectory" if passes else "reject_balanced_selector",
        "passes_all_frozen_gates": passes,
        "primary_median_low11_rmse_improvement": primary_median,
        "primary_improving_pair_fraction": improving_fraction,
        "region_median_improvements": {str(key): float(value) for key, value in region_medians.items()},
        "fold_median_improvements": {str(int(key)): float(value) for key, value in fold_medians.items()},
        "product_median_improvements": {str(key): float(value) for key, value in product_medians.items()},
        "experiment1_median_improvement": exp1_median,
        "gates": gates,
        "selection_overlap": selection_overlap,
        "hxb848_closure": {str(network): float(operators[network]["hxb848_closure"]) for network in (6, 9)},
        "paleo_candidate_generated": paleo_summary is not None,
        "paleo_summary": paleo_summary,
        "locked_1921_1939_accessed": False,
        "jionglacuo_accessed": False,
        "production_enkf_run": False,
        "oggm_run": False,
        "all_declared_sources_unchanged": sources_unchanged,
        "interpretation_boundary": "A deterministic background-selector test. Passing does not promote a production climate field.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after_stat": after}, indent=2), encoding="utf-8")
    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    paleo_line = ""
    if paleo_summary is not None:
        paleo_line = (
            f"- Current/balanced east 1901-1920 trends: "
            f"`{paleo_summary['trends_degC_per_century']['current_global']['east_ge90']:+.3f}` / "
            f"`{paleo_summary['trends_degC_per_century']['balanced_equal_bands']['east_ge90']:+.3f} degC/century`.\n"
        )
    (output / "FINAL_RESULT.md").write_text(
        "# Spatially Balanced Adaptive Background OOF\n\n"
        f"- Decision: `{summary['decision']}`.\n"
        f"- Median primary low-pass RMSE improvement: `{100 * primary_median:+.3f}%`.\n"
        f"- Improving region-product-fold pairs: `{100 * improving_fraction:.1f}%`.\n"
        f"- Experiment-1 median improvement: `{100 * exp1_median:+.3f}%`.\n"
        + paleo_line
        + "\n## Frozen gates\n\n" + gate_lines
        + "\n\nNo locked-period, production EnKF, Jionglacuo, or OGGM access occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
