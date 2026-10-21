"""Nested blocked-field validation of an annual OOF proxy filter."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_precipitation_nested_seasonal_tree_psm_cv_v1 as psm
from run_climate_reconstruction_round03_ao9_operator_oof_v1 import (
    REGIONS,
    anomaly,
    area_series,
    clean,
    correlation,
    source_grid,
    state_coordinates,
    to_cube,
)


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_nested_blocked_field_validation_v1"
YEARS = np.arange(1941, 2001, dtype=int)
BLOCK = (YEARS - YEARS[0]) // 10
BRANCHES = ("equal_six_control", "blocked_all_tree", "blocked_annual_oof_filter")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def source_state(path: Path, hashed: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if hashed:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        result["sha256"] = value.hexdigest()
    return result


def fit(response: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray, selected: np.ndarray) -> np.ndarray | None:
    valid = selected & np.isfinite(response) & np.isfinite(temperature) & np.isfinite(precipitation)
    if int(valid.sum()) < 30:
        return None
    design = np.column_stack((np.ones(valid.sum()), temperature[valid], precipitation[valid]))
    return np.linalg.lstsq(design, response[valid], rcond=None)[0]


def predict(coefficients: np.ndarray | None, temperature: np.ndarray, precipitation: np.ndarray, selected: np.ndarray) -> np.ndarray:
    result = np.full(len(response_placeholder := temperature), np.nan)
    valid = selected & np.isfinite(temperature) & np.isfinite(precipitation)
    if coefficients is not None:
        result[valid] = coefficients[0] + coefficients[1] * temperature[valid] + coefficients[2] * precipitation[valid]
    return result


def metric_nrmse(observed: np.ndarray, predicted: np.ndarray, selected: np.ndarray) -> float:
    valid = selected & np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 30 or np.std(observed[valid]) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)) / np.std(observed[valid]))


def metric_correlation(observed: np.ndarray, predicted: np.ndarray, selected: np.ndarray) -> float:
    valid = selected & np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 3 or np.std(observed[valid]) == 0 or np.std(predicted[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1])


def classify_proxy(response: np.ndarray, bea_t: np.ndarray, bea_p: np.ndarray, cru_t: np.ndarray, cru_p: np.ndarray, outer_train: np.ndarray) -> tuple[bool, dict[str, float]]:
    predictions = {name: np.full(len(YEARS), np.nan) for name in ("bea", "cru", "bea_to_cru", "cru_to_bea")}
    for inner_block in np.unique(BLOCK[outer_train]):
        test = outer_train & (BLOCK == inner_block)
        train = outer_train & ~test
        bea_coef = fit(response, bea_t, bea_p, train)
        cru_coef = fit(response, cru_t, cru_p, train)
        predictions["bea"][test] = predict(bea_coef, bea_t, bea_p, test)[test]
        predictions["bea_to_cru"][test] = predict(bea_coef, cru_t, cru_p, test)[test]
        predictions["cru"][test] = predict(cru_coef, cru_t, cru_p, test)[test]
        predictions["cru_to_bea"][test] = predict(cru_coef, bea_t, bea_p, test)[test]
    metrics = {
        "nrmse_bea": metric_nrmse(response, predictions["bea"], outer_train),
        "nrmse_cru": metric_nrmse(response, predictions["cru"], outer_train),
        "nrmse_bea_to_cru": metric_nrmse(response, predictions["bea_to_cru"], outer_train),
        "nrmse_cru_to_bea": metric_nrmse(response, predictions["cru_to_bea"], outer_train),
        "correlation_bea": metric_correlation(response, predictions["bea"], outer_train),
        "correlation_cru": metric_correlation(response, predictions["cru"], outer_train),
    }
    admissible = bool(
        all(np.isfinite(value) for value in metrics.values())
        and all(metrics[name] < 1 for name in ("nrmse_bea", "nrmse_cru", "nrmse_bea_to_cru", "nrmse_cru_to_bea"))
        and metrics["correlation_bea"] > 0
        and metrics["correlation_cru"] > 0
    )
    return admissible, metrics


def analysis_mean(prior: np.ndarray, hxb: np.ndarray, observation: np.ndarray, error_variance: np.ndarray) -> np.ndarray:
    members = prior.shape[1]
    prior_mean = prior.mean(axis=1)
    hxb_mean = hxb.mean(axis=1)
    prior_prime = prior - prior_mean[:, None]
    hxb_prime = hxb - hxb_mean[:, None]
    pbht = prior_prime @ hxb_prime.T / (members - 1)
    covariance = hxb_prime @ hxb_prime.T / (members - 1) + np.diag(error_variance)
    innovation = observation - hxb_mean
    try:
        factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
        solved = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
    except np.linalg.LinAlgError:
        solved = scipy.linalg.solve(covariance, innovation, assume_a="sym", check_finite=False)
    return prior_mean + pbht @ solved


def main() -> int:
    project = root()
    result_root = project / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    loop = result_root / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop") / "".join(chr(v) for v in (25991, 20214, 25972, 29702)) / ("1 " + "".join(chr(v) for v in (35770, 25991))) / "work1/compared_result_all"
    sources = {
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "metadata": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "replay": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "berkeley_temperature": psm.BERKELEY_TEMPERATURE,
        "gpcc_precipitation": psm.GPCC_PRECIPITATION,
        "cru_temperature": psm.CRU_TEMPERATURE,
        "cru_precipitation": psm.CRU_PRECIPITATION,
        "netcdf": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": result_root / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.intersect1d(YEARS, np.arange(1921, 1940)).size:
        raise RuntimeError("locked-period-requested")
    before = {name: source_state(path, hashed=name in {"allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}

    allowed = scipy.io.loadmat(sources["allowed_proxy6"])
    allowed_years = np.asarray(allowed["years"]).reshape(-1).astype(int)
    allowed_obs = np.asarray(allowed["observations"], dtype=float)
    row_index = np.asarray([int(np.flatnonzero(allowed_years == year)[0]) for year in YEARS])
    observations = allowed_obs[row_index]
    sample = np.asarray(allowed["closure_indices"]).reshape(-1).astype(int) - 1
    archived_closure_hxb = np.asarray(allowed["closure_hxb"], dtype=float)
    if observations.shape != (60, 202) or sample.shape != (500,) or archived_closure_hxb.shape != (202, 500):
        raise RuntimeError("allowed-input-shape-conflict")
    replay = scipy.io.loadmat(sources["replay"], variable_names=["R0v_filled"])
    error_variance = np.nanvar(np.asarray(replay["R0v_filled"], dtype=float), axis=1, ddof=1)
    if np.any(~np.isfinite(error_variance)) or np.any(error_variance <= 0):
        raise RuntimeError("invalid-regem-error-variance")
    with h5py.File(sources["prior"], "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T[:, sample]

    metadata = pd.read_csv(sources["metadata"])
    tree = metadata[(metadata["proxy_type"] == "tree") & (metadata["coefficient_count"] == 3)].sort_values("proxy_index")
    if len(tree) != 201 or not np.array_equal(tree["proxy_index"].to_numpy(int), np.arange(1, 202)):
        raise RuntimeError("tree-metadata-contract-failed")
    latitudes = tree["latitude"].to_numpy(float)
    longitudes = tree["longitude"].to_numpy(float)
    response = observations[:, :201]

    bea_monthly, _, _ = psm.read_berkeley_points(latitudes, longitudes)
    gpcc_monthly, _, _ = psm.read_netcdf_points(psm.GPCC_PRECIPITATION, "precip", 1891, latitudes, longitudes)
    cru_t_monthly, _, _ = psm.read_netcdf_points(psm.CRU_TEMPERATURE, "tmp", 1901, latitudes, longitudes)
    cru_p_monthly, _, _ = psm.read_netcdf_points(psm.CRU_PRECIPITATION, "pre", 1901, latitudes, longitudes)
    bea_t = psm.center_reference(psm.aggregate_temperature(bea_monthly, psm.SEASONS["annual"]))
    bea_p = psm.center_reference(psm.aggregate_precipitation(gpcc_monthly, psm.SEASONS["annual"]))
    cru_t = psm.center_reference(psm.aggregate_temperature(cru_t_monthly, psm.SEASONS["annual"]))
    cru_p = psm.center_reference(psm.aggregate_precipitation(cru_p_monthly, psm.SEASONS["annual"]))

    mt = np.full((201, 500), np.nan)
    mp = np.full((201, 500), np.nan)
    with h5py.File(sources["proxy6"], "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column in range(201):
            proxy = handle[refs[column]]
            mt[column] = np.asarray(proxy["M_t"]).reshape(-1, order="F")[sample]
            mp[column] = np.asarray(proxy["M_p"]).reshape(-1, order="F")[sample]
    if not (np.isfinite(mt).all() and np.isfinite(mp).all()):
        raise RuntimeError("prior-predictor-nonfinite")

    blocked_all = np.full((prior.shape[0], len(YEARS)), np.nan)
    blocked_filter = np.full_like(blocked_all, np.nan)
    fold_rows: list[dict[str, object]] = []
    proxy_rows: list[dict[str, object]] = []
    for outer_block in np.unique(BLOCK):
        test = BLOCK == outer_block
        train = ~test
        hxb_full = np.empty((202, 500), dtype=float)
        hxb_full[201] = archived_closure_hxb[201]
        admissible = np.zeros(201, dtype=bool)
        fitted = np.zeros(201, dtype=bool)
        for column in range(201):
            is_admissible, metrics = classify_proxy(response[:, column], bea_t[:, column], bea_p[:, column], cru_t[:, column], cru_p[:, column], train)
            coefficients = fit(response[:, column], bea_t[:, column], bea_p[:, column], train)
            fitted[column] = coefficients is not None
            admissible[column] = is_admissible and fitted[column]
            if coefficients is None:
                hxb_full[column] = archived_closure_hxb[column].mean()
            else:
                hxb_full[column] = coefficients[0] + coefficients[1] * mt[column] + coefficients[2] * mp[column]
            proxy_rows.append({
                "outer_fold": int(outer_block + 1),
                "test_start": int(YEARS[test][0]),
                "test_end": int(YEARS[test][-1]),
                "proxy_index": int(column + 1),
                "full_training_fit_available": bool(fitted[column]),
                "annual_oof_admissible": bool(admissible[column]),
                **metrics,
            })
        hxb_filter = hxb_full.copy()
        means = hxb_filter.mean(axis=1)
        failing = np.flatnonzero(~admissible)
        hxb_filter[failing] = means[failing, None]
        fold_rows.append({
            "outer_fold": int(outer_block + 1),
            "test_start": int(YEARS[test][0]),
            "test_end": int(YEARS[test][-1]),
            "training_year_count": int(train.sum()),
            "fitted_tree_count": int(fitted.sum()),
            "admissible_tree_count": int(admissible.sum()),
        })
        for year_index in np.flatnonzero(test):
            active = np.flatnonzero(np.isfinite(observations[year_index]))
            blocked_all[:, year_index] = analysis_mean(prior, hxb_full[active], observations[year_index, active], error_variance[active])
            blocked_filter[:, year_index] = analysis_mean(prior, hxb_filter[active], observations[year_index, active], error_variance[active])

    if not (np.isfinite(blocked_all).all() and np.isfinite(blocked_filter).all()):
        raise RuntimeError("blocked-analysis-incomplete")
    grid_lat, grid_lon = state_coordinates()
    all_cube = anomaly(to_cube(blocked_all, len(grid_lat), len(grid_lon)))
    filter_cube = anomaly(to_cube(blocked_filter, len(grid_lat), len(grid_lon)))
    with h5py.File(sources["netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        control05 = clean(np.asarray(handle["pr_ensemble_mean_rec_result"][indices], dtype=float))
        cru05 = clean(np.asarray(handle["pr_obs_cru_pre_05deg"][indices], dtype=float))
        gpcc05 = clean(np.asarray(handle["pr_obs_gpcp_05deg"][indices], dtype=float))
    control_cube = anomaly(source_grid(control05, source_lat, source_lon, grid_lat, grid_lon))
    observed = {
        "CRU": anomaly(source_grid(cru05, source_lat, source_lon, grid_lat, grid_lon)),
        "GPCC": anomaly(source_grid(gpcc05, source_lat, source_lon, grid_lat, grid_lon)),
    }
    mask05 = np.load(sources["hma_mask"]).astype(float)
    mask2 = source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    lon2 = np.broadcast_to(grid_lon[None], mask2.shape)
    masks = {name: mask2 & (lon2 >= lower) & (lon2 < upper) for name, (lower, upper) in REGIONS.items()}
    fields = {"equal_six_control": control_cube, "blocked_all_tree": all_cube, "blocked_annual_oof_filter": filter_cube}

    curve_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, np.ndarray]] = {name: {} for name in fields}
    obs_curves: dict[str, dict[str, np.ndarray]] = {name: {} for name in observed}
    for region, mask in masks.items():
        for branch, field in fields.items():
            values = area_series(field, grid_lat, mask)
            curves[branch][region] = values
            curve_rows.extend({"year": int(year), "region": region, "series_type": "reconstruction", "series": branch, "value_mm_day": float(value)} for year, value in zip(YEARS, values))
        for name, field in observed.items():
            values = area_series(field, grid_lat, mask)
            obs_curves[name][region] = values
            curve_rows.extend({"year": int(year), "region": region, "series_type": "observation", "series": name, "value_mm_day": float(value)} for year, value in zip(YEARS, values))
        for branch in fields:
            for name in observed:
                left = curves[branch][region]
                right = obs_curves[name][region]
                valid = np.isfinite(left) & np.isfinite(right)
                delta = left[valid] - right[valid]
                score_rows.append({
                    "branch": branch,
                    "region": region,
                    "observation": name,
                    "valid_year_count": int(valid.sum()),
                    "rmse_mm_day": float(np.sqrt(np.mean(delta ** 2))),
                    "correlation": correlation(left, right),
                })
    scores = pd.DataFrame(score_rows)
    control_scores = scores[scores["branch"] == "equal_six_control"].set_index(["region", "observation"])
    control_median = float(control_scores["rmse_mm_day"].median())
    aggregate_rows = []
    for branch in BRANCHES:
        local = scores[scores["branch"] == branch].copy()
        regressions = []
        for _, row in local.iterrows():
            baseline = float(control_scores.loc[(row["region"], row["observation"]), "rmse_mm_day"])
            regressions.append(float(row["rmse_mm_day"] / baseline - 1))
        median = float(local["rmse_mm_day"].median())
        improvement = float(1 - median / control_median)
        maximum_regression = float(max(regressions))
        aggregate_rows.append({
            "branch": branch,
            "median_region_reference_rmse_mm_day": median,
            "relative_improvement_vs_equal_six": improvement,
            "maximum_pair_relative_regression": maximum_regression,
            "passes_5pct_primary": bool(improvement >= 0.05),
            "passes_5pct_no_regression": bool(maximum_regression <= 0.05 + 1e-12),
            "passes_gate": bool(improvement >= 0.05 and maximum_regression <= 0.05 + 1e-12),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    candidate = aggregate[aggregate["branch"] == "blocked_annual_oof_filter"].iloc[0]
    decision = "advance_to_1901_1920_candidate_reconstruction" if bool(candidate["passes_gate"]) else "close_annual_oof_filter_without_early_candidate"

    output.mkdir(parents=True)
    pd.DataFrame(fold_rows).to_csv(output / "outer_fold_proxy_counts.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(output / "outer_fold_proxy_admissibility.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "regional_curves_1941_2000.csv", index=False)
    scores.to_csv(output / "regional_reference_scores.csv", index=False)
    aggregate.to_csv(output / "aggregate_gate_scores.csv", index=False)
    np.savez_compressed(output / "blocked_field_arrays.npz", years=YEARS, blocked_all_tree=blocked_all, blocked_annual_oof_filter=blocked_filter, grid_lat=grid_lat, grid_lon=grid_lon, hma_mask=mask2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, constrained_layout=True)
    colors = {"equal_six_control": "#111111", "blocked_all_tree": "#D55E00", "blocked_annual_oof_filter": "#009E73", "CRU": "#0072B2", "GPCC": "#CC79A7"}
    labels = {"equal_six_control": "Existing equal-six", "blocked_all_tree": "Blocked all-tree", "blocked_annual_oof_filter": "Blocked annual OOF filter", "CRU": "CRU", "GPCC": "GPCC"}
    for axis, region in zip(axes.ravel(), REGIONS):
        for branch in BRANCHES:
            axis.plot(YEARS, curves[branch][region], color=colors[branch], linewidth=2.1 if branch == "equal_six_control" else 1.7, label=labels[branch])
        for name in observed:
            axis.plot(YEARS, obs_curves[name][region], color=colors[name], linestyle="--", linewidth=1.3, label=labels[name])
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set(title=region, ylabel="Precipitation anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "nested_blocked_field_curves_1941_2000.png", dpi=220)
    plt.close(fig)

    after = {name: source_state(path, hashed=name in {"allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1941, 2000],
        "outer_blocks": [[int(YEARS[BLOCK == block][0]), int(YEARS[BLOCK == block][-1])] for block in np.unique(BLOCK)],
        "locked_1921_1939_accessed": False,
        "admissibility_is_training_fold_only": True,
        "operator_fit_product": "Berkeley Earth annual temperature plus GPCC annual precipitation",
        "regem_error_frozen": True,
        "prior_members_frozen": 500,
        "fold_proxy_counts": fold_rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "candidate_gate": candidate.to_dict(),
        "decision": decision,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "This is the decisive modern-period field gate. Only a pass authorizes constructing an early-period candidate; it does not by itself replace production.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Annual OOF Nested Blocked-Field Contract\n\nUse only 1941-2000. For every outer ten-year block, derive proxy admissibility and fit annual PSMs from the other 50 years, then reconstruct only the held-out block. Score against CRU and GPCC over HMA/west/central/east. Advance only if median RMSE improves by at least 5% versus equal-six and no region/reference pair regresses by more than 5%. Do not access 1921-1939 or tune from 1901-1920.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Annual OOF Nested Blocked-Field Validation\n\n"
        f"- Candidate median RMSE: `{candidate['median_region_reference_rmse_mm_day']:.6f} mm/day`; improvement versus equal-six: `{candidate['relative_improvement_vs_equal_six']:.2%}`.\n"
        f"- Maximum region/reference regression: `{candidate['maximum_pair_relative_regression']:.2%}`.\n"
        f"- Gate pass: `{bool(candidate['passes_gate'])}`; decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-blocked-field-validation")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
