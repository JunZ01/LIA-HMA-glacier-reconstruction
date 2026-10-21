"""Evaluate blocked ridge and precipitation-only proxy9 PSMs through AO-EnKF."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_PATH = Path(__file__).with_name("run_climate_reconstruction_round03_ao9_operator_oof_v1.py")
SPEC = importlib.util.spec_from_file_location("round03_oof_tools", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("round03-tools-load-failed")
tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tools)

YEARS = tools.YEARS
BLOCKS = tools.BLOCKS
REGIONS = tools.REGIONS


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ao(
    prior: np.ndarray,
    sample: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    residual_variance: np.ndarray,
    branch_by_year: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]], float]:
    background = prior[:, sample]
    analysis = np.full((prior.shape[0], len(YEARS)), np.nan)
    selected = np.full((len(YEARS), 100), -1, dtype=int)
    active_rows: list[dict[str, int]] = []
    closure = 0.0
    for year_index, branch in enumerate(branch_by_year):
        operator_valid = (
            np.isfinite(residual_variance[:, branch])
            & (residual_variance[:, branch] > 0)
            & np.all(np.isfinite(hxb[:, branch][:, sample]), axis=1)
        )
        observed = np.isfinite(observations[:, year_index])
        active = np.flatnonzero(observed & operator_valid)
        active_rows.append({
            "year": int(YEARS[year_index]),
            "psm_branch": int(branch),
            "observed_proxy_count": int(observed.sum()),
            "valid_operator_proxy_count": int(operator_valid.sum()),
            "assimilated_proxy_count": int(len(active)),
        })
        if len(active) == 0:
            analysis[:, year_index] = np.mean(background, axis=1)
            continue
        current_hxb = hxb[active, branch][:, sample]
        local = tools.select_analogues(observations[active, year_index], current_hxb)
        selected[year_index] = local
        current, current_closure = tools.analysis_mean(
            background[:, local],
            current_hxb[:, local],
            observations[active, year_index],
            residual_variance[active, branch],
        )
        analysis[:, year_index] = current
        closure = max(closure, current_closure)
    return analysis, selected, active_rows, closure


def main() -> int:
    project = tools.root()
    loop_v1 = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1"
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_01_psm_stability_repair"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    paths = {
        "variants": output / "psm_stability_variants_v1.mat",
        "proxy_input": loop_v1 / "round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "parent_arrays": loop_v1 / "round_03_nested_ensemble_selection/ao9_operator_oof_arrays_v1.npz",
        "parent_summary": loop_v1 / "round_03_nested_ensemble_selection/operator_oof_summary_v1.json",
        "prior": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "indices": archive / "fixed_proxy_experiments_69/pr/indices/all_Xb_indices.mat",
        "netcdf": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "lock": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "config": output / "config.json",
        "contract": output / "ROUND_CONTRACT.md",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-loop-v2-round01-inputs:" + ";".join(missing))
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if lock.get("status") != "locked_not_evaluated" or np.any((YEARS >= 1921) & (YEARS <= 1939)):
        raise RuntimeError("locked-period-contract-failed")
    parent = json.loads(paths["parent_summary"].read_text(encoding="utf-8"))
    before = {name: state(path) for name, path in paths.items()}

    with h5py.File(paths["variants"], "r") as handle:
        ridge_hxb = np.asarray(handle["ridge_hxb"], dtype=float).transpose(2, 1, 0)
        precip_hxb = np.asarray(handle["precip_only_hxb"], dtype=float).transpose(2, 1, 0)
        ridge_variance = np.asarray(handle["ridge_residual_variance"], dtype=float).T
        precip_variance = np.asarray(handle["precip_only_residual_variance"], dtype=float).T
        ridge_lambda = np.asarray(handle["ridge_lambda"], dtype=float).T
        ridge_holdout = np.asarray(handle["ridge_holdout_rmse"], dtype=float).T
        precip_holdout = np.asarray(handle["precip_only_holdout_rmse"], dtype=float).T
    proxy = scipy.io.loadmat(paths["proxy_input"], variable_names=["observations"])
    observations = np.asarray(proxy["observations"], dtype=float).T
    with h5py.File(paths["prior"], "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    with np.load(paths["parent_arrays"]) as saved:
        parent_analysis = np.asarray(saved["oof_analysis"], dtype=float)

    branch_by_year = np.full(len(YEARS), -1, dtype=int)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        branch_by_year[(YEARS >= start) & (YEARS <= end)] = fold
    if np.any(branch_by_year < 0):
        raise RuntimeError("blocked-branch-coverage-failed")
    ridge_analysis, ridge_selected, ridge_active, ridge_closure = run_ao(
        prior, sample, observations, ridge_hxb, ridge_variance, branch_by_year
    )
    precip_analysis, precip_selected, precip_active, precip_closure = run_ao(
        prior, sample, observations, precip_hxb, precip_variance, branch_by_year
    )

    grid_lat, grid_lon = tools.state_coordinates()
    parent_cube = tools.anomaly(tools.to_cube(parent_analysis, len(grid_lat), len(grid_lon)))
    ridge_cube = tools.anomaly(tools.to_cube(ridge_analysis, len(grid_lat), len(grid_lon)))
    precip_cube = tools.anomaly(tools.to_cube(precip_analysis, len(grid_lat), len(grid_lon)))
    with h5py.File(paths["netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        control05 = tools.clean(np.asarray(handle["pr_ensemble_mean_rec_result"][indices], dtype=float))
        cru05 = tools.clean(np.asarray(handle["pr_obs_cru_pre_05deg"][indices], dtype=float))
        gpcc05 = tools.clean(np.asarray(handle["pr_obs_gpcp_05deg"][indices], dtype=float))
    control_cube = tools.anomaly(tools.source_grid(control05, source_lat, source_lon, grid_lat, grid_lon))
    observed_fields = {
        "CRU": tools.anomaly(tools.source_grid(cru05, source_lat, source_lon, grid_lat, grid_lon)),
        "GPCC": tools.anomaly(tools.source_grid(gpcc05, source_lat, source_lon, grid_lat, grid_lon)),
    }
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = tools.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    lon2 = np.broadcast_to(grid_lon[None], mask2.shape)
    region_masks = {
        label: mask2 & (lon2 >= lower) & (lon2 < upper)
        for label, (lower, upper) in REGIONS.items()
    }
    branches = {
        "equal_six_control": control_cube,
        "ao9_current_bivariate_ols_oof": parent_cube,
        "ao9_ridge_bivariate_oof": ridge_cube,
        "ao9_precipitation_only_tree_oof": precip_cube,
    }
    curves: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in branches}
    observed_curves: dict[str, dict[str, np.ndarray]] = {name: {} for name in observed_fields}
    score_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for region, mask in region_masks.items():
        for branch, field in branches.items():
            values = tools.area_series(field, grid_lat, mask)
            curves[branch][region] = values
            for year, value in zip(YEARS, values):
                curve_rows.append({"year": int(year), "region": region, "series_type": "reconstruction", "series": branch, "value_mm_day": float(value)})
        for observation, field in observed_fields.items():
            values = tools.area_series(field, grid_lat, mask)
            observed_curves[observation][region] = values
            for year, value in zip(YEARS, values):
                curve_rows.append({"year": int(year), "region": region, "series_type": "observation", "series": observation, "value_mm_day": float(value)})
        for branch in branches:
            for observation in observed_fields:
                left = curves[branch][region]
                right = observed_curves[observation][region]
                valid = np.isfinite(left) & np.isfinite(right)
                delta = left[valid] - right[valid]
                score_rows.append({
                    "branch": branch,
                    "region": region,
                    "observation": observation,
                    "valid_year_count": int(valid.sum()),
                    "bias_mm_day": float(np.mean(delta)),
                    "rmse_mm_day": float(np.sqrt(np.mean(delta * delta))),
                    "correlation": tools.correlation(left, right),
                })

    controls = {(row["region"], row["observation"]): row for row in score_rows if row["branch"] == "equal_six_control"}
    aggregate_rows: list[dict[str, object]] = []
    for branch in branches:
        selected_rows = [row for row in score_rows if row["branch"] == branch]
        median_rmse = float(np.median([row["rmse_mm_day"] for row in selected_rows]))
        control_median = float(np.median([row["rmse_mm_day"] for row in controls.values()]))
        regressions = [row["rmse_mm_day"] / controls[(row["region"], row["observation"])]["rmse_mm_day"] - 1.0 for row in selected_rows]
        improvement = 1.0 - median_rmse / control_median
        aggregate_rows.append({
            "branch": branch,
            "median_region_reference_annual_rmse_mm_day": median_rmse,
            "control_median_rmse_mm_day": control_median,
            "relative_improvement": float(improvement),
            "maximum_pair_relative_regression": float(max(regressions)),
            "passes_5pct_primary": bool(improvement >= 0.05),
            "passes_5pct_no_regression": bool(max(regressions) <= 0.05 + 1e-12),
            "passes_gate": bool(improvement >= 0.05 and max(regressions) <= 0.05 + 1e-12),
        })
    parent_gate = next(row for row in aggregate_rows if row["branch"] == "ao9_current_bivariate_ols_oof")
    parent_expected = parent["selected_oof_gate"]
    parent_closure = abs(parent_gate["median_region_reference_annual_rmse_mm_day"] - parent_expected["median_region_reference_annual_rmse_mm_day"])
    if parent_closure > 1e-12:
        raise RuntimeError("parent-score-closure-failed")
    candidates = [row for row in aggregate_rows if row["branch"] in {"ao9_ridge_bivariate_oof", "ao9_precipitation_only_tree_oof"}]
    passed = [row for row in candidates if row["passes_gate"]]
    selected = min(passed, key=lambda row: row["median_region_reference_annual_rmse_mm_day"]) if passed else None
    decision = "advance_best_psm_repair_to_50_pool_test" if selected else "close_psm_stability_repair"

    lambda_rows = []
    for branch in range(1, 6):
        values = ridge_lambda[:, branch]
        for value in np.unique(values[np.isfinite(values)]):
            lambda_rows.append({"outer_fold": branch, "lambda": float(value), "proxy_count": int(np.sum(values == value))})
    proxy_holdout = []
    for fold in range(5):
        valid = np.isfinite(ridge_holdout[:, fold]) & np.isfinite(precip_holdout[:, fold])
        proxy_holdout.append({
            "outer_fold": fold + 1,
            "paired_proxy_count": int(valid.sum()),
            "ridge_median_proxy_rmse": float(np.median(ridge_holdout[valid, fold])),
            "precip_only_median_proxy_rmse": float(np.median(precip_holdout[valid, fold])),
            "ridge_better_fraction": float(np.mean(ridge_holdout[valid, fold] < precip_holdout[valid, fold])),
        })
    active_rows = [{"variant": "ridge_bivariate", **row} for row in ridge_active] + [{"variant": "precipitation_only_tree", **row} for row in precip_active]
    write_csv(output / "scores.csv", score_rows)
    write_csv(output / "aggregate_scores.csv", aggregate_rows)
    write_csv(output / "regional_curves.csv", curve_rows)
    write_csv(output / "ridge_lambda_selection.csv", lambda_rows)
    write_csv(output / "proxy_holdout_summary.csv", proxy_holdout)
    write_csv(output / "active_proxy_counts.csv", active_rows)
    np.savez_compressed(
        output / "psm_stability_ao_oof_arrays.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        ridge_analysis=ridge_analysis,
        precip_only_analysis=precip_analysis,
        ridge_selected_local_indices=ridge_selected,
        precip_only_selected_local_indices=precip_selected,
        psm_branch=branch_by_year,
    )

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    styles = {
        "equal_six_control": ("#222222", 2.5),
        "ao9_current_bivariate_ols_oof": ("#D55E00", 1.6),
        "ao9_ridge_bivariate_oof": ("#0072B2", 2.1),
        "ao9_precipitation_only_tree_oof": ("#009E73", 2.1),
    }
    labels = {
        "equal_six_control": "Existing equal-six control",
        "ao9_current_bivariate_ols_oof": "AO_9 current bivariate OLS",
        "ao9_ridge_bivariate_oof": "AO_9 ridge bivariate",
        "ao9_precipitation_only_tree_oof": "AO_9 precipitation-only trees",
    }
    for branch, (color, width) in styles.items():
        axis.plot(YEARS, curves[branch]["hma"], color=color, linewidth=width, label=labels[branch])
    axis.plot(YEARS, observed_curves["CRU"]["hma"], color="#666666", linestyle="--", linewidth=1.2, label="CRU")
    axis.plot(YEARS, observed_curves["GPCC"]["hma"], color="#AA4499", linestyle="--", linewidth=1.2, label="GPCC")
    axis.axhline(0, color="#AAAAAA", linewidth=0.7)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set(xlabel="Year", ylabel="Precipitation anomaly (mm/day)", title="HMA precipitation: blocked PSM stability repairs")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "figure_hma_psm_stability_repairs_1940_2000.png", dpi=220)
    plt.close(fig)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-changed-during-loop-v2-round01")
    summary = {
        "experiment_id": "climate_reconstruction_improvement_loop_v2/round_01_psm_stability_repair",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": decision,
        "selected": selected,
        "locked_1921_1939_accessed": False,
        "parent_score_closure_abs": parent_closure,
        "enkf_mean_closure_max_abs": max(ridge_closure, precip_closure),
        "aggregate": aggregate_rows,
        "ridge_lambda_counts_all_outer_folds": {
            str(float(value)): int(np.sum(ridge_lambda[:, 1:][np.isfinite(ridge_lambda[:, 1:])] == value))
            for value in np.unique(ridge_lambda[:, 1:][np.isfinite(ridge_lambda[:, 1:])])
        },
        "proxy_holdout": proxy_holdout,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "A one-pool nested PSM stability screen. Passing authorizes 50-pool testing only.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "before": before,
        "after": after,
        "sha256": {name: digest(path) for name, path in paths.items() if name != "netcdf"},
        "netcdf_hash_omitted_reason": "large immutable source; size and mtime checked before and after",
        "driver_sha256": digest(Path(__file__).resolve()),
        "matlab_driver_sha256": digest(Path(__file__).with_name("fit_climate_reconstruction_loop_v2_round01_psm_stability_v1.m")),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidate_lines = "\n".join(
        f"- {row['branch']}: improvement `{100 * row['relative_improvement']:.3f}%`, max regression `{100 * row['maximum_pair_relative_regression']:.3f}%`, pass `{str(row['passes_gate']).lower()}`."
        for row in candidates
    )
    (output / "FINAL_RESULT.md").write_text(
        "# PSM Stability Repair\n\n"
        f"- Decision: `{decision}`\n"
        f"{candidate_lines}\n"
        "- Locked 1921-1939 values accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
