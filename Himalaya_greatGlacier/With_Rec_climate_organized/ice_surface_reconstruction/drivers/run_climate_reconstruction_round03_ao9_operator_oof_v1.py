"""Round 03: operator-level blocked OOF test of the AO_9 precipitation branch."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result <= -9000) | (result >= 1e20)] = np.nan
    return result


def source_grid(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    lat2, lon2 = np.meshgrid(target_lat, target_lon, indexing="ij")
    fy = (lat2 - lat[0]) / (lat[1] - lat[0])
    fx = (lon2 - lon[0]) / (lon[1] - lon[0])
    inside = (fy >= 0) & (fy <= len(lat) - 1) & (fx >= 0) & (fx <= len(lon) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(lon) - 2)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    a = field[:, y0, x0]
    b = field[:, y0, x0 + 1]
    c = field[:, y0 + 1, x0]
    d = field[:, y0 + 1, x0 + 1]
    values = (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b) + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    values[:, ~inside] = np.nan
    return values


def state_coordinates() -> tuple[np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    return all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)], all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)]


def to_cube(analysis: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    return analysis.reshape(nlon, nlat, analysis.shape[1]).transpose(2, 1, 0)


def anomaly(values: np.ndarray) -> np.ndarray:
    reference = (YEARS >= 1961) & (YEARS <= 1990)
    return values - np.nanmean(values[reference], axis=0)[None]


def area_series(field: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(field) & mask[None]
    numerator = np.sum(np.where(valid, field, 0.0) * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def select_analogues(observation: np.ndarray, hxb: np.ndarray, count: int = 100) -> np.ndarray:
    y_anomaly = observation - np.mean(observation)
    hxb_anomaly = hxb - np.mean(hxb, axis=0)
    denominator = np.sqrt(np.sum(y_anomaly * y_anomaly)) * np.sqrt(np.sum(hxb_anomaly * hxb_anomaly, axis=0))
    scores = np.divide(y_anomaly @ hxb_anomaly, denominator, out=np.full(hxb.shape[1], -1.0), where=denominator > 0)
    return np.argsort(-scores, kind="stable")[:count]


def analysis_mean(prior: np.ndarray, hxb: np.ndarray, observation: np.ndarray, error_variance: np.ndarray) -> tuple[np.ndarray, float]:
    members = prior.shape[1]
    prior_mean = np.mean(prior, axis=1)
    hxb_mean = np.mean(hxb, axis=1)
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
    result = prior_mean + pbht @ solved
    closure = float(np.max(np.abs(result - (prior_mean + pbht @ solved))))
    return result, closure


def run_branch(prior: np.ndarray, sample: np.ndarray, observations: np.ndarray, hxb: np.ndarray, residual_variance: np.ndarray, branch_by_year: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    background = prior[:, sample]
    analysis = np.full((prior.shape[0], len(YEARS)), np.nan)
    selected = np.full((len(YEARS), 100), -1, dtype=int)
    maximum_closure = 0.0
    for year_index, branch in enumerate(branch_by_year):
        active = np.flatnonzero(np.isfinite(observations[:, year_index]))
        if len(active) == 0:
            analysis[:, year_index] = np.mean(background, axis=1)
            continue
        branch_hxb = hxb[active, branch][:, sample]
        local = select_analogues(observations[active, year_index], branch_hxb)
        selected[year_index] = local
        variance = residual_variance[active, branch]
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise RuntimeError(f"invalid-residual-variance:{int(YEARS[year_index])}:{int(branch)}")
        result, closure = analysis_mean(background[:, local], branch_hxb[:, local], observations[active, year_index], variance)
        analysis[:, year_index] = result
        maximum_closure = max(maximum_closure, closure)
    return analysis, selected, maximum_closure


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    paths = {
        "psm": output / "proxy9_blocked_psm_v2.mat",
        "proxy_input": output / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "production_replay": output / "proxy9_production_operator_rng4236_v3.mat",
        "prior": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "indices": archive / "fixed_proxy_experiments_69/pr/indices/all_Xb_indices.mat",
        "archived_ao": archive / "fixed_proxy_experiments_69/pr/results/proxy9/AOEnKF_ind_exp1.mat",
        "netcdf": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "lock": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-round03-inputs:" + ";".join(missing))
    final_path = output / "operator_oof_summary_v1.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if lock.get("status") != "locked_not_evaluated" or np.any((YEARS >= 1921) & (YEARS <= 1939)):
        raise RuntimeError("locked-period-contract-failed")
    before = {name: state(path) for name, path in paths.items()}

    with h5py.File(paths["psm"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        residual_variance = np.asarray(handle["residual_variance"], dtype=float).T
        baseline_hxb_closure = float(np.asarray(handle["baseline_hxb_max_abs"]).reshape(-1)[0])
    proxy = scipy.io.loadmat(paths["proxy_input"], variable_names=["observations"])
    observations = np.asarray(proxy["observations"], dtype=float).T
    with h5py.File(paths["prior"], "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    with h5py.File(paths["archived_ao"], "r") as handle:
        archived_ao = np.asarray(handle["AOEnKF_ind"], dtype=int).T[940:1001] - 1
    with h5py.File(paths["production_replay"], "r") as handle:
        production_hxb = np.asarray(handle["hxb"], dtype=float).T
        production_observations = np.asarray(handle["observations"], dtype=float)
    production_observations = production_observations.T
    if production_observations.shape != (61, 297):
        raise RuntimeError(f"production-observation-shape-failed:{production_observations.shape}")
    observation_closure = float(np.nanmax(np.abs(production_observations - observations.T)))
    psm_to_production_hxb = float(np.nanmax(np.abs(hxb[:, 0] - production_hxb)))
    if baseline_hxb_closure > 1e-9 or observation_closure > 1e-12:
        raise RuntimeError("operator-input-closure-failed")

    full_branches = np.zeros(len(YEARS), dtype=int)
    crossfit_branches = np.full(len(YEARS), -1, dtype=int)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        crossfit_branches[(YEARS >= start) & (YEARS <= end)] = fold
    if np.any(crossfit_branches < 0):
        raise RuntimeError("crossfit-branch-coverage-failed")

    full_analysis, full_indices, full_closure = run_branch(prior, sample, observations, hxb, residual_variance, full_branches)
    oof_analysis, oof_indices, oof_closure = run_branch(prior, sample, observations, hxb, residual_variance, crossfit_branches)

    ordered_years = int(np.sum(np.all(full_indices == archived_ao, axis=1)))
    overlaps = np.asarray([len(set(current.tolist()) & set(old.tolist())) for current, old in zip(full_indices, archived_ao)])
    set_exact_years = int(np.sum(overlaps == 100))
    ao_closure = {
        "archived_ordered_exact_years_of_61": ordered_years,
        "archived_set_exact_years_of_61": set_exact_years,
        "selected_member_overlap_min_of_100": int(np.min(overlaps)),
        "selected_member_overlap_median_of_100": float(np.median(overlaps)),
        "selected_member_overlap_mean_of_100": float(np.mean(overlaps)),
        "allowed_observation_max_abs_difference": observation_closure,
        "round03_full_hxb_vs_rng4236_production_replay_max_abs": psm_to_production_hxb,
        "interpretation": "The archived AO indices use an unrecovered historical PSM version. Code-path replay is near but not exact and is documented rather than used as the OOF gate.",
    }

    grid_lat, grid_lon = state_coordinates()
    full_cube = anomaly(to_cube(full_analysis, len(grid_lat), len(grid_lon)))
    oof_cube = anomaly(to_cube(oof_analysis, len(grid_lat), len(grid_lon)))
    with h5py.File(paths["netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        control05 = clean(np.asarray(handle["pr_ensemble_mean_rec_result"][time_indices], dtype=float))
        cru05 = clean(np.asarray(handle["pr_obs_cru_pre_05deg"][time_indices], dtype=float))
        gpcc05 = clean(np.asarray(handle["pr_obs_gpcp_05deg"][time_indices], dtype=float))
    control_cube = anomaly(source_grid(control05, source_lat, source_lon, grid_lat, grid_lon))
    observations_2deg = {
        "CRU": anomaly(source_grid(cru05, source_lat, source_lon, grid_lat, grid_lon)),
        "GPCC": anomaly(source_grid(gpcc05, source_lat, source_lon, grid_lat, grid_lon)),
    }
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    lon2 = np.broadcast_to(grid_lon[None], mask2.shape)
    region_masks = {
        label: mask2 & (lon2 >= lower) & (lon2 < upper)
        for label, (lower, upper) in REGIONS.items()
    }
    branches = {
        "equal_six_control": control_cube,
        "ao9_full_calibration_exp1": full_cube,
        "ao9_blocked_operator_oof_exp1": oof_cube,
    }
    curve_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, np.ndarray]] = {name: {} for name in branches}
    observed_curves: dict[str, dict[str, np.ndarray]] = {name: {} for name in observations_2deg}
    for region, mask in region_masks.items():
        for branch, field in branches.items():
            values = area_series(field, grid_lat, mask)
            curves[branch][region] = values
            for year, value in zip(YEARS, values):
                curve_rows.append({"year": int(year), "region": region, "series_type": "reconstruction", "series": branch, "value_mm_day": float(value)})
        for observation, field in observations_2deg.items():
            values = area_series(field, grid_lat, mask)
            observed_curves[observation][region] = values
            for year, value in zip(YEARS, values):
                curve_rows.append({"year": int(year), "region": region, "series_type": "observation", "series": observation, "value_mm_day": float(value)})
        for branch, branch_curves in curves.items():
            for observation, obs_curves in observed_curves.items():
                left = branch_curves[region]
                right = obs_curves[region]
                valid = np.isfinite(left) & np.isfinite(right)
                delta = left[valid] - right[valid]
                score_rows.append({
                    "branch": branch,
                    "region": region,
                    "observation": observation,
                    "valid_year_count": int(valid.sum()),
                    "bias_mm_day": float(np.mean(delta)),
                    "rmse_mm_day": float(np.sqrt(np.mean(delta * delta))),
                    "correlation": correlation(left, right),
                })

    aggregate = []
    controls = {(row["region"], row["observation"]): row for row in score_rows if row["branch"] == "equal_six_control"}
    for branch in branches:
        selected_rows = [row for row in score_rows if row["branch"] == branch]
        median_rmse = float(np.median([row["rmse_mm_day"] for row in selected_rows]))
        control_median = float(np.median([row["rmse_mm_day"] for row in controls.values()]))
        regressions = [row["rmse_mm_day"] / controls[(row["region"], row["observation"])]["rmse_mm_day"] - 1.0 for row in selected_rows]
        improvement = 1.0 - median_rmse / control_median
        aggregate.append({
            "branch": branch,
            "median_region_reference_annual_rmse_mm_day": median_rmse,
            "control_median_rmse_mm_day": control_median,
            "relative_improvement": float(improvement),
            "maximum_pair_relative_regression": float(max(regressions)),
            "passes_5pct_primary": bool(improvement >= 0.05),
            "passes_5pct_no_regression": bool(max(regressions) <= 0.05 + 1e-12),
            "passes_gate": bool(improvement >= 0.05 and max(regressions) <= 0.05 + 1e-12),
        })
    oof_gate = next(row for row in aggregate if row["branch"] == "ao9_blocked_operator_oof_exp1")
    decision = "advance_ao9_precipitation_to_round04" if oof_gate["passes_gate"] else "reject_ao9_operator_repair"

    write_csv(output / "ao9_operator_oof_scores_v1.csv", score_rows)
    write_csv(output / "ao9_operator_oof_aggregate_v1.csv", aggregate)
    write_csv(output / "ao9_operator_oof_regional_curves_v1.csv", curve_rows)
    (output / "ao_index_closure_v1.json").write_text(json.dumps(ao_closure, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "ao9_operator_oof_arrays_v1.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        full_analysis=full_analysis,
        oof_analysis=oof_analysis,
        full_selected_local_indices=full_indices,
        oof_selected_local_indices=oof_indices,
        crossfit_psm_branch=crossfit_branches,
    )

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.plot(YEARS, curves["equal_six_control"]["hma"], color="#222222", linewidth=2.3, label="Existing equal-six control")
    axis.plot(YEARS, curves["ao9_full_calibration_exp1"]["hma"], color="#0072B2", linewidth=1.5, label="AO_9 full calibration")
    axis.plot(YEARS, curves["ao9_blocked_operator_oof_exp1"]["hma"], color="#D55E00", linewidth=2.1, label="AO_9 blocked OOF")
    axis.plot(YEARS, observed_curves["CRU"]["hma"], color="#009E73", linestyle="--", linewidth=1.3, label="CRU")
    axis.plot(YEARS, observed_curves["GPCC"]["hma"], color="#CC79A7", linestyle="--", linewidth=1.3, label="GPCC")
    axis.axhline(0, color="#AAAAAA", linewidth=0.7)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set(xlabel="Year", ylabel="Precipitation anomaly (mm/day)", title="HMA precipitation: AO_9 operator-level blocked validation")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "figure_hma_ao9_operator_oof_1940_2000_v1.png", dpi=220)
    plt.close(fig)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-changed-during-round03-oof")
    summary = {
        "experiment_id": "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "official_full_psm_hxb_closure_max_abs": baseline_hxb_closure,
        "enkf_mean_closure_max_abs": max(full_closure, oof_closure),
        "historical_ao_index_closure": ao_closure,
        "aggregate": aggregate,
        "selected_oof_gate": oof_gate,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "This is a one-prior-pool operator-level blocked test. Passing authorizes the 50-pool Round 04 reconstruction; it is not final product acceptance.",
    }
    final_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "before": before,
        "after": after,
        "sha256": {name: digest(path) for name, path in paths.items() if name != "netcdf"},
        "netcdf_hash_omitted_reason": "large immutable source; size and mtime were checked before and after",
        "driver_sha256": digest(Path(__file__).resolve()),
    }
    (output / "operator_oof_source_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT_OPERATOR_OOF_V1.md").write_text(
        "# Round 03 AO_9 Operator OOF\n\n"
        f"- Decision: `{decision}`\n"
        f"- OOF median RMSE improvement: `{100 * oof_gate['relative_improvement']:.3f}%`\n"
        f"- Maximum region/reference regression: `{100 * oof_gate['maximum_pair_relative_regression']:.3f}%`\n"
        f"- Passed both frozen gates: `{str(oof_gate['passes_gate']).lower()}`\n"
        "- Locked 1921-1939 values accessed: `false`.\n"
        "- Historical AO index replay is documented separately and is not exact because its generating PSM version was not retained.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
