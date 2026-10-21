"""Loop v2 Round 00: blocked OOF standard-EnKF test against AO_9 and legacy."""

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


def run_standard_enkf(
    prior: np.ndarray,
    sample: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    residual_variance: np.ndarray,
    branch_by_year: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]], float]:
    background = prior[:, sample]
    analysis = np.full((prior.shape[0], len(YEARS)), np.nan)
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
            "ensemble_member_count": int(background.shape[1]),
        })
        if len(active) == 0:
            analysis[:, year_index] = np.mean(background, axis=1)
            continue
        current, current_closure = tools.analysis_mean(
            background,
            hxb[active, branch][:, sample],
            observations[active, year_index],
            residual_variance[active, branch],
        )
        analysis[:, year_index] = current
        closure = max(closure, current_closure)
    return analysis, active_rows, closure


def main() -> int:
    project = tools.root()
    loop_v1 = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1"
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_00_standard_enkf_operator_oof"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    paths = {
        "psm": loop_v1 / "round_03_nested_ensemble_selection/proxy9_blocked_psm_v2.mat",
        "proxy_input": loop_v1 / "round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ao_arrays": loop_v1 / "round_03_nested_ensemble_selection/ao9_operator_oof_arrays_v1.npz",
        "ao_summary": loop_v1 / "round_03_nested_ensemble_selection/operator_oof_summary_v1.json",
        "prior": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "indices": archive / "fixed_proxy_experiments_69/pr/indices/all_Xb_indices.mat",
        "netcdf": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "lock": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    if output.exists():
        raise FileExistsError(output)
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-loop-v2-round00-inputs:" + ";".join(missing))
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if lock.get("status") != "locked_not_evaluated" or np.any((YEARS >= 1921) & (YEARS <= 1939)):
        raise RuntimeError("locked-period-contract-failed")
    previous = json.loads(paths["ao_summary"].read_text(encoding="utf-8"))
    if previous.get("decision") != "reject_ao9_operator_repair":
        raise RuntimeError("unexpected-parent-decision")
    before = {name: state(path) for name, path in paths.items()}

    output.mkdir(parents=True)
    config = {
        "experiment_id": "climate_reconstruction_improvement_loop_v2/round_00_standard_enkf_operator_oof",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "Does removing AO analogue-member selection recover OOF precipitation skill?",
        "hypothesis": "AO selection overfits proxy spatial patterns; standard EnKF with all 500 fixed members will outperform both AO_9 OOF and the legacy equal-six control.",
        "branches": ["equal_six_control", "ao9_blocked_operator_oof_exp1", "standard_enkf9_blocked_operator_oof_exp1"],
        "development_period": [1940, 2000],
        "blocks": BLOCKS,
        "forbidden_period": [1921, 1939],
        "primary_gate": "median regional/reference annual RMSE improvement >= 5 percent versus equal_six_control",
        "no_regression_gate": "no regional/reference annual RMSE regression > 5 percent",
        "prior_pool": "fixed precipitation experiment 1, 500 members",
        "proxy_network": 9,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Loop v2 Round 00 Contract\n\n"
        "Decision: determine whether AO analogue-member selection, rather than proxy9 PSM calibration alone, causes the blocked-validation failure.\n\n"
        "Control: legacy equal-six mean. Variants: existing AO_9 blocked OOF and standard EnKF using all 500 fixed prior members. Only 1940-2000 is read; 1921-1939 remains forbidden.\n",
        encoding="utf-8",
    )

    with h5py.File(paths["psm"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        residual_variance = np.asarray(handle["residual_variance"], dtype=float).T
        psm_closure = float(np.asarray(handle["baseline_hxb_max_abs"]).reshape(-1)[0])
    proxy = scipy.io.loadmat(paths["proxy_input"], variable_names=["observations"])
    observations = np.asarray(proxy["observations"], dtype=float).T
    with h5py.File(paths["prior"], "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    with np.load(paths["ao_arrays"]) as saved:
        ao_analysis = np.asarray(saved["oof_analysis"], dtype=float)

    branch_by_year = np.full(len(YEARS), -1, dtype=int)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        branch_by_year[(YEARS >= start) & (YEARS <= end)] = fold
    if np.any(branch_by_year < 0):
        raise RuntimeError("blocked-branch-coverage-failed")
    standard_analysis, active_rows, enkf_closure = run_standard_enkf(
        prior, sample, observations, hxb, residual_variance, branch_by_year
    )

    grid_lat, grid_lon = tools.state_coordinates()
    ao_cube = tools.anomaly(tools.to_cube(ao_analysis, len(grid_lat), len(grid_lon)))
    standard_cube = tools.anomaly(tools.to_cube(standard_analysis, len(grid_lat), len(grid_lon)))
    with h5py.File(paths["netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        control05 = tools.clean(np.asarray(handle["pr_ensemble_mean_rec_result"][time_indices], dtype=float))
        cru05 = tools.clean(np.asarray(handle["pr_obs_cru_pre_05deg"][time_indices], dtype=float))
        gpcc05 = tools.clean(np.asarray(handle["pr_obs_gpcp_05deg"][time_indices], dtype=float))
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
        "ao9_blocked_operator_oof_exp1": ao_cube,
        "standard_enkf9_blocked_operator_oof_exp1": standard_cube,
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
        selected = [row for row in score_rows if row["branch"] == branch]
        median_rmse = float(np.median([row["rmse_mm_day"] for row in selected]))
        control_median = float(np.median([row["rmse_mm_day"] for row in controls.values()]))
        regressions = [row["rmse_mm_day"] / controls[(row["region"], row["observation"])]["rmse_mm_day"] - 1.0 for row in selected]
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
    standard_gate = next(row for row in aggregate_rows if row["branch"] == "standard_enkf9_blocked_operator_oof_exp1")
    ao_gate = next(row for row in aggregate_rows if row["branch"] == "ao9_blocked_operator_oof_exp1")
    previous_ao = previous["selected_oof_gate"]
    control_closure = abs(aggregate_rows[0]["median_region_reference_annual_rmse_mm_day"] - previous_ao["control_median_rmse_mm_day"])
    ao_score_closure = abs(ao_gate["median_region_reference_annual_rmse_mm_day"] - previous_ao["median_region_reference_annual_rmse_mm_day"])
    if control_closure > 1e-12 or ao_score_closure > 1e-12 or psm_closure > 1e-9:
        raise RuntimeError("parent-score-or-psm-closure-failed")
    decision = "advance_standard_enkf9_to_50_pool_test" if standard_gate["passes_gate"] else "close_analogue_selection_removal_test"

    write_csv(output / "scores.csv", score_rows)
    write_csv(output / "aggregate_scores.csv", aggregate_rows)
    write_csv(output / "regional_curves.csv", curve_rows)
    write_csv(output / "active_proxy_counts.csv", active_rows)
    np.savez_compressed(
        output / "standard_enkf_oof_arrays.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        standard_analysis=standard_analysis,
        psm_branch=branch_by_year,
    )

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.plot(YEARS, curves["equal_six_control"]["hma"], color="#222222", linewidth=2.4, label="Existing equal-six control")
    axis.plot(YEARS, curves["ao9_blocked_operator_oof_exp1"]["hma"], color="#D55E00", linewidth=1.8, label="AO_9 blocked OOF, 100 members")
    axis.plot(YEARS, curves["standard_enkf9_blocked_operator_oof_exp1"]["hma"], color="#0072B2", linewidth=2.2, label="Standard EnKF blocked OOF, 500 members")
    axis.plot(YEARS, observed_curves["CRU"]["hma"], color="#009E73", linestyle="--", linewidth=1.3, label="CRU")
    axis.plot(YEARS, observed_curves["GPCC"]["hma"], color="#CC79A7", linestyle="--", linewidth=1.3, label="GPCC")
    axis.axhline(0, color="#AAAAAA", linewidth=0.7)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set(xlabel="Year", ylabel="Precipitation anomaly (mm/day)", title="HMA precipitation: standard EnKF versus AO analogue selection")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "figure_hma_standard_enkf_vs_ao_oof_1940_2000.png", dpi=220)
    plt.close(fig)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-changed-during-loop-v2-round00")
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "psm_closure_max_abs": psm_closure,
        "parent_control_score_closure_abs": control_closure,
        "parent_ao_score_closure_abs": ao_score_closure,
        "enkf_mean_closure_max_abs": enkf_closure,
        "aggregate": aggregate_rows,
        "selected_standard_enkf_gate": standard_gate,
        "standard_minus_ao_median_rmse_mm_day": standard_gate["median_region_reference_annual_rmse_mm_day"] - ao_gate["median_region_reference_annual_rmse_mm_day"],
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "One fixed 500-member prior pool. Passing would authorize a 50-pool stability test, not production release.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "before": before,
        "after": after,
        "sha256": {name: digest(path) for name, path in paths.items() if name != "netcdf"},
        "netcdf_hash_omitted_reason": "large immutable source; size and mtime checked before and after",
        "driver_sha256": digest(Path(__file__).resolve()),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Standard EnKF Operator OOF\n\n"
        f"- Decision: `{decision}`\n"
        f"- Standard EnKF median RMSE improvement versus legacy: `{100 * standard_gate['relative_improvement']:.3f}%`\n"
        f"- Maximum region/reference regression: `{100 * standard_gate['maximum_pair_relative_regression']:.3f}%`\n"
        f"- Standard EnKF minus AO median RMSE: `{summary['standard_minus_ao_median_rmse_mm_day']:.6f} mm/day`\n"
        f"- Passed both frozen gates: `{str(standard_gate['passes_gate']).lower()}`\n"
        "- Locked 1921-1939 values accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
