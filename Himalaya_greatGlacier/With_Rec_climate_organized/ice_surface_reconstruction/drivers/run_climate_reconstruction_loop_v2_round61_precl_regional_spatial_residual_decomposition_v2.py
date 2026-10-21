"""Correct NaN-safe Round 61 regional/spatial residual decomposition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1948, 2001)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("round58_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("round58-helper-import-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    x = left - np.sum(weights * left)
    y = right - np.sum(weights * right)
    denominator = np.sqrt(np.sum(weights * x**2) * np.sum(weights * y**2))
    return float(np.sum(weights * x * y) / denominator) if denominator > 0 else np.nan


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_61_precl_regional_spatial_residual_decomposition_v2"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round60 = loop / "round_60_precl_western_spatial_transfer_v1"
    round61v1 = loop / "round_61_precl_regional_spatial_residual_decomposition_v1"
    driver58 = Path(__file__).resolve().with_name("run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py")
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round60_summary": round60 / "summary.json",
        "invalid_v1_summary": round61v1 / "summary.json",
        "udel": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc",
        "public_reference_source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round58_driver": driver58,
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    invalid_v1 = json.loads(paths["invalid_v1_summary"].read_text(encoding="utf-8"))
    if not np.isnan(float(invalid_v1["target_mse_components"]["total"])):
        raise RuntimeError("v1-failure-signature-not-confirmed")
    helpers = load_helpers(driver58)

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((source_years >= 1948) & (source_years <= 2000))
        if not np.array_equal(source_years[indices], YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((source_years[indices] >= 1921) & (source_years[indices] <= 1939)):
            raise RuntimeError("locked-period-index-requested")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        observations = {name: helpers.recenter(helpers.clean(handle[variable], selection)) for name, variable in helpers.OBSERVATIONS.items()}
        members = {name: helpers.recenter(helpers.clean(handle[variable], selection)) for name, variable in helpers.MEMBERS.items()}
    observations["UDel"] = helpers.recenter(helpers.annualize_udel(paths["udel"], latitudes, longitudes))
    observations["PREC/L"] = helpers.recenter(helpers.annualize_precl(precl_paths, latitudes, longitudes))
    candidates = {"O": members["O"], "local_enkf": helpers.finite_mean([members["HGAO"], members["O"]])}
    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    expected_cells = int(json.loads(paths["round60_summary"].read_text(encoding="utf-8"))["common_western_cell_count"])
    if int(common.sum()) != expected_cells:
        raise RuntimeError("common-support-closure-failed")
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], common.shape)
    weights = weights_2d[common].astype(float)
    weights /= np.sum(weights)
    target = observations["PREC/L"][:, common]
    if not np.all(np.isfinite(target)):
        raise RuntimeError("nonfinite-target-on-common-support")
    target_mean = target @ weights
    target_residual = target - target_mean[:, None]
    target_total_energy = float(np.mean(np.sum(target**2 * weights[None], axis=1)))
    target_regional_energy = float(np.mean(target_mean**2))
    target_residual_energy = float(np.mean(np.sum(target_residual**2 * weights[None], axis=1)))
    baseline_closure = abs(target_total_energy - target_regional_energy - target_residual_energy)

    annual_rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    for candidate, field in candidates.items():
        estimate = field[:, common]
        if not np.all(np.isfinite(estimate)):
            raise RuntimeError(f"nonfinite-candidate-on-common-support:{candidate}")
        estimate_mean = estimate @ weights
        estimate_residual = estimate - estimate_mean[:, None]
        difference = estimate - target
        regional_difference = estimate_mean - target_mean
        residual_difference = estimate_residual - target_residual
        total_mse = float(np.mean(np.sum(difference**2 * weights[None], axis=1)))
        regional_mse = float(np.mean(regional_difference**2))
        residual_mse = float(np.mean(np.sum(residual_difference**2 * weights[None], axis=1)))
        closure = abs(total_mse - regional_mse - residual_mse)
        total_skill = 1.0 - total_mse / target_total_energy
        regional_skill = 1.0 - regional_mse / target_regional_energy
        residual_skill = 1.0 - residual_mse / target_residual_energy
        correlations = np.asarray([weighted_correlation(estimate_residual[index], target_residual[index], weights) for index in range(len(YEARS))])
        if not np.all(np.isfinite(np.r_[total_mse, regional_mse, residual_mse, total_skill, regional_skill, residual_skill, closure, baseline_closure, correlations])):
            raise RuntimeError(f"nonfinite-core-statistic:{candidate}")
        for index, year in enumerate(YEARS):
            annual_rows.append({"year": int(year), "candidate": candidate, "candidate_west_mean_mm_day": float(estimate_mean[index]), "precl_west_mean_mm_day": float(target_mean[index]), "spatial_pattern_correlation": float(correlations[index])})
        gates = {
            "mse_decomposition_closure_at_most_1e_12": closure <= 1e-12 and baseline_closure <= 1e-12,
            "residual_rmse_skill_positive": residual_skill > 0,
            "median_pattern_correlation_at_least_0_10": float(np.median(correlations)) >= 0.10,
            "at_least_60pct_years_positive_pattern_correlation": float(np.mean(correlations > 0)) >= 0.60,
        }
        results[candidate] = {
            "total_rmse_skill": total_skill,
            "regional_mean_rmse_skill": regional_skill,
            "spatial_residual_rmse_skill": residual_skill,
            "median_annual_spatial_pattern_correlation": float(np.median(correlations)),
            "positive_annual_pattern_correlation_fraction": float(np.mean(correlations > 0)),
            "mse_decomposition_closure": closure,
            "baseline_mse_decomposition_closure": baseline_closure,
            "gates": gates,
            "spatial_residual_supported": bool(all(gates.values())),
        }

    annual = pd.DataFrame(annual_rows)
    annual.to_csv(output / "annual_regional_spatial_decomposition.csv", index=False)
    pd.DataFrame([{"candidate": name, **{key: value for key, value in result.items() if key != "gates"}} for name, result in results.items()]).to_csv(output / "regional_spatial_decomposition_summary.csv", index=False)
    supported = [name for name, result in results.items() if result["spatial_residual_supported"]]
    summary = {
        "status": "complete",
        "compatibility_rerun_of_invalid_v1": True,
        "v1_invalid_reason": "out_of_support_nan_multiplied_by_zero_weight",
        "locked_1921_1939_accessed": False,
        "common_western_cell_count": int(common.sum()),
        "target_mse_components": {"total": target_total_energy, "regional_mean": target_regional_energy, "spatial_residual": target_residual_energy, "closure": baseline_closure},
        "candidate_results": results,
        "spatial_residual_supported_candidates": supported,
        "decision": "current_skill_is_regional_mean_only" if not supported else "retain_candidate_spatial_residual_signal",
        "production_field_selected": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True)
    first = annual[annual.candidate == "O"]
    axes[0].plot(first.year, first.precl_west_mean_mm_day, color="#111111", linewidth=2.3, label="PREC/L")
    for candidate, color in (("O", "#0072b2"), ("local_enkf", "#d55e00")):
        selected = annual[annual.candidate == candidate]
        axes[0].plot(selected.year, selected.candidate_west_mean_mm_day, color=color, linewidth=1.5, label=candidate)
        axes[1].plot(selected.year, selected.spatial_pattern_correlation, color=color, linewidth=1.3, label=candidate)
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(title="Western regional-mean curves", ylabel="Anomaly (mm/day)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[1].axhline(0.1, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set(title="Annual within-west spatial-pattern correlation", xlabel="Year", ylabel="Weighted correlation")
    axes[1].legend(frameon=False)
    figure.savefig(output / "precl_regional_spatial_residual_decomposition.png", dpi=180)
    plt.close(figure)

    lines = ["# PREC/L Regional And Spatial-Residual Decomposition v2", "", "V1 is invalid because NaNs outside the common mask propagated through zero weights. V2 extracts finite cells first and changes no scientific gate.", ""]
    for candidate, result in results.items():
        lines.append(f"- {candidate}: total/regional/residual skills `{result['total_rmse_skill']:.2%}/{result['regional_mean_rmse_skill']:.2%}/{result['spatial_residual_rmse_skill']:.2%}`, median annual pattern correlation `{result['median_annual_spatial_pattern_correlation']:.3f}`, positive years `{result['positive_annual_pattern_correlation_fraction']:.1%}`, residual supported `{str(result['spatial_residual_supported']).lower()}`.")
    lines.extend(["", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
