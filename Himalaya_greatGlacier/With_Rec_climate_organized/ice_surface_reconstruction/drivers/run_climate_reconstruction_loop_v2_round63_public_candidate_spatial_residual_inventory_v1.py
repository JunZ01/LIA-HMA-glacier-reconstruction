"""Inventory cross-product spatial-residual skill in fixed public candidates."""

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
PRODUCTS = ("CRU", "GPCC", "UDel", "PREC/L")


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


def residualize(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return field - (field @ weights)[:, None]


def residual_mse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(np.sum(values**2 * weights[None], axis=1)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_63_public_candidate_spatial_residual_inventory_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round62 = loop / "round_62_western_observation_spatial_residual_compatibility_v1"
    driver58 = Path(__file__).resolve().with_name("run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py")
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round62_summary": round62 / "summary.json",
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
    summary62 = json.loads(paths["round62_summary"].read_text(encoding="utf-8"))
    if not summary62["precl_is_valid_cross_product_spatial_adjudicator"]:
        raise RuntimeError("observation-spatial-compatibility-not-established")
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
    candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": helpers.finite_mean([members["HGAO"], members["O"]]),
        "member_equal": helpers.finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": helpers.finite_mean([helpers.finite_mean([members["HGAO"], members["O"]]), members["LMR"]]),
    }
    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if int(common.sum()) != int(summary62["common_western_cell_count"]):
        raise RuntimeError("common-support-closure-failed")
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], common.shape)
    weights = weights_2d[common].astype(float)
    weights /= np.sum(weights)
    observation_residuals = {product: residualize(observations[product][:, common], weights) for product in PRODUCTS}

    rows: list[dict[str, object]] = []
    candidate_results: dict[str, dict[str, object]] = {}
    for candidate, field in candidates.items():
        candidate_residual = residualize(field[:, common], weights)
        path_results: dict[str, dict[str, object]] = {}
        skills: list[float] = []
        for product in PRODUCTS:
            target = observation_residuals[product]
            skill = 1.0 - residual_mse(candidate_residual - target, weights) / residual_mse(target, weights)
            correlations = np.asarray([weighted_correlation(candidate_residual[index], target[index], weights) for index in range(len(YEARS))])
            if not np.all(np.isfinite(correlations)) or not np.isfinite(skill):
                raise RuntimeError(f"nonfinite-residual-statistic:{candidate}:{product}")
            median_corr = float(np.median(correlations))
            positive_fraction = float(np.mean(correlations > 0))
            gates = {"residual_mse_skill_positive": skill > 0, "median_pattern_correlation_at_least_0_10": median_corr >= 0.10, "at_least_60pct_years_positive": positive_fraction >= 0.60}
            path_passes = bool(all(gates.values()))
            path_results[product] = {"residual_mse_skill": skill, "median_annual_pattern_correlation": median_corr, "positive_year_fraction": positive_fraction, "gates": gates, "passes": path_passes}
            skills.append(skill)
            rows.append({"candidate": candidate, "observation": product, "residual_mse_skill": skill, "median_annual_pattern_correlation": median_corr, "positive_year_fraction": positive_fraction, "passes": path_passes})
        passing_count = int(sum(result["passes"] for result in path_results.values()))
        partial_gates = {"at_least_3_of_4_paths_pass": passing_count >= 3, "median_four_product_residual_skill_positive": float(np.median(skills)) > 0, "no_product_residual_skill_below_minus_5pct": float(np.min(skills)) >= -0.05}
        candidate_results[candidate] = {"product_paths": path_results, "passing_product_path_count": passing_count, "median_four_product_residual_skill": float(np.median(skills)), "minimum_product_residual_skill": float(np.min(skills)), "robust_all_four_products": passing_count == 4, "diagnostic_partial_component": bool(all(partial_gates.values())), "partial_gates": partial_gates}

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "public_candidate_spatial_residual_inventory.csv", index=False)
    robust = [name for name, result in candidate_results.items() if result["robust_all_four_products"]]
    partial = [name for name, result in candidate_results.items() if result["diagnostic_partial_component"]]
    summary = {"status": "complete", "locked_1921_1939_accessed": False, "common_western_cell_count": int(common.sum()), "candidate_results": candidate_results, "robust_spatial_residual_candidates": robust, "diagnostic_partial_residual_candidates": partial, "decision": "advance_robust_residual_to_separate_component_test" if robust else "close_existing_public_fields_as_spatial_residual_source", "new_field_constructed": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pivot = frame.pivot(index="candidate", columns="observation", values="residual_mse_skill").reindex(index=list(candidates), columns=PRODUCTS)
    figure, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=10)
    axis.set_xticks(np.arange(len(PRODUCTS)), labels=PRODUCTS)
    axis.set_yticks(np.arange(len(candidates)), labels=list(candidates))
    for row in range(len(candidates)):
        for column in range(len(PRODUCTS)):
            axis.text(column, row, f"{pivot.iloc[row, column] * 100:.1f}", ha="center", va="center", fontsize=9)
    axis.set_title("Within-west spatial-residual MSE skill (%)")
    figure.colorbar(image, ax=axis, label="Improvement over zero residual (%)")
    figure.savefig(output / "public_candidate_spatial_residual_inventory.png", dpi=180)
    plt.close(figure)

    lines = ["# Public Candidate Spatial-Residual Inventory", ""]
    for candidate, result in candidate_results.items():
        lines.append(f"- {candidate}: paths `{result['passing_product_path_count']}/4`, median residual skill `{result['median_four_product_residual_skill']:.2%}`, minimum `{result['minimum_product_residual_skill']:.2%}`, robust `{str(result['robust_all_four_products']).lower()}`, partial `{str(result['diagnostic_partial_component']).lower()}`.")
    lines.extend(["", f"Robust residual candidates: `{robust}`.", f"Decision: `{summary['decision']}`."])
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
