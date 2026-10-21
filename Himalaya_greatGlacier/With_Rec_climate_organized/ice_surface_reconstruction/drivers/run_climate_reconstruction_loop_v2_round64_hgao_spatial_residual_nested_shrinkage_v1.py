"""Nested cross-product test of one HGAO spatial-residual shrinkage scalar."""

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
BLOCKS = ((1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))


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


def residualize(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return field - (field @ weights)[:, None]


def mse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(np.sum(values**2 * weights[None], axis=1)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_64_hgao_spatial_residual_nested_shrinkage_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round63 = loop / "round_63_public_candidate_spatial_residual_inventory_v1"
    driver58 = Path(__file__).resolve().with_name("run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py")
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round63_summary": round63 / "summary.json",
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
    summary63 = json.loads(paths["round63_summary"].read_text(encoding="utf-8"))
    if summary63["decision"] != "close_existing_public_fields_as_spatial_residual_source":
        raise RuntimeError("unexpected-round63-route")
    ranked = sorted(summary63["candidate_results"], key=lambda name: (summary63["candidate_results"][name]["passing_product_path_count"], summary63["candidate_results"][name]["median_four_product_residual_skill"]), reverse=True)
    if ranked[0] != "HGAO":
        raise RuntimeError(f"hgao-not-objective-round63-leader:{ranked}")
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
        hgao = helpers.recenter(helpers.clean(handle[helpers.MEMBERS["HGAO"]], selection))
    observations["UDel"] = helpers.recenter(helpers.annualize_udel(paths["udel"], latitudes, longitudes))
    observations["PREC/L"] = helpers.recenter(helpers.annualize_precl(precl_paths, latitudes, longitudes))
    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if int(common.sum()) != int(summary63["common_western_cell_count"]):
        raise RuntimeError("common-support-closure-failed")
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], common.shape)
    weights = weights_2d[common].astype(float)
    weights /= np.sum(weights)
    hgao_residual = residualize(hgao[:, common], weights)
    targets = {product: residualize(observations[product][:, common], weights) for product in PRODUCTS}

    alpha_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    alphas: list[float] = []
    for fold, (start, end) in enumerate(BLOCKS, start=2):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        predictor_train = hgao_residual[train]
        denominator = float(np.sum(predictor_train**2 * weights[None]))
        if denominator <= 0:
            raise RuntimeError("zero-hgao-training-energy")
        for fit_product in PRODUCTS:
            numerator = float(np.sum(predictor_train * targets[fit_product][train] * weights[None]))
            alpha_raw = numerator / denominator
            alpha = float(np.clip(alpha_raw, 0.0, 1.0))
            alphas.append(alpha)
            alpha_rows.append({"fold": fold, "start_year": start, "end_year": end, "fit_product": fit_product, "alpha_raw": alpha_raw, "alpha": alpha})
            for evaluation_product in PRODUCTS:
                if evaluation_product == fit_product:
                    continue
                target = targets[evaluation_product][test]
                scaled_error = alpha * hgao_residual[test] - target
                unscaled_error = hgao_residual[test] - target
                zero_mse = mse(target, weights)
                scaled_mse = mse(scaled_error, weights)
                unscaled_mse = mse(unscaled_error, weights)
                score_rows.append({"fold": fold, "start_year": start, "end_year": end, "fit_product": fit_product, "evaluation_product": evaluation_product, "alpha": alpha, "skill_over_zero": 1.0 - scaled_mse / zero_mse, "improvement_over_unscaled_hgao": 1.0 - scaled_mse / unscaled_mse})

    alpha_frame = pd.DataFrame(alpha_rows)
    scores = pd.DataFrame(score_rows)
    alpha_frame.to_csv(output / "nested_shrinkage_coefficients.csv", index=False)
    scores.to_csv(output / "nested_cross_product_scores.csv", index=False)
    alpha_array = np.asarray(alphas)
    median_alpha = float(np.median(alpha_array))
    alpha_cv = float(np.std(alpha_array, ddof=1) / np.mean(alpha_array)) if np.mean(alpha_array) > 0 else np.inf
    product_medians = scores.groupby("evaluation_product").skill_over_zero.median().to_dict()
    gates = {
        "median_alpha_at_least_0_05": median_alpha >= 0.05,
        "alpha_coefficient_of_variation_at_most_0_50": alpha_cv <= 0.50,
        "median_cross_product_skill_positive": float(scores.skill_over_zero.median()) > 0,
        "at_least_60pct_paths_improve_over_zero": float((scores.skill_over_zero > 0).mean()) >= 0.60,
        "every_evaluation_product_median_skill_positive": all(value > 0 for value in product_medians.values()),
        "no_path_regresses_more_than_5pct": float(scores.skill_over_zero.min()) >= -0.05,
        "median_improvement_over_unscaled_hgao_positive": float(scores.improvement_over_unscaled_hgao.median()) > 0,
    }
    passes = bool(all(gates.values()))
    summary = {"status": "complete", "locked_1921_1939_accessed": False, "common_western_cell_count": int(common.sum()), "coefficient_count": len(alpha_frame), "cross_product_score_count": len(scores), "median_alpha": median_alpha, "alpha_coefficient_of_variation": alpha_cv, "median_cross_product_skill_over_zero": float(scores.skill_over_zero.median()), "improving_cross_product_fraction": float((scores.skill_over_zero > 0).mean()), "minimum_cross_product_skill": float(scores.skill_over_zero.min()), "median_improvement_over_unscaled_hgao": float(scores.improvement_over_unscaled_hgao.median()), "evaluation_product_median_skills": {key: float(value) for key, value in product_medians.items()}, "gates": gates, "passes_nested_spatial_residual_shrinkage": passes, "decision": "advance_shrunk_hgao_residual_to_component_combination" if passes else "close_hgao_spatial_residual_shrinkage", "production_field_selected": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for product, selected in alpha_frame.groupby("fit_product", sort=False):
        axes[0].plot(selected.fold, selected.alpha, marker="o", label=product)
    axes[0].set(title="Nested HGAO residual shrinkage", xlabel="Held-out fold", ylabel="Training alpha", xticks=[2, 3, 4, 5])
    axes[0].legend(frameon=False, ncol=2)
    scores.boxplot(column="skill_over_zero", by="evaluation_product", ax=axes[1], grid=False)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].axhline(-0.05, color="#b2182b", linestyle="--", linewidth=0.9)
    axes[1].set(title="Held-out cross-product residual skill", xlabel="Evaluation product", ylabel="MSE improvement over zero")
    figure.suptitle("")
    figure.savefig(output / "hgao_spatial_residual_nested_shrinkage.png", dpi=180)
    plt.close(figure)

    lines = ["# HGAO Spatial-Residual Nested Shrinkage", "", f"- Median alpha / CV: `{median_alpha:.3f}/{alpha_cv:.3f}`.", f"- Median/minimum held-out skill: `{scores.skill_over_zero.median():.2%}/{scores.skill_over_zero.min():.2%}`.", f"- Improving paths: `{(scores.skill_over_zero > 0).sum()}/{len(scores)}`.", f"- Median improvement over unscaled HGAO: `{scores.improvement_over_unscaled_hgao.median():.2%}`.", f"- Passes: `{str(passes).lower()}`.", "", f"Decision: `{summary['decision']}`."]
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
