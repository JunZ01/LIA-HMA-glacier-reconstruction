"""Nested cross-product validation of a regional-mean spatial fingerprint."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
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


def weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    x = left - np.sum(weights * left)
    y = right - np.sum(weights * right)
    denominator = np.sqrt(np.sum(weights * x**2) * np.sum(weights * y**2))
    return float(np.sum(weights * x * y) / denominator) if denominator > 0 else np.nan


def mse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(np.sum(values**2 * weights[None], axis=1)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_65_observation_regional_spatial_fingerprint_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round64 = loop / "round_64_hgao_spatial_residual_nested_shrinkage_v1"
    driver58 = Path(__file__).resolve().with_name("run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py")
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round64_summary": round64 / "summary.json",
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
    if json.loads(paths["round64_summary"].read_text(encoding="utf-8"))["passes_nested_spatial_residual_shrinkage"]:
        raise RuntimeError("round64-unexpected-pass")
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
    observations["UDel"] = helpers.recenter(helpers.annualize_udel(paths["udel"], latitudes, longitudes))
    observations["PREC/L"] = helpers.recenter(helpers.annualize_precl(precl_paths, latitudes, longitudes))
    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if int(common.sum()) != 418:
        raise RuntimeError(f"common-support-closure-failed:{int(common.sum())}")
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], common.shape)
    weights = weights_2d[common].astype(float)
    weights /= np.sum(weights)
    means: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        field = observations[product][:, common]
        means[product] = field @ weights
        residuals[product] = field - means[product][:, None]

    full_fingerprints: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        denominator = float(np.sum(means[product] ** 2))
        full_fingerprints[product] = np.sum(means[product][:, None] * residuals[product], axis=0) / denominator
    fingerprint_rows = []
    for left, right in itertools.combinations(PRODUCTS, 2):
        fingerprint_rows.append({"left": left, "right": right, "weighted_fingerprint_correlation": weighted_correlation(full_fingerprints[left], full_fingerprints[right], weights)})

    score_rows: list[dict[str, object]] = []
    annual_corr_values: list[float] = []
    for fold, (start, end) in enumerate(BLOCKS, start=2):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        for fit_product in PRODUCTS:
            fit_mean = means[fit_product][train]
            denominator = float(np.sum(fit_mean**2))
            if denominator <= 0:
                raise RuntimeError(f"zero-regional-training-energy:{fit_product}:{fold}")
            fingerprint = np.sum(fit_mean[:, None] * residuals[fit_product][train], axis=0) / denominator
            for evaluation_product in PRODUCTS:
                if evaluation_product == fit_product:
                    continue
                prediction = means[evaluation_product][test, None] * fingerprint[None]
                target = residuals[evaluation_product][test]
                skill = 1.0 - mse(prediction - target, weights) / mse(target, weights)
                correlations = [weighted_correlation(prediction[index], target[index], weights) for index in range(prediction.shape[0])]
                if not np.all(np.isfinite(np.r_[skill, correlations])):
                    raise RuntimeError(f"nonfinite-fingerprint-statistic:{fit_product}:{evaluation_product}:{fold}")
                annual_corr_values.extend(correlations)
                score_rows.append({"fold": fold, "start_year": start, "end_year": end, "fit_product": fit_product, "evaluation_product": evaluation_product, "heldout_residual_skill": skill, "median_annual_pattern_correlation": float(np.median(correlations)), "positive_annual_pattern_fraction": float(np.mean(np.asarray(correlations) > 0))})

    scores = pd.DataFrame(score_rows)
    fingerprints = pd.DataFrame(fingerprint_rows)
    scores.to_csv(output / "nested_cross_product_fingerprint_scores.csv", index=False)
    fingerprints.to_csv(output / "full_training_fingerprint_agreement.csv", index=False)
    product_medians = scores.groupby("evaluation_product").heldout_residual_skill.median().to_dict()
    fingerprint_median = float(fingerprints.weighted_fingerprint_correlation.median())
    gates = {
        "median_cross_product_skill_positive": float(scores.heldout_residual_skill.median()) > 0,
        "at_least_60pct_paths_improve": float((scores.heldout_residual_skill > 0).mean()) >= 0.60,
        "every_evaluation_product_median_skill_positive": all(value > 0 for value in product_medians.values()),
        "no_path_regresses_more_than_5pct": float(scores.heldout_residual_skill.min()) >= -0.05,
        "median_annual_pattern_correlation_at_least_0_10": float(np.median(annual_corr_values)) >= 0.10,
        "median_fingerprint_pair_correlation_at_least_0_50": fingerprint_median >= 0.50,
    }
    passes = bool(all(gates.values()))
    summary = {"status": "complete", "locked_1921_1939_accessed": False, "common_western_cell_count": int(common.sum()), "cross_product_path_count": len(scores), "median_heldout_residual_skill": float(scores.heldout_residual_skill.median()), "improving_path_fraction": float((scores.heldout_residual_skill > 0).mean()), "minimum_path_skill": float(scores.heldout_residual_skill.min()), "evaluation_product_median_skills": {key: float(value) for key, value in product_medians.items()}, "median_annual_spatial_pattern_correlation": float(np.median(annual_corr_values)), "median_full_training_fingerprint_pair_correlation": fingerprint_median, "gates": gates, "passes_regional_to_spatial_fingerprint_recoverability": passes, "decision": "advance_fingerprint_to_reconstructed_regional_driver_test" if passes else "close_single_regional_mean_spatial_fingerprint", "production_field_selected": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    scores.boxplot(column="heldout_residual_skill", by="evaluation_product", ax=axes[0], grid=False)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].axhline(-0.05, color="#b2182b", linestyle="--", linewidth=0.9)
    axes[0].set(title="Held-out regional-fingerprint skill", xlabel="Evaluation product", ylabel="Residual MSE improvement")
    for product, fingerprint in full_fingerprints.items():
        order = np.argsort(longitudes[np.where(common)[1]])
        axes[1].plot(np.arange(len(fingerprint)), fingerprint[order], linewidth=1.0, label=product)
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[1].set(title="Full-training product fingerprints", xlabel="Common cells ordered by longitude", ylabel="Residual loading")
    axes[1].legend(frameon=False, ncol=2)
    figure.suptitle("")
    figure.savefig(output / "observation_regional_spatial_fingerprint.png", dpi=180)
    plt.close(figure)

    lines = ["# Observation Regional-To-Spatial Fingerprint", "", f"- Median/minimum held-out residual skill: `{scores.heldout_residual_skill.median():.2%}/{scores.heldout_residual_skill.min():.2%}`.", f"- Improving paths: `{(scores.heldout_residual_skill > 0).sum()}/{len(scores)}`.", f"- Median annual pattern correlation: `{np.median(annual_corr_values):.3f}`.", f"- Median fingerprint-pair correlation: `{fingerprint_median:.3f}`.", f"- Passes: `{str(passes).lower()}`.", "", f"Decision: `{summary['decision']}`."]
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
