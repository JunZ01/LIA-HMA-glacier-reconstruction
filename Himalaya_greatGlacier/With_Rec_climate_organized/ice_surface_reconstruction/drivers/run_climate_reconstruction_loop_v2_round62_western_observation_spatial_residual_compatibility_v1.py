"""Audit observation-product agreement in western spatial precipitation residuals."""

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
    output = loop / "round_62_western_observation_spatial_residual_compatibility_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round61 = loop / "round_61_precl_regional_spatial_residual_decomposition_v2"
    driver58 = Path(__file__).resolve().with_name("run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py")
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round61_summary": round61 / "summary.json",
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
    summary61 = json.loads(paths["round61_summary"].read_text(encoding="utf-8"))
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
    if int(common.sum()) != int(summary61["common_western_cell_count"]):
        raise RuntimeError("common-support-closure-failed")
    weights_2d = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], common.shape)
    weights = weights_2d[common].astype(float)
    weights /= np.sum(weights)
    residuals: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        field = observations[product][:, common]
        if not np.all(np.isfinite(field)):
            raise RuntimeError(f"nonfinite-product-on-common-support:{product}")
        residuals[product] = field - (field @ weights)[:, None]

    annual_rows: list[dict[str, object]] = []
    pair_results: dict[str, dict[str, object]] = {}
    for left, right in itertools.combinations(PRODUCTS, 2):
        correlations = np.asarray([weighted_correlation(residuals[left][index], residuals[right][index], weights) for index in range(len(YEARS))])
        if not np.all(np.isfinite(correlations)):
            raise RuntimeError(f"nonfinite-pair-correlation:{left}:{right}")
        median = float(np.median(correlations))
        positive_fraction = float(np.mean(correlations > 0))
        gates = {
            "median_pattern_correlation_at_least_0_10": median >= 0.10,
            "at_least_60pct_years_positive": positive_fraction >= 0.60,
        }
        key = f"{left}__{right}"
        pair_results[key] = {"left": left, "right": right, "median_annual_pattern_correlation": median, "positive_year_fraction": positive_fraction, "gates": gates, "passes_spatial_compatibility": bool(all(gates.values()))}
        for year, correlation in zip(YEARS, correlations):
            annual_rows.append({"year": int(year), "left": left, "right": right, "pair": key, "spatial_pattern_correlation": float(correlation)})

    precl_pairs = [result for result in pair_results.values() if "PREC/L" in (result["left"], result["right"])]
    precl_valid = len(precl_pairs) == 3 and all(result["passes_spatial_compatibility"] for result in precl_pairs)
    all_pair_pass_count = int(sum(result["passes_spatial_compatibility"] for result in pair_results.values()))
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(output / "annual_observation_spatial_pattern_correlations.csv", index=False)
    pd.DataFrame([{key: value for key, value in result.items() if key != "gates"} for result in pair_results.values()]).to_csv(output / "observation_spatial_pair_summary.csv", index=False)
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "common_western_cell_count": int(common.sum()),
        "pair_results": pair_results,
        "passing_pair_count": all_pair_pass_count,
        "total_pair_count": len(pair_results),
        "precl_is_valid_cross_product_spatial_adjudicator": precl_valid,
        "round61_spatial_rejection_scope": "cross_product" if precl_valid else "precl_specific_only",
        "decision": "accept_round61_as_cross_product_spatial_rejection" if precl_valid else "do_not_generalize_round61_spatial_rejection",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (pair, selected) in enumerate(annual.groupby("pair", sort=False)):
        axis.plot(selected.year, selected.spatial_pattern_correlation, linewidth=1.1, color=colors[index], label=pair.replace("__", " / "))
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axhline(0.1, color="#333333", linestyle="--", linewidth=0.9)
    axis.set(title="Western observation annual spatial-residual agreement", xlabel="Year", ylabel="Weighted spatial-pattern correlation")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(output / "western_observation_spatial_residual_compatibility.png", dpi=180)
    plt.close(figure)

    lines = ["# Western Observation Spatial-Residual Compatibility", ""]
    for result in pair_results.values():
        lines.append(f"- {result['left']} / {result['right']}: median correlation `{result['median_annual_pattern_correlation']:.3f}`, positive years `{result['positive_year_fraction']:.1%}`, passes `{str(result['passes_spatial_compatibility']).lower()}`.")
    lines.extend(["", f"PREC/L valid spatial adjudicator: `{str(precl_valid).lower()}`.", f"Decision: `{summary['decision']}`."])
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
