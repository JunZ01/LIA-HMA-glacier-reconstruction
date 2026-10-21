from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVER_DIR = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_220_netp_mjj_cross_prior_spatial_oof_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
ROUND134_DRIVER = DRIVER_DIR / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py"
ROUND218 = LOOP / "round_218_netp_mjj_prior_covariance_spatial_oof_v2"
ROUND219 = LOOP / "round_219_ccsm4_netp_mjj_prior_preflight_v1"
CAL_YEARS = np.arange(1940, 2001, dtype=int)
TARGET_LAT = np.asarray([35, 37, 39, 41] * 8, dtype=float)
TARGET_LON = np.repeat(np.arange(93, 109, 2), 4).astype(float)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 100_000_000:
        result["sha256"] = sha256(path)
    return result


def pattern_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) <= 1e-12 or np.std(predicted) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def run_branch(round134, proxy: np.ndarray, groups: list[np.ndarray], observed_cells: dict[str, np.ndarray], weights: np.ndarray, loadings: np.ndarray):
    prediction = {product: np.full((61, 32), np.nan) for product in round134.PRODUCTS}
    baseline = {product: np.full((61, 32), np.nan) for product in round134.PRODUCTS}
    observed_smooth = {product: np.full((61, 32), np.nan) for product in round134.PRODUCTS}
    fold_rows = []
    for product in round134.PRODUCTS:
        cells = observed_cells[product]
        regional = np.average(cells, axis=1, weights=weights)
        for fold_name, fit_start, fit_end, test_start, test_end in round134.FOLDS:
            fit = (CAL_YEARS >= fit_start) & (CAL_YEARS <= fit_end)
            test = (CAL_YEARS >= test_start) & (CAL_YEARS <= test_end)
            fold_proxy = np.fmax(round134.smooth_inside_mask(proxy, fit), round134.smooth_inside_mask(proxy, test))
            fold_regional = np.fmax(round134.smooth_inside_mask(regional, fit), round134.smooth_inside_mask(regional, test))
            fold_cells = np.fmax(round134.smooth_inside_mask(cells, fit), round134.smooth_inside_mask(cells, test))
            fit_valid = fit & np.isfinite(fold_regional)
            test_valid = test & np.isfinite(fold_regional)
            comp, _, _, _ = round134.composite(fold_proxy, fold_regional, fit_valid, groups)
            regional_prediction, coefficients = round134.linear_fit_predict(comp, fold_regional, fit_valid, test_valid)
            regional_center = float(np.mean(fold_regional[fit_valid]))
            cell_center = np.mean(fold_cells[fit_valid], axis=0)
            prediction[product][test_valid] = cell_center[None, :] + (regional_prediction[test_valid] - regional_center)[:, None] * loadings[None, :]
            baseline[product][test_valid] = cell_center[None, :]
            observed_smooth[product][test_valid] = fold_cells[test_valid]
            candidate_error = fold_cells[test_valid] - prediction[product][test_valid]
            baseline_error = fold_cells[test_valid] - baseline[product][test_valid]
            fold_rows.append({
                "product": product,
                "fold": fold_name,
                "field_skill": 1.0 - float(np.sum(candidate_error**2) / np.sum(baseline_error**2)),
                "regional_fit_slope": float(coefficients[1]),
                "test_center_count": int(test_valid.sum()),
            })

    fold_frame = pd.DataFrame(fold_rows)
    cell_rows = []
    pattern_rows = []
    metrics = {}
    for product in round134.PRODUCTS:
        valid_rows = np.isfinite(prediction[product]).all(axis=1)
        obs = observed_smooth[product][valid_rows]
        pred = prediction[product][valid_rows]
        base = baseline[product][valid_rows]
        pooled_skill = 1.0 - float(np.sum((obs - pred) ** 2) / np.sum((obs - base) ** 2))
        cell_skill = 1.0 - np.sum((obs - pred) ** 2, axis=0) / np.sum((obs - base) ** 2, axis=0)
        patterns = []
        for local_index, year_index in enumerate(np.flatnonzero(valid_rows)):
            value = pattern_correlation(obs[local_index] - base[local_index], pred[local_index] - base[local_index])
            patterns.append(value)
            pattern_rows.append({"product": product, "year": int(CAL_YEARS[year_index]), "spatial_pattern_correlation": value})
        metrics[product] = {
            "pooled_field_skill": pooled_skill,
            "median_cell_skill": float(np.median(cell_skill)),
            "positive_cell_fraction": float(np.mean(cell_skill > 0)),
            "median_yearly_spatial_pattern_correlation": float(np.nanmedian(patterns)),
            "regional_series_correlation": round134.correlation(np.average(obs, axis=1, weights=weights), np.average(pred, axis=1, weights=weights)),
        }
        for cell in range(32):
            cell_rows.append({"product": product, "latitude": TARGET_LAT[cell], "longitude": TARGET_LON[cell], "cell_skill": cell_skill[cell]})
    common = np.isfinite(prediction["GPCC"]) & np.isfinite(prediction["CRU"])
    gpcc_anomaly = prediction["GPCC"] - baseline["GPCC"]
    cru_anomaly = prediction["CRU"] - baseline["CRU"]
    cross_product = round134.correlation(gpcc_anomaly[common], cru_anomaly[common])
    gates = {
        "both_pooled_field_skills_ge_0_05": all(metrics[p]["pooled_field_skill"] >= 0.05 for p in round134.PRODUCTS),
        "both_median_cell_skills_positive": all(metrics[p]["median_cell_skill"] > 0 for p in round134.PRODUCTS),
        "both_positive_cell_fractions_ge_0_75": all(metrics[p]["positive_cell_fraction"] >= 0.75 for p in round134.PRODUCTS),
        "both_median_spatial_pattern_correlations_ge_0_30": all(metrics[p]["median_yearly_spatial_pattern_correlation"] >= 0.30 for p in round134.PRODUCTS),
        "all_four_fold_field_skills_positive": bool((fold_frame.field_skill > 0).all()),
        "cross_product_predicted_anomaly_correlation_ge_0_80": cross_product >= 0.80,
    }
    return {
        "prediction": prediction,
        "baseline": baseline,
        "observed": observed_smooth,
        "folds": fold_frame,
        "cells": pd.DataFrame(cell_rows),
        "patterns": pd.DataFrame(pattern_rows),
        "metrics": metrics,
        "cross_product": cross_product,
        "gates": gates,
        "passes": bool(all(gates.values())),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    round134 = load_module("round134_for_round220", ROUND134_DRIVER)
    audit = round134.load_module("round220_archive", DRIVER_DIR / "audit_positive_b2_event_seasonal_psm_v1.py")
    r118 = round134.load_module("round220_r118", DRIVER_DIR / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "contract": CONTRACT,
        "driver": Path(__file__).resolve(),
        "round134_driver": ROUND134_DRIVER,
        "round218_summary": ROUND218 / "summary.json",
        "round218_cells": ROUND218 / "cell_skill_and_prior_loadings.csv",
        "round219_summary": ROUND219 / "summary.json",
        "round219_prior": ROUND219 / "ccsm4_netp_mjj_prior_850_1850.npz",
        "proxy6": audit.archive_proxy_path(),
        "metadata": ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}
    summary218 = json.loads(paths["round218_summary"].read_text(encoding="utf-8"))
    summary219 = json.loads(paths["round219_summary"].read_text(encoding="utf-8"))
    if summary218["passes_all_gates"] or summary219["decision"] != "admit_ccsm4_as_independent_netp_mjj_prior_covariance":
        raise RuntimeError("Upstream branch state mismatch")

    cesm_frame = pd.read_csv(paths["round218_cells"])
    cesm_gpcc = cesm_frame[cesm_frame["product"] == "GPCC"].sort_values(["longitude", "latitude"])
    cesm_cru = cesm_frame[cesm_frame["product"] == "CRU"].sort_values(["longitude", "latitude"])
    if len(cesm_gpcc) != 32 or not np.array_equal(cesm_gpcc.prior_loading.to_numpy(), cesm_cru.prior_loading.to_numpy()):
        raise RuntimeError("CESM loading duplication mismatch")
    if not np.array_equal(cesm_gpcc.latitude.to_numpy(), TARGET_LAT) or not np.array_equal(cesm_gpcc.longitude.to_numpy(), TARGET_LON):
        raise RuntimeError("CESM coordinates mismatch")
    cesm = cesm_gpcc.prior_loading.to_numpy(float)
    with np.load(paths["round219_prior"]) as archive:
        if not np.array_equal(archive["latitude"], TARGET_LAT) or not np.array_equal(archive["longitude"], TARGET_LON):
            raise RuntimeError("CCSM coordinates mismatch")
        ccsm = archive["prior_loading"].astype(float)
    equal = 0.5 * (cesm + ccsm)
    weights = np.cos(np.deg2rad(TARGET_LAT))
    loadings = {"cesm_lme010": cesm, "ccsm4_past1000": ccsm, "equal_prior": equal}
    for name, values in loadings.items():
        if abs(float(np.average(values, weights=weights)) - 1.0) > 1e-12:
            raise RuntimeError(f"Loading closure failed: {name}")

    metadata = pd.read_csv(paths["metadata"])
    _, groups = round134.groups_from_metadata(metadata)
    proxy = round134.read_proxy_values(paths["proxy6"], round134.TRAIN_IDS, CAL_YEARS)
    observed_cells = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, TARGET_LAT, TARGET_LON)
        observed_cells[product] = round134.aggregate_mjj(monthly)
        if observed_cells[product].shape != (61, 32) or not np.isfinite(observed_cells[product]).all():
            raise RuntimeError(f"Observed cell read failed: {product}")

    results = {name: run_branch(round134, proxy, groups, observed_cells, weights, values) for name, values in loadings.items()}
    branch_rows = []
    fold_frames = []
    cell_frames = []
    pattern_frames = []
    for branch, result in results.items():
        for product in round134.PRODUCTS:
            row = {"branch": branch, "product": product, **result["metrics"][product]}
            row["cross_product_predicted_anomaly_field_correlation"] = result["cross_product"]
            row["branch_passes_all_gates"] = result["passes"]
            branch_rows.append(row)
        frame = result["folds"].copy(); frame.insert(0, "branch", branch); fold_frames.append(frame)
        frame = result["cells"].copy(); frame.insert(0, "branch", branch); cell_frames.append(frame)
        frame = result["patterns"].copy(); frame.insert(0, "branch", branch); pattern_frames.append(frame)
    branch_metrics = pd.DataFrame(branch_rows)
    branch_metrics.to_csv(OUT / "cross_prior_branch_metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUT / "cross_prior_fold_metrics.csv", index=False)
    pd.concat(cell_frames, ignore_index=True).to_csv(OUT / "cross_prior_cell_skill.csv", index=False)
    pd.concat(pattern_frames, ignore_index=True).to_csv(OUT / "cross_prior_yearly_pattern_correlation.csv", index=False)
    pd.DataFrame({"latitude": TARGET_LAT, "longitude": TARGET_LON, **{name: values for name, values in loadings.items()}}).to_csv(OUT / "prior_loading_comparison.csv", index=False)

    loading_correlation = float(np.corrcoef(cesm, ccsm)[0, 1])
    loading_rmse = float(np.sqrt(np.mean((cesm - ccsm) ** 2)))
    equal_passes = results["equal_prior"]["passes"]
    decision = "advance_equal_prior_netp_mjj_spatial_field_to_paleo_prototype" if equal_passes else "close_two_prior_netp_mjj_spatialization_retain_temporal_curve"

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.26})
    colors = {"cesm_lme010": "#B33A3A", "ccsm4_past1000": "#147D78", "equal_prior": "#222222"}
    markers = {"cesm_lme010": "o", "ccsm4_past1000": "s", "equal_prior": "^"}
    x = np.arange(2)
    for offset, branch in zip((-0.22, 0.0, 0.22), loadings):
        rows = branch_metrics[branch_metrics.branch == branch].set_index("product")
        axes[0].bar(x + offset, [rows.loc[p, "pooled_field_skill"] for p in round134.PRODUCTS], width=0.2, color=colors[branch], label=branch)
        axes[1].plot(x, [rows.loc[p, "median_yearly_spatial_pattern_correlation"] for p in round134.PRODUCTS], color=colors[branch], marker=markers[branch], linewidth=1.8, label=branch)
    axes[0].axhline(0.05, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, round134.PRODUCTS); axes[0].set_ylabel("Pooled field skill"); axes[0].set_title("Field amplitude skill")
    axes[1].axhline(0.30, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, round134.PRODUCTS); axes[1].set_ylabel("Median yearly spatial correlation"); axes[1].set_title("Spatial-pattern gate")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("NETP MJJ cross-prior spatial OOF", y=0.98)
    figure.savefig(OUT / "netp_mjj_cross_prior_spatial_oof.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("A source changed")
    summary = {
        "round": 220,
        "decision": decision,
        "prior_loading_comparison": {"correlation": loading_correlation, "rmse": loading_rmse},
        "branch_results": {branch: {"metrics": result["metrics"], "cross_product": result["cross_product"], "gates": result["gates"], "passes_all_gates": result["passes"]} for branch, result in results.items()},
        "equal_prior_production_gate_passed": equal_passes,
        "years_read": [1940, 2000],
        "paleo_proxy_values_read": False,
        "locked_1901_1939_values_read": False,
        "external_validation_reserve_opened": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
        "source_states_unchanged": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    equal_metrics = results["equal_prior"]["metrics"]
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 220 Result\n\n"
        f"`{decision}`\n\n"
        f"CESM/CCSM loading correlation is `{loading_correlation:.3f}`. The equal-prior GPCC/CRU pooled skills are "
        f"`{equal_metrics['GPCC']['pooled_field_skill']:+.3f}` / `{equal_metrics['CRU']['pooled_field_skill']:+.3f}`; "
        f"median yearly spatial correlations are `{equal_metrics['GPCC']['median_yearly_spatial_pattern_correlation']:+.3f}` / "
        f"`{equal_metrics['CRU']['median_yearly_spatial_pattern_correlation']:+.3f}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
