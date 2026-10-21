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


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVER_DIR = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_218_netp_mjj_prior_covariance_spatial_oof_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
ROUND134_DRIVER = DRIVER_DIR / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py"
PRIOR = LOOP / "round_134_netp_mjj_exact_prior_v1/netp_mjj_exact_prior_fields.mat"
PRIOR_SUMMARY = LOOP / "round_134_netp_mjj_exact_prior_v2/summary.json"
CAL_YEARS = np.arange(1940, 2001, dtype=int)


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
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def field(handle: h5py.File, name: str) -> np.ndarray:
    raw = np.asarray(handle[name], dtype=float)
    result = raw.T if raw.shape == (848, 336) else raw
    if result.shape != (336, 848):
        raise RuntimeError(f"Unexpected field shape: {result.shape}")
    return result


def pattern_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) <= 1e-12 or np.std(predicted) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    round134 = load_module("round134_for_round218", ROUND134_DRIVER)
    audit = round134.load_module(
        "round218_archive", DRIVER_DIR / "audit_positive_b2_event_seasonal_psm_v1.py"
    )
    r118 = round134.load_module(
        "round218_r118",
        DRIVER_DIR / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "contract": CONTRACT,
        "driver": Path(__file__).resolve(),
        "round134_driver": ROUND134_DRIVER,
        "prior": PRIOR,
        "prior_summary": PRIOR_SUMMARY,
        "proxy6": audit.archive_proxy_path(),
        "metadata": ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior_summary = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    if not prior_summary["passes_all_gates"] or prior_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("Round 134 exact-prior gate failed")
    before = {name: state(path) for name, path in paths.items()}

    metadata = pd.read_csv(paths["metadata"])
    group_keys, groups = round134.groups_from_metadata(metadata)
    if len(group_keys) != 15:
        raise RuntimeError("Unexpected proxy group count")
    proxy = round134.read_proxy_values(paths["proxy6"], round134.TRAIN_IDS, CAL_YEARS)
    if proxy.shape != (61, 19) or not np.isfinite(proxy).any():
        raise RuntimeError("Modern proxy read failed")

    with h5py.File(PRIOR, "r") as handle:
        prior = field(handle, "mjj_precipitation")
        lat_axis = np.asarray(handle["lat"], dtype=float).ravel()
        lon_axis = np.asarray(handle["lon"], dtype=float).ravel()
    state_lat = np.tile(lat_axis, len(lon_axis))
    state_lon = np.repeat(lon_axis, len(lat_axis))
    netp = (state_lat >= 35) & (state_lat <= 42) & (state_lon >= 92) & (state_lon <= 108)
    prior_cells = prior[netp]
    cell_lat = state_lat[netp]
    cell_lon = state_lon[netp]
    expected_lat = np.asarray([35, 37, 39, 41] * 8, dtype=float)
    expected_lon = np.repeat(np.arange(93, 109, 2), 4).astype(float)
    if prior_cells.shape != (32, 848) or not np.array_equal(cell_lat, expected_lat) or not np.array_equal(cell_lon, expected_lon):
        raise RuntimeError("NETP prior cell ordering mismatch")
    weights = np.cos(np.deg2rad(cell_lat))
    prior_regional = np.average(prior_cells, axis=0, weights=weights)
    regional_anomaly = prior_regional - prior_regional.mean()
    cell_anomaly = prior_cells - prior_cells.mean(axis=1, keepdims=True)
    loadings = cell_anomaly @ regional_anomaly / np.sum(regional_anomaly**2)
    loading_closure = float(np.average(loadings, weights=weights))
    if abs(loading_closure - 1.0) > 1e-12:
        raise RuntimeError(f"Prior loading regional closure failed: {loading_closure}")

    observed_cells = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, cell_lat, cell_lon)
        cells = round134.aggregate_mjj(monthly)
        if cells.shape != (61, 32) or not np.isfinite(cells).all():
            raise RuntimeError(f"Modern MJJ cells failed: {product}")
        observed_cells[product] = cells

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
            error_candidate = fold_cells[test_valid] - prediction[product][test_valid]
            error_baseline = fold_cells[test_valid] - baseline[product][test_valid]
            skill = 1.0 - float(np.sum(error_candidate**2) / np.sum(error_baseline**2))
            fold_rows.append({
                "product": product,
                "fold": fold_name,
                "fit_years": f"{fit_start}-{fit_end}",
                "test_years": f"{test_start}-{test_end}",
                "field_skill": skill,
                "regional_fit_slope": float(coefficients[1]),
                "test_center_count": int(test_valid.sum()),
            })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "fold_field_skill.csv", index=False)

    cell_rows = []
    product_summary = {}
    yearly_pattern_rows = []
    for product in round134.PRODUCTS:
        valid_rows = np.isfinite(prediction[product]).all(axis=1)
        obs = observed_smooth[product][valid_rows]
        pred = prediction[product][valid_rows]
        base = baseline[product][valid_rows]
        pooled_skill = 1.0 - float(np.sum((obs - pred) ** 2) / np.sum((obs - base) ** 2))
        cell_sse = np.sum((obs - pred) ** 2, axis=0)
        base_sse = np.sum((obs - base) ** 2, axis=0)
        cell_skill = 1.0 - cell_sse / base_sse
        for index in range(32):
            cell_rows.append({"product": product, "latitude": cell_lat[index], "longitude": cell_lon[index], "prior_loading": loadings[index], "cell_skill": cell_skill[index]})
        pattern_values = []
        for local_index, year_index in enumerate(np.flatnonzero(valid_rows)):
            value = pattern_correlation(obs[local_index] - base[local_index], pred[local_index] - base[local_index])
            pattern_values.append(value)
            yearly_pattern_rows.append({"product": product, "year": int(CAL_YEARS[year_index]), "spatial_pattern_correlation": value})
        product_summary[product] = {
            "pooled_field_skill": pooled_skill,
            "median_cell_skill": float(np.median(cell_skill)),
            "positive_cell_fraction": float(np.mean(cell_skill > 0)),
            "median_yearly_spatial_pattern_correlation": float(np.nanmedian(pattern_values)),
            "regional_series_correlation": round134.correlation(
                np.average(obs, axis=1, weights=weights), np.average(pred, axis=1, weights=weights)
            ),
        }
    cells_frame = pd.DataFrame(cell_rows)
    cells_frame.to_csv(OUT / "cell_skill_and_prior_loadings.csv", index=False)
    pd.DataFrame(yearly_pattern_rows).to_csv(OUT / "yearly_spatial_pattern_correlations.csv", index=False)

    common = np.isfinite(prediction["GPCC"]) & np.isfinite(prediction["CRU"])
    gpcc_anomaly = prediction["GPCC"] - baseline["GPCC"]
    cru_anomaly = prediction["CRU"] - baseline["CRU"]
    cross_product = round134.correlation(gpcc_anomaly[common], cru_anomaly[common])
    gates = {
        "both_pooled_field_skills_ge_0_05": all(product_summary[p]["pooled_field_skill"] >= 0.05 for p in round134.PRODUCTS),
        "both_median_cell_skills_positive": all(product_summary[p]["median_cell_skill"] > 0 for p in round134.PRODUCTS),
        "both_positive_cell_fractions_ge_0_75": all(product_summary[p]["positive_cell_fraction"] >= 0.75 for p in round134.PRODUCTS),
        "both_median_spatial_pattern_correlations_ge_0_30": all(product_summary[p]["median_yearly_spatial_pattern_correlation"] >= 0.30 for p in round134.PRODUCTS),
        "all_four_fold_field_skills_positive": bool((folds.field_skill > 0).all()),
        "cross_product_predicted_anomaly_correlation_ge_0_80": cross_product >= 0.80,
        "prior_loading_regional_closure": abs(loading_closure - 1.0) <= 1e-12,
    }
    passed = bool(all(gates.values()))
    decision = "advance_netp_mjj_prior_covariance_field_to_paleo_prototype" if passed else "close_fixed_netp_mjj_prior_covariance_spatialization_retain_temporal_curve"

    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), gridspec_kw={"hspace": 0.32, "wspace": 0.24})
    for column, product in enumerate(round134.PRODUCTS):
        skill = cells_frame[cells_frame.product == product].cell_skill.to_numpy().reshape(8, 4).T
        image = axes[0, column].imshow(skill, origin="lower", aspect="auto", cmap="RdBu", vmin=-0.5, vmax=0.5, extent=[92, 108, 34, 42])
        axes[0, column].set_title(f"{product} cell skill")
        axes[0, column].set_xlabel("Longitude")
        axes[0, column].set_ylabel("Latitude")
        figure.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.04)
        valid = np.isfinite(prediction[product]).all(axis=1)
        obs_reg = np.average(observed_smooth[product][valid], axis=1, weights=weights)
        pred_reg = np.average(prediction[product][valid], axis=1, weights=weights)
        axes[1, column].plot(CAL_YEARS[valid], obs_reg, color="#222222", linewidth=1.8, label="Observed MJJ")
        axes[1, column].plot(CAL_YEARS[valid], pred_reg, color="#B33A3A", linewidth=1.5, linestyle="--", label="Prior-covariance OOF")
        axes[1, column].set_title(f"{product} regional OOF")
        axes[1, column].set_xlabel("Year")
        axes[1, column].set_ylabel("mm/day")
        axes[1, column].grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axes[1, column].legend(frameon=False, fontsize=8.5)
    figure.suptitle("NETP MJJ fixed-prior covariance spatialization test", y=0.98)
    figure.savefig(OUT / "netp_mjj_prior_covariance_spatial_oof.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("A source changed")
    summary = {
        "round": 218,
        "decision": decision,
        "product_metrics": product_summary,
        "cross_product_predicted_anomaly_field_correlation": cross_product,
        "prior_loading": {"minimum": float(loadings.min()), "maximum": float(loadings.max()), "area_weighted_mean": loading_closure},
        "gates": gates,
        "passes_all_gates": passed,
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
    result = f"""# Round 218 Result

## Decision

`{decision}`

GPCC/CRU pooled spatial-field skills are
`{product_summary['GPCC']['pooled_field_skill']:+.3f}` and
`{product_summary['CRU']['pooled_field_skill']:+.3f}`. Median cell skills are
`{product_summary['GPCC']['median_cell_skill']:+.3f}` and
`{product_summary['CRU']['median_cell_skill']:+.3f}`; positive-cell fractions
are `{product_summary['GPCC']['positive_cell_fraction']:.1%}` and
`{product_summary['CRU']['positive_cell_fraction']:.1%}`.

The GPCC/CRU predicted anomaly fields correlate `{cross_product:.3f}`. No
paleo, locked-period, Jionglacuo, EnKF, production, or OGGM value was used.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
