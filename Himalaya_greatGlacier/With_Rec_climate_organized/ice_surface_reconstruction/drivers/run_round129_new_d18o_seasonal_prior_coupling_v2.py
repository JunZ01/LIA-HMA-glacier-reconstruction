"""Test seasonal d18O proxy operators against the annual HMA precipitation prior."""

from __future__ import annotations

import calendar
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITES = {
    "mimei": {"latitude": 29.45, "longitude": 96.43, "operator": "JJA"},
    "xinpu": {"latitude": 30.42, "longitude": 95.07, "operator": "JJA"},
    "ganesh": {"latitude": 28.183, "longitude": 85.183, "operator": "annual"},
    "keylong": {"latitude": 32.75, "longitude": 76.62, "operator": "JJA_ONDJFMA"},
}
MEMBER_YEARS = np.arange(1001, 1849, dtype=int)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    item = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        item["sha256"] = digest.hexdigest()
    return item


def bilinear_points(dataset: xr.Dataset, variable: str, start: int, stop: int, points: list[tuple[float, float]]) -> np.ndarray:
    lat = np.asarray(dataset["lat"].values, dtype=float)
    lon = np.asarray(dataset["lon"].values, dtype=float)
    brackets = []
    lat_ids: set[int] = set()
    lon_ids: set[int] = set()
    for latitude, longitude in points:
        i1 = int(np.searchsorted(lat, latitude))
        j1 = int(np.searchsorted(lon, longitude))
        i0, j0 = i1 - 1, j1 - 1
        if i0 < 0 or i1 >= len(lat) or j0 < 0 or j1 >= len(lon):
            raise RuntimeError(f"point-outside-model-grid:{latitude},{longitude}")
        brackets.append((i0, i1, j0, j1, latitude, longitude))
        lat_ids.update((i0, i1))
        lon_ids.update((j0, j1))
    lat_ids_sorted = sorted(lat_ids)
    lon_ids_sorted = sorted(lon_ids)
    lat_lookup = {value: index for index, value in enumerate(lat_ids_sorted)}
    lon_lookup = {value: index for index, value in enumerate(lon_ids_sorted)}
    block = np.asarray(
        dataset[variable].isel(time=slice(start, stop), lat=lat_ids_sorted, lon=lon_ids_sorted).values,
        dtype=float,
    )
    output = np.empty((stop - start, len(points)), dtype=float)
    for column, (i0, i1, j0, j1, latitude, longitude) in enumerate(brackets):
        wy = (latitude - lat[i0]) / (lat[i1] - lat[i0])
        wx = (longitude - lon[j0]) / (lon[j1] - lon[j0])
        q00 = block[:, lat_lookup[i0], lon_lookup[j0]]
        q01 = block[:, lat_lookup[i0], lon_lookup[j1]]
        q10 = block[:, lat_lookup[i1], lon_lookup[j0]]
        q11 = block[:, lat_lookup[i1], lon_lookup[j1]]
        output[:, column] = (
            (1 - wy) * (1 - wx) * q00
            + (1 - wy) * wx * q01
            + wy * (1 - wx) * q10
            + wy * wx * q11
        )
    return output


def seasonal_mean(monthly: np.ndarray, source_years: np.ndarray, mode: str, precipitation: bool) -> np.ndarray:
    lookup = {(int(year), int(month)): index for index, (year, month) in enumerate(zip(source_years[:, 0], source_years[:, 1]))}
    result = np.full((len(MEMBER_YEARS), monthly.shape[1]), np.nan)
    for row, year in enumerate(MEMBER_YEARS):
        if mode == "annual":
            keys = [(year, month) for month in range(1, 13)]
        elif mode == "JJA":
            keys = [(year, month) for month in (6, 7, 8)]
        elif mode == "JJAS":
            keys = [(year, month) for month in (6, 7, 8, 9)]
        elif mode == "ONDJFMA":
            keys = [(year - 1, month) for month in (10, 11, 12)] + [(year, month) for month in (1, 2, 3, 4)]
        else:
            raise ValueError(mode)
        indices = [lookup[key] for key in keys]
        values = monthly[indices]
        if precipitation:
            weights = np.asarray([calendar.monthrange(key[0], key[1])[1] for key in keys], dtype=float)
            result[row] = np.average(values, axis=0, weights=weights)
        else:
            result[row] = np.mean(values, axis=0)
    return result


def correlations(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = matrix - np.mean(matrix, axis=1, keepdims=True)
    y = vector - np.mean(vector)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    return np.divide(x @ y, denominator, out=np.full(matrix.shape[0], np.nan), where=denominator > 0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1]) if int(valid.sum()) >= 3 else float("nan")


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_129_new_d18o_seasonal_prior_coupling_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    round128_driver = driver / "run_round128_new_d18o_documented_season_oof_v1.py"
    round128 = load_module("round129_round128", round128_driver)
    r118 = load_module("round129_r118", driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py")
    r123 = load_module("round129_r123", driver / "qualify_round123_southeast_tibet_d18o_v1.py")
    r125 = load_module("round129_r125", driver / "qualify_round125_ganesh_keylong_d18o_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    r123_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    r125_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round129_new_d18o_seasonal_prior_coupling_v2_contract.md",
        "driver": Path(__file__).resolve(),
        "round128_driver": round128_driver,
        "round128_summary": loop / "round_128_new_d18o_season_oof_v1/summary.json",
        "round128_metrics": loop / "round_128_new_d18o_season_oof_v1/documented_season_oof_metrics.csv",
        "temperature_prior": Path(r"D:\VM\gcm_7\cesm\cesm_TREFHT_merged.nc"),
        "precc_prior": Path(r"D:\VM\gcm_7\cesm\cesm_PRECC_merged.nc"),
        "precl_prior": Path(r"D:\VM\gcm_7\cesm\cesm_PRECL_merged.nc"),
        "annual_precipitation_prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "berkeley_temperature": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru_temperature": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "cru_precipitation": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "gpcc_precipitation": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "mimei": r123_dir / "raw_unopened/mimei_tree_ring_d18o.txt",
        "xinpu": r123_dir / "raw_unopened/xinpu_tree_ring_d18o.txt",
        "ganesh": r125_dir / "raw_unopened/ganesh2018d18o-noaa.txt",
        "keylong": r125_dir / "raw_unopened/keylong2020d18o-noaa.txt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: fingerprint(path) for name, path in paths.items()}

    prior128 = json.loads(paths["round128_summary"].read_text(encoding="utf-8"))
    if prior128["status"] != "complete" or prior128["locked_1921_1939_accessed"] or not prior128["source_hashes_unchanged"]:
        raise RuntimeError("round128-contract-failed")
    metrics128 = pd.read_csv(paths["round128_metrics"])

    site_names = list(SITES)
    latitudes = np.asarray([SITES[name]["latitude"] for name in site_names])
    longitudes = np.asarray([SITES[name]["longitude"] for name in site_names])
    responses = {}
    for name in ("mimei", "xinpu"):
        responses[name], audit = r123.parse_candidate(paths[name])
        if audit["locked_1921_1939_values_decoded"]:
            raise RuntimeError(f"locked-candidate-value-decoded:{name}")
    for name in ("ganesh", "keylong"):
        responses[name], audit = r125.parse_noaa_candidate(paths[name], r118.YEARS)
        if audit["locked_1921_1939_values_decoded"]:
            raise RuntimeError(f"locked-candidate-value-decoded:{name}")

    bea_monthly = r118.read_berkeley_points(paths["berkeley_temperature"], latitudes, longitudes)
    cru_t_monthly = r118.read_netcdf3_points(paths["cru_temperature"], "tmp", 1901, latitudes, longitudes)
    cru_p_monthly = r118.read_netcdf3_points(paths["cru_precipitation"], "pre", 1901, latitudes, longitudes)
    gpcc_monthly = r118.read_netcdf3_points(paths["gpcc_precipitation"], "precip", 1891, latitudes, longitudes)
    annual_months = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * len(site_names)
    jja_months = [(6, 7, 8)] * len(site_names)
    products = {
        "BEA_GPCC": {
            "annual_t": r118.center(r118.aggregate_temperature(bea_monthly, annual_months)),
            "annual_p": r118.center(r118.aggregate_precipitation(gpcc_monthly, annual_months)),
            "jja_t": r118.center(r118.aggregate_temperature(bea_monthly, jja_months)),
            "jja_p": r118.center(r118.aggregate_precipitation(gpcc_monthly, jja_months)),
        },
        "CRU": {
            "annual_t": r118.center(r118.aggregate_temperature(cru_t_monthly, annual_months)),
            "annual_p": r118.center(r118.aggregate_precipitation(cru_p_monthly, annual_months)),
            "jja_t": r118.center(r118.aggregate_temperature(cru_t_monthly, jja_months)),
            "jja_p": r118.center(r118.aggregate_precipitation(cru_p_monthly, jja_months)),
        },
    }
    keylong = site_names.index("keylong")
    for product, monthly in (("BEA_GPCC", gpcc_monthly), ("CRU", cru_p_monthly)):
        values = round128.aggregate_cross_year_winter(monthly[:, [keylong]], r118.YEARS)
        products[product]["winter_p"] = r118.center(values)[:, 0]

    with h5py.File(paths["annual_precipitation_prior"], "r") as handle:
        raw_prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float)
        target_lat = np.asarray(handle["lat"], dtype=float).ravel()
        target_lon = np.asarray(handle["lon"], dtype=float).ravel()
        stored_years = np.asarray(handle["member_years"], dtype=int).ravel()
    annual_prior = raw_prior.T if raw_prior.shape == (848, 336) else raw_prior
    if annual_prior.shape != (336, 848) or not np.array_equal(stored_years, MEMBER_YEARS):
        raise RuntimeError(f"annual-prior-shape-or-year-failed:{annual_prior.shape}")
    target_points = []
    target_indices = []
    for latitude, longitude in zip(latitudes, longitudes):
        ilat = int(np.argmin(np.abs(target_lat - latitude)))
        ilon = int(np.argmin(np.abs(target_lon - longitude)))
        target_points.append((float(target_lat[ilat]), float(target_lon[ilon])))
        target_indices.append(ilat + len(target_lat) * ilon)
    model_points = list(zip(latitudes.tolist(), longitudes.tolist())) + target_points

    with xr.open_dataset(paths["temperature_prior"], decode_times=False) as tds, xr.open_dataset(paths["precc_prior"], decode_times=False) as cds, xr.open_dataset(paths["precl_prior"], decode_times=False) as lds:
        dates = np.asarray(tds["date"].values, dtype=int)
        years = dates // 10000 if int(np.nanmedian(dates)) > 1_000_000 else dates // 100
        months = (dates // 100) % 100 if int(np.nanmedian(dates)) > 1_000_000 else dates % 100
        use = np.flatnonzero((years >= 1000) & (years <= 1848))
        if len(use) != 849 * 12 or not np.all(np.diff(use) == 1):
            raise RuntimeError(f"unexpected-prior-month-selection:{len(use)}")
        start, stop = int(use[0]), int(use[-1] + 1)
        source_year_month = np.column_stack((years[use], months[use]))
        model_t = bilinear_points(tds, "TREFHT", start, stop, model_points)
        model_p = (bilinear_points(cds, "PRECC", start, stop, model_points) + bilinear_points(lds, "PRECL", start, stop, model_points)) * 86_400_000.0

    annual_target_point_model = seasonal_mean(model_p[:, 4:], source_year_month, "annual", True)
    closure_rows = []
    for column, name in enumerate(site_names):
        reconstructed = annual_target_point_model[:, column]
        archived = annual_prior[target_indices[column]]
        reconstructed -= np.mean(reconstructed)
        archived = archived - np.mean(archived)
        closure_rows.append({
            "site": name,
            "target_latitude": target_points[column][0],
            "target_longitude": target_points[column][1],
            "correlation": correlation(reconstructed, archived),
            "normalized_rmse": float(np.sqrt(np.mean((reconstructed - archived) ** 2)) / np.std(archived, ddof=1)),
        })
    closure = pd.DataFrame(closure_rows)
    closure_pass = bool((closure.correlation >= 0.995).all() and (closure.normalized_rmse <= 0.10).all())
    if not closure_pass:
        closure.to_csv(output / "annual_prior_extraction_closure.csv", index=False)
        raise RuntimeError("annual-prior-extraction-closure-failed")

    weights = np.tile(np.cos(np.deg2rad(target_lat)), len(target_lon))
    hma_mean = np.average(annual_prior, axis=0, weights=weights)
    operator_series: dict[tuple[str, str], np.ndarray] = {}
    coefficient_rows = []
    metric_rows = []
    map_rows = []
    for column, name in enumerate(site_names):
        mode = SITES[name]["operator"]
        if mode == "annual":
            model_t_mode = seasonal_mean(model_t[:, :4], source_year_month, "annual", False)[:, column]
            model_p_mode = seasonal_mean(model_p[:, :4], source_year_month, "annual", True)[:, column]
            modern_t_key, modern_p_key, metric_model = "annual_t", "annual_p", "annual"
        elif mode == "JJA":
            model_t_mode = seasonal_mean(model_t[:, :4], source_year_month, "JJA", False)[:, column]
            model_p_mode = seasonal_mean(model_p[:, :4], source_year_month, "JJA", True)[:, column]
            modern_t_key, modern_p_key, metric_model = "jja_t", "jja_p", "seasonal"
        elif mode == "JJA_ONDJFMA":
            model_t_mode = seasonal_mean(model_t[:, :4], source_year_month, "JJA", False)[:, column]
            model_p_mode = seasonal_mean(model_p[:, :4], source_year_month, "ONDJFMA", True)[:, column]
            modern_t_key, modern_p_key, metric_model = "jja_t", "winter_p", "seasonal"
        else:
            raise ValueError(mode)
        model_t_mode -= np.mean(model_t_mode)
        model_p_mode -= np.mean(model_p_mode)
        for product, path_name in (("BEA_GPCC", "bea_self"), ("CRU", "cru_self")):
            modern_t = products[product][modern_t_key][:, column] if products[product][modern_t_key].ndim == 2 else products[product][modern_t_key]
            modern_p = products[product][modern_p_key][:, column] if products[product][modern_p_key].ndim == 2 else products[product][modern_p_key]
            coeff = r118.fit(responses[name], modern_t, modern_p, np.ones(len(r118.YEARS), dtype=bool))
            if coeff is None:
                raise RuntimeError(f"full-fit-failed:{name}:{product}")
            hx = coeff[1] * model_t_mode + coeff[2] * model_p_mode
            operator_series[(name, product)] = hx
            oof = metrics128[(metrics128.site == name) & (metrics128.path == path_name)].iloc[0]
            rmse = float(oof[f"{metric_model}_rmse"])
            corr_map = correlations(annual_prior, hx)
            info_ratio = float(np.var(hx, ddof=1) / (np.var(hx, ddof=1) + rmse**2))
            metric_rows.append({
                "site": name,
                "operator": mode,
                "calibration_product": product,
                "temperature_coefficient": float(coeff[1]),
                "precipitation_coefficient": float(coeff[2]),
                "oof_rmse": rmse,
                "operator_standard_deviation": float(np.std(hx, ddof=1)),
                "information_ratio": info_ratio,
                "hma_mean_correlation": correlation(hx, hma_mean),
                "fraction_cells_abs_correlation_ge_0_20": float(np.mean(np.abs(corr_map) >= 0.20)),
                "median_abs_cell_correlation": float(np.median(np.abs(corr_map))),
                "maximum_abs_cell_correlation": float(np.max(np.abs(corr_map))),
            })
            coefficient_rows.append({"site": name, "product": product, "intercept": float(coeff[0]), "temperature_coefficient": float(coeff[1]), "precipitation_coefficient": float(coeff[2])})
            for cell, value in enumerate(corr_map):
                ilon, ilat = divmod(cell, len(target_lat))
                map_rows.append({"site": name, "product": product, "cell": cell, "latitude": target_lat[ilat], "longitude": target_lon[ilon], "correlation": value})

    metrics = pd.DataFrame(metric_rows)
    maps = pd.DataFrame(map_rows)
    decisions = []
    for name in site_names:
        subset = metrics[metrics.site == name].set_index("calibration_product")
        bea_hx = operator_series[(name, "BEA_GPCC")]
        cru_hx = operator_series[(name, "CRU")]
        bea_map = maps[(maps.site == name) & (maps["product"] == "BEA_GPCC")].correlation.to_numpy()
        cru_map = maps[(maps.site == name) & (maps["product"] == "CRU")].correlation.to_numpy()
        gates = {
            "operator_product_correlation_ge_0_90": correlation(bea_hx, cru_hx) >= 0.90,
            "target_map_product_correlation_ge_0_80": correlation(bea_map, cru_map) >= 0.80,
            "both_abs_hma_mean_correlation_ge_0_10": bool((subset.hma_mean_correlation.abs() >= 0.10).all()),
            "both_fraction_cells_ge_0_10": bool((subset.fraction_cells_abs_correlation_ge_0_20 >= 0.10).all()),
            "both_information_ratio_ge_0_10": bool((subset.information_ratio >= 0.10).all()),
        }
        passed = bool(all(gates.values()))
        decisions.append({
            "site": name,
            "operator": SITES[name]["operator"],
            "operator_product_correlation": correlation(bea_hx, cru_hx),
            "target_map_product_correlation": correlation(bea_map, cru_map),
            **gates,
            "passes_all_gates": passed,
            "decision": "advance_to_frozen_enkf_addition_sensitivity" if passed else "close_for_current_annual_precipitation_prior",
        })
    decisions_frame = pd.DataFrame(decisions)

    closure.to_csv(output / "annual_prior_extraction_closure.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "full_calibration_coefficients.csv", index=False)
    metrics.to_csv(output / "operator_coupling_metrics.csv", index=False)
    maps.to_csv(output / "operator_target_correlation_maps.csv", index=False)
    decisions_frame.to_csv(output / "candidate_decisions.csv", index=False)
    curves = []
    hma_z = (hma_mean - np.mean(hma_mean)) / np.std(hma_mean, ddof=1)
    for name in site_names:
        for product in ("BEA_GPCC", "CRU"):
            hx = operator_series[(name, product)]
            hx_z = (hx - np.mean(hx)) / np.std(hx, ddof=1)
            curves.extend({"year": int(year), "site": name, "series": product, "standardized_value": float(value)} for year, value in zip(MEMBER_YEARS, hx_z))
        curves.extend({"year": int(year), "site": name, "series": "annual_HMA_precipitation_prior", "standardized_value": float(value)} for year, value in zip(MEMBER_YEARS, hma_z))
    pd.DataFrame(curves).to_csv(output / "operator_and_hma_prior_curves.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), constrained_layout=True, sharex=True)
    for axis, name in zip(axes.flat, site_names):
        axis.plot(MEMBER_YEARS, hma_z, color="#222222", linewidth=1.0, alpha=0.65, label="annual HMA precipitation prior")
        for product, color in (("BEA_GPCC", "#0072B2"), ("CRU", "#D55E00")):
            hx = operator_series[(name, product)]
            axis.plot(MEMBER_YEARS, (hx - np.mean(hx)) / np.std(hx, ddof=1), color=color, linewidth=1.05, label=product)
        axis.set_title(f"{name}: {SITES[name]['operator']}")
        axis.set_ylabel("standardized anomaly")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    figure.savefig(output / "round129_operator_vs_annual_hma_prior_curves.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True)
    for axis, name in zip(axes.flat, site_names):
        subset = maps[maps.site == name].groupby(["latitude", "longitude"], as_index=False).correlation.mean().pivot(index="latitude", columns="longitude", values="correlation")
        image = axis.imshow(subset.to_numpy(), origin="lower", extent=[target_lon.min(), target_lon.max(), target_lat.min(), target_lat.max()], vmin=-0.7, vmax=0.7, cmap="RdBu_r", aspect="auto")
        axis.scatter(SITES[name]["longitude"], SITES[name]["latitude"], c="black", s=18)
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes, label="operator vs annual precipitation prior correlation", shrink=0.85)
    figure.savefig(output / "round129_operator_target_correlation_maps.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round129")
    summary = {
        "experiment_id": "round129_new_d18o_seasonal_prior_coupling_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "member_years": [1001, 1848],
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "annual_prior_extraction_closure_passed": closure_pass,
        "source_fingerprints_unchanged": unchanged,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = [
        "# Round 129 New d18O Seasonal-Prior Coupling",
        "",
        f"- Annual-prior extraction closure passed: `{str(closure_pass).lower()}`.",
        "- Locked 1921-1939 accessed: `false`.",
        "- External validation reserve opened: `false`.",
        f"- Source fingerprints unchanged: `{str(unchanged).lower()}`.",
        "",
    ]
    for item in decisions:
        site_metrics = metrics[metrics.site == item["site"]]
        lines.extend([
            f"## {item['site']}",
            "",
            f"- Operator: `{item['operator']}`.",
            f"- Product operator correlation: `{item['operator_product_correlation']:.3f}`.",
            f"- Product target-map correlation: `{item['target_map_product_correlation']:.3f}`.",
            f"- HMA-mean correlations: `{', '.join(f'{v:.3f}' for v in site_metrics.hma_mean_correlation)}`.",
            f"- Information ratios: `{', '.join(f'{v:.3f}' for v in site_metrics.information_ratio)}`.",
            f"- Decision: `{item['decision']}`.",
            "",
        ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

