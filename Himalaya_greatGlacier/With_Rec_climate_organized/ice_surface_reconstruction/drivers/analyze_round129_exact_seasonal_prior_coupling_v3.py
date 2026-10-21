"""Analyze exact seasonal proxy operators against the annual HMA precipitation prior."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITES = {
    "mimei": {"latitude": 29.45, "longitude": 96.43, "operator": "JJA"},
    "xinpu": {"latitude": 30.42, "longitude": 95.07, "operator": "JJA"},
    "ganesh": {"latitude": 28.183, "longitude": 85.183, "operator": "annual"},
    "keylong": {"latitude": 32.75, "longitude": 76.62, "operator": "JJA_ONDJFMA"},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orient_field(handle: h5py.File, name: str) -> np.ndarray:
    raw = np.asarray(handle[name], dtype=float)
    field = raw.T if raw.shape == (848, 336) else raw
    if field.shape != (336, 848):
        raise RuntimeError(f"unexpected-field-shape:{name}:{field.shape}")
    return field


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_129_new_d18o_exact_seasonal_prior_coupling_v3"
    exact_prior = output / "exact_seasonal_prior_fields.mat"
    if not exact_prior.exists():
        raise FileNotFoundError(exact_prior)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    base = load_module("round129_v3_base", driver / "run_round129_new_d18o_seasonal_prior_coupling_v2.py")
    round128 = load_module("round129_v3_round128", driver / "run_round128_new_d18o_documented_season_oof_v1.py")
    r118 = load_module("round129_v3_r118", driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py")
    r123 = load_module("round129_v3_r123", driver / "qualify_round123_southeast_tibet_d18o_v1.py")
    r125 = load_module("round129_v3_r125", driver / "qualify_round125_ganesh_keylong_d18o_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    r123_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    r125_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round129_new_d18o_exact_seasonal_prior_coupling_v3_contract.md",
        "driver": Path(__file__).resolve(),
        "matlab_exporter": driver / "export_round129_exact_seasonal_prior_v3.m",
        "exact_seasonal_prior": exact_prior,
        "round128_summary": loop / "round_128_new_d18o_season_oof_v1/summary.json",
        "round128_metrics": loop / "round_128_new_d18o_season_oof_v1/documented_season_oof_metrics.csv",
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
    before = {name: base.fingerprint(path) for name, path in paths.items()}
    prior128 = json.loads(paths["round128_summary"].read_text(encoding="utf-8"))
    if prior128["status"] != "complete" or prior128["locked_1921_1939_accessed"] or not prior128["source_hashes_unchanged"]:
        raise RuntimeError("round128-contract-failed")
    metrics128 = pd.read_csv(paths["round128_metrics"])

    names = list(SITES)
    latitudes = np.asarray([SITES[name]["latitude"] for name in names])
    longitudes = np.asarray([SITES[name]["longitude"] for name in names])
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
    annual_months = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * len(names)
    jja_months = [(6, 7, 8)] * len(names)
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
    keylong_column = names.index("keylong")
    for product, monthly in (("BEA_GPCC", gpcc_monthly), ("CRU", cru_p_monthly)):
        products[product]["winter_p"] = r118.center(round128.aggregate_cross_year_winter(monthly[:, [keylong_column]], r118.YEARS))[:, 0]

    with h5py.File(exact_prior, "r") as handle:
        annual_p = orient_field(handle, "annual_precipitation")
        jja_p = orient_field(handle, "jja_precipitation")
        winter_p = orient_field(handle, "ondjfma_precipitation")
        annual_t = orient_field(handle, "annual_temperature")
        jja_t = orient_field(handle, "jja_temperature")
        target_lat = np.asarray(handle["lat"], dtype=float).ravel()
        target_lon = np.asarray(handle["lon"], dtype=float).ravel()
        member_years = np.asarray(handle["member_years"], dtype=int).ravel()
        closure_max = float(np.asarray(handle["annual_closure_max_abs"]).ravel()[0])
        closure_rmse = float(np.asarray(handle["annual_closure_rmse"]).ravel()[0])
        closure_changed = int(np.asarray(handle["annual_closure_changed_count"]).ravel()[0])
    if closure_max != 0 or closure_rmse != 0 or closure_changed != 0 or not np.array_equal(member_years, np.arange(1001, 1849)):
        raise RuntimeError("exact-annual-prior-closure-or-year-failed")

    weights = np.tile(np.cos(np.deg2rad(target_lat)), len(target_lon))
    hma_mean = np.average(annual_p, axis=0, weights=weights)
    operator_series = {}
    coefficient_rows = []
    metric_rows = []
    map_rows = []
    for column, name in enumerate(names):
        ilat = int(np.argmin(np.abs(target_lat - SITES[name]["latitude"])))
        ilon = int(np.argmin(np.abs(target_lon - SITES[name]["longitude"])))
        cell = ilat + len(target_lat) * ilon
        mode = SITES[name]["operator"]
        if mode == "annual":
            prior_t, prior_p = annual_t[cell], annual_p[cell]
            modern_t_key, modern_p_key, metric_model = "annual_t", "annual_p", "annual"
        elif mode == "JJA":
            prior_t, prior_p = jja_t[cell], jja_p[cell]
            modern_t_key, modern_p_key, metric_model = "jja_t", "jja_p", "seasonal"
        elif mode == "JJA_ONDJFMA":
            prior_t, prior_p = jja_t[cell], winter_p[cell]
            modern_t_key, modern_p_key, metric_model = "jja_t", "winter_p", "seasonal"
        else:
            raise ValueError(mode)
        prior_t = prior_t - np.mean(prior_t)
        prior_p = prior_p - np.mean(prior_p)
        for product, path_name in (("BEA_GPCC", "bea_self"), ("CRU", "cru_self")):
            modern_t_values = products[product][modern_t_key]
            modern_p_values = products[product][modern_p_key]
            modern_t = modern_t_values[:, column] if modern_t_values.ndim == 2 else modern_t_values
            modern_p = modern_p_values[:, column] if modern_p_values.ndim == 2 else modern_p_values
            coeff = r118.fit(responses[name], modern_t, modern_p, np.ones(len(r118.YEARS), dtype=bool))
            if coeff is None:
                raise RuntimeError(f"full-fit-failed:{name}:{product}")
            hx = coeff[1] * prior_t + coeff[2] * prior_p
            operator_series[(name, product)] = hx
            oof = metrics128[(metrics128.site == name) & (metrics128.path == path_name)].iloc[0]
            rmse = float(oof[f"{metric_model}_rmse"])
            corr_map = base.correlations(annual_p, hx)
            signal_variance = float(np.var(hx, ddof=1))
            metric_rows.append({
                "site": name,
                "operator": mode,
                "calibration_product": product,
                "proxy_grid_latitude": target_lat[ilat],
                "proxy_grid_longitude": target_lon[ilon],
                "temperature_coefficient": float(coeff[1]),
                "precipitation_coefficient": float(coeff[2]),
                "oof_rmse": rmse,
                "operator_standard_deviation": float(np.sqrt(signal_variance)),
                "information_ratio": signal_variance / (signal_variance + rmse**2),
                "hma_mean_correlation": base.correlation(hx, hma_mean),
                "fraction_cells_abs_correlation_ge_0_20": float(np.mean(np.abs(corr_map) >= 0.20)),
                "median_abs_cell_correlation": float(np.median(np.abs(corr_map))),
                "maximum_abs_cell_correlation": float(np.max(np.abs(corr_map))),
            })
            coefficient_rows.append({"site": name, "product": product, "intercept": float(coeff[0]), "temperature_coefficient": float(coeff[1]), "precipitation_coefficient": float(coeff[2])})
            for map_cell, value in enumerate(corr_map):
                map_ilon, map_ilat = divmod(map_cell, len(target_lat))
                map_rows.append({"site": name, "product": product, "cell": map_cell, "latitude": target_lat[map_ilat], "longitude": target_lon[map_ilon], "correlation": value})

    metrics = pd.DataFrame(metric_rows)
    maps = pd.DataFrame(map_rows)
    decisions = []
    for name in names:
        subset = metrics[metrics.site == name].set_index("calibration_product")
        bea_hx, cru_hx = operator_series[(name, "BEA_GPCC")], operator_series[(name, "CRU")]
        bea_map = maps[(maps.site == name) & (maps["product"] == "BEA_GPCC")].correlation.to_numpy()
        cru_map = maps[(maps.site == name) & (maps["product"] == "CRU")].correlation.to_numpy()
        gates = {
            "operator_product_correlation_ge_0_90": base.correlation(bea_hx, cru_hx) >= 0.90,
            "target_map_product_correlation_ge_0_80": base.correlation(bea_map, cru_map) >= 0.80,
            "both_abs_hma_mean_correlation_ge_0_10": bool((subset.hma_mean_correlation.abs() >= 0.10).all()),
            "both_fraction_cells_ge_0_10": bool((subset.fraction_cells_abs_correlation_ge_0_20 >= 0.10).all()),
            "both_information_ratio_ge_0_10": bool((subset.information_ratio >= 0.10).all()),
        }
        passed = bool(all(gates.values()))
        decisions.append({
            "site": name,
            "operator": SITES[name]["operator"],
            "operator_product_correlation": base.correlation(bea_hx, cru_hx),
            "target_map_product_correlation": base.correlation(bea_map, cru_map),
            **gates,
            "passes_all_gates": passed,
            "decision": "advance_to_frozen_enkf_addition_sensitivity" if passed else "close_for_current_annual_precipitation_prior",
        })
    decisions_frame = pd.DataFrame(decisions)

    pd.DataFrame(coefficient_rows).to_csv(output / "full_calibration_coefficients.csv", index=False)
    metrics.to_csv(output / "operator_coupling_metrics.csv", index=False)
    maps.to_csv(output / "operator_target_correlation_maps.csv", index=False)
    decisions_frame.to_csv(output / "candidate_decisions.csv", index=False)
    curves = []
    hma_z = (hma_mean - np.mean(hma_mean)) / np.std(hma_mean, ddof=1)
    for name in names:
        for product in ("BEA_GPCC", "CRU"):
            hx = operator_series[(name, product)]
            hx_z = (hx - np.mean(hx)) / np.std(hx, ddof=1)
            curves.extend({"year": int(year), "site": name, "series": product, "standardized_value": float(value)} for year, value in zip(member_years, hx_z))
        curves.extend({"year": int(year), "site": name, "series": "annual_HMA_precipitation_prior", "standardized_value": float(value)} for year, value in zip(member_years, hma_z))
    pd.DataFrame(curves).to_csv(output / "operator_and_hma_prior_curves.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), constrained_layout=True, sharex=True)
    for axis, name in zip(axes.flat, names):
        axis.plot(member_years, hma_z, color="#222222", linewidth=0.9, alpha=0.6, label="annual HMA precipitation prior")
        for product, color in (("BEA_GPCC", "#0072B2"), ("CRU", "#D55E00")):
            hx = operator_series[(name, product)]
            axis.plot(member_years, (hx - np.mean(hx)) / np.std(hx, ddof=1), color=color, linewidth=1.0, label=product)
        axis.set_title(f"{name}: {SITES[name]['operator']}")
        axis.set_ylabel("standardized anomaly")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    figure.savefig(output / "round129_exact_operator_vs_annual_hma_prior_curves.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        subset = maps[maps.site == name].groupby(["latitude", "longitude"], as_index=False).correlation.mean().pivot(index="latitude", columns="longitude", values="correlation")
        image = axis.imshow(subset.to_numpy(), origin="lower", extent=[target_lon.min(), target_lon.max(), target_lat.min(), target_lat.max()], vmin=-0.7, vmax=0.7, cmap="RdBu_r", aspect="auto")
        axis.scatter(SITES[name]["longitude"], SITES[name]["latitude"], c="black", s=18)
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes, label="operator vs annual precipitation prior correlation", shrink=0.85)
    figure.savefig(output / "round129_exact_operator_target_correlation_maps.png", dpi=220)
    plt.close(figure)

    after = {name: base.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round129-v3-analysis")
    summary = {
        "experiment_id": "round129_new_d18o_exact_seasonal_prior_coupling_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "member_years": [1001, 1848],
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "exact_annual_prior_closure": {"max_abs": closure_max, "rmse": closure_rmse, "changed_count": closure_changed},
        "source_fingerprints_unchanged": unchanged,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = ["# Round 129 Exact Seasonal-Prior Coupling V3", "", "- Exact annual-prior closure: `true` (zero changed values).", "- Locked 1921-1939 accessed: `false`.", "- External validation reserve opened: `false`.", f"- Source fingerprints unchanged: `{str(unchanged).lower()}`.", ""]
    for item in decisions:
        site_metrics = metrics[metrics.site == item["site"]]
        lines.extend([
            f"## {item['site']}", "",
            f"- Operator: `{item['operator']}`.",
            f"- Product operator correlation: `{item['operator_product_correlation']:.3f}`.",
            f"- Product target-map correlation: `{item['target_map_product_correlation']:.3f}`.",
            f"- HMA-mean correlations: `{', '.join(f'{value:.3f}' for value in site_metrics.hma_mean_correlation)}`.",
            f"- Information ratios: `{', '.join(f'{value:.3f}' for value in site_metrics.information_ratio)}`.",
            f"- Decision: `{item['decision']}`.", "",
        ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
