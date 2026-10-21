"""Compare the 1915-1918 positive-b2 proxy event with local climate products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


YEARS = np.arange(1905, 1921)
BASELINE = (1905, 1914)
EVENT = (1915, 1918)
CALIBRATION_YEARS = np.arange(1940, 2001)
FILL_VALUE_LIMIT = -9000.0
RNG_SEED = 20260814
PERMUTATIONS = 10000
CLIMATE = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all" / "python_data" / "proxy_data_6.mat"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def permutation_correlation(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    observed = correlation(left, right)
    if not np.isfinite(observed):
        return observed, float("nan")
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        null[index] = correlation(left, rng.permutation(right))
    return observed, float((1 + np.sum(np.abs(null) >= abs(observed))) / (PERMUTATIONS + 1))


def period_change(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= BASELINE[0]) & (YEARS <= BASELINE[1])
    event = (YEARS >= EVENT[0]) & (YEARS <= EVENT[1])
    return np.mean(values[event], axis=0) - np.mean(values[baseline], axis=0)


def sign_class(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.where(
        (left < 0) & (right < 0),
        "both_decline",
        np.where((left >= 0) & (right >= 0), "both_non_decline", "products_disagree"),
    )


def fraction_of_negative_effect(table: pd.DataFrame, mask: np.ndarray) -> float:
    denominator = table["negative_effect_magnitude"].sum()
    return float(table.loc[mask, "negative_effect_magnitude"].sum() / denominator) if denominator > 0 else float("nan")


def main() -> int:
    root = project_root()
    event_table_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_1915_1918_spatial_source_v1/positive_b2_event_proxy_table.csv"
    proxy6 = archive_proxy_path()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1"
    sources = {"event_table": event_table_path, "proxy6": proxy6, "climate": CLIMATE}
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    event_hash_before = digest(event_table_path)
    source_stats_before = {name: source_stat(path) for name, path in sources.items() if name != "event_table"}
    table = pd.read_csv(event_table_path).sort_values("proxy_index").reset_index(drop=True)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")
    selected = table["proxy_index"].to_numpy(dtype=int)

    proxy_raw = np.full((len(YEARS), len(table)), np.nan)
    calibration = np.full((len(CALIBRATION_YEARS), len(table)), np.nan)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(selected):
            proxy = handle[refs[proxy_index - 1]]
            event_slice = slice(1904, 1920)
            calibration_slice = slice(1939, 2000)
            if not np.array_equal(np.asarray(proxy["age_yng"][0, event_slice], dtype=float), YEARS):
                raise RuntimeError(f"proxy-event-index-contract-failed:{proxy_index}")
            if not np.array_equal(np.asarray(proxy["age_yng"][0, calibration_slice], dtype=float), CALIBRATION_YEARS):
                raise RuntimeError(f"proxy-calibration-index-contract-failed:{proxy_index}")
            proxy_raw[:, column] = np.asarray(proxy["data"][event_slice, 0], dtype=float)
            calibration[:, column] = np.asarray(proxy["data"][calibration_slice, 0], dtype=float)

    calibration_mean = np.nanmean(calibration, axis=0)
    calibration_std = np.nanstd(calibration, axis=0)
    if not np.all(np.isfinite(calibration_std) & (calibration_std > 0)):
        raise RuntimeError("proxy-calibration-scale-failed")
    proxy_z = (proxy_raw - calibration_mean[None, :]) / calibration_std[None, :]
    if not np.all(np.isfinite(proxy_z)):
        raise RuntimeError("proxy-event-window-missingness")
    if not np.allclose(period_change(proxy_z), table["event_drop_z"].to_numpy(), atol=1e-12, rtol=0):
        raise RuntimeError("proxy-event-table-closure-failed")

    variables = {
        "temperature_bea": "tas_obs_bea_05deg",
        "temperature_cru": "tas_obs_cru_tmp_05deg",
        "precipitation_gpcc": "pr_obs_gpcp_05deg",
        "precipitation_cru": "pr_obs_cru_pre_05deg",
    }
    local = {name: np.full((len(YEARS), len(table)), np.nan) for name in variables}
    with h5py.File(CLIMATE, "r") as handle:
        time = np.asarray(handle["time"], dtype=int)
        indices = np.flatnonzero((time >= YEARS[0]) & (time <= YEARS[-1]))
        if not np.array_equal(time[indices], YEARS):
            raise RuntimeError("climate-event-index-contract-failed")
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        if missing := [variable for variable in variables.values() if variable not in handle]:
            raise KeyError(f"missing-climate-variables:{missing}")
        fields = {name: np.asarray(handle[variable][indices], dtype=float) for name, variable in variables.items()}
        for values in fields.values():
            values[values <= FILL_VALUE_LIMIT] = np.nan
        selected_latitudes = np.empty(len(table))
        selected_longitudes = np.empty(len(table))
        for column, row in enumerate(table.itertuples(index=False)):
            lat_index = int(np.argmin(np.abs(latitudes - row.latitude)))
            lon_index = int(np.argmin(np.abs(longitudes - row.longitude)))
            selected_latitudes[column] = latitudes[lat_index]
            selected_longitudes[column] = longitudes[lon_index]
            for name in variables:
                local[name][:, column] = fields[name][:, lat_index, lon_index]
    if not all(np.all(np.isfinite(values)) for values in local.values()):
        raise RuntimeError("incomplete-local-instrumental-event-values")

    changes = {name: period_change(values) for name, values in local.items()}
    b1 = table["coefficient_b1"].to_numpy()
    b2 = table["coefficient_b2"].to_numpy()
    production_temperature_component_z = b1 * changes["temperature_bea"] / calibration_std
    production_precipitation_component_z = b2 * changes["precipitation_gpcc"] / calibration_std
    cru_temperature_component_z = b1 * changes["temperature_cru"] / calibration_std
    cru_precipitation_component_z = b2 * changes["precipitation_cru"] / calibration_std
    production_predicted_z = production_temperature_component_z + production_precipitation_component_z
    cru_predicted_z = cru_temperature_component_z + cru_precipitation_component_z
    actual_z = period_change(proxy_z)

    table["selected_latitude"] = selected_latitudes
    table["selected_longitude"] = selected_longitudes
    for name, values in changes.items():
        table[f"{name}_event_change"] = values
    table["production_temperature_component_event_change_z"] = production_temperature_component_z
    table["production_precipitation_component_event_change_z"] = production_precipitation_component_z
    table["cru_temperature_component_event_change_z"] = cru_temperature_component_z
    table["cru_precipitation_component_event_change_z"] = cru_precipitation_component_z
    table["production_psm_predicted_event_change_z"] = production_predicted_z
    table["cru_psm_predicted_event_change_z"] = cru_predicted_z
    table["precipitation_product_sign_class"] = sign_class(changes["precipitation_gpcc"], changes["precipitation_cru"])
    table["temperature_product_sign_class"] = sign_class(changes["temperature_bea"], changes["temperature_cru"])
    table["psm_pair_sign_class"] = sign_class(production_predicted_z, cru_predicted_z)
    table["actual_proxy_decline"] = actual_z < 0

    rng = np.random.default_rng(RNG_SEED)
    correlation_inputs = {
        "production_psm": production_predicted_z,
        "cru_psm": cru_predicted_z,
        "production_temperature_component": production_temperature_component_z,
        "production_precipitation_component": production_precipitation_component_z,
        "cru_temperature_component": cru_temperature_component_z,
        "cru_precipitation_component": cru_precipitation_component_z,
    }
    correlation_rows = []
    for name, values in correlation_inputs.items():
        value, pvalue = permutation_correlation(actual_z, values, rng)
        correlation_rows.append(
            {
                "predictor": name,
                "actual_proxy_event_change_correlation": value,
                "permutation_pvalue": pvalue,
                "sign_agreement_fraction": float(np.mean(np.sign(actual_z) == np.sign(values))),
            }
        )
    correlations = pd.DataFrame(correlation_rows)

    precipitation_class_counts = table["precipitation_product_sign_class"].value_counts().to_dict()
    psm_class_counts = table["psm_pair_sign_class"].value_counts().to_dict()
    actual_decline = table["actual_proxy_decline"].to_numpy(dtype=bool)
    both_precip_decline = table["precipitation_product_sign_class"].eq("both_decline").to_numpy()
    both_psm_decline = table["psm_pair_sign_class"].eq("both_decline").to_numpy()
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1",
        "analyzed_periods": [[1905, 1920], [1940, 2000]],
        "baseline_period": list(BASELINE),
        "event_period": list(EVENT),
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed": False,
        "proxy_count": len(table),
        "production_psm": "b1*BerkeleyEarth_temperature + b2*GPCC_derived_precipitation",
        "independent_cru_combination": "b1*CRU_temperature + b2*CRU_precipitation",
        "product_event_changes": {
            "mean_local_gpcc_precipitation_mm_day": float(np.mean(changes["precipitation_gpcc"])),
            "mean_local_cru_precipitation_mm_day": float(np.mean(changes["precipitation_cru"])),
            "gpcc_vs_cru_site_change_correlation": correlation(changes["precipitation_gpcc"], changes["precipitation_cru"]),
            "mean_local_bea_temperature_c": float(np.mean(changes["temperature_bea"])),
            "mean_local_cru_temperature_c": float(np.mean(changes["temperature_cru"])),
            "bea_vs_cru_site_change_correlation": correlation(changes["temperature_bea"], changes["temperature_cru"]),
        },
        "precipitation_product_sign_counts": {str(key): int(value) for key, value in precipitation_class_counts.items()},
        "psm_pair_sign_counts": {str(key): int(value) for key, value in psm_class_counts.items()},
        "actual_declining_proxy_count": int(actual_decline.sum()),
        "actual_declining_proxies_with_both_precip_products_declining_count": int(np.sum(actual_decline & both_precip_decline)),
        "actual_declining_proxies_with_both_precip_products_declining_fraction": float(np.sum(actual_decline & both_precip_decline) / np.sum(actual_decline)),
        "negative_enkf_effect_fraction_at_both_precip_decline_sites": fraction_of_negative_effect(table, both_precip_decline),
        "negative_enkf_effect_fraction_at_actual_and_both_precip_decline_sites": fraction_of_negative_effect(table, actual_decline & both_precip_decline),
        "actual_declining_proxies_with_both_psm_combinations_declining_count": int(np.sum(actual_decline & both_psm_decline)),
        "actual_declining_proxies_with_both_psm_combinations_declining_fraction": float(np.sum(actual_decline & both_psm_decline) / np.sum(actual_decline)),
        "negative_enkf_effect_fraction_at_both_psm_decline_sites": fraction_of_negative_effect(table, both_psm_decline),
        "correlation_tests": correlations.to_dict(orient="records"),
        "interpretation_limit": "CRU and GPCC-derived precipitation are uncertain in the early twentieth century and are treated as independent product evidence, not truth. Nearest-grid extraction is a site diagnostic and does not reproduce every production PSM spatial extraction detail.",
    }

    actual_network = np.mean(proxy_z, axis=1)
    production_annual_z = (
        b1[None, :] * local["temperature_bea"] + b2[None, :] * local["precipitation_gpcc"]
    ) / calibration_std[None, :]
    cru_annual_z = (
        b1[None, :] * local["temperature_cru"] + b2[None, :] * local["precipitation_cru"]
    ) / calibration_std[None, :]
    baseline_mask = (YEARS >= BASELINE[0]) & (YEARS <= BASELINE[1])
    production_annual_z -= np.mean(production_annual_z[baseline_mask], axis=0)[None, :]
    cru_annual_z -= np.mean(cru_annual_z[baseline_mask], axis=0)[None, :]
    actual_centered = proxy_z - np.mean(proxy_z[baseline_mask], axis=0)[None, :]
    annual = pd.DataFrame(
        {
            "year": YEARS,
            "actual_proxy_network_mean_z_from_baseline": np.mean(actual_centered, axis=1),
            "production_psm_network_mean_z_from_baseline": np.mean(production_annual_z, axis=1),
            "cru_psm_network_mean_z_from_baseline": np.mean(cru_annual_z, axis=1),
            "gpcc_local_precipitation_mean_mm_day_from_baseline": np.mean(local["precipitation_gpcc"] - np.mean(local["precipitation_gpcc"][baseline_mask], axis=0)[None, :], axis=1),
            "cru_local_precipitation_mean_mm_day_from_baseline": np.mean(local["precipitation_cru"] - np.mean(local["precipitation_cru"][baseline_mask], axis=0)[None, :], axis=1),
            "bea_local_temperature_mean_c_from_baseline": np.mean(local["temperature_bea"] - np.mean(local["temperature_bea"][baseline_mask], axis=0)[None, :], axis=1),
            "cru_local_temperature_mean_c_from_baseline": np.mean(local["temperature_cru"] - np.mean(local["temperature_cru"][baseline_mask], axis=0)[None, :], axis=1),
        }
    )

    output.mkdir(parents=True)
    table.to_csv(output / "site_instrumental_event_consistency.csv", index=False)
    correlations.to_csv(output / "proxy_climate_event_correlation_tests.csv", index=False)
    annual.to_csv(output / "site_network_event_curves_1905_1920.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    axes[0, 0].plot(YEARS, annual["actual_proxy_network_mean_z_from_baseline"], color="#1B1B1B", linewidth=2.2, label="Stored proxies")
    axes[0, 0].plot(YEARS, annual["production_psm_network_mean_z_from_baseline"], color="#0072B2", linewidth=1.8, label="BEA T + GPCC P")
    axes[0, 0].plot(YEARS, annual["cru_psm_network_mean_z_from_baseline"], color="#D55E00", linewidth=1.8, label="CRU T + CRU P")
    axes[0, 0].axhline(0, color="#999999", linewidth=0.8); axes[0, 0].axvspan(1915, 1918, color="#D9D9D9", alpha=0.45)
    axes[0, 0].set_ylabel("Network mean change (proxy calibration z)"); axes[0, 0].set_title("Observed proxy event versus local PSM predictions"); axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(YEARS, annual["gpcc_local_precipitation_mean_mm_day_from_baseline"], color="#0072B2", linewidth=1.9, label="GPCC-derived")
    axes[0, 1].plot(YEARS, annual["cru_local_precipitation_mean_mm_day_from_baseline"], color="#D55E00", linewidth=1.9, label="CRU")
    axes[0, 1].axhline(0, color="#999999", linewidth=0.8); axes[0, 1].axvspan(1915, 1918, color="#D9D9D9", alpha=0.45)
    axes[0, 1].set_ylabel("Local-site mean precipitation change (mm/day)"); axes[0, 1].set_title("Precipitation products at proxy grid cells"); axes[0, 1].legend(frameon=False)
    limit = max(0.25, float(np.max(np.abs(np.concatenate([actual_z, production_predicted_z, cru_predicted_z])))))
    axes[1, 0].scatter(production_predicted_z, actual_z, color="#0072B2", alpha=0.72, s=34, label="BEA T + GPCC P")
    axes[1, 0].scatter(cru_predicted_z, actual_z, facecolors="none", edgecolors="#D55E00", alpha=0.82, s=38, label="CRU T + CRU P")
    axes[1, 0].plot([-limit, limit], [-limit, limit], color="#666666", linewidth=0.9)
    axes[1, 0].axhline(0, color="#999999", linewidth=0.7); axes[1, 0].axvline(0, color="#999999", linewidth=0.7)
    axes[1, 0].set_xlim(-limit, limit); axes[1, 0].set_ylim(-limit, limit); axes[1, 0].set_xlabel("Instrumental PSM event prediction (z)"); axes[1, 0].set_ylabel("Stored proxy event change (z)"); axes[1, 0].set_title("Site-by-site event consistency"); axes[1, 0].legend(frameon=False)
    colors = {"both_decline": "#0072B2", "products_disagree": "#E69F00", "both_non_decline": "#999999"}
    for label, group in table.groupby("precipitation_product_sign_class"):
        axes[1, 1].scatter(group["longitude"], group["latitude"], color=colors[label], s=38 + 100 * group["negative_effect_magnitude"] / max(table["negative_effect_magnitude"].max(), 1e-12), alpha=0.8, edgecolor="#333333", linewidth=0.35, label=label.replace("_", " "))
    axes[1, 1].set_xlabel("Longitude (degrees east)"); axes[1, 1].set_ylabel("Latitude (degrees north)"); axes[1, 1].set_title("CRU-GPCC precipitation event-sign agreement"); axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "positive_b2_event_site_instrumental_consistency.png", dpi=180)
    plt.close(figure)

    prod_row = correlations.loc[correlations["predictor"] == "production_psm"].iloc[0]
    cru_row = correlations.loc[correlations["predictor"] == "cru_psm"].iloc[0]
    report = [
        "# Positive-b2 Event Site Instrumental Consistency",
        "",
        "The diagnostic reads 1905-1920 and 1940-2000 only. The locked 1921-1939 period was not accessed.",
        "",
        f"- Actual proxy declines: `{int(actual_decline.sum())}/60`.",
        f"- Both precipitation products decline at `{int(both_precip_decline.sum())}/60` proxy locations.",
        f"- Declining proxies with dual-product precipitation support: `{int(np.sum(actual_decline & both_precip_decline))}/{int(actual_decline.sum())}`.",
        f"- Those jointly supported proxies account for `{summary['negative_enkf_effect_fraction_at_actual_and_both_precip_decline_sites']:.1%}` of the total negative EnKF pathway effect.",
        f"- Actual versus production-PSM event correlation/sign agreement: `{prod_row['actual_proxy_event_change_correlation']:.3f}` / `{prod_row['sign_agreement_fraction']:.1%}`.",
        f"- Actual versus CRU-combination event correlation/sign agreement: `{cru_row['actual_proxy_event_change_correlation']:.3f}` / `{cru_row['sign_agreement_fraction']:.1%}`.",
        "",
        "CRU and GPCC-derived precipitation are treated as an uncertainty pair, not as ground truth. Agreement supports a climate interpretation; disagreement leaves the site unresolved.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    event_hash_after = digest(event_table_path)
    source_stats_after = {name: source_stat(path) for name, path in sources.items() if name != "event_table"}
    if event_hash_before != event_hash_after or source_stats_before != source_stats_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    manifest = {
        "event_table_sha256_before": event_hash_before,
        "event_table_sha256_after": event_hash_after,
        "source_stats_before": source_stats_before,
        "source_stats_after": source_stats_after,
        "climate_sha256_from_existing_psm_audit_contract": "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90",
        "analysis_driver_sha256": digest(Path(__file__)),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
