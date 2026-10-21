"""Build corrected retained-32 mean curves with interval-area MB weighting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import plot_lia_191_retained_32_mean_geometry_mass_balance_v1 as base


OUTPUT = base.RESULTS / "lia_191_retained_32_mean_geometry_mass_balance_v2"


def main() -> None:
    retained_ids = set(pd.read_csv(base.RETAINED)["system_id"].astype(int))
    metrics = {int(row["system_id"]): row for row in base.read_csv(base.REGISTRY)}
    evidence = {int(row["system_id"]): row for row in base.read_csv(base.INDEX)}
    if len(retained_ids) != 32:
        raise RuntimeError("retained list is not exactly 32 unique systems")

    records: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for system_id in sorted(retained_ids):
        metric = metrics[system_id]
        source, length_key = base.resolve_source(metric, evidence)
        arrays = base.interpolate_source(source, length_key)
        area_m2 = arrays["area_km2"] * 1.0e6
        volume_m3 = arrays["volume_km3"] * 1.0e9
        reference_area_m2 = 0.5 * (area_m2[1:] + area_m2[:-1])
        delta_volume_m3 = volume_m3[1:] - volume_m3[:-1]
        net_mb = base.ICE_DENSITY_KG_M3 * delta_volume_m3 / reference_area_m2

        source_relative = str(source.relative_to(base.ROOT))
        manifest.append(
            {
                "system_id": system_id,
                "rgi_id": metric["rgi_id"],
                "route": metric["route"],
                "source": source_relative,
                "direct_surface_mass_balance_available": False,
            }
        )
        for offset, year in enumerate(base.OUTPUT_YEARS, start=1):
            records.append(
                {
                    "system_id": system_id,
                    "rgi_id": metric["rgi_id"],
                    "route": metric["route"],
                    "year": int(year),
                    "length_km": arrays["length_km"][offset],
                    "area_km2": arrays["area_km2"][offset],
                    "volume_km3": arrays["volume_km3"][offset],
                    "mass_balance_reference_area_km2": reference_area_m2[offset - 1] / 1.0e6,
                    "net_volume_change_equivalent_mb_kg_m2_yr": net_mb[offset - 1],
                    "source": source_relative,
                }
            )

    long_table = pd.DataFrame(records)
    numeric = long_table[
        [
            "length_km",
            "area_km2",
            "volume_km3",
            "mass_balance_reference_area_km2",
            "net_volume_change_equivalent_mb_kg_m2_yr",
        ]
    ].to_numpy()
    if len(long_table) != 32 * len(base.OUTPUT_YEARS) or not np.isfinite(numeric).all():
        raise RuntimeError("incomplete or non-finite retained-32 annual table")

    year_1500 = long_table[long_table["year"] == 1500].set_index("system_id")
    for key in ["length_km", "area_km2", "volume_km3"]:
        denominator = long_table["system_id"].map(year_1500[key])
        long_table[f"{key}_relative_1500"] = long_table[key] / denominator

    annual: list[dict[str, float | int]] = []
    variable_names = {
        "length_km": "length",
        "area_km2": "area",
        "volume_km3": "volume",
    }
    for year, group in long_table.groupby("year", sort=True):
        row: dict[str, float | int] = {"year": int(year), "system_count": len(group)}
        for key, short_name in variable_names.items():
            row.update(base.ensemble_statistics(group[key].to_numpy(float), key))
            row[f"mean_{short_name}_relative_1500"] = float(
                group[f"{key}_relative_1500"].mean()
            )
        mb = group["net_volume_change_equivalent_mb_kg_m2_yr"].to_numpy(float)
        row["mean_net_mb_kg_m2_yr"] = float(np.mean(mb))
        row["median_net_mb_kg_m2_yr"] = float(np.median(mb))
        row["q25_net_mb_kg_m2_yr"] = float(np.quantile(mb, 0.25))
        row["q75_net_mb_kg_m2_yr"] = float(np.quantile(mb, 0.75))
        weights = group["mass_balance_reference_area_km2"].to_numpy(float)
        row["area_weighted_net_mb_kg_m2_yr"] = float(np.average(mb, weights=weights))
        annual.append(row)
    stats = pd.DataFrame(annual)
    stats["mean_net_mb_21yr_kg_m2_yr"] = stats["mean_net_mb_kg_m2_yr"].rolling(
        base.MB_SMOOTHING_YEARS, center=True, min_periods=1
    ).mean()
    stats["area_weighted_net_mb_21yr_kg_m2_yr"] = stats[
        "area_weighted_net_mb_kg_m2_yr"
    ].rolling(base.MB_SMOOTHING_YEARS, center=True, min_periods=1).mean()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT = OUTPUT
    long_path = OUTPUT / "retained_32_annual_geometry_and_net_mass_balance.csv"
    stats_path = OUTPUT / "retained_32_annual_ensemble_statistics.csv"
    manifest_path = OUTPUT / "source_manifest.csv"
    long_table.to_csv(long_path, index=False)
    stats.to_csv(stats_path, index=False)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    absolute_plot = base.plot_absolute(stats)
    normalized_plot = base.plot_normalized(stats)

    start, end = stats.iloc[0], stats.iloc[-1]
    summary = {
        "status": "retained-32-mean-curves-complete-corrected-interval-area-weighting",
        "system_count": 32,
        "year_start": 1500,
        "year_end": 2000,
        "ice_density_kg_m3": base.ICE_DENSITY_KG_M3,
        "mass_balance_definition": (
            "annual net volume-change-equivalent specific mass balance: "
            "rho_ice * (V_t - V_t-1) / mean(A_t, A_t-1)"
        ),
        "ensemble_means": {
            "geometry": "equal glacier weighting",
            "net_mass_balance": "both equal glacier and reference-area weighting are reported",
        },
        "direct_climatic_surface_mass_balance_available": False,
        "mass_balance_warning": (
            "Net/geodetic-equivalent balance derived from annual geometry; "
            "not a directly saved climatic surface mass-balance diagnostic."
        ),
        "mean_1500": {
            "length_km": float(start["mean_length_km"]),
            "area_km2": float(start["mean_area_km2"]),
            "volume_km3": float(start["mean_volume_km3"]),
        },
        "mean_2000": {
            "length_km": float(end["mean_length_km"]),
            "area_km2": float(end["mean_area_km2"]),
            "volume_km3": float(end["mean_volume_km3"]),
        },
        "equal_glacier_mean_relative_change_1500_2000_percent": {
            "length": float(100.0 * (end["mean_length_relative_1500"] - 1.0)),
            "area": float(100.0 * (end["mean_area_relative_1500"] - 1.0)),
            "volume": float(100.0 * (end["mean_volume_relative_1500"] - 1.0)),
        },
        "mean_net_mb_1500_2000_kg_m2_yr": float(stats["mean_net_mb_kg_m2_yr"].mean()),
        "area_weighted_net_mb_1500_2000_kg_m2_yr": float(
            stats["area_weighted_net_mb_kg_m2_yr"].mean()
        ),
        "formal_accepted_count_unchanged": 58,
        "registry_or_controller_modified": False,
        "outputs": {
            "absolute_plot": absolute_plot.name,
            "normalized_plot": normalized_plot.name,
            "annual_ensemble_statistics": stats_path.name,
            "annual_glacier_table": long_path.name,
            "source_manifest": manifest_path.name,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
