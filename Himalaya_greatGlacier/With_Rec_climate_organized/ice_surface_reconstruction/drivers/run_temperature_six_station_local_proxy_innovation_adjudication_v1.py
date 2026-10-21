"""Compare local temperature-oriented proxy innovations with six direct stations."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round28_temperature_proxy_innovation_concentration_v1 as round28
import run_temperature_six_station_oenkf_innovation_gain_attribution_v1 as attribution


RADII_KM = (100, 200, 300)


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi1 - phi2
    dlambda = np.deg2rad(lon1 - lon2)
    value = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def slope(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    root = round25.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_six_station_local_proxy_innovation_adjudication_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    manifest = json.loads((loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json").read_text(encoding="utf-8"))["sources_before"]
    recovered = {key: Path(manifest[key]["path"]) for key in ("prior", "indices", "proxy6", "proxy9", "regem6", "regem9")}
    paths = {
        **recovered,
        "station_selection": loop / "temperature_early_station_homogeneity_audit_v2/early_station_metrics.csv",
        "member_annual": loop / "temperature_six_station_prior_increment_level_attribution_v1/station_member_annual_components.csv",
        "orientation_summary": loop / "temperature_six_station_psm_covariance_orientation_audit_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    if json.loads(paths["orientation_summary"].read_text(encoding="utf-8"))["locked_1921_1939_accessed"]:
        raise RuntimeError("upstream-lock-boundary-failed")

    stations = pd.read_csv(paths["station_selection"])[["station_id", "station_name", "latitude", "longitude"]]
    station_values = pd.read_csv(paths["member_annual"])
    station_values = station_values[station_values.member.eq("O_6") & station_values.year.isin(attribution.YEARS)][["station_id", "year", "qcf_station_degC"]]
    if len(stations) != 6 or station_values.year.between(1921, 1939).any():
        raise RuntimeError("station-lock-contract-failed")

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    target_prior = attribution.sample_prior_points(prior, stations)
    annual_rows = []
    metric_rows = []
    network_rows = []
    max_closure = 0.0
    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(paths[f"proxy{network}"], attribution.YEARS)
        if locked_reads:
            raise RuntimeError("locked-proxy-read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        result = attribution.run_all_pools(target_prior, hxb, observations, np.nanvar(residuals, axis=1, ddof=1), pools)
        max_closure = max(max_closure, float(result["closure"]))
        innovations = np.asarray(result["innovations"])
        meta = round28.metadata(paths[f"proxy{network}"])
        baseline_mean = np.full(len(meta), np.nan)
        baseline_std = np.full(len(meta), np.nan)
        for index in range(len(meta)):
            values = innovations[index, attribution.BASELINE]
            finite = values[np.isfinite(values)]
            if len(finite) >= 10 and np.std(finite, ddof=1) > 0:
                baseline_mean[index] = np.mean(finite)
                baseline_std[index] = np.std(finite, ddof=1)
        oriented = np.sign(meta.b1_temperature.to_numpy(float))[:, None] * (innovations - baseline_mean[:, None]) / baseline_std[:, None]
        meta["location_group_id"] = meta.latitude.round(4).astype(str) + "_" + meta.longitude.round(4).astype(str)
        valid_proxy = np.isfinite(baseline_std) & np.isfinite(meta.b1_temperature) & (meta.b1_temperature != 0)

        for station in stations.itertuples(index=False):
            distances = haversine(meta.latitude.to_numpy(float), meta.longitude.to_numpy(float), station.latitude, station.longitude)
            station_curve = station_values[station_values.station_id.eq(station.station_id)].set_index("year").qcf_station_degC
            for radius in RADII_KM:
                selected_records = np.flatnonzero(valid_proxy & (distances <= radius))
                group_ids = meta.loc[selected_records, "location_group_id"].unique()
                proxy_curve = np.full(len(attribution.YEARS), np.nan)
                if len(group_ids):
                    group_curves = []
                    for group_id in group_ids:
                        records = selected_records[meta.loc[selected_records, "location_group_id"].to_numpy() == group_id]
                        group_curves.append(np.nanmean(oriented[records], axis=0))
                    proxy_curve = np.nanmean(np.stack(group_curves), axis=0)
                station_curve_array = np.asarray([station_curve.get(int(year), np.nan) for year in attribution.YEARS])
                for index, year in enumerate(attribution.YEARS):
                    annual_rows.append({
                        "network": network,
                        "station_id": station.station_id,
                        "station_name": station.station_name,
                        "radius_km": radius,
                        "year": int(year),
                        "qcf_station_degC": station_curve_array[index],
                        "local_temperature_oriented_proxy_innovation_z": proxy_curve[index],
                        "proxy_record_count": int(len(selected_records)),
                        "proxy_location_group_count": int(len(group_ids)),
                    })
                historical = attribution.HISTORICAL
                valid = historical & np.isfinite(station_curve_array) & np.isfinite(proxy_curve)
                metric_rows.append({
                    "network": network,
                    "station_id": station.station_id,
                    "station_name": station.station_name,
                    "radius_km": radius,
                    "proxy_record_count": int(len(selected_records)),
                    "proxy_location_group_count": int(len(group_ids)),
                    "matched_years": int(valid.sum()),
                    "station_mean_degC": float(np.mean(station_curve_array[valid])) if valid.any() else np.nan,
                    "proxy_innovation_mean_z": float(np.mean(proxy_curve[valid])) if valid.any() else np.nan,
                    "mean_sign_agreement": bool(np.sign(np.mean(station_curve_array[valid])) == np.sign(np.mean(proxy_curve[valid]))) if valid.any() else False,
                    "annual_correlation": correlation(station_curve_array[valid], proxy_curve[valid]) if valid.any() else np.nan,
                    "station_trend_degC_per_century": slope(attribution.YEARS[valid], station_curve_array[valid]) if valid.any() else np.nan,
                    "proxy_innovation_trend_z_per_century": slope(attribution.YEARS[valid], proxy_curve[valid]) if valid.any() else np.nan,
                })

        network_rows.append({
            "network": network,
            "proxy_count": int(len(meta)),
            "proxy_with_valid_baseline_standardization_count": int(valid_proxy.sum()),
            "contribution_closure_max_abs_degC": float(result["closure"]),
        })

    annual = pd.DataFrame(annual_rows)
    metrics = pd.DataFrame(metric_rows)
    annual.to_csv(output / "station_local_proxy_innovation_annual.csv", index=False)
    metrics.to_csv(output / "station_local_proxy_innovation_metrics.csv", index=False)
    radius_rows = []
    for (network, radius), group in metrics.groupby(["network", "radius_km"]):
        eligible = group[(group.matched_years >= 5) & (group.proxy_location_group_count >= 1)]
        cold_station = eligible.station_mean_degC < 0
        warm_proxy = eligible.proxy_innovation_mean_z > 0
        radius_rows.append({
            "network": int(network),
            "radius_km": int(radius),
            "eligible_station_count": int(len(eligible)),
            "mean_sign_agreement_station_count": int(eligible.mean_sign_agreement.sum()),
            "cold_station_warm_proxy_conflict_count": int((cold_station & warm_proxy).sum()),
            "median_annual_correlation": float(eligible.annual_correlation.median()) if len(eligible) else np.nan,
            "median_station_mean_degC": float(eligible.station_mean_degC.median()) if len(eligible) else np.nan,
            "median_proxy_innovation_mean_z": float(eligible.proxy_innovation_mean_z.median()) if len(eligible) else np.nan,
            "median_proxy_location_group_count": float(eligible.proxy_location_group_count.median()) if len(eligible) else np.nan,
        })
    radius_table = pd.DataFrame(radius_rows)
    radius_table.to_csv(output / "radius_adjudication_summary.csv", index=False)
    primary = radius_table[radius_table.radius_km.eq(200)]
    adequate = bool((primary.eligible_station_count >= 5).all())
    conflict = bool(adequate and (primary.cold_station_warm_proxy_conflict_count >= 4).all())
    sign_agreement = bool(adequate and (primary.mean_sign_agreement_station_count >= 4).all())
    if conflict:
        decision = "local_proxy_innovations_conflict_with_direct_station_level"
    elif sign_agreement:
        decision = "local_proxy_innovations_support_direct_station_level"
    else:
        decision = "local_proxy_station_adjudication_mixed_or_insufficient"

    after = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary = {
        "experiment": "temperature_six_station_local_proxy_innovation_adjudication_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1901, 1920],
        "baseline": [1961, 1990],
        "radii_km": list(RADII_KM),
        "primary_radius_km": 200,
        "locked_1921_1939_accessed": False,
        "maximum_algebraic_closure_degC": max_closure,
        "network_preflight": network_rows,
        "primary_radius_metrics": primary.to_dict(orient="records"),
        "decision": decision,
        "interpretation_boundary": "Temperature-oriented innovation uses sign(b1) and each proxy's 1961-1990 innovation standard deviation. It is directional, not a temperature-magnitude reconstruction. Equal coordinate groups prevent repeated chronologies at one site from receiving duplicate spatial weight.",
        "source_states_unchanged": unchanged,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "unchanged": unchanged, "inherited_sha256": {key: manifest[key]["sha256"] for key in recovered}}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.8), sharex=True, constrained_layout=True)
    selected = annual[(annual.network == 9) & (annual.radius_km == 200) & (annual.year <= 1920)]
    for axis, (station_id, values) in zip(axes.flat, selected.groupby("station_id", sort=False)):
        axis.plot(values.year, values.qcf_station_degC, color="#111111", marker="o", ms=2.8, lw=1.4, label="GHCN qcf (degC)")
        axis2 = axis.twinx()
        axis2.plot(values.year, values.local_temperature_oriented_proxy_innovation_z, color="#B23A48", lw=1.4, label="local proxy innovation (z)")
        axis.axhline(0, color="#999999", lw=0.6)
        axis2.axhline(0, color="#D0A0A5", lw=0.5)
        axis.set_title(values.station_name.iloc[0])
        axis.set_ylabel("Station degC")
        axis2.set_ylabel("Proxy innovation z")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0, 0].twinx().set_yticks([])
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.savefig(output / "local_proxy_innovation_vs_station_curves_200km_proxy9.png", dpi=220)
    plt.close(fig)

    lines = ["# Local proxy innovation versus direct stations", ""]
    for row in primary.itertuples(index=False):
        lines.append(f"- proxy{row.network}, 200 km: `{row.eligible_station_count}` eligible stations, `{row.mean_sign_agreement_station_count}` mean-sign agreements, `{row.cold_station_warm_proxy_conflict_count}` cold-station/warm-proxy conflicts, median annual r `{row.median_annual_correlation:+.3f}`.")
    lines += ["", f"Decision: `{decision}`.", ""]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
