"""Attribute six-station temperature level and trend residuals to background/update."""

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


REQUESTED_YEARS = np.r_[
    np.arange(1880, 1921),
    np.arange(1940, 1951),
    np.arange(1961, 1991),
]
PERIODS = ((1880, 1900), (1901, 1920), (1940, 1950))


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def slope(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(years) & np.isfinite(values)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def sample_points(fields: np.ndarray, lat: np.ndarray, lon: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    """Bilinearly sample time-lat-lon fields at station coordinates."""
    result = np.full((fields.shape[0], len(stations)), np.nan, dtype=float)
    for station_index, row in enumerate(stations.itertuples(index=False)):
        fy = (float(row.latitude) - lat[0]) / (lat[1] - lat[0])
        fx = (float(row.longitude) - lon[0]) / (lon[1] - lon[0])
        if fy < 0 or fy > len(lat) - 1 or fx < 0 or fx > len(lon) - 1:
            continue
        y0 = min(int(np.floor(fy)), len(lat) - 2)
        x0 = min(int(np.floor(fx)), len(lon) - 2)
        wy = fy - y0
        wx = fx - x0
        result[:, station_index] = (
            (1.0 - wy) * ((1.0 - wx) * fields[:, y0, x0] + wx * fields[:, y0, x0 + 1])
            + wy * ((1.0 - wx) * fields[:, y0 + 1, x0] + wx * fields[:, y0 + 1, x0 + 1])
        )
    return result


def component_fields(flat: np.ndarray) -> np.ndarray:
    return round25.anomaly_field(flat, REQUESTED_YEARS)


def main() -> int:
    root = round25.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_six_station_prior_increment_level_attribution_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    round25_manifest_path = loop / "round_25_temperature_prior_update_decomposition_v1/source_manifest.json"
    old_manifest = json.loads(round25_manifest_path.read_text(encoding="utf-8"))
    old_sources = old_manifest["sources_before"]
    recovered_keys = [
        "prior", "indices", "proxy6", "proxy9", "archive_config",
        *[f"posterior_{member}" for member in round25.MEMBERS],
    ]
    recovered = {key: Path(old_sources[key]["path"]) for key in recovered_keys}
    paths = {
        **recovered,
        "facts": root / "LIA_PROJECT_FACTS.md",
        "round25_manifest": round25_manifest_path,
        "station_curves": loop / "temperature_station_annual_anomaly_benchmark_v4/station_annual_anomalies_and_products.csv",
        "station_selection": loop / "temperature_early_station_homogeneity_audit_v2/early_station_metrics.csv",
        "station_audit_summary": loop / "temperature_early_station_homogeneity_audit_v2/summary.json",
        "contract": DRIVER_DIR / "temperature_six_station_prior_increment_level_attribution_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: state(path) for name, path in paths.items()}

    audit = json.loads(paths["station_audit_summary"].read_text(encoding="utf-8"))
    if audit["selected_station_count"] != 6 or audit["locked_1921_1939_accessed"]:
        raise RuntimeError("station-selection-contract-failed")
    stations = pd.read_csv(paths["station_selection"])[
        ["station_id", "station_name", "latitude", "longitude", "evidence_grade"]
    ].copy()
    if len(stations) != 6 or stations.station_id.duplicated().any():
        raise RuntimeError("six-station-identity-failed")

    station_values = pd.read_csv(paths["station_curves"])
    station_values = station_values[
        station_values.station_id.isin(stations.station_id)
        & station_values.year.isin(REQUESTED_YEARS)
    ][["station_id", "year", "qcf_station_degC", "qcu_station_degC"]].copy()
    if station_values.station_id.nunique() != 6:
        raise RuntimeError("station-value-coverage-failed")
    if station_values.year.between(1921, 1939).any():
        raise RuntimeError("locked-station-year-present")

    config = scipy.io.loadmat(recovered["archive_config"], squeeze_me=True, struct_as_record=False)["config"]
    config_tuple = (
        int(config.num_experiments), str(config.aoenkf_strategy), float(config.alpha),
        int(config.r_o), int(config.r_f),
    )
    if config_tuple != (50, "C", 0.3, 1000, 2000):
        raise RuntimeError(f"archive-config-contract-failed:{config_tuple}")

    prior = np.load(recovered["prior"])
    with h5py.File(recovered["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or experiment_indices.shape != (50, 500):
        raise RuntimeError("prior-or-index-shape-failed")

    proxies: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(recovered[f"proxy{network}"], REQUESTED_YEARS)
        if locked_reads != 0:
            raise RuntimeError("locked-proxy-read")
        proxies[network] = (hxb, observations)
    adaptive = {
        network: round25.ao_background_mean(prior, experiment_indices, *proxies[network])[0]
        for network in (6, 9)
    }
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed = np.repeat(fixed_one[None, :], len(REQUESTED_YEARS), axis=0)

    lat, lon = round25.native_coordinates()
    component_samples: dict[str, dict[str, np.ndarray]] = {}
    closure_max = 0.0
    annual_rows: list[dict[str, object]] = []
    for member, (method, network) in round25.MEMBERS.items():
        posterior = component_fields(round25.read_posterior_mean(recovered[f"posterior_{member}"], REQUESTED_YEARS))
        background = component_fields(fixed if method == "O" else adaptive[network])
        increment = posterior - background
        sampled = {
            "posterior": sample_points(posterior, lat, lon, stations),
            "background": sample_points(background, lat, lon, stations),
            "increment": sample_points(increment, lat, lon, stations),
        }
        closure_max = max(
            closure_max,
            float(np.nanmax(np.abs(sampled["posterior"] - sampled["background"] - sampled["increment"])))
        )
        component_samples[member] = sampled
        for station_index, station in enumerate(stations.itertuples(index=False)):
            for year_index, year in enumerate(REQUESTED_YEARS):
                annual_rows.append({
                    "station_id": station.station_id,
                    "station_name": station.station_name,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "evidence_grade": station.evidence_grade,
                    "year": int(year),
                    "member": member,
                    "method": method,
                    "proxy_network": network,
                    "posterior_degC": sampled["posterior"][year_index, station_index],
                    "background_degC": sampled["background"][year_index, station_index],
                    "increment_degC": sampled["increment"][year_index, station_index],
                })

    annual = pd.DataFrame(annual_rows).merge(
        station_values, on=["station_id", "year"], how="left", validate="many_to_one"
    )
    annual["posterior_closure_degC"] = annual.posterior_degC - annual.background_degC - annual.increment_degC
    if annual.year.between(1921, 1939).any() or closure_max > 1e-12:
        raise RuntimeError(f"annual-contract-failed:{closure_max}")
    annual.to_csv(output / "station_member_annual_components.csv", index=False)

    ensemble = annual.groupby(
        ["station_id", "station_name", "latitude", "longitude", "evidence_grade", "year"],
        as_index=False,
    )[["posterior_degC", "background_degC", "increment_degC", "qcf_station_degC", "qcu_station_degC"]].mean()
    ensemble.to_csv(output / "station_ensemble_annual_components.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for (station_id, station_name), group in ensemble.groupby(["station_id", "station_name"], sort=False):
        for start, end in PERIODS:
            period = group[group.year.between(start, end)].copy()
            for target in ("qcf_station_degC", "qcu_station_degC"):
                selected = period[np.isfinite(period[target]) & np.isfinite(period.posterior_degC)]
                years = selected.year.to_numpy(float)
                observation = selected[target].to_numpy(float)
                posterior = selected.posterior_degC.to_numpy(float)
                background = selected.background_degC.to_numpy(float)
                increment = selected.increment_degC.to_numpy(float)
                posterior_residual = posterior - observation
                background_residual = background - observation
                metric_rows.append({
                    "station_id": station_id,
                    "station_name": station_name,
                    "period": f"{start}-{end}",
                    "target": target.replace("_station_degC", ""),
                    "matched_years": int(len(selected)),
                    "station_mean_degC": float(np.mean(observation)) if len(selected) else np.nan,
                    "posterior_mean_degC": float(np.mean(posterior)) if len(selected) else np.nan,
                    "background_mean_degC": float(np.mean(background)) if len(selected) else np.nan,
                    "increment_mean_degC": float(np.mean(increment)) if len(selected) else np.nan,
                    "posterior_minus_station_degC": float(np.mean(posterior_residual)) if len(selected) else np.nan,
                    "background_minus_station_degC": float(np.mean(background_residual)) if len(selected) else np.nan,
                    "level_closure_degC": float(np.mean(posterior_residual - background_residual - increment)) if len(selected) else np.nan,
                    "station_trend_degC_per_century": slope(years, observation),
                    "posterior_trend_degC_per_century": slope(years, posterior),
                    "background_trend_degC_per_century": slope(years, background),
                    "increment_trend_degC_per_century": slope(years, increment),
                    "trend_residual_degC_per_century": slope(years, posterior) - slope(years, observation),
                    "background_minus_station_trend_degC_per_century": slope(years, background) - slope(years, observation),
                    "trend_closure_degC_per_century": (
                        slope(years, posterior) - slope(years, observation)
                        - (slope(years, background) - slope(years, observation))
                        - slope(years, increment)
                    ),
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "station_period_level_trend_attribution.csv", index=False)

    summary_rows = []
    for (period, target), group in metrics.groupby(["period", "target"], sort=False):
        eligible = group[group.matched_years >= 5].copy()
        summary_rows.append({
            "period": period,
            "target": target,
            "eligible_station_count": int(len(eligible)),
            "median_posterior_minus_station_degC": float(eligible.posterior_minus_station_degC.median()) if len(eligible) else np.nan,
            "median_background_minus_station_degC": float(eligible.background_minus_station_degC.median()) if len(eligible) else np.nan,
            "median_increment_mean_degC": float(eligible.increment_mean_degC.median()) if len(eligible) else np.nan,
            "positive_posterior_residual_station_count": int((eligible.posterior_minus_station_degC > 0).sum()),
            "median_trend_residual_degC_per_century": float(eligible.trend_residual_degC_per_century.median()) if len(eligible) else np.nan,
            "median_background_minus_station_trend_degC_per_century": float(eligible.background_minus_station_trend_degC_per_century.median()) if len(eligible) else np.nan,
            "median_increment_trend_degC_per_century": float(eligible.increment_trend_degC_per_century.median()) if len(eligible) else np.nan,
        })
    period_summary = pd.DataFrame(summary_rows)
    period_summary.to_csv(output / "period_attribution_summary.csv", index=False)

    qcf_early = period_summary[(period_summary.period == "1901-1920") & (period_summary.target == "qcf")].iloc[0]
    level_terms = {
        "background_minus_station_degC": float(qcf_early.median_background_minus_station_degC),
        "increment_degC": float(qcf_early.median_increment_mean_degC),
    }
    level_dominant = max(level_terms, key=lambda key: abs(level_terms[key]))
    if qcf_early.eligible_station_count < 5:
        decision = "insufficient_station_support"
    elif abs(level_terms["increment_degC"]) > abs(level_terms["background_minus_station_degC"]):
        decision = "assimilation_increment_dominates_1901_1920_warm_level_residual"
    else:
        decision = "background_relative_to_stations_dominates_1901_1920_warm_level_residual"

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary = {
        "experiment": "temperature_six_station_prior_increment_level_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "selected_station_count": 6,
        "requested_years": ["1880-1920", "1940-1950", "1961-1990"],
        "locked_1921_1939_accessed": False,
        "component_closure_max_abs_degC": closure_max,
        "qcf_1901_1920": {key: (int(value) if key.endswith("count") else float(value)) for key, value in qcf_early.to_dict().items() if key not in ("period", "target")},
        "qcf_1901_1920_level_terms": level_terms,
        "dominant_absolute_level_term": level_dominant,
        "decision": decision,
        "interpretation_boundary": "Background-minus-station includes disagreement between the model prior anomaly and direct station anomaly; increment is the exact archived assimilation contribution. Strategy-C adaptive backgrounds are deterministic recoveries because selected yearly analog indices were not archived.",
        "source_states_unchanged": unchanged,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "unchanged": unchanged, "inherited_sha256": {key: old_sources[key].get("sha256") for key in recovered_keys}}, indent=2),
        encoding="utf-8",
    )

    colors = {"station": "#111111", "posterior": "#D1495B", "background": "#00798C", "increment": "#7A5195"}
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.0), sharex=True, constrained_layout=True)
    for axis, (station_id, group) in zip(axes.flat, ensemble.groupby("station_id", sort=False)):
        group = group.sort_values("year")
        allowed = group.year <= 1920
        axis.plot(group.loc[allowed, "year"], group.loc[allowed, "qcf_station_degC"], color=colors["station"], marker="o", ms=2.8, lw=1.5, label="GHCN qcf")
        for component in ("posterior", "background", "increment"):
            axis.plot(group.loc[allowed, "year"], group.loc[allowed, f"{component}_degC"], color=colors[component], lw=1.35, label=component)
        axis.axhline(0, color="#AAAAAA", lw=0.7)
        axis.set_title(group.station_name.iloc[0])
        axis.set_ylabel("Anomaly (degC)")
        axis.grid(axis="y", color="#E5E5E5", lw=0.5)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.savefig(output / "six_station_prior_increment_curves_1880_1920.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    qcf = period_summary[period_summary.target.eq("qcf")].set_index("period")
    positions = np.arange(len(PERIODS))
    labels = [f"{start}-{end}" for start, end in PERIODS]
    axes[0].bar(positions, qcf.loc[labels, "median_background_minus_station_degC"], color=colors["background"], label="background - station")
    axes[0].bar(positions, qcf.loc[labels, "median_increment_mean_degC"], bottom=qcf.loc[labels, "median_background_minus_station_degC"], color=colors["increment"], label="increment")
    axes[0].plot(positions, qcf.loc[labels, "median_posterior_minus_station_degC"], color=colors["station"], marker="o", label="posterior - station")
    axes[0].set(xticks=positions, xticklabels=labels, ylabel="Median level term (degC)", title="QCF level residual decomposition")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(positions, qcf.loc[labels, "median_background_minus_station_trend_degC_per_century"], color=colors["background"], label="background - station")
    axes[1].bar(positions, qcf.loc[labels, "median_increment_trend_degC_per_century"], bottom=qcf.loc[labels, "median_background_minus_station_trend_degC_per_century"], color=colors["increment"], label="increment")
    axes[1].plot(positions, qcf.loc[labels, "median_trend_residual_degC_per_century"], color=colors["station"], marker="o", label="posterior - station")
    axes[1].set(xticks=positions, xticklabels=labels, ylabel="Median trend term (degC/century)", title="QCF trend residual decomposition")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.axhline(0, color="#777777", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "six_station_level_trend_attribution.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Six-station prior/increment level attribution\n\n"
        f"- Exact posterior closure: `{closure_max:.3e} degC`.\n"
        f"- 1901-1920 qcf eligible stations: `{int(qcf_early.eligible_station_count)}`.\n"
        f"- Median posterior-minus-station residual: `{qcf_early.median_posterior_minus_station_degC:+.3f} degC`.\n"
        f"- Median background-minus-station term: `{qcf_early.median_background_minus_station_degC:+.3f} degC`.\n"
        f"- Median assimilation-increment term: `{qcf_early.median_increment_mean_degC:+.3f} degC`.\n"
        f"- Dominant absolute level term: `{level_dominant}`.\n\n"
        f"Decision: `{decision}`.\n\n"
        "This is a mechanism attribution, not authorization to edit the production reconstruction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
