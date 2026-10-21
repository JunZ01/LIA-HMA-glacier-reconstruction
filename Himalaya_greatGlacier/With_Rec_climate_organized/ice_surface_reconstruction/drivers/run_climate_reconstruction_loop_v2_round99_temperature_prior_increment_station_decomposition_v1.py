"""Decompose 1901-1920 western temperature into background and increment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1 as round97

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


EXPECTED_FACTS_SHA256 = "d523009867d5481aed48e5c3f622eb2727848da5cfa9ca2be194d63cc78efce3"
REQUESTED_YEARS = np.r_[np.arange(1901, 1921), np.arange(1961, 1991)]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def station_curve(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, annual: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    sample = round97.sample_station_cells(field[:20], lat, lon, annual, name)
    curve = sample.groupby("year")[name].mean().reindex(round97.YEARS).to_numpy(float)
    return sample, round97.centered(curve)


def main() -> int:
    root = round97.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_99_temperature_prior_increment_station_decomposition_v1"
    round25_manifest_path = loop / "round_25_temperature_prior_update_decomposition_v1/source_manifest.json"
    round98_curves_path = loop / "round_98_project_temperature_member_attribution_v1/member_and_composite_curves.csv"
    station_path = loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1/station_annual_anomalies_and_products.csv"
    facts = root / "LIA_PROJECT_FACTS.md"
    reference_grid = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    contract = output / "ROUND_CONTRACT.md"
    old_manifest = json.loads(round25_manifest_path.read_text(encoding="utf-8"))
    old_sources = old_manifest["sources_before"]
    keys = ["prior", "indices", "proxy6", "proxy9", "archive_config", "recovered_algorithm", *[f"posterior_{name}" for name in round25.MEMBERS]]
    recovered = {key: Path(old_sources[key]["path"]) for key in keys}
    paths = {"facts": facts, "round25_manifest": round25_manifest_path, "round98_curves": round98_curves_path, "station_annual": station_path, "reference_grid": reference_grid, "contract": contract, "driver": Path(__file__).resolve(), **recovered}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(facts) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(facts)}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for key in keys:
        if before[key]["sha256"] != old_sources[key]["sha256"]:
            raise RuntimeError(f"round25-source-hash-mismatch:{key}")

    config = scipy.io.loadmat(recovered["archive_config"], squeeze_me=True, struct_as_record=False)["config"]
    if (int(config.num_experiments), str(config.aoenkf_strategy), float(config.alpha), int(config.r_o), int(config.r_f)) != (50, "C", 0.3, 1000, 2000):
        raise RuntimeError("archive-config-contract-failed")
    prior = np.load(recovered["prior"])
    with h5py.File(recovered["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or experiment_indices.shape != (50, 500):
        raise RuntimeError("prior-or-index-shape-failed")

    proxies = {}
    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(recovered[f"proxy{network}"], REQUESTED_YEARS)
        if locked_reads != 0:
            raise RuntimeError("locked-proxy-read")
        proxies[network] = (hxb, observations)
    adaptive = {network: round25.ao_background_mean(prior, experiment_indices, *proxies[network])[0] for network in (6, 9)}
    fixed = np.mean(np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0)
    fixed = np.repeat(fixed[None], len(REQUESTED_YEARS), axis=0)

    native_lat, native_lon = round25.native_coordinates()
    with netcdf_file(reference_grid, "r", mmap=False) as dataset:
        target_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        target_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    annual = pd.read_csv(station_path)
    export = pd.read_csv(round98_curves_path)
    curves = pd.DataFrame({"year": round97.YEARS, "station_network_degC": export.station_network_degC})
    component_fields: dict[str, dict[str, np.ndarray]] = {}
    sample_rows = []
    metric_rows = []

    for member, (method, network) in round25.MEMBERS.items():
        posterior_native = round25.anomaly_field(round25.read_posterior_mean(recovered[f"posterior_{member}"], REQUESTED_YEARS), REQUESTED_YEARS)
        background_flat = fixed if method == "O" else adaptive[network]
        background_native = round25.anomaly_field(background_flat, REQUESTED_YEARS)
        increment_native = posterior_native - background_native
        component_fields[member] = {}
        for component, native in {"posterior": posterior_native, "background": background_native, "increment": increment_native}.items():
            projected = round25.interpolate_fields(native, native_lat, native_lon, target_lat, target_lon)
            component_fields[member][component] = projected
            sampled, curve = station_curve(projected, target_lat, target_lon, annual, f"{member}_{component}_degC")
            sampled["member"] = member
            sampled["component"] = component
            sampled = sampled.rename(columns={f"{member}_{component}_degC": "anomaly_degC"})
            sample_rows.append(sampled)
            curves[f"{member}_{component}_degC"] = curve
        closure = curves[f"{member}_posterior_degC"] - curves[f"{member}_background_degC"] - curves[f"{member}_increment_degC"]
        recovered_curve = curves[f"{member}_posterior_degC"].to_numpy(float)
        export_curve = export[f"member_{member}_degC"].to_numpy(float)
        metric_rows.append({
            "member": member,
            "method": method,
            "proxy_network": network,
            "closure_max_abs_degC": float(np.max(np.abs(closure))),
            "export_reproduction_correlation": round97.correlation(recovered_curve, export_curve),
            "export_reproduction_rmse_degC": float(np.sqrt(np.mean((recovered_curve - export_curve) ** 2))),
            "posterior_trend_degC_per_century": round97.slope_century(round97.YEARS, recovered_curve),
            "background_trend_degC_per_century": round97.slope_century(round97.YEARS, curves[f"{member}_background_degC"].to_numpy(float)),
            "increment_trend_degC_per_century": round97.slope_century(round97.YEARS, curves[f"{member}_increment_degC"].to_numpy(float)),
        })

    for component in ("posterior", "background", "increment"):
        curves[f"ensemble_{component}_degC"] = curves[[f"{member}_{component}_degC" for member in round25.MEMBERS]].mean(axis=1)
    curves.to_csv(output / "annual_station_prior_increment_curves.csv", index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(output / "station_component_anomalies.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "member_prior_increment_metrics.csv", index=False)

    closure_max = float(metrics.closure_max_abs_degC.max())
    median_reproduction = float(metrics.export_reproduction_correlation.median())
    ensemble_trends = {component: round97.slope_century(round97.YEARS, curves[f"ensemble_{component}_degC"].to_numpy(float)) for component in ("posterior", "background", "increment")}
    ordinary = metrics[metrics.method.eq("O")]
    observed_trend = round97.slope_century(round97.YEARS, curves.station_network_degC.to_numpy(float))
    ordinary_increment_wrong = bool((np.sign(ordinary.increment_trend_degC_per_century) != np.sign(observed_trend)).all())
    update_fraction = float(abs(ensemble_trends["increment"]) / abs(ensemble_trends["posterior"])) if ensemble_trends["posterior"] != 0 else np.nan
    reproduction_pass = bool(closure_max <= 1e-10 and median_reproduction >= 0.90)
    update_dominant = bool(reproduction_pass and ordinary_increment_wrong and np.sign(ensemble_trends["increment"]) == np.sign(ensemble_trends["posterior"]) and update_fraction >= 0.80)
    if not reproduction_pass:
        decision = "halt_native_to_export_reproduction_failed"
    elif update_dominant:
        decision = "proxy_assimilation_increment_dominant_route_to_innovation_gain_attribution"
    elif abs(ensemble_trends["background"]) >= 0.5 * abs(ensemble_trends["posterior"]):
        decision = "adaptive_background_material_route_to_analog_selection"
    else:
        decision = "mixed_background_increment_route"
    event = curves[curves.year.eq(1917)].iloc[0]
    summary = {
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "component_closure_max_abs_degC": closure_max,
        "median_native_export_reproduction_correlation": median_reproduction,
        "minimum_native_export_reproduction_correlation": float(metrics.export_reproduction_correlation.min()),
        "reproduction_passes": reproduction_pass,
        "observed_trend_degC_per_century": observed_trend,
        "ensemble_trends_degC_per_century": ensemble_trends,
        "increment_absolute_trend_fraction": update_fraction,
        "both_ordinary_members_increment_wrong_sign": ordinary_increment_wrong,
        "year_1917_ensemble_decomposition_degC": {component: float(event[f"ensemble_{component}_degC"]) for component in ("posterior", "background", "increment")},
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(curves.year, curves.station_network_degC, color="#111111", marker="o", linewidth=2.3, label="GHCN-M stations")
    axes[0].plot(curves.year, curves.ensemble_posterior_degC, color="#D55E00", marker="s", linewidth=2.0, label="Recovered posterior")
    axes[0].plot(curves.year, curves.ensemble_background_degC, color="#7B2CBF", marker="^", linewidth=1.7, label="Background")
    axes[0].plot(curves.year, curves.ensemble_increment_degC, color="#0072B2", marker="v", linewidth=1.7, label="Increment")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(ylabel="Centered anomaly (degC)", title="Western station-support prior/update decomposition")
    axes[0].legend(frameon=False, ncol=4)
    for member in round25.MEMBERS:
        axes[1].plot(curves.year, curves[f"{member}_increment_degC"], linewidth=1.2, label=member)
    axes[1].plot(curves.year, curves.station_network_degC, color="#111111", linewidth=2.3, marker="o", label="Stations")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Year", ylabel="Centered anomaly (degC)", title="Member assimilation increments", xticks=round97.YEARS[::2])
    axes[1].legend(frameon=False, ncol=4)
    fig.savefig(output / "temperature_prior_increment_station_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Early Temperature Prior/Increment Decomposition",
            "",
            f"- Native/export median correlation: {median_reproduction:.3f}; closure: {closure_max:.3e} degC.",
            f"- Observed trend: {observed_trend:+.3f} degC/century.",
            f"- Posterior/background/increment trends: {ensemble_trends['posterior']:+.3f} / {ensemble_trends['background']:+.3f} / {ensemble_trends['increment']:+.3f} degC/century.",
            f"- Increment absolute trend fraction: {update_fraction:.1%}; both ordinary-member increment trends have the wrong sign: {str(ordinary_increment_wrong).lower()}.",
            f"- 1917 posterior/background/increment: {summary['year_1917_ensemble_decomposition_degC']['posterior']:+.3f} / {summary['year_1917_ensemble_decomposition_degC']['background']:+.3f} / {summary['year_1917_ensemble_decomposition_degC']['increment']:+.3f} degC.",
            "",
            f"Decision: {decision}.",
        ]) + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"local_sources_before": before, "local_sources_after": after, "local_source_states_unchanged": before == after, "parsed_years": ["1901-1920", "1961-1990"], "locked_1921_1939_accessed": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
