"""Test whether central proxies control the 1500-1879 warm background level."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import decompose_temperature_lia_regional_prior_update_v1 as prior_tools
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as base


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
PARENT = RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1"
OUTPUT = RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1500_1879_v1"
STATE_FILE = PARENT / "four_prior_state_pools_1500_1848_v1.npz"
HISTORICAL = np.arange(1500, 1880, dtype=int)
BASELINE = np.arange(1961, 1991, dtype=int)
REQUEST_YEARS = np.r_[HISTORICAL, BASELINE]
POOL_YEARS = np.arange(1500, 1849, dtype=int)
MODELS = base.MODELS
REGIONS = ("hma", "west_lt80", "central_80_90", "east_ge90")
BRANCHES = ("full", "central_only", "omit_central", "full_complete_operator_only")
WINDOWS = [(start, start + 49) for start in range(1500, 1850, 50)] + [(1850, 1879)]


base.TARGET_YEARS = REQUEST_YEARS
base.POOL_YEARS = POOL_YEARS


def file_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def regional_weights(states: Any) -> dict[str, np.ndarray]:
    native_lat = np.asarray(states["native_lat"], dtype=float)
    native_lon = np.asarray(states["native_lon"], dtype=float)
    with h5py.File(prior_tools.PUBLIC_REFERENCE_SOURCE, "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(prior_tools.HMA_MASK).astype(float)
    mask = round25.interpolate_fields(mask05[None], source_lat, source_lon, native_lat, native_lon)[0] >= 0.5
    lon_grid = np.broadcast_to(native_lon[None, :], mask.shape)
    masks = {
        "hma": mask,
        "west_lt80": mask & (lon_grid < 80.0),
        "central_80_90": mask & (lon_grid >= 80.0) & (lon_grid < 90.0),
        "east_ge90": mask & (lon_grid >= 90.0),
    }
    output = {}
    for name, region in masks.items():
        if not region.any():
            raise RuntimeError(f"empty-region:{name}")
        output[name] = (np.cos(np.deg2rad(native_lat))[:, None] * region).reshape(-1, order="F")
    return output


def state_series(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = state - np.mean(state, axis=0, keepdims=True)
    valid = weights > 0
    return np.average(centered[:, valid], axis=1, weights=weights[valid])


def anomaly(values: np.ndarray) -> np.ndarray:
    baseline = np.isin(REQUEST_YEARS, BASELINE)
    if int(baseline.sum()) != 30:
        raise RuntimeError("baseline-count-failed")
    return values - np.mean(values[baseline])


def smooth(values: np.ndarray, width: int = 31) -> np.ndarray:
    return pd.Series(values).rolling(width, center=True, min_periods=width).mean().to_numpy()


def exact_archived_backgrounds(proxies: dict[int, dict[str, Any]], weights: dict[str, np.ndarray]) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    prior = np.load(prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy")
    with h5py.File(prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat", "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    result: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for network, proxy in proxies.items():
        central = (proxy["locations"][:, 1] >= 80.0) & (proxy["locations"][:, 1] < 90.0)
        masks = {"full": np.ones(len(central), dtype=bool), "central_only": central, "omit_central": ~central}
        result[network] = {}
        for branch, record_mask in masks.items():
            flat, _ = round25.ao_background_mean(prior, indices, proxy["hxb_pool_full"][record_mask], proxy["observations"][record_mask])
            result[network][branch] = {}
            for region, region_weights in weights.items():
                valid = region_weights > 0
                series = np.average(flat[:, valid], axis=1, weights=region_weights[valid])
                result[network][branch][region] = anomaly(series)
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    round25_output = RESULTS / "climate_reconstruction_improvement_loop_v2/round_25_temperature_prior_update_decomposition_v1"
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md", "state_pool": STATE_FILE,
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "hma_mask": prior_tools.HMA_MASK, "public_grid": prior_tools.PUBLIC_REFERENCE_SOURCE,
        "round25_annual": round25_output / "annual_native_state_decomposition.csv",
        "driver": DRIVER,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}
    states = np.load(STATE_FILE)
    if not np.array_equal(np.asarray(states["years"], dtype=int), POOL_YEARS):
        raise RuntimeError("pool-year-axis-failed")
    member_years = np.arange(1001, 1849, dtype=int)
    pool_indices = np.asarray([int(np.flatnonzero(member_years == year)[0]) for year in POOL_YEARS])
    proxies = {network: base.load_proxy(network, pool_indices) for network in (6, 9)}
    for network in (6, 9):
        hxb, observations, locked = round25.read_proxy_allowed(proxies[network]["path"], REQUEST_YEARS)
        if locked:
            raise RuntimeError("locked-proxy-value-read")
        proxies[network]["hxb_pool_full"] = hxb
        proxies[network]["observations"] = observations

    weights = regional_weights(states)
    model_domain_states = {
        model: {region: state_series(np.asarray(states[f"{model}_tas_state"], dtype=float), region_weights) for region, region_weights in weights.items()}
        for model in MODELS
    }
    annual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, int, str, str], np.ndarray] = {}
    for model, model_label in MODELS.items():
        for network, proxy in proxies.items():
            hxb = base.rebuild_hxb(model, network, proxy, states)
            central = (proxy["locations"][:, 1] >= 80.0) & (proxy["locations"][:, 1] < 90.0)
            masks = {
                "full": np.ones(len(central), dtype=bool), "central_only": central,
                "omit_central": ~central, "full_complete_operator_only": proxy["complete"],
            }
            for branch, record_mask in masks.items():
                for region in REGIONS:
                    selected, projection, counts, _ = base.select_background(hxb, proxy["observations"], record_mask, model_domain_states[model][region])
                    values = anomaly(selected)
                    curves[(model_label, network, branch, region)] = values
                    for index, year in enumerate(REQUEST_YEARS):
                        annual_rows.append({
                            "model": model_label, "proxy_network": network, "branch": branch, "region": region,
                            "year": int(year), "background_anomaly_degC": float(values[index]),
                            "score_region_projection_correlation": float(projection[index]), "active_proxy_count": int(counts[index]),
                        })
                    historical_values = values[: len(HISTORICAL)]
                    metric_rows.append({
                        "model": model_label, "proxy_network": network, "branch": branch, "region": region,
                        "historical_mean_background_anomaly_degC": float(np.mean(historical_values)),
                        "historical_positive_year_fraction": float(np.mean(historical_values > 0)),
                        "historical_min_degC": float(np.min(historical_values)), "historical_max_degC": float(np.max(historical_values)),
                    })

    annual = pd.DataFrame(annual_rows)
    metrics = pd.DataFrame(metric_rows)
    annual.to_csv(OUTPUT / "annual_four_prior_background_1500_1879.csv", index=False)
    metrics.to_csv(OUTPUT / "historical_background_metrics.csv", index=False)

    window_rows: list[dict[str, Any]] = []
    for model_label in MODELS.values():
        for network in (6, 9):
            for region in REGIONS:
                full = curves[(model_label, network, "full", region)][: len(HISTORICAL)]
                omit = curves[(model_label, network, "omit_central", region)][: len(HISTORICAL)]
                central_only = curves[(model_label, network, "central_only", region)][: len(HISTORICAL)]
                complete = curves[(model_label, network, "full_complete_operator_only", region)][: len(HISTORICAL)]
                for start, end in WINDOWS:
                    selected = (HISTORICAL >= start) & (HISTORICAL <= end)
                    window_rows.append({
                        "model": model_label, "proxy_network": network, "region": region,
                        "start_year": start, "end_year": end,
                        "full_background_degC": float(np.mean(full[selected])),
                        "central_only_background_degC": float(np.mean(central_only[selected])),
                        "omit_central_background_degC": float(np.mean(omit[selected])),
                        "central_effect_full_minus_omit_degC": float(np.mean((full - omit)[selected])),
                        "component_free_operator_effect_full_minus_complete_degC": float(np.mean((full - complete)[selected])),
                    })
    windows = pd.DataFrame(window_rows)
    windows.to_csv(OUTPUT / "four_prior_50year_background_windows.csv", index=False)

    exact = exact_archived_backgrounds(proxies, weights)
    exact_rows: list[dict[str, Any]] = []
    exact_six: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in ("full", "central_only", "omit_central")}
    for branch in exact_six:
        for region in REGIONS:
            exact_six[branch][region] = (exact[6][branch][region] + exact[9][branch][region]) / 3.0
            for index, year in enumerate(REQUEST_YEARS):
                exact_rows.append({"branch": branch, "region": region, "year": int(year), "six_member_background_anomaly_degC": float(exact_six[branch][region][index])})
    pd.DataFrame(exact_rows).to_csv(OUTPUT / "exact_archived_cesm_six_member_background.csv", index=False)

    round25_annual = pd.read_csv(sources["round25_annual"]).set_index("year").loc[REQUEST_YEARS]
    archived_closure = float(np.max(np.abs(round25_annual.native_six_member_background_degC.to_numpy(float) - exact_six["full"]["hma"])))
    if archived_closure > 1e-10:
        raise RuntimeError(f"archived-background-closure:{archived_closure}")
    warm_gap = round25_annual.native_six_member_posterior_degC.to_numpy(float) - round25_annual.reference_family_central_degC.to_numpy(float)
    increment = round25_annual.native_six_member_increment_degC.to_numpy(float)
    exact_window_rows = []
    for start, end in WINDOWS:
        selected = (REQUEST_YEARS >= start) & (REQUEST_YEARS <= end)
        full = exact_six["full"]["hma"]
        omit = exact_six["omit_central"]["hma"]
        effect = full - omit
        gap_mean = float(np.mean(warm_gap[selected]))
        effect_mean = float(np.mean(effect[selected]))
        exact_window_rows.append({
            "start_year": start, "end_year": end,
            "posterior_minus_reference_warm_gap_degC": gap_mean,
            "full_six_member_background_degC": float(np.mean(full[selected])),
            "omit_central_six_member_background_degC": float(np.mean(omit[selected])),
            "central_background_effect_degC": effect_mean,
            "assimilation_increment_degC": float(np.mean(increment[selected])),
            "central_effect_over_warm_gap_diagnostic": effect_mean / gap_mean if gap_mean != 0 else np.nan,
        })
    exact_windows = pd.DataFrame(exact_window_rows)
    exact_windows.to_csv(OUTPUT / "archived_warm_gap_central_background_attribution.csv", index=False)

    hma_six_rows = []
    for model_label in MODELS.values():
        for branch in ("full", "omit_central"):
            values = (curves[(model_label, 6, branch, "hma")] + curves[(model_label, 9, branch, "hma")]) / 3.0
            for index, year in enumerate(REQUEST_YEARS):
                hma_six_rows.append({"model": model_label, "branch": branch, "year": int(year), "six_member_background_anomaly_degC": float(values[index])})
    hma_six = pd.DataFrame(hma_six_rows)
    hma_six.to_csv(OUTPUT / "four_prior_hma_six_member_background.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, model_label in zip(axes.flat, MODELS.values()):
        rows = hma_six[(hma_six.model == model_label) & hma_six.year.isin(HISTORICAL)]
        for branch, color in (("full", "#111111"), ("omit_central", "#0072B2")):
            values = rows[rows.branch == branch].sort_values("year").six_member_background_anomaly_degC.to_numpy(float)
            axis.plot(HISTORICAL, smooth(values), color=color, lw=2.0, label=branch.replace("_", " "))
        axis.axhline(0, color="#999999", lw=0.8)
        axis.set_title(model_label)
        axis.set_ylabel("Background anomaly (degC)")
        axis.grid(axis="y", color="#E5E5E5", lw=0.6)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(OUTPUT / "four_prior_hma_background_full_vs_omit_central_1500_1879.png", dpi=220)
    plt.close(fig)

    hist = slice(0, len(HISTORICAL))
    fig, axis = plt.subplots(figsize=(12.5, 5.3), constrained_layout=True)
    axis.plot(HISTORICAL, smooth(warm_gap[hist]), color="#D55E00", lw=2.3, label="Posterior - reference")
    axis.plot(HISTORICAL, smooth(exact_six["full"]["hma"][hist]), color="#111111", lw=2.0, label="Full background")
    axis.plot(HISTORICAL, smooth(exact_six["omit_central"]["hma"][hist]), color="#0072B2", lw=1.8, label="Background without central proxies")
    axis.plot(HISTORICAL, smooth((exact_six["full"]["hma"] - exact_six["omit_central"]["hma"])[hist]), color="#009E73", lw=1.7, label="Central background effect")
    axis.axhline(0, color="#888888", lw=0.8)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="Archived CESM: warm gap and central-proxy background contribution")
    axis.grid(axis="y", color="#E5E5E5", lw=0.6)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(OUTPUT / "archived_warm_gap_central_background_decomposition_1500_1879.png", dpi=220)
    plt.close(fig)

    external_window_summary = windows.groupby(["model", "proxy_network", "region"], as_index=False).agg(
        full_positive_window_count=("full_background_degC", lambda values: int(np.sum(np.asarray(values) > 0))),
        omit_positive_window_count=("omit_central_background_degC", lambda values: int(np.sum(np.asarray(values) > 0))),
        central_effect_positive_window_count=("central_effect_full_minus_omit_degC", lambda values: int(np.sum(np.asarray(values) > 0))),
        median_central_effect_degC=("central_effect_full_minus_omit_degC", "median"),
    )
    external_window_summary.to_csv(OUTPUT / "window_sign_summary.csv", index=False)
    hma_summary = external_window_summary[external_window_summary.region == "hma"].copy()
    all_model_network_central_positive = bool((hma_summary.central_effect_positive_window_count == len(WINDOWS)).all())
    archived_gap_positive_count = int((exact_windows.posterior_minus_reference_warm_gap_degC > 0).sum())
    archived_central_positive_count = int((exact_windows.central_background_effect_degC > 0).sum())
    median_ratio = float(exact_windows.central_effect_over_warm_gap_diagnostic.median())

    psm_closure = max(float(record.get("closure", 0.0)) for proxy in proxies.values() for record in proxy["records"])
    max_component_sensitivity = float(windows.component_free_operator_effect_full_minus_complete_degC.abs().max())
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in sources)
    summary = {
        "experiment_id": "climate_reconstruction_temperature_four_prior_strategy_c_replay_1500_1879_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "historical_period": [1500, 1879], "baseline_period": [1961, 1990],
        "harmonized_state_pool": [1500, 1848], "state_count": 349, "selection_size": 100,
        "window_count": len(WINDOWS), "all_model_network_hma_central_effect_positive_in_all_windows": all_model_network_central_positive,
        "archived_warm_gap_positive_window_count": archived_gap_positive_count,
        "archived_central_background_effect_positive_window_count": archived_central_positive_count,
        "median_archived_central_effect_over_warm_gap_diagnostic": median_ratio,
        "archived_six_member_background_closure_degC": archived_closure,
        "maximum_archived_bivariate_psm_closure": psm_closure,
        "maximum_component_free_operator_window_sensitivity_degC": max_component_sensitivity,
        "locked_1921_1939_values_accessed": 0, "enkf_run": False, "oggm_run": False,
        "production_modified": False, "source_hashes_unchanged": unchanged,
        "interpretation_boundary": "The experiment attributes only strategy-C background selection. It does not recompute EnKF increments or prove that posterior-reference disagreement is an error.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    lines = [
        "# Four-Prior Long-Horizon Strategy-C Replay", "",
        f"- HMA central effect positive in every window for every model/network: `{str(all_model_network_central_positive).lower()}`.",
        f"- Archived posterior-reference warm-gap positive windows: `{archived_gap_positive_count}/{len(WINDOWS)}`.",
        f"- Archived central-background effect positive windows: `{archived_central_positive_count}/{len(WINDOWS)}`.",
        f"- Median central-effect/warm-gap diagnostic ratio: `{median_ratio:+.3f}`.",
        f"- Archived background closure: `{archived_closure:.3e} degC`.", "",
        "## Archived HMA window attribution", "",
        "| window | warm gap | full background | omit central | central effect | increment |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in exact_windows.itertuples():
        lines.append(f"| {row.start_year}-{row.end_year} | {row.posterior_minus_reference_warm_gap_degC:+.3f} | {row.full_six_member_background_degC:+.3f} | {row.omit_central_six_member_background_degC:+.3f} | {row.central_background_effect_degC:+.3f} | {row.assimilation_increment_degC:+.3f} |")
    lines.extend(["", "No EnKF or OGGM was run; production and source files remain unchanged; 1921-1939 was not accessed."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
