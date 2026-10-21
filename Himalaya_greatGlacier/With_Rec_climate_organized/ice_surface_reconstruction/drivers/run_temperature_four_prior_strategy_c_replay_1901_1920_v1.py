"""Replay strategy-C background selection with four harmonized prior pools."""

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


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1"
STATE_FILE = OUTPUT / "four_prior_state_pools_1500_1848_v1.npz"
TARGET_YEARS = np.arange(1901, 1921, dtype=int)
POOL_YEARS = np.arange(1500, 1849, dtype=int)
MODELS = {"cesm": "CESM", "mode": "ModE-Sim", "mpi": "MPI", "miroc": "MIROC"}
BRANCHES = ("full", "central_only", "omit_central", "full_complete_operator_only")


def file_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def text(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value))


def slope(values: np.ndarray) -> float:
    if values.shape != TARGET_YEARS.shape or not np.all(np.isfinite(values)):
        raise RuntimeError("trend-coverage-failed")
    return float(np.polyfit(TARGET_YEARS, values, 1)[0] * 100.0)


def load_proxy(network: int, pool_indices: np.ndarray) -> dict[str, Any]:
    path = prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
    hxb, observations, locked = round25.read_proxy_allowed(path, TARGET_YEARS)
    if locked:
        raise RuntimeError("locked-proxy-value-read")
    records: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, ref in enumerate(refs):
            group = handle[ref]
            location = np.asarray(group["location"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(group["b"], dtype=float).reshape(-1, order="F")
            has_components = all(name in group for name in ("M_t", "M_p")) and coefficients.size == 3
            record: dict[str, Any] = {
                "record_1based": index + 1,
                "latitude": float(location[0]),
                "longitude": float(location[1]),
                "ptype": text(group["ptype"]),
                "psm_method": text(group["psm_mthd"]),
                "b": coefficients,
                "has_components": has_components,
            }
            if has_components:
                mt = np.asarray(group["M_t"], dtype=float).reshape(-1, order="F")
                mp = np.asarray(group["M_p"], dtype=float).reshape(-1, order="F")
                rebuilt = coefficients[0] + coefficients[1] * mt + coefficients[2] * mp
                closure = float(np.max(np.abs(rebuilt - np.asarray(group["HXb"], dtype=float).reshape(-1, order="F"))))
                record.update({"mt_mean": float(np.mean(mt)), "mp_mean": float(np.mean(mp)), "closure": closure})
            records.append(record)
    return {
        "path": path,
        "hxb_pool": hxb[:, pool_indices],
        "observations": observations,
        "records": records,
        "locations": np.asarray([[item["latitude"], item["longitude"]] for item in records]),
        "complete": np.asarray([item["has_components"] for item in records], dtype=bool),
    }


def rebuild_hxb(model: str, network: int, proxy: dict[str, Any], states: Any) -> np.ndarray:
    if model == "cesm":
        return proxy["hxb_pool"].copy()
    unique_indices = np.asarray(states[f"proxy{network}_unique_point_indices"], dtype=int)
    raw_t = np.asarray(states[f"{model}_tas_proxy"], dtype=float)[:, unique_indices].T
    raw_p = np.asarray(states[f"{model}_pr_proxy"], dtype=float)[:, unique_indices].T
    t_anomaly = raw_t - np.mean(raw_t, axis=1, keepdims=True)
    p_anomaly = raw_p - np.mean(raw_p, axis=1, keepdims=True)
    output = np.empty((len(proxy["records"]), len(POOL_YEARS)), dtype=float)
    for index, record in enumerate(proxy["records"]):
        if record["has_components"]:
            b = record["b"]
            mt = t_anomaly[index] + record["mt_mean"]
            mp = p_anomaly[index] + record["mp_mean"]
            output[index] = b[0] + b[1] * mt + b[2] * mp
        else:
            # The archive contains no recoverable predictor components for these
            # eight legacy rows. Preserve their exact archived operator, then
            # quantify a complete-operator-only sensitivity separately.
            output[index] = proxy["hxb_pool"][index]
    if not np.all(np.isfinite(output)):
        raise RuntimeError(f"nonfinite-hxb:{model}:{network}")
    return output


def hma_east_mask(states: Any) -> tuple[np.ndarray, np.ndarray]:
    native_lat = np.asarray(states["native_lat"], dtype=float)
    native_lon = np.asarray(states["native_lon"], dtype=float)
    with h5py.File(prior_tools.PUBLIC_REFERENCE_SOURCE, "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(prior_tools.HMA_MASK).astype(float)
    mask = round25.interpolate_fields(mask05[None], source_lat, source_lon, native_lat, native_lon)[0] >= 0.5
    lon_grid = np.broadcast_to(native_lon[None, :], mask.shape)
    east = mask & (lon_grid >= 90.0)
    if not east.any():
        raise RuntimeError("empty-east-mask")
    weights = np.cos(np.deg2rad(native_lat))[:, None] * east
    return east, weights.reshape(-1, order="F")


def east_state_series(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = state - np.mean(state, axis=0, keepdims=True)
    valid = weights > 0
    return np.average(centered[:, valid], axis=1, weights=weights[valid])


def select_background(hxb: np.ndarray, observations: np.ndarray, record_mask: np.ndarray, east_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    background = np.empty(len(TARGET_YEARS), dtype=float)
    projection = np.empty(len(TARGET_YEARS), dtype=float)
    active_counts = np.empty(len(TARGET_YEARS), dtype=int)
    selected_years = np.empty((len(TARGET_YEARS), 100), dtype=int)
    for year_index in range(len(TARGET_YEARS)):
        active = record_mask & np.isfinite(observations[:, year_index])
        active_counts[year_index] = int(active.sum())
        if active_counts[year_index] < 3:
            raise RuntimeError(f"insufficient-active-proxies:{TARGET_YEARS[year_index]}:{active_counts[year_index]}")
        y = observations[active, year_index]
        y_anomaly = y - np.mean(y)
        model = hxb[active]
        model_anomaly = model - np.mean(model, axis=0, keepdims=True)
        numerator = y_anomaly @ model_anomaly
        denominator = np.sqrt(np.sum(y_anomaly**2) * np.sum(model_anomaly**2, axis=0))
        scores = np.full(model.shape[1], -1.0, dtype=float)
        valid = denominator > 0
        scores[valid] = numerator[valid] / denominator[valid]
        chosen = np.argsort(-scores, kind="stable")[:100]
        selected_years[year_index] = POOL_YEARS[chosen]
        background[year_index] = float(np.mean(east_state[chosen]))
        projection[year_index] = float(np.corrcoef(scores[valid], east_state[valid])[0, 1])
    return background, projection, active_counts, selected_years


def exact_archived_cesm(proxy: dict[str, Any], east_weights: np.ndarray) -> np.ndarray:
    prior = np.load(prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy")
    with h5py.File(prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat", "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    background, _ = round25.ao_background_mean(prior, indices, np.asarray(proxy["hxb_pool_full"]), proxy["observations"])
    valid = east_weights > 0
    return np.average(background[:, valid], axis=1, weights=east_weights[valid])


def main() -> int:
    if not STATE_FILE.exists():
        raise FileNotFoundError(STATE_FILE)
    target = OUTPUT / "summary.json"
    if target.exists():
        raise FileExistsError(target)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md", "state_pool": STATE_FILE,
        "state_summary": OUTPUT / "state_pool_acquisition_summary.json",
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "hma_mask": prior_tools.HMA_MASK, "public_grid": prior_tools.PUBLIC_REFERENCE_SOURCE,
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
    proxies = {network: load_proxy(network, pool_indices) for network in (6, 9)}
    # Preserve the unsliced operator only for the exact archived CESM replay.
    for network in (6, 9):
        full_hxb, _, _ = round25.read_proxy_allowed(proxies[network]["path"], TARGET_YEARS)
        proxies[network]["hxb_pool_full"] = full_hxb

    _, east_weights = hma_east_mask(states)
    east_states = {model: east_state_series(np.asarray(states[f"{model}_tas_state"], dtype=float), east_weights) for model in MODELS}
    psm_closure = max(float(record.get("closure", 0.0)) for proxy in proxies.values() for record in proxy["records"])
    if psm_closure > 1e-12:
        raise RuntimeError(f"archived-psm-closure:{psm_closure}")

    annual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, int, str], np.ndarray] = {}
    for model, model_label in MODELS.items():
        for network in (6, 9):
            proxy = proxies[network]
            hxb = rebuild_hxb(model, network, proxy, states)
            central = (proxy["locations"][:, 1] >= 80.0) & (proxy["locations"][:, 1] < 90.0)
            masks = {
                "full": np.ones(len(central), dtype=bool),
                "central_only": central,
                "omit_central": ~central,
                "full_complete_operator_only": proxy["complete"],
            }
            for branch, record_mask in masks.items():
                curve, projection, counts, selected_years = select_background(hxb, proxy["observations"], record_mask, east_states[model])
                selected[(model, network, branch)] = selected_years
                for index, year in enumerate(TARGET_YEARS):
                    annual_rows.append({
                        "model": model_label, "proxy_network": network, "branch": branch, "year": int(year),
                        "east_background_degC": float(curve[index]), "score_east_projection_correlation": float(projection[index]),
                        "active_proxy_count": int(counts[index]),
                    })
                metric_rows.append({
                    "model": model_label, "proxy_network": network, "branch": branch,
                    "east_background_trend_degC_per_century": slope(curve),
                    "mean_score_east_projection_correlation": float(np.mean(projection)),
                    "median_score_east_projection_correlation": float(np.median(projection)),
                    "active_proxy_count_min": int(counts.min()), "active_proxy_count_max": int(counts.max()),
                })

    annual = pd.DataFrame(annual_rows)
    metrics = pd.DataFrame(metric_rows)
    annual.to_csv(OUTPUT / "annual_east_background_projection.csv", index=False)
    metrics.to_csv(OUTPUT / "east_background_projection_metrics.csv", index=False)

    effect_rows: list[dict[str, Any]] = []
    for model_label in MODELS.values():
        for network in (6, 9):
            table = metrics[(metrics.model == model_label) & (metrics.proxy_network == network)].set_index("branch")
            full = float(table.loc["full", "east_background_trend_degC_per_century"])
            omit = float(table.loc["omit_central", "east_background_trend_degC_per_century"])
            central = float(table.loc["central_only", "east_background_trend_degC_per_century"])
            complete = float(table.loc["full_complete_operator_only", "east_background_trend_degC_per_century"])
            effect_rows.append({
                "model": model_label, "proxy_network": network, "full_trend": full,
                "central_only_trend": central, "omit_central_trend": omit,
                "central_omission_effect_full_minus_omit": full - omit,
                "component_free_operator_sensitivity_full_minus_complete": full - complete,
            })
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUTPUT / "central_proxy_effects.csv", index=False)

    exact_rows = []
    for network in (6, 9):
        curve = exact_archived_cesm(proxies[network], east_weights)
        exact_rows.append({"proxy_network": network, "exact_archived_50x500_east_trend_degC_per_century": slope(curve)})
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(OUTPUT / "exact_archived_cesm_control.csv", index=False)

    network_inventory = []
    for network, proxy in proxies.items():
        central = (proxy["locations"][:, 1] >= 80.0) & (proxy["locations"][:, 1] < 90.0)
        network_inventory.append({
            "proxy_network": network, "record_count": len(proxy["records"]), "central_80_90_record_count": int(central.sum()),
            "complete_bivariate_operator_count": int(proxy["complete"].sum()), "component_free_frozen_operator_count": int((~proxy["complete"]).sum()),
        })
    pd.DataFrame(network_inventory).to_csv(OUTPUT / "proxy_network_inventory.csv", index=False)

    full_pivot = effects.pivot(index="model", columns="proxy_network", values="full_trend")
    warming_by_model = {model: bool((full_pivot.loc[model] > 0).all()) for model in MODELS.values()}
    all_network_trends = effects.groupby("model")["full_trend"].mean()
    if all(value > 0 for value in all_network_trends):
        direction_classification = "all_four_model_backgrounds_warm"
    elif all_network_trends["CESM"] > 0 and all(all_network_trends[model] <= 0 for model in ("ModE-Sim", "MPI", "MIROC")):
        direction_classification = "only_cesm_background_warms"
    else:
        direction_classification = "background_direction_changes_with_prior_model"
    proxy9 = effects[effects.proxy_network == 9].set_index("model")
    central_proxy9_drives_all = bool(all(
        proxy9.loc[model, "full_trend"] > 0
        and proxy9.loc[model, "omit_central_trend"] <= 0
        and proxy9.loc[model, "central_omission_effect_full_minus_omit"] > 0
        for model in MODELS.values()
    ))

    colors = {"full": "#111111", "central_only": "#D55E00", "omit_central": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, model_label in zip(axes.flat, MODELS.values()):
        for network, style in ((6, "--"), (9, "-")):
            for branch in ("full", "central_only", "omit_central"):
                rows = annual[(annual.model == model_label) & (annual.proxy_network == network) & (annual.branch == branch)].sort_values("year")
                values = rows.east_background_degC.to_numpy(float)
                label = f"proxy{network} {branch.replace('_', ' ')}"
                axis.plot(rows.year, values - values[0], color=colors[branch], ls=style, lw=2.0 if branch == "full" else 1.35, label=label)
        axis.axhline(0, color="#999999", lw=0.8)
        axis.set_title(model_label)
        axis.grid(axis="y", color="#E5E5E5", lw=0.6)
        axis.set_ylabel("Change from 1901 (degC)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(OUTPUT / "four_prior_east_background_curves_1901_1920.png", dpi=220)
    plt.close(fig)

    plot = effects[effects.proxy_network.isin([6, 9])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True, constrained_layout=True)
    for axis, network in zip(axes, (6, 9)):
        rows = plot[plot.proxy_network == network].set_index("model").loc[list(MODELS.values())]
        x = np.arange(len(rows))
        axis.bar(x - 0.25, rows.full_trend, width=0.25, color="#111111", label="Full")
        axis.bar(x, rows.central_only_trend, width=0.25, color="#D55E00", label="Central only")
        axis.bar(x + 0.25, rows.omit_central_trend, width=0.25, color="#0072B2", label="Omit central")
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set_xticks(x, rows.index, rotation=20, ha="right")
        axis.set_title(f"proxy{network}")
        axis.set_ylabel("East background trend (degC/century)")
    axes[0].legend(frameon=False, ncol=3)
    fig.savefig(OUTPUT / "four_prior_east_background_trends.png", dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in sources)
    summary = {
        "experiment_id": "climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "target_period": [1901, 1920], "harmonized_pool_period": [1500, 1848], "state_count_per_model": 349,
        "selection_size": 100, "selection_fraction": 100.0 / 349.0,
        "direction_classification": direction_classification,
        "central_proxy9_drives_all_models_under_strict_sign_rule": central_proxy9_drives_all,
        "warming_both_networks_by_model": warming_by_model,
        "maximum_archived_bivariate_psm_closure": psm_closure,
        "maximum_component_free_operator_trend_sensitivity": float(effects.component_free_operator_sensitivity_full_minus_complete.abs().max()),
        "exact_archived_cesm_control": exact.to_dict(orient="records"),
        "locked_1921_1939_values_accessed": 0, "enkf_run": False, "oggm_run": False,
        "production_modified": False, "source_hashes_unchanged": unchanged,
        "interpretation_boundary": "Background-selection mechanism sensitivity only. The four model pools use identical 1500-1848 years and top-100 selection; this does not validate or replace proxy data, PSMs, production reconstruction, or EnKF output.",
    }
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest_final.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    lines = [
        "# Four-Prior Strategy-C Replay", "",
        f"- Direction classification: `{direction_classification}`.",
        f"- Central proxy9 strict all-model driver test: `{str(central_proxy9_drives_all).lower()}`.",
        f"- Maximum archived bivariate PSM closure: `{psm_closure:.3e}`.",
        f"- Maximum sensitivity to the eight component-free legacy operators: `{summary['maximum_component_free_operator_trend_sensitivity']:.3f} degC/century`.", "",
        "## East background trends (degC/century)", "",
        "| model | network | full | central only | omit central | full - omit |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in effects.itertuples():
        lines.append(f"| {row.model} | {row.proxy_network} | {row.full_trend:+.3f} | {row.central_only_trend:+.3f} | {row.omit_central_trend:+.3f} | {row.central_omission_effect_full_minus_omit:+.3f} |")
    lines.extend(["", "No EnKF or OGGM run was performed; no production or source file was modified; 1921-1939 was not accessed."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
