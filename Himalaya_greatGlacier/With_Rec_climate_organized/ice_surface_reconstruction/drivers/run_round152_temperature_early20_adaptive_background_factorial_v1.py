"""Factor support and proxy-value effects in the AO/HGAO background selection."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import decompose_temperature_lia_regional_prior_update_v1 as prior_tools
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


YEARS = np.arange(1901, 1921, dtype=int)
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
BRANCHES = (
    "dynamic_support_dynamic_values",
    "common_support_dynamic_values",
    "dynamic_support_frozen_values",
    "common_support_frozen_values",
)


def slope(values: np.ndarray) -> float:
    if len(values) != len(YEARS) or not np.all(np.isfinite(values)):
        raise RuntimeError("slope-coverage-failed")
    return float(np.polyfit(YEARS, values, 1)[0] * 100.0)


def branch_observations(observations: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    finite = np.isfinite(observations)
    common = finite.all(axis=1)
    available = finite.any(axis=1)
    means = np.full(observations.shape[0], np.nan, dtype=float)
    means[available] = np.nanmean(observations[available], axis=1)
    common_dynamic = np.where(common[:, None], observations, np.nan)
    dynamic_frozen = np.where(finite, means[:, None], np.nan)
    common_frozen = np.where(common[:, None], means[:, None], np.nan)
    branches = {
        "dynamic_support_dynamic_values": observations.copy(),
        "common_support_dynamic_values": common_dynamic,
        "dynamic_support_frozen_values": dynamic_frozen,
        "common_support_frozen_values": common_frozen,
    }
    summary = {
        "proxy_count": int(observations.shape[0]),
        "ever_available_count": int(available.sum()),
        "common_support_count": int(common.sum()),
        "annual_active_min": int(finite.sum(axis=0).min()),
        "annual_active_max": int(finite.sum(axis=0).max()),
        "annual_active_counts": finite.sum(axis=0).astype(int).tolist(),
    }
    if summary["common_support_count"] < 10:
        raise RuntimeError(f"insufficient-common-support:{summary}")
    return branches, summary


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_temperature_early20_adaptive_background_factorial_1901_1920_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round152_temperature_early20_adaptive_background_factorial_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    round151 = validation / "climate_reconstruction_temperature_early20_prior_update_export_attribution_1901_1920_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "prior": prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "public_source": prior_tools.PUBLIC_REFERENCE_SOURCE,
        "hma_mask": prior_tools.HMA_MASK,
        "round151_summary": round151 / "summary.json",
        "round151_members": round151 / "temperature_early20_stage_attribution_members.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: prior_tools.file_state(path) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or experiment_indices.shape != (50, 500):
        raise RuntimeError("prior-or-index-shape-failed")
    proxy_inputs = {}
    support_summary = {}
    for network in (6, 9):
        hxb, observations, locked = round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked:
            raise RuntimeError("locked-proxy-values-read")
        branches, support = branch_observations(observations)
        proxy_inputs[network] = (hxb, branches)
        support_summary[str(network)] = support

    native_lat, native_lon = round25.native_coordinates()
    with h5py.File(paths["public_source"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = round25.interpolate_fields(mask05[None], source_lat, source_lon, native_lat, native_lon)[0] >= 0.5
    lon_grid = np.broadcast_to(native_lon[None, :], mask2.shape)
    masks = {
        name: mask2 if name == "hma" else mask2 & (lon_grid >= lower) & (lon_grid < upper)
        for name, (lower, upper) in REGIONS.items()
    }
    if any(not mask.any() for mask in masks.values()):
        raise RuntimeError("empty-region-mask")

    network_fields: dict[int, dict[str, np.ndarray]] = {6: {}, 9: {}}
    network_series: dict[int, dict[str, dict[str, np.ndarray]]] = {6: {}, 9: {}}
    for network in (6, 9):
        hxb, branch_inputs = proxy_inputs[network]
        for branch in BRANCHES:
            flat, _ = round25.ao_background_mean(
                prior, experiment_indices, hxb, branch_inputs[branch]
            )
            field = flat.reshape(len(YEARS), len(native_lat), len(native_lon), order="F")
            network_fields[network][branch] = field
            network_series[network][branch] = {
                region: round25.weighted_series(field, mask, native_lat)
                for region, mask in masks.items()
            }

    fixed_flat = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed_field = np.repeat(
        fixed_flat.reshape(1, len(native_lat), len(native_lon), order="F"), len(YEARS), axis=0
    )
    fixed_series = {
        region: round25.weighted_series(fixed_field, mask, native_lat)
        for region, mask in masks.items()
    }
    ensemble_series = {
        branch: {
            region: (
                network_series[6][branch][region]
                + network_series[9][branch][region]
                + fixed_series[region]
            ) / 3.0
            for region in masks
        }
        for branch in BRANCHES
    }

    factorial = {}
    annual_rows = []
    metric_rows = []
    closure_max = 0.0
    for region in masks:
        dd = ensemble_series["dynamic_support_dynamic_values"][region]
        cd = ensemble_series["common_support_dynamic_values"][region]
        df = ensemble_series["dynamic_support_frozen_values"][region]
        cf = ensemble_series["common_support_frozen_values"][region]
        terms = {
            "constant_control": cf,
            "support": df - cf,
            "values": cd - cf,
            "interaction": dd - cd - df + cf,
            "full": dd,
        }
        closure = terms["full"] - terms["constant_control"] - terms["support"] - terms["values"] - terms["interaction"]
        closure_max = max(closure_max, float(np.max(np.abs(closure))))
        factorial[region] = terms
        for index, year in enumerate(YEARS):
            row = {"region": region, "year": int(year)}
            for name, values in terms.items():
                row[f"{name}_degC"] = values[index]
            row["closure_degC"] = closure[index]
            annual_rows.append(row)
        component_slopes = {name: slope(values) for name, values in terms.items()}
        dominant = max(("support", "values", "interaction"), key=lambda name: abs(component_slopes[name]))
        metric_rows.append({
            "region": region,
            **{f"{name}_trend_degC_per_century": value for name, value in component_slopes.items()},
            "trend_closure_degC_per_century": component_slopes["full"] - sum(component_slopes[name] for name in ("constant_control", "support", "values", "interaction")),
            "dominant_absolute_factorial_component": dominant,
        })
    if closure_max > 1e-12:
        raise RuntimeError(f"factorial-closure-failed:{closure_max}")

    network_rows = []
    for network in (6, 9):
        for branch in BRANCHES:
            for region in masks:
                network_rows.append({
                    "proxy_network": network,
                    "branch": branch,
                    "region": region,
                    "trend_degC_per_century": slope(network_series[network][branch][region]),
                })
    annual = pd.DataFrame(annual_rows)
    metrics = pd.DataFrame(metric_rows)
    networks = pd.DataFrame(network_rows)
    annual.to_csv(output / "adaptive_background_factorial_annual.csv", index=False)
    metrics.to_csv(output / "adaptive_background_factorial_regional_metrics.csv", index=False)
    networks.to_csv(output / "adaptive_background_factorial_network_metrics.csv", index=False)
    (output / "proxy_support_summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")

    r151 = json.loads(paths["round151_summary"].read_text(encoding="utf-8"))
    expected_east = float(r151["east_contribution_trends_degC_per_century"]["background"])
    expected_hma = float(
        pd.read_csv(round151 / "temperature_early20_stage_attribution_regional.csv")
        .set_index("region")
        .loc["hma", "background_trend_1901_1920_degC_per_century"]
    )
    metric_index = metrics.set_index("region")
    closure_round151 = {
        "east_ge90": abs(float(metric_index.loc["east_ge90", "full_trend_degC_per_century"]) - expected_east),
        "hma": abs(float(metric_index.loc["hma", "full_trend_degC_per_century"]) - expected_hma),
    }
    if max(closure_round151.values()) > 1e-10:
        raise RuntimeError(f"round151-background-closure:{closure_round151}")

    colors = {
        "full": "#111111",
        "support": "#00798C",
        "values": "#D1495B",
        "interaction": "#EDA93B",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, region in zip(axes.flat, ("hma", "west_lt80", "central_80_90", "east_ge90")):
        for component in ("full", "support", "values", "interaction"):
            values = factorial[region][component]
            axis.plot(YEARS, values - values[0], color=colors[component], linewidth=2.2 if component == "full" else 1.5, label=component)
        axis.axhline(0, color="#B8B8B8", linewidth=0.8)
        axis.set_title(region)
        axis.set_ylabel("change from 1901 (degC)")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "temperature_early20_adaptive_background_factorial.png", dpi=220)
    plt.close(fig)

    east = metric_index.loc["east_ge90"]
    east_terms = {
        name: float(east[f"{name}_trend_degC_per_century"])
        for name in ("constant_control", "support", "values", "interaction", "full")
    }
    dominant = str(east["dominant_absolute_factorial_component"])
    after = {name: prior_tools.file_state(path) for name, path in paths.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in paths)
    if not unchanged:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment_id": "round152_temperature_early20_adaptive_background_factorial_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1901, 1920],
        "east_factorial_trends_degC_per_century": east_terms,
        "east_dominant_absolute_factorial_component": dominant,
        "proxy_support_summary": support_summary,
        "maximum_annual_factorial_closure_degC": closure_max,
        "round151_background_trend_closure": closure_round151,
        "decision": f"adaptive_background_{dominant}_dominant",
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_rerun": False,
        "oggm_run": False,
        "source_hashes_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 152 Adaptive-Background Factorial\n\n"
        f"- East full adaptive-background trend: `{east_terms['full']:+.3f} degC/century`.\n"
        f"- Support effect: `{east_terms['support']:+.3f}`.\n"
        f"- Proxy-value effect: `{east_terms['values']:+.3f}`.\n"
        f"- Support-value interaction: `{east_terms['interaction']:+.3f}`.\n"
        f"- Constant-pattern control: `{east_terms['constant_control']:+.3f}`.\n"
        f"- Dominant absolute component: `{dominant}`.\n"
        f"- Annual factorial closure: `{closure_max:.3e} degC`.\n"
        "- Locked 1921--1939 accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
