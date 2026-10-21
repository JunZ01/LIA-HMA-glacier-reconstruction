"""Attribute the 1901-1920 public temperature trend to production stages."""

from __future__ import annotations

import importlib.util
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


HISTORICAL_YEARS = np.arange(1800, 1921, dtype=int)
REQUESTED_YEARS = np.r_[HISTORICAL_YEARS, np.arange(1961, 1991, dtype=int)]
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
PERIODS = {
    "late_19c_1881_1900": (1881, 1900),
    "early_20c_1901_1920": (1901, 1920),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(years) & np.isfinite(values)
    if int(finite.sum()) < 3:
        return np.nan
    return float(np.polyfit(years[finite], values[finite], 1)[0] * 100.0)


def mean_between(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end)
    if int(selected.sum()) != end - start + 1 or not np.all(np.isfinite(values[selected])):
        raise RuntimeError(f"period-coverage-failed:{start}:{end}")
    return float(np.mean(values[selected]))


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_temperature_early20_prior_update_export_attribution_1901_1920_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round151_temperature_early20_prior_update_export_attribution_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    round122 = validation / "climate_reconstruction_external_low_frequency_benchmark_20260818_v2"
    round150 = validation / "climate_reconstruction_noaa_temperature_trajectory_extension_20260819_v1"
    base = load_module(
        "round151_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    common = base.load_common(driver)
    paths: dict[str, Path] = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "prior": prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "public_source": prior_tools.PUBLIC_REFERENCE_SOURCE,
        "hma_mask": prior_tools.HMA_MASK,
        "round122_current": round122 / "current_regional_curves_allowed_years.csv",
        "round122_manifest": round122 / "source_manifest.json",
        "round150_summary": round150 / "summary.json",
    }
    for member, (method, network) in prior_tools.MEMBERS.items():
        archive_method = {"AO": "AOEnKF", "HGAO": "HGAOEnKF", "O": "EnKF"}[method]
        paths[f"posterior_{member}"] = prior_tools.RESULT_ROOT / f"proxy{network}/{archive_method}_results.mat"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: prior_tools.file_state(path) for name, path in paths.items()}

    round150_summary = json.loads(paths["round150_summary"].read_text(encoding="utf-8"))
    if round150_summary["locked_1921_1939_accessed"] or round150_summary["jionglacuo_external_validation_file_opened"]:
        raise RuntimeError("round150-boundary-failed")
    prior = np.load(paths["prior"])
    if prior.shape != (336, 848):
        raise RuntimeError(f"prior-shape:{prior.shape}")
    with h5py.File(paths["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if experiment_indices.shape != (50, 500):
        raise RuntimeError(f"indices-shape:{experiment_indices.shape}")

    proxies = {}
    locked_reads = 0
    for network in (6, 9):
        hxb, observations, count = round25.read_proxy_allowed(paths[f"proxy{network}"], REQUESTED_YEARS)
        proxies[network] = (hxb, observations)
        locked_reads += count
    if locked_reads != 0:
        raise RuntimeError("locked-proxy-values-read")
    adaptive_background = {
        network: round25.ao_background_mean(
            prior, experiment_indices, proxies[network][0], proxies[network][1]
        )[0]
        for network in (6, 9)
    }
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed_background = np.repeat(fixed_one[None, :], len(REQUESTED_YEARS), axis=0)

    native_lat, native_lon = round25.native_coordinates()
    grid_lat, grid_lon = common.state_coordinates()
    if not np.array_equal(native_lat, grid_lat) or not np.array_equal(native_lon, grid_lon):
        raise RuntimeError("native-round122-grid-mismatch")
    with h5py.File(paths["public_source"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    lon_grid = np.broadcast_to(grid_lon[None, :], mask2.shape)
    masks = {
        name: mask2 if name == "hma" else mask2 & (lon_grid >= lower) & (lon_grid < upper)
        for name, (lower, upper) in REGIONS.items()
    }
    if any(not mask.any() for mask in masks.values()):
        raise RuntimeError("empty-region-mask")

    member_series: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for member, (method, network) in prior_tools.MEMBERS.items():
        posterior_field = round25.anomaly_field(
            round25.read_posterior_mean(paths[f"posterior_{member}"], REQUESTED_YEARS),
            REQUESTED_YEARS,
        )
        background_flat = fixed_background if method == "O" else adaptive_background[network]
        background_field = round25.anomaly_field(background_flat, REQUESTED_YEARS)
        increment_field = posterior_field - background_field
        member_series[member] = {
            region: {
                "posterior": round25.weighted_series(posterior_field, mask, native_lat),
                "background": round25.weighted_series(background_field, mask, native_lat),
                "increment": round25.weighted_series(increment_field, mask, native_lat),
            }
            for region, mask in masks.items()
        }
    ensemble = {
        region: {
            component: np.mean(
                np.stack([member_series[member][region][component] for member in prior_tools.MEMBERS]),
                axis=0,
            )
            for component in ("posterior", "background", "increment")
        }
        for region in masks
    }

    public_table = pd.read_csv(paths["round122_current"])
    public_table = public_table[public_table.climate_variable.eq("temperature")]
    if ((public_table.year >= 1921) & (public_table.year <= 1939)).any():
        raise RuntimeError("locked-year-in-public-allowed-export")
    public = {}
    for region in masks:
        subset = public_table[public_table.region.eq(region)].set_index("year").value
        public[region] = np.asarray([subset.get(int(year), np.nan) for year in REQUESTED_YEARS], dtype=float)
        if not np.all(np.isfinite(public[region])):
            raise RuntimeError(f"public-coverage:{region}")

    hist = REQUESTED_YEARS <= 1920
    years = REQUESTED_YEARS[hist]
    annual_rows = []
    regional_rows = []
    member_rows = []
    posterior_closure_max = 0.0
    public_closure_max = 0.0
    curves = {}
    for region in masks:
        posterior = ensemble[region]["posterior"][hist]
        background = ensemble[region]["background"][hist]
        increment = ensemble[region]["increment"][hist]
        public_values = public[region][hist]
        export_residual = public_values - posterior
        posterior_closure = posterior - background - increment
        public_closure = public_values - background - increment - export_residual
        posterior_closure_max = max(posterior_closure_max, float(np.max(np.abs(posterior_closure))))
        public_closure_max = max(public_closure_max, float(np.max(np.abs(public_closure))))
        curves[region] = {
            "public": public_values,
            "posterior": posterior,
            "background": background,
            "increment": increment,
            "export_residual": export_residual,
        }
        for index, year in enumerate(years):
            annual_rows.append({
                "region": region,
                "year": int(year),
                "public_current_degC": public_values[index],
                "native_posterior_degC": posterior[index],
                "background_degC": background[index],
                "enkf_increment_degC": increment[index],
                "export_residual_degC": export_residual[index],
                "posterior_closure_degC": posterior_closure[index],
                "public_closure_degC": public_closure[index],
            })
        selected = (years >= 1901) & (years <= 1920)
        slopes = {
            component: slope_per_century(years[selected], values[selected])
            for component, values in curves[region].items()
        }
        mean_changes = {
            component: mean_between(years, values, 1901, 1920) - mean_between(years, values, 1881, 1900)
            for component, values in curves[region].items()
        }
        contribution_slopes = {
            "background": slopes["background"],
            "enkf_increment": slopes["increment"],
            "export_residual": slopes["export_residual"],
        }
        dominant = max(contribution_slopes, key=lambda key: abs(contribution_slopes[key]))
        regional_rows.append({
            "region": region,
            "public_trend_1901_1920_degC_per_century": slopes["public"],
            "native_posterior_trend_1901_1920_degC_per_century": slopes["posterior"],
            "background_trend_1901_1920_degC_per_century": slopes["background"],
            "enkf_increment_trend_1901_1920_degC_per_century": slopes["increment"],
            "export_residual_trend_1901_1920_degC_per_century": slopes["export_residual"],
            "trend_closure_degC_per_century": slopes["public"] - slopes["background"] - slopes["increment"] - slopes["export_residual"],
            "public_mean_change_1901_1920_minus_1881_1900_degC": mean_changes["public"],
            "background_mean_change_degC": mean_changes["background"],
            "enkf_increment_mean_change_degC": mean_changes["increment"],
            "export_residual_mean_change_degC": mean_changes["export_residual"],
            "dominant_absolute_trend_component": dominant,
        })
        for member, (method, network) in prior_tools.MEMBERS.items():
            item = member_series[member][region]
            posterior_member = item["posterior"][hist]
            background_member = item["background"][hist]
            increment_member = item["increment"][hist]
            member_rows.append({
                "region": region,
                "member": member,
                "method": method,
                "proxy_network": network,
                "posterior_trend_1901_1920_degC_per_century": slope_per_century(years[selected], posterior_member[selected]),
                "background_trend_1901_1920_degC_per_century": slope_per_century(years[selected], background_member[selected]),
                "increment_trend_1901_1920_degC_per_century": slope_per_century(years[selected], increment_member[selected]),
            })
    if posterior_closure_max > 1e-12 or public_closure_max > 1e-12:
        raise RuntimeError(f"closure-failed:{posterior_closure_max}:{public_closure_max}")

    annual = pd.DataFrame(annual_rows)
    regional = pd.DataFrame(regional_rows)
    members = pd.DataFrame(member_rows)
    annual.to_csv(output / "temperature_early20_stage_attribution_annual.csv", index=False)
    regional.to_csv(output / "temperature_early20_stage_attribution_regional.csv", index=False)
    members.to_csv(output / "temperature_early20_stage_attribution_members.csv", index=False)

    colors = {
        "public": "#111111",
        "posterior": "#6C757D",
        "background": "#00798C",
        "increment": "#D1495B",
        "export_residual": "#EDA93B",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, region in zip(axes.flat, ("hma", "west_lt80", "central_80_90", "east_ge90")):
        base_index = int(np.flatnonzero(years == 1901)[0])
        for component in ("public", "posterior", "background", "increment", "export_residual"):
            values = curves[region][component]
            axis.plot(years, values - values[base_index], color=colors[component], linewidth=2.2 if component == "public" else 1.45, label=component.replace("_", " "))
        axis.axvline(1901, color="#888888", linestyle=":", linewidth=0.9)
        axis.axhline(0, color="#B8B8B8", linewidth=0.8)
        axis.set_xlim(1881, 1920)
        axis.set_title(region)
        axis.set_ylabel("change from 1901 (degC)")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "temperature_early20_prior_update_export_attribution.png", dpi=220)
    plt.close(fig)

    east = regional.set_index("region").loc["east_ge90"].to_dict()
    current_trend = float(east["public_trend_1901_1920_degC_per_century"])
    round150_expected = next(
        item["current_trend_per_century_native_units"]
        for item in round150_summary["new_metrics"]
        if item["period"] == "early_1901_1920"
    )
    round150_closure = abs(current_trend - float(round150_expected))
    if round150_closure > 1e-10:
        raise RuntimeError(f"round150-current-trend-closure:{round150_closure}")
    contributions = {
        "background": float(east["background_trend_1901_1920_degC_per_century"]),
        "enkf_increment": float(east["enkf_increment_trend_1901_1920_degC_per_century"]),
        "export_residual": float(east["export_residual_trend_1901_1920_degC_per_century"]),
    }
    dominant = max(contributions, key=lambda key: abs(contributions[key]))
    after = {name: prior_tools.file_state(path) for name, path in paths.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in paths)
    if not unchanged:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment_id": "round151_temperature_early20_prior_update_export_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1901, 1920],
        "east_public_trend_degC_per_century": current_trend,
        "east_contribution_trends_degC_per_century": contributions,
        "east_dominant_absolute_trend_component": dominant,
        "east_native_posterior_trend_degC_per_century": float(east["native_posterior_trend_1901_1920_degC_per_century"]),
        "round150_trend_closure_abs_degC_per_century": round150_closure,
        "maximum_annual_posterior_closure_degC": posterior_closure_max,
        "maximum_annual_public_closure_degC": public_closure_max,
        "instrumental_target_role": "1940_2000_psm_calibration_only_not_annual_1901_1920_state_input",
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_rerun": False,
        "oggm_run": False,
        "source_hashes_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 151 Early-20th-Century Temperature Attribution\n\n"
        f"- East public trend: `{current_trend:+.3f} degC/century`.\n"
        f"- Background contribution: `{contributions['background']:+.3f}`.\n"
        f"- EnKF increment contribution: `{contributions['enkf_increment']:+.3f}`.\n"
        f"- Export/resampling residual: `{contributions['export_residual']:+.3f}`.\n"
        f"- Dominant absolute component: `{dominant}`.\n"
        f"- Annual public closure: `{public_closure_max:.3e} degC`.\n"
        "- Locked 1921--1939 accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
