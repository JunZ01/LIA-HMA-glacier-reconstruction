"""Group leave-out audit of adaptive-background warming in 1901-1920."""

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


def slope(values: np.ndarray) -> float:
    if len(values) != len(YEARS) or not np.all(np.isfinite(values)):
        raise RuntimeError("slope-coverage-failed")
    return float(np.polyfit(YEARS, values, 1)[0] * 100.0)


def harmonized_metadata(network: int, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if network == 6:
        result = pd.DataFrame({
            "record": frame.proxy_index.astype(int),
            "longitude": frame.longitude.astype(float),
            "b1": frame.coefficient_b1.astype(float),
            "b2": frame.coefficient_b2.astype(float),
            "pvalue": frame.psm_pvalue.astype(float),
        })
    else:
        result = pd.DataFrame({
            "record": frame.proxy9_record_1based.astype(int),
            "longitude": frame.longitude.astype(float),
            "b1": frame.psm_temp_coefficient_b1.astype(float),
            "b2": frame.psm_prcp_coefficient_b2.astype(float),
            "pvalue": frame.psm_p_value.astype(float),
        })
    expected = np.arange(1, len(result) + 1)
    if not np.array_equal(result.record.to_numpy(int), expected):
        raise RuntimeError(f"metadata-record-order:{network}")
    return result


def groups(metadata: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    lon = metadata.longitude.to_numpy(float)
    b1 = metadata.b1.to_numpy(float)
    b2 = metadata.b2.to_numpy(float)
    pvalue = metadata.pvalue.to_numpy(float)
    return {
        "longitude_west_lt80": ("longitude", lon < 80.0),
        "longitude_central_80_90": ("longitude", (lon >= 80.0) & (lon < 90.0)),
        "longitude_east_ge90": ("longitude", lon >= 90.0),
        "temperature_b1_negative": ("temperature_coefficient_sign", b1 < 0.0),
        "temperature_b1_nonnegative": ("temperature_coefficient_sign", b1 >= 0.0),
        "precipitation_b2_negative": ("precipitation_coefficient_sign", b2 < 0.0),
        "precipitation_b2_nonnegative": ("precipitation_coefficient_sign", b2 >= 0.0),
        "psm_significant_p05": ("psm_significance", pvalue <= 0.05),
        "psm_nonsignificant_gt05": ("psm_significance", pvalue > 0.05),
    }


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_temperature_early20_adaptive_background_group_loo_1901_1920_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round153_temperature_early20_adaptive_background_group_loo_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    round152 = validation / "climate_reconstruction_temperature_early20_adaptive_background_factorial_1901_1920_v2"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "prior": prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "proxy6_metadata": validation / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9_metadata": validation / "climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "public_source": prior_tools.PUBLIC_REFERENCE_SOURCE,
        "hma_mask": prior_tools.HMA_MASK,
        "round152_summary": round152 / "summary.json",
        "round152_networks": round152 / "adaptive_background_factorial_network_metrics.csv",
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
    native_lat, native_lon = round25.native_coordinates()
    with h5py.File(paths["public_source"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = round25.interpolate_fields(mask05[None], source_lat, source_lon, native_lat, native_lon)[0] >= 0.5
    lon_grid = np.broadcast_to(native_lon[None, :], mask2.shape)
    east_mask = mask2 & (lon_grid >= 90.0)
    hma_mask = mask2
    if not east_mask.any() or not hma_mask.any():
        raise RuntimeError("empty-mask")

    metadata = {
        6: harmonized_metadata(6, paths["proxy6_metadata"]),
        9: harmonized_metadata(9, paths["proxy9_metadata"]),
    }
    proxy_inputs = {}
    full_fields = {}
    full_series = {6: {}, 9: {}}
    for network in (6, 9):
        hxb, observations, locked = round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked:
            raise RuntimeError("locked-proxy-values-read")
        if len(metadata[network]) != observations.shape[0]:
            raise RuntimeError(f"metadata-count:{network}:{len(metadata[network])}:{observations.shape[0]}")
        proxy_inputs[network] = (hxb, observations)
        flat, _ = round25.ao_background_mean(prior, experiment_indices, hxb, observations)
        field = flat.reshape(len(YEARS), len(native_lat), len(native_lon), order="F")
        full_fields[network] = field
        full_series[network] = {
            "east_ge90": round25.weighted_series(field, east_mask, native_lat),
            "hma": round25.weighted_series(field, hma_mask, native_lat),
        }

    fixed_flat = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed_field = np.repeat(
        fixed_flat.reshape(1, len(native_lat), len(native_lon), order="F"), len(YEARS), axis=0
    )
    fixed_series = {
        "east_ge90": round25.weighted_series(fixed_field, east_mask, native_lat),
        "hma": round25.weighted_series(fixed_field, hma_mask, native_lat),
    }
    full_ensemble = {
        region: (full_series[6][region] + full_series[9][region] + fixed_series[region]) / 3.0
        for region in ("east_ge90", "hma")
    }

    round152_networks = pd.read_csv(paths["round152_networks"])
    round152_full = round152_networks[round152_networks.branch.eq("dynamic_support_dynamic_values")]
    network_closure = {}
    for network in (6, 9):
        for region in ("east_ge90", "hma"):
            expected = float(round152_full[(round152_full.proxy_network == network) & (round152_full.region == region)].trend_degC_per_century.iloc[0])
            actual = slope(full_series[network][region])
            network_closure[f"proxy{network}_{region}"] = abs(actual - expected)
    if max(network_closure.values()) > 1e-10:
        raise RuntimeError(f"round152-network-closure:{network_closure}")

    rows = []
    curve_rows = []
    full_trends = {region: slope(values) for region, values in full_ensemble.items()}
    for network in (6, 9):
        hxb, observations = proxy_inputs[network]
        other = 9 if network == 6 else 6
        for group_name, (family, omitted) in groups(metadata[network]).items():
            count = int(omitted.sum())
            if count == 0 or count == len(omitted):
                raise RuntimeError(f"invalid-group-size:{network}:{group_name}:{count}")
            branch_observations = observations.copy()
            branch_observations[omitted] = np.nan
            flat, _ = round25.ao_background_mean(
                prior, experiment_indices, hxb, branch_observations
            )
            field = flat.reshape(len(YEARS), len(native_lat), len(native_lon), order="F")
            for region, mask in (("east_ge90", east_mask), ("hma", hma_mask)):
                omitted_network_series = round25.weighted_series(field, mask, native_lat)
                ensemble = (omitted_network_series + full_series[other][region] + fixed_series[region]) / 3.0
                branch_trend = slope(ensemble)
                effect = full_trends[region] - branch_trend
                rows.append({
                    "proxy_network": network,
                    "group_family": family,
                    "group": group_name,
                    "group_record_count": count,
                    "retained_record_count": int(len(omitted) - count),
                    "region": region,
                    "full_six_member_background_trend_degC_per_century": full_trends[region],
                    "leave_group_out_trend_degC_per_century": branch_trend,
                    "group_warming_effect_degC_per_century": effect,
                    "effect_fraction_of_full_trend": effect / full_trends[region] if abs(full_trends[region]) > 1e-12 else np.nan,
                })
                for index, year in enumerate(YEARS):
                    curve_rows.append({
                        "proxy_network": network,
                        "group_family": family,
                        "group": group_name,
                        "region": region,
                        "year": int(year),
                        "full_background_degC": full_ensemble[region][index],
                        "leave_group_out_background_degC": ensemble[index],
                    })
    effects = pd.DataFrame(rows)
    effects.to_csv(output / "adaptive_background_group_leave_out_effects.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "adaptive_background_group_leave_out_curves.csv", index=False)

    east = effects[effects.region.eq("east_ge90")].copy()
    east["absolute_effect"] = east.group_warming_effect_degC_per_century.abs()
    top = east.sort_values("absolute_effect", ascending=False).iloc[0]
    distributed = bool(top.absolute_effect < 0.5 * abs(full_trends["east_ge90"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True, sharey=True)
    for axis, network in zip(axes, (6, 9)):
        subset = east[east.proxy_network.eq(network)].sort_values("group_warming_effect_degC_per_century")
        colors = np.where(subset.group_warming_effect_degC_per_century >= 0, "#D1495B", "#00798C")
        axis.barh(subset.group, subset.group_warming_effect_degC_per_century, color=colors)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_title(f"proxy{network}")
        axis.set_xlabel("full - leave-out trend (degC/century)")
        axis.grid(axis="x", color="#E2E2E2", linewidth=0.5)
    fig.savefig(output / "temperature_early20_adaptive_background_group_leave_out.png", dpi=220)
    plt.close(fig)

    after = {name: prior_tools.file_state(path) for name, path in paths.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in paths)
    if not unchanged:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment_id": "round153_temperature_early20_adaptive_background_group_loo_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1901, 1920],
        "east_full_six_member_background_trend_degC_per_century": full_trends["east_ge90"],
        "top_absolute_group_effect": {
            "proxy_network": int(top.proxy_network),
            "group_family": str(top.group_family),
            "group": str(top.group),
            "group_record_count": int(top.group_record_count),
            "warming_effect_degC_per_century": float(top.group_warming_effect_degC_per_century),
            "effect_fraction_of_full_trend": float(top.effect_fraction_of_full_trend),
        },
        "single_group_explains_at_least_half_full_trend": not distributed,
        "interpretation": "distributed_group_sensitivity" if distributed else "one_predefined_group_has_large_sensitivity",
        "round152_network_trend_closure": network_closure,
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_rerun": False,
        "oggm_run": False,
        "source_hashes_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 153 Adaptive-Background Group Leave-Out\n\n"
        f"- East full background trend: `{full_trends['east_ge90']:+.3f} degC/century`.\n"
        f"- Largest absolute leave-out effect: `proxy{int(top.proxy_network)} / {top.group}` = `{float(top.group_warming_effect_degC_per_century):+.3f} degC/century`.\n"
        f"- Fraction of full trend: `{float(top.effect_fraction_of_full_trend):+.1%}`.\n"
        f"- Interpretation: `{summary['interpretation']}`.\n"
        "- Effects from overlapping group families are not additive.\n"
        "- Locked 1921--1939 accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
