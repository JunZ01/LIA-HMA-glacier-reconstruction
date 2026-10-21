"""Decompose the 1800-1900 transition mismatch into prior and EnKF update."""

from __future__ import annotations

import json

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import decompose_temperature_lia_regional_prior_update_v1 as prior_tools
import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


OUTPUT = prior_tools.RESULTS / "climate_reconstruction_temperature_transition_prior_update_1800_1900_v1"
HISTORICAL_YEARS = np.arange(1800, 1901)
REQUESTED_YEARS = np.r_[HISTORICAL_YEARS, np.arange(1961, 1991)]
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
PERIODS = {
    "baseline_1800_1860": (1800, 1860),
    "early_transition_1861_1880": (1861, 1880),
    "late_transition_1881_1900": (1881, 1900),
}


def period_mean(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end)
    return float(np.mean(values[selected]))


def trailing_mean(values: np.ndarray, width: int = 11) -> np.ndarray:
    return pd.Series(values).rolling(width, min_periods=width).mean().to_numpy(float)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Read reconstruction years 1800-1900 and modern baseline 1961-1990 only.\n"
        "- Decompose posterior-reference exactly as (background-reference) + EnKF increment.\n"
        "- Attribute transition using 1881-1900 minus 1800-1860 means.\n"
        "- Use frozen longitude regions; do not access 1901-1920 or 1921-1939.\n"
        "- No OGGM and no source modification.\n",
        encoding="utf-8",
    )
    paths = {
        "prior": prior_tools.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": prior_tools.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": prior_tools.ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": prior_tools.ARCHIVE / "python_data/proxy_data_9.mat",
        "public_reference_source": prior_tools.PUBLIC_REFERENCE_SOURCE,
        "hma_mask": prior_tools.HMA_MASK,
        "contract": contract,
    }
    for member, (method, network) in prior_tools.MEMBERS.items():
        archive_method = {"AO": "AOEnKF", "HGAO": "HGAOEnKF", "O": "EnKF"}[method]
        paths[f"posterior_{member}"] = prior_tools.RESULT_ROOT / f"proxy{network}/{archive_method}_results.mat"
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: prior_tools.file_state(path) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    proxies = {}
    locked_reads = 0
    for network in (6, 9):
        hxb, observations, count = round25.read_proxy_allowed(paths[f"proxy{network}"], REQUESTED_YEARS)
        proxies[network] = (hxb, observations)
        locked_reads += count
    if locked_reads:
        raise RuntimeError("Locked proxy values were read")

    adaptive_background = {}
    for network in (6, 9):
        adaptive_background[network], _ = round25.ao_background_mean(
            prior, experiment_indices, proxies[network][0], proxies[network][1]
        )
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed_background = np.repeat(fixed_one[None, :], len(REQUESTED_YEARS), axis=0)

    with h5py.File(paths["public_reference_source"], "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        target_lat = np.asarray(handle["lat"], dtype=float).ravel()
        target_lon = np.asarray(handle["lon"], dtype=float).ravel()
        reference_members = {
            name: round21.read_allowed_member(handle[variable], all_years)
            for name, variable in round21.MEMBERS.items()
        }
    families = {
        "local_enkf": round21.finite_mean([reference_members["HGAO"], reference_members["O"]]),
        "lmr": reference_members["LMR"],
        "neukom": round21.finite_mean(
            [field for name, field in reference_members.items() if name.startswith("Nk_")]
        ),
        "phyda": reference_members["PHYDA"],
    }
    reference = round21.finite_mean([families[name] for name in prior_tools.FAMILY_ORDER])
    reference = reference[REQUESTED_YEARS - 1000]

    hma_mask = np.load(paths["hma_mask"]).astype(bool)
    lon_grid = np.broadcast_to(target_lon[None, :], hma_mask.shape)
    masks = {}
    for name, (lower, upper) in REGIONS.items():
        mask = hma_mask if name == "hma" else hma_mask & (lon_grid >= lower) & (lon_grid < upper)
        if not mask.any():
            raise RuntimeError(f"Empty mask: {name}")
        masks[name] = mask

    reference_series = {
        name: round21.weighted_series(reference, mask, target_lat)
        for name, mask in masks.items()
    }
    native_lat, native_lon = round25.native_coordinates()
    member_series = {}
    for member, (method, network) in prior_tools.MEMBERS.items():
        posterior_field = round25.anomaly_field(
            round25.read_posterior_mean(paths[f"posterior_{member}"], REQUESTED_YEARS),
            REQUESTED_YEARS,
        )
        background_flat = fixed_background if method == "O" else adaptive_background[network]
        background_field = round25.anomaly_field(background_flat, REQUESTED_YEARS)
        projected_posterior = round25.interpolate_fields(
            posterior_field, native_lat, native_lon, target_lat, target_lon
        )
        projected_background = round25.interpolate_fields(
            background_field, native_lat, native_lon, target_lat, target_lon
        )
        member_series[member] = {
            region: {
                "posterior": round21.weighted_series(projected_posterior, mask, target_lat),
                "background": round21.weighted_series(projected_background, mask, target_lat),
            }
            for region, mask in masks.items()
        }

    hist = REQUESTED_YEARS <= 1900
    years = REQUESTED_YEARS[hist]
    annual_rows = []
    metric_rows = []
    plotted = {}
    max_closure = 0.0
    for region in masks:
        posterior = np.mean(
            np.stack([member_series[m][region]["posterior"] for m in prior_tools.MEMBERS]), axis=0
        )[hist]
        background = np.mean(
            np.stack([member_series[m][region]["background"] for m in prior_tools.MEMBERS]), axis=0
        )[hist]
        increment = posterior - background
        ref = reference_series[region][hist]
        background_gap = background - ref
        posterior_gap = posterior - ref
        closure = posterior_gap - (background_gap + increment)
        max_closure = max(max_closure, float(np.max(np.abs(closure))))
        baseline = (years >= 1800) & (years <= 1860)
        bg_adjusted = background_gap - np.mean(background_gap[baseline])
        inc_adjusted = increment - np.mean(increment[baseline])
        total_adjusted = posterior_gap - np.mean(posterior_gap[baseline])
        plotted[region] = {
            "total": trailing_mean(total_adjusted),
            "background": trailing_mean(bg_adjusted),
            "increment": trailing_mean(inc_adjusted),
        }
        for index, year in enumerate(years):
            annual_rows.append(
                {
                    "region": region,
                    "year": int(year),
                    "posterior_minus_reference_adjusted_degC": total_adjusted[index],
                    "background_minus_reference_adjusted_degC": bg_adjusted[index],
                    "increment_adjusted_degC": inc_adjusted[index],
                    "closure_degC": closure[index],
                }
            )
        base_bg = period_mean(years, background_gap, 1800, 1860)
        base_inc = period_mean(years, increment, 1800, 1860)
        for period, (start, end) in PERIODS.items():
            bg_change = period_mean(years, background_gap, start, end) - base_bg
            inc_change = period_mean(years, increment, start, end) - base_inc
            total_change = bg_change + inc_change
            metric_rows.append(
                {
                    "region": region,
                    "period": period,
                    "background_reference_change_degC": bg_change,
                    "enkf_increment_change_degC": inc_change,
                    "posterior_reference_change_degC": total_change,
                    "background_share_of_total": np.nan if abs(total_change) < 1e-12 else bg_change / total_change,
                    "increment_share_of_total": np.nan if abs(total_change) < 1e-12 else inc_change / total_change,
                }
            )

    pd.DataFrame(annual_rows).to_csv(OUTPUT / "transition_prior_update_annual_1800_1900.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "transition_prior_update_period_decomposition.csv", index=False)

    order = ["hma", "west_lt80", "central_80_90", "east_ge90"]
    labels = {"hma": "HMA", "west_lt80": "West (<80E)", "central_80_90": "Central (80-90E)", "east_ge90": "East (>=90E)"}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, region in zip(axes.flat, order):
        ax.plot(years, plotted[region]["total"], color="#111111", linewidth=2.3, label="Posterior - reference")
        ax.plot(years, plotted[region]["background"], color="#276FBF", linewidth=1.8, label="Background - reference")
        ax.plot(years, plotted[region]["increment"], color="#B92D27", linewidth=1.8, label="EnKF increment")
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_title(labels[region])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("11-year trailing change (degC)")
    axes[1, 0].set_ylabel("11-year trailing change (degC)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Late-19th-century temperature mismatch: background versus EnKF update")
    fig.savefig(OUTPUT / "temperature_transition_prior_update_decomposition.png", dpi=260)
    plt.close(fig)

    after = {name: prior_tools.file_state(path) for name, path in paths.items()}
    late = metrics.loc[metrics.period.eq("late_transition_1881_1900")].set_index("region")
    summary = {
        "status": "complete",
        "maximum_algebraic_closure_degC": max_closure,
        "late_transition_1881_1900": late.to_dict(orient="index"),
        "input_hashes_unchanged": all(before[name]["sha256"] == after[name]["sha256"] for name in paths),
        "years_read": [1800, 1900, 1961, 1990],
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(late.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
