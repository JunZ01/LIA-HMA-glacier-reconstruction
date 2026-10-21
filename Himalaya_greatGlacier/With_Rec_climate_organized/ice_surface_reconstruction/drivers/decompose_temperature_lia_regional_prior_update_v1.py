"""Map the confirmed LIA warm bias into background and EnKF increments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_lia_regional_prior_update_decomposition_v1"
ARCHIVE = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
RESULT_ROOT = ARCHIVE / "fixed_proxy_experiments_69/tas/results"
SOURCE_RECOVERY = RESULTS / "enkf_reconstruction_source_recovery_audit_v2"
PUBLIC_REFERENCE_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
GTNG_ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
GTNG_SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"

MEMBERS = round25.MEMBERS
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
REQUESTED_YEARS = np.r_[np.arange(1500, 1900), np.arange(1961, 1991)]
FAMILY_ORDER = ["local_enkf", "lmr", "neukom", "phyda"]


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def window_mean(values: np.ndarray, start: int, end: int) -> float:
    selected = (REQUESTED_YEARS >= start) & (REQUESTED_YEARS <= end)
    if int(selected.sum()) != 50 or not np.all(np.isfinite(values[selected])):
        raise RuntimeError(f"Window coverage failed: {start}-{end}")
    return float(np.mean(values[selected]))


def mechanism(background_gap: float, increment: float, posterior_gap: float) -> str:
    if increment < 0:
        return "increment_offsets_background"
    if posterior_gap <= 0:
        return "no_positive_warm_gap"
    background_share = background_gap / posterior_gap
    increment_share = increment / posterior_gap
    if background_share >= 0.75:
        return "background_dominant"
    if increment_share >= 0.75:
        return "increment_dominant"
    return "joint"


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "prior": SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": ARCHIVE / "python_data/proxy_data_6.mat",
        "proxy9": ARCHIVE / "python_data/proxy_data_9.mat",
        "archive_config": ARCHIVE / "fixed_proxy_experiments_69/tas/experiment_config.mat",
        "public_reference_source": PUBLIC_REFERENCE_SOURCE,
        "hma_mask": HMA_MASK,
        "gtng_assignment": GTNG_ASSIGNMENT,
        "gtng_support": GTNG_SUPPORT,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for member, (method, network) in MEMBERS.items():
        archive_method = {"AO": "AOEnKF", "HGAO": "HGAOEnKF", "O": "EnKF"}[method]
        paths[f"posterior_{member}"] = RESULT_ROOT / f"proxy{network}/{archive_method}_results.mat"
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in paths.items()}

    config = scipy.io.loadmat(
        paths["archive_config"], squeeze_me=True, struct_as_record=False
    )["config"]
    recovered_config = {
        "num_experiments": int(config.num_experiments),
        "analysis_ensemble_size": 100,
        "aoenkf_strategy": str(config.aoenkf_strategy),
        "alpha": float(config.alpha),
        "start_year": int(config.r_o),
        "end_year": int(config.r_f),
    }
    if recovered_config != {
        "num_experiments": 50,
        "analysis_ensemble_size": 100,
        "aoenkf_strategy": "C",
        "alpha": 0.3,
        "start_year": 1000,
        "end_year": 2000,
    }:
        raise RuntimeError(f"Archive config contract failed: {recovered_config}")

    prior = np.load(paths["prior"])
    if prior.shape != (336, 848):
        raise RuntimeError(f"Unexpected prior shape: {prior.shape}")
    with h5py.File(paths["indices"], "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if experiment_indices.shape != (50, 500):
        raise RuntimeError(f"Unexpected experiment index shape: {experiment_indices.shape}")

    proxies: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    locked_reads = 0
    for network in (6, 9):
        hxb, observations, network_locked_reads = round25.read_proxy_allowed(
            paths[f"proxy{network}"], REQUESTED_YEARS
        )
        proxies[network] = (hxb, observations)
        locked_reads += network_locked_reads
    if locked_reads != 0:
        raise RuntimeError("Locked 1921-1939 proxy values were read")

    adaptive_background: dict[int, np.ndarray] = {}
    for network in (6, 9):
        adaptive_background[network], _ = round25.ao_background_mean(
            prior, experiment_indices, proxies[network][0], proxies[network][1]
        )
    fixed_background_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]),
        axis=0,
    )
    fixed_background = np.repeat(
        fixed_background_one[None, :], len(REQUESTED_YEARS), axis=0
    )

    with h5py.File(PUBLIC_REFERENCE_SOURCE, "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        target_lat = np.asarray(handle["lat"], dtype=float).ravel()
        target_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(all_years, np.arange(1000, 2001)):
            raise RuntimeError("Reference year contract failed")
        reference_members = {
            name: round21.read_allowed_member(handle[variable], all_years)
            for name, variable in round21.MEMBERS.items()
        }
    reference_families = {
        "local_enkf": round21.finite_mean(
            [reference_members["HGAO"], reference_members["O"]]
        ),
        "lmr": reference_members["LMR"],
        "neukom": round21.finite_mean(
            [field for name, field in reference_members.items() if name.startswith("Nk_")]
        ),
        "phyda": reference_members["PHYDA"],
    }
    reference_central = round21.finite_mean(
        [reference_families[name] for name in FAMILY_ORDER]
    )
    requested_indices = REQUESTED_YEARS - 1000

    hma_mask = np.load(HMA_MASK).astype(bool)
    if hma_mask.shape != (len(target_lat), len(target_lon)):
        raise RuntimeError("HMA mask/reference grid shape mismatch")
    assignment = pd.read_csv(GTNG_ASSIGNMENT)
    support = pd.read_csv(GTNG_SUPPORT).sort_values("o2region").reset_index(drop=True)
    domain_masks: dict[str, np.ndarray] = {"hma": hma_mask}
    domain_names = {"hma": "High Mountain Asia"}
    for region in support.itertuples(index=False):
        cells = assignment[assignment["o2region"].eq(region.o2region)]
        mask = np.zeros_like(hma_mask)
        mask[cells["grid_row"].to_numpy(int), cells["grid_col"].to_numpy(int)] = True
        if int(mask.sum()) != int(region.hma_grid_cell_count):
            raise RuntimeError(f"GTN-G mask closure failed: {region.o2region}")
        domain_masks[region.o2region] = mask
        domain_names[region.o2region] = region.full_name

    reference_series = {
        domain: round21.weighted_series(
            reference_central[requested_indices], mask, target_lat
        )
        for domain, mask in domain_masks.items()
    }
    native_lat, native_lon = round25.native_coordinates()
    member_series: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for member, (method, network) in MEMBERS.items():
        posterior_field = round25.anomaly_field(
            round25.read_posterior_mean(paths[f"posterior_{member}"], REQUESTED_YEARS),
            REQUESTED_YEARS,
        )
        background_flat = fixed_background if method == "O" else adaptive_background[network]
        background_field = round25.anomaly_field(background_flat, REQUESTED_YEARS)
        increment_field = posterior_field - background_field
        projected = {
            "posterior": round25.interpolate_fields(
                posterior_field, native_lat, native_lon, target_lat, target_lon
            ),
            "background": round25.interpolate_fields(
                background_field, native_lat, native_lon, target_lat, target_lon
            ),
            "increment": round25.interpolate_fields(
                increment_field, native_lat, native_lon, target_lat, target_lon
            ),
        }
        member_series[member] = {
            domain: {
                component: round21.weighted_series(field, mask, target_lat)
                for component, field in projected.items()
            }
            for domain, mask in domain_masks.items()
        }

    ensemble_series = {
        domain: {
            component: np.mean(
                np.stack(
                    [member_series[member][domain][component] for member in MEMBERS]
                ),
                axis=0,
            )
            for component in ("posterior", "background", "increment")
        }
        for domain in domain_masks
    }

    annual_rows: list[dict[str, object]] = []
    for domain in domain_masks:
        for year_index, year in enumerate(REQUESTED_YEARS):
            annual_rows.append(
                {
                    "domain": domain,
                    "domain_name": domain_names[domain],
                    "year": int(year),
                    "reference_four_family_degC": reference_series[domain][year_index],
                    "posterior_six_member_degC": ensemble_series[domain]["posterior"][year_index],
                    "background_six_member_degC": ensemble_series[domain]["background"][year_index],
                    "increment_six_member_degC": ensemble_series[domain]["increment"][year_index],
                }
            )
    pd.DataFrame(annual_rows).to_csv(
        OUTPUT / "annual_gtng15_prior_update_curves.csv", index=False
    )

    window_rows: list[dict[str, object]] = []
    closure_errors: list[float] = []
    for domain in domain_masks:
        for branch in ["six_member", *MEMBERS.keys()]:
            values = ensemble_series[domain] if branch == "six_member" else member_series[branch][domain]
            method, network = ("ensemble", "6+9") if branch == "six_member" else MEMBERS[branch]
            for start, end in WINDOWS:
                reference_mean = window_mean(reference_series[domain], start, end)
                posterior_mean = window_mean(values["posterior"], start, end)
                background_mean = window_mean(values["background"], start, end)
                increment_mean = window_mean(values["increment"], start, end)
                posterior_gap = posterior_mean - reference_mean
                background_gap = background_mean - reference_mean
                closure = posterior_gap - background_gap - increment_mean
                closure_errors.append(abs(closure))
                window_rows.append(
                    {
                        "domain": domain,
                        "domain_name": domain_names[domain],
                        "branch": branch,
                        "method": method,
                        "proxy_network": network,
                        "start_year": start,
                        "end_year": end,
                        "reference_degC": reference_mean,
                        "posterior_degC": posterior_mean,
                        "background_degC": background_mean,
                        "increment_degC": increment_mean,
                        "posterior_minus_reference_degC": posterior_gap,
                        "background_minus_reference_degC": background_gap,
                        "closure_error_degC": closure,
                    }
                )
    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(OUTPUT / "gtng15_member_50year_decomposition.csv", index=False)
    closure_max = float(max(closure_errors))
    if closure_max > 1e-12:
        raise RuntimeError(f"Decomposition closure failed: {closure_max}")

    ensemble_windows = window_table[window_table["branch"].eq("six_member")].copy()
    region_rows: list[dict[str, object]] = []
    for domain in domain_masks:
        rows = ensemble_windows[ensemble_windows["domain"].eq(domain)]
        posterior_gap = float(rows["posterior_minus_reference_degC"].median())
        background_gap = float(rows["background_minus_reference_degC"].median())
        increment = float(rows["increment_degC"].median())
        region_rows.append(
            {
                "domain": domain,
                "domain_name": domain_names[domain],
                "median_posterior_minus_reference_degC": posterior_gap,
                "median_background_minus_reference_degC": background_gap,
                "median_increment_degC": increment,
                "background_share_of_positive_gap": (
                    background_gap / posterior_gap if posterior_gap > 0 else np.nan
                ),
                "increment_positive_window_count": int((rows["increment_degC"] > 0).sum()),
                "background_gap_positive_window_count": int(
                    (rows["background_minus_reference_degC"] > 0).sum()
                ),
                "mechanism": mechanism(background_gap, increment, posterior_gap),
            }
        )
    regional_summary = pd.DataFrame(region_rows)
    regional_summary.to_csv(OUTPUT / "regional_mechanism_summary.csv", index=False)

    branch_rows: list[dict[str, object]] = []
    for (domain, branch, method, network), rows in window_table.groupby(
        ["domain", "branch", "method", "proxy_network"], sort=False
    ):
        branch_rows.append(
            {
                "domain": domain,
                "domain_name": domain_names[domain],
                "branch": branch,
                "method": method,
                "proxy_network": network,
                "median_posterior_minus_reference_degC": float(
                    rows["posterior_minus_reference_degC"].median()
                ),
                "median_background_minus_reference_degC": float(
                    rows["background_minus_reference_degC"].median()
                ),
                "median_increment_degC": float(rows["increment_degC"].median()),
                "increment_positive_window_count": int((rows["increment_degC"] > 0).sum()),
            }
        )
    branch_summary = pd.DataFrame(branch_rows)
    branch_summary.to_csv(OUTPUT / "member_network_mechanism_summary.csv", index=False)

    regions_only = regional_summary[~regional_summary["domain"].eq("hma")].copy()
    region_order = support["o2region"].tolist()
    regions_only = regions_only.set_index("domain").loc[region_order].reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    x = np.arange(len(region_order))
    axes[0].bar(
        x,
        regions_only["median_background_minus_reference_degC"],
        color="#0072B2",
        label="Background minus reference",
    )
    axes[0].bar(
        x,
        regions_only["median_increment_degC"],
        bottom=regions_only["median_background_minus_reference_degC"],
        color="#D55E00",
        label="EnKF increment",
    )
    axes[0].scatter(
        x,
        regions_only["median_posterior_minus_reference_degC"],
        color="#111111",
        zorder=3,
        label="Posterior minus reference",
    )
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_xticks(x, region_order)
    axes[0].set_ylabel("Median 50-year contribution (degC)")
    axes[0].set_title("Regional closure of the 1500-1899 warm bias")
    axes[0].legend(frameon=False, ncol=3)

    matrix_background = ensemble_windows.pivot(
        index="domain", columns="start_year", values="background_minus_reference_degC"
    ).loc[region_order]
    image1 = axes[1].imshow(
        matrix_background.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7
    )
    axes[1].set_yticks(x, region_order)
    axes[1].set_xticks(
        np.arange(len(WINDOWS)), [f"{start}-{end}" for start, end in WINDOWS], rotation=30, ha="right"
    )
    axes[1].set_title("Background minus four-family reference")
    fig.colorbar(image1, ax=axes[1], shrink=0.82, label="degC")

    matrix_increment = ensemble_windows.pivot(
        index="domain", columns="start_year", values="increment_degC"
    ).loc[region_order]
    limit = max(0.15, float(np.nanmax(np.abs(matrix_increment.to_numpy()))))
    image2 = axes[2].imshow(
        matrix_increment.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axes[2].set_yticks(x, [f"{code} {domain_names[code]}" for code in region_order])
    axes[2].set_xticks(
        np.arange(len(WINDOWS)), [f"{start}-{end}" for start, end in WINDOWS], rotation=30, ha="right"
    )
    axes[2].set_title("EnKF increment by region and window")
    fig.colorbar(image2, ax=axes[2], shrink=0.82, label="degC")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "temperature_lia_regional_prior_update_decomposition.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    hma_summary = regional_summary[regional_summary["domain"].eq("hma")].iloc[0]
    mechanism_counts = regions_only["mechanism"].value_counts().to_dict()
    summary = {
        "status": "complete",
        "period": [1500, 1899],
        "window_count": 8,
        "region_count": 15,
        "closure_max_abs_degC": closure_max,
        "locked_1921_1939_values_read": False,
        "phase_problem_used_for_tuning": False,
        "hma_median_posterior_minus_reference_degC": float(
            hma_summary["median_posterior_minus_reference_degC"]
        ),
        "hma_median_background_minus_reference_degC": float(
            hma_summary["median_background_minus_reference_degC"]
        ),
        "hma_median_increment_degC": float(hma_summary["median_increment_degC"]),
        "hma_mechanism": str(hma_summary["mechanism"]),
        "regional_mechanism_counts": {str(k): int(v) for k, v in mechanism_counts.items()},
        "regions_background_gap_positive_all_8_windows": int(
            (regions_only["background_gap_positive_window_count"] == 8).sum()
        ),
        "regions_increment_positive_at_least_6_windows": int(
            (regions_only["increment_positive_window_count"] >= 6).sum()
        ),
        "decision": (
            "prior_low_frequency_state_is_primary_target"
            if str(hma_summary["mechanism"]) == "background_dominant"
            else "joint_prior_and_psm_update_target"
        ),
        "interpretation_limit": (
            "AO/HGAO yearly backgrounds are deterministic reconstructions from archived strategy-C; "
            "their selected-member indices were not archived. The experiment diagnoses production behavior "
            "but does not validate a replacement prior."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Regional LIA Temperature Prior/Update Decomposition",
        "",
        f"- HMA posterior minus reference: `{summary['hma_median_posterior_minus_reference_degC']:+.3f} degC`.",
        f"- HMA background minus reference: `{summary['hma_median_background_minus_reference_degC']:+.3f} degC`.",
        f"- HMA EnKF increment: `{summary['hma_median_increment_degC']:+.3f} degC`.",
        f"- HMA mechanism: `{summary['hma_mechanism']}`.",
        f"- Regional mechanisms: `{summary['regional_mechanism_counts']}`.",
        f"- Maximum closure error: `{summary['closure_max_abs_degC']:.3e} degC`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: file_state(path) for name, path in paths.items()}
    unchanged = all(before[name]["sha256"] == after[name]["sha256"] for name in paths)
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "sources_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
