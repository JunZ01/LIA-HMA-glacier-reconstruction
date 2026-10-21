"""Confirm the 1500-1899 HMA temperature warm bias with frozen curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_lia_warm_bias_confirmation_v1"
PUBLIC_REFERENCE_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
FIVE_FAMILY = LOOP / (
    "round_72_mode_ra_five_family_temperature_reference_v1/"
    "temperature_reference_five_family_coarse_allowed_v1.nc"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
GTNG_ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
GTNG_SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
PROJECT_HMA_CURVES = LOOP / (
    "round_23_corrected_reconstruction_reference_envelope_v1/"
    "corrected_reconstruction_reference_curves.csv"
)
PROJECT_GTNG15_CURVES = RESULTS / (
    "climate_reconstruction_temperature_gtng15_curves_1500_2000_v1/"
    "temperature_gtng15_curves_1500_2000.csv"
)
PROJECT_GTNG15_MANIFEST = RESULTS / (
    "climate_reconstruction_temperature_gtng15_curves_1500_2000_v1/"
    "source_manifest.json"
)

WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
FAMILY_ORDER = ["local_enkf", "lmr", "neukom", "phyda"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window_mean(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end)
    if int(selected.sum()) != 50 or not np.all(np.isfinite(values[selected])):
        raise RuntimeError(f"Window coverage failed: {start}-{end}")
    return float(np.mean(values[selected]))


def read_project_series(
    years: np.ndarray, support: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    hma = pd.read_csv(PROJECT_HMA_CURVES)
    if not np.array_equal(hma["year"].to_numpy(int), years):
        raise RuntimeError("Frozen HMA project curve year contract failed")
    project = {"hma": hma["corrected_six_member_hma_degC"].to_numpy(float)}
    names = {"hma": "High Mountain Asia"}

    regional = pd.read_csv(PROJECT_GTNG15_CURVES)
    expected_years = np.arange(1500, 2001)
    for region in support.itertuples(index=False):
        rows = regional[regional["o2region"].eq(region.o2region)].sort_values("year")
        if not np.array_equal(rows["year"].to_numpy(int), expected_years):
            raise RuntimeError(f"Frozen GTN-G curve year contract failed: {region.o2region}")
        values = np.full(years.shape, np.nan)
        values[np.isin(years, expected_years)] = rows["project_six_member_degC"].to_numpy(float)
        project[region.o2region] = values
        names[region.o2region] = region.full_name
    return project, names


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "public_reference_source": PUBLIC_REFERENCE_SOURCE,
        "five_family_reference": FIVE_FAMILY,
        "hma_mask": HMA_MASK,
        "gtng_assignment": GTNG_ASSIGNMENT,
        "gtng_support": GTNG_SUPPORT,
        "project_hma_curves": PROJECT_HMA_CURVES,
        "project_gtng15_curves": PROJECT_GTNG15_CURVES,
        "project_gtng15_manifest": PROJECT_GTNG15_MANIFEST,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    before = {name: sha256(path) for name, path in sources.items()}

    years = np.arange(1000, 2001)
    with h5py.File(PUBLIC_REFERENCE_SOURCE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        latitude = np.asarray(handle["lat"], dtype=float).ravel()
        longitude = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(source_years, years):
            raise RuntimeError("Reference source year contract failed")
        members = {
            name: round21.read_allowed_member(handle[variable], years)
            for name, variable in round21.MEMBERS.items()
        }
    families = {
        "local_enkf": round21.finite_mean([members["HGAO"], members["O"]]),
        "lmr": members["LMR"],
        "neukom": round21.finite_mean(
            [members[name] for name in members if name.startswith("Nk_")]
        ),
        "phyda": members["PHYDA"],
    }

    hma_mask = np.load(HMA_MASK).astype(bool)
    if hma_mask.shape != (len(latitude), len(longitude)):
        raise RuntimeError("HMA mask/reference grid shape mismatch")
    assignment = pd.read_csv(GTNG_ASSIGNMENT)
    support = pd.read_csv(GTNG_SUPPORT).sort_values("o2region").reset_index(drop=True)
    if len(support) != 15:
        raise RuntimeError("GTN-G 15-region contract failed")
    domain_masks: dict[str, np.ndarray] = {"hma": hma_mask}
    for region in support.itertuples(index=False):
        cells = assignment[assignment["o2region"].eq(region.o2region)]
        mask = np.zeros_like(hma_mask)
        mask[cells["grid_row"].to_numpy(int), cells["grid_col"].to_numpy(int)] = True
        if int(mask.sum()) != int(region.hma_grid_cell_count):
            raise RuntimeError(f"GTN-G mask closure failed: {region.o2region}")
        domain_masks[region.o2region] = mask

    project_series, domain_names = read_project_series(years, support)
    family_series = {
        domain: {
            family: round21.weighted_series(field, mask, latitude)
            for family, field in families.items()
        }
        for domain, mask in domain_masks.items()
    }

    rows: list[dict[str, object]] = []
    for domain in domain_masks:
        for start, end in WINDOWS:
            family_means = {
                family: window_mean(years, family_series[domain][family], start, end)
                for family in FAMILY_ORDER
            }
            project_mean = window_mean(years, project_series[domain], start, end)
            central = float(np.mean(list(family_means.values())))
            row: dict[str, object] = {
                "domain": domain,
                "domain_name": domain_names[domain],
                "start_year": start,
                "end_year": end,
                "project_mean_degC": project_mean,
                "four_family_central_degC": central,
                "project_minus_four_family_central_degC": project_mean - central,
                "warmer_than_individual_family_count": int(
                    sum(project_mean > value for value in family_means.values())
                ),
                "project_above_four_family_range": bool(project_mean > max(family_means.values())),
            }
            for family, value in family_means.items():
                row[f"{family}_mean_degC"] = value
                row[f"project_minus_{family}_degC"] = project_mean - value
                loo = float(np.mean([v for name, v in family_means.items() if name != family]))
                row[f"loo_without_{family}_central_degC"] = loo
                row[f"project_minus_loo_without_{family}_degC"] = project_mean - loo
            rows.append(row)
    window_table = pd.DataFrame(rows)
    window_table.to_csv(OUTPUT / "four_family_gtng15_50year_confirmation.csv", index=False)

    with h5py.File(FIVE_FAMILY, "r") as handle:
        five_years = np.asarray(handle["time"], dtype=int).ravel()
        five_values = np.asarray(handle["temperature_reference_five_family"], dtype=float)
        cell_latitude = np.asarray(handle["lat"], dtype=float).ravel()
        cell_count = np.asarray(handle["fine_hma_cell_count"], dtype=float).ravel()
    weights = np.cos(np.deg2rad(cell_latitude)) * cell_count
    valid = np.isfinite(five_values)
    five_series = np.nansum(five_values * weights[None, :], axis=1) / np.where(
        np.sum(valid * weights[None, :], axis=1) == 0,
        np.nan,
        np.sum(valid * weights[None, :], axis=1),
    )
    five_rows: list[dict[str, object]] = []
    for start, end in WINDOWS:
        project_mean = window_mean(years, project_series["hma"], start, end)
        five_mean = window_mean(five_years, five_series, start, end)
        five_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "project_mean_degC": project_mean,
                "five_family_mean_degC": five_mean,
                "project_minus_five_family_degC": project_mean - five_mean,
            }
        )
    five_table = pd.DataFrame(five_rows)
    five_table.to_csv(OUTPUT / "five_family_hma_50year_confirmation.csv", index=False)

    hma = window_table[window_table["domain"].eq("hma")].sort_values("start_year")
    regional = window_table[~window_table["domain"].eq("hma")].copy()
    loo_positive_counts = {
        family: int((hma[f"project_minus_loo_without_{family}_degC"] > 0).sum())
        for family in FAMILY_ORDER
    }
    regional_positive_counts = (
        regional.assign(positive=regional["project_minus_four_family_central_degC"] > 0)
        .groupby("start_year")["positive"]
        .sum()
        .astype(int)
    )
    gates = {
        "hma_four_family_positive_all_8_windows": bool(
            (hma["project_minus_four_family_central_degC"] > 0).all()
        ),
        "hma_four_family_median_gap_at_least_0p20_degC": bool(
            hma["project_minus_four_family_central_degC"].median() >= 0.20
        ),
        "every_loo_positive_at_least_7_of_8_windows": bool(
            min(loo_positive_counts.values()) >= 7
        ),
        "warmer_than_at_least_3_individual_families_in_at_least_7_windows": bool(
            (hma["warmer_than_individual_family_count"] >= 3).sum() >= 7
        ),
        "at_least_12_of_15_regions_positive_in_at_least_7_windows": bool(
            (regional_positive_counts >= 12).sum() >= 7
        ),
        "five_family_positive_at_least_7_of_8_windows": bool(
            (five_table["project_minus_five_family_degC"] > 0).sum() >= 7
        ),
        "five_family_median_gap_at_least_0p20_degC": bool(
            five_table["project_minus_five_family_degC"].median() >= 0.20
        ),
    }
    confirmed = bool(all(gates.values()))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    centers = (hma["start_year"].to_numpy() + hma["end_year"].to_numpy()) / 2.0
    axes[0].plot(
        centers,
        hma["project_mean_degC"],
        color="#D55E00",
        marker="o",
        linewidth=2.4,
        label="Project six-member",
    )
    colors = {
        "local_enkf": "#0072B2",
        "lmr": "#009E73",
        "neukom": "#CC79A7",
        "phyda": "#E69F00",
    }
    for family in FAMILY_ORDER:
        axes[0].plot(
            centers,
            hma[f"{family}_mean_degC"],
            color=colors[family],
            marker=".",
            linewidth=1.35,
            label=family,
        )
    axes[0].plot(
        centers,
        hma["four_family_central_degC"],
        color="#222222",
        linewidth=2.0,
        linestyle="--",
        label="Four-family central",
    )
    axes[0].plot(
        centers,
        five_table["five_family_mean_degC"],
        color="#666666",
        linewidth=2.0,
        linestyle=":",
        label="Five-family central",
    )
    axes[0].axhline(0, color="#444444", linewidth=0.7)
    axes[0].set(title="HMA 50-year temperature means", ylabel="Temperature anomaly (degC)")
    axes[0].legend(ncol=4, frameon=False)

    region_order = support["o2region"].tolist()
    matrix = regional.pivot(
        index="domain", columns="start_year", values="project_minus_four_family_central_degC"
    ).loc[region_order]
    image = axes[1].imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[1].set_yticks(
        np.arange(len(region_order)),
        [f"{code} {domain_names[code]}" for code in region_order],
    )
    axes[1].set_xticks(
        np.arange(len(WINDOWS)),
        [f"{start}-{end}" for start, end in WINDOWS],
        rotation=30,
        ha="right",
    )
    axes[1].set_title("Project minus four-family central by GTN-G region")
    fig.colorbar(image, ax=axes[1], shrink=0.82, label="Temperature difference (degC)")

    region_median = (
        regional.groupby(["domain", "domain_name"])["project_minus_four_family_central_degC"]
        .median()
        .reset_index()
        .set_index("domain")
        .loc[region_order]
        .reset_index()
    )
    axes[2].bar(
        region_median["domain"],
        region_median["project_minus_four_family_central_degC"],
        color="#D55E00",
    )
    axes[2].axhline(0.20, color="#333333", linewidth=1.0, linestyle="--", label="Material scale: +0.20 degC")
    axes[2].axhline(0, color="#444444", linewidth=0.7)
    axes[2].set(
        title="Median 1500-1899 regional warm gap",
        xlabel="GTN-G region",
        ylabel="Project minus reference (degC)",
    )
    axes[2].legend(frameon=False)
    for axis in (axes[0], axes[2]):
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "temperature_lia_warm_bias_confirmation.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "driver_version": "v2_frozen_project_curves",
        "period": [1500, 1899],
        "window_count": len(WINDOWS),
        "reference_families": FAMILY_ORDER,
        "gtng_region_count": 15,
        "hma_four_family_median_project_minus_reference_degC": float(
            hma["project_minus_four_family_central_degC"].median()
        ),
        "hma_five_family_median_project_minus_reference_degC": float(
            five_table["project_minus_five_family_degC"].median()
        ),
        "hma_project_above_four_family_range_window_count": int(
            hma["project_above_four_family_range"].sum()
        ),
        "loo_positive_window_counts": loo_positive_counts,
        "regional_positive_counts_by_window": {
            str(int(year)): int(count) for year, count in regional_positive_counts.items()
        },
        "regional_median_gap_range_degC": [
            float(region_median["project_minus_four_family_central_degC"].min()),
            float(region_median["project_minus_four_family_central_degC"].max()),
        ],
        "gates": gates,
        "systematic_lia_warm_bias_confirmed": confirmed,
        "decision": (
            "authorize_regional_prior_update_innovation_decomposition"
            if confirmed
            else "stop_bias_correction_as_reference_dependent"
        ),
        "locked_1921_1939_accessed": False,
        "phase_problem_used_for_tuning": False,
        "interpretation_limit": (
            "Confirmation concerns robust regional-mean direction and material scale. "
            "It does not validate every reference pixel and does not authorize empirical subtraction."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# LIA Temperature Warm-Bias Confirmation",
        "",
        f"- Systematic warm bias confirmed: `{str(confirmed).lower()}`.",
        f"- HMA median project minus four-family central: `{summary['hma_four_family_median_project_minus_reference_degC']:+.3f} degC`.",
        f"- HMA median project minus five-family central: `{summary['hma_five_family_median_project_minus_reference_degC']:+.3f} degC`.",
        f"- Project above full four-family range: `{summary['hma_project_above_four_family_range_window_count']}/8` windows.",
        f"- Regional median-gap range: `{summary['regional_median_gap_range_degC'][0]:+.3f}` to `{summary['regional_median_gap_range_degC'][1]:+.3f} degC`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in sources.items()}
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "sources_unchanged": before == after,
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
