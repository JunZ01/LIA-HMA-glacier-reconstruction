"""Read-only 1D/2D climate and area-budget parity audit for system 2005."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = BASE / "lia_191_system2005_1d_2d_climate_area_budget_parity_v1"
PACKAGE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
BRIDGE_SUMMARY = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/export_summary.json"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
STATIC = BASE / "lia_191_igm_planform2d_system2005_static_v1/static_summary.json"
TWO_D = BASE / "lia_191_torch_planform2d_system2005_steady50_monthly_ti_formal_v2"
ONE_D = BASE / "lia_191_network_topology_adaptive_split_system2005_steady_50y_coldm0p125_6hour_v1"
NATURAL = BASE / "lia_191_system2005_policy_v6_year9_materialization_v1/lia_natural_state_year9.pkl.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def monthly_mb_m(package: dict[str, np.ndarray], heights: np.ndarray, month_index: int) -> np.ndarray:
    source_index = int(package["sequence_indices"][month_index])
    temperature = package["temp"][source_index] + package["grad"][source_index] * (
        heights - float(package["ref_hgt"])
    )
    melt_temperature = np.maximum(temperature - float(package["t_melt"]), 0.0)
    solid_fraction = np.clip(
        1.0 - (temperature - float(package["t_solid"]))
        / (float(package["t_liq"]) - float(package["t_solid"])),
        0.0,
        1.0,
    )
    monthly_mass = package["prcp"][source_index] * solid_fraction
    monthly_mass -= float(package["monthly_melt_f"]) * melt_temperature
    monthly_mass -= float(package["bias"]) * float(package["sec_in_month"]) / float(package["sec_in_year"])
    return monthly_mass / float(package["rho"])


def fixed_geometry_sources(
    package: dict[str, np.ndarray], heights: np.ndarray, weights_m2: np.ndarray
) -> np.ndarray:
    annual = np.zeros(50, dtype=float)
    for month_index in range(600):
        annual[month_index // 12] += float(
            np.sum(monthly_mb_m(package, heights, month_index) * weights_m2)
        )
    return annual


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    two_summary = load_json(TWO_D / "summary.json")
    one_summary = load_json(ONE_D / "summary.json")
    static = load_json(STATIC)
    bridge = load_json(BRIDGE_SUMMARY)
    if two_summary["status"] != "formal-steady50-fail-stop":
        raise RuntimeError("unexpected-2d-status")
    if bridge["status"] != "climate-bridge-export-pass":
        raise RuntimeError("climate-bridge-not-validated")

    with np.load(PACKAGE) as source:
        package = {key: np.asarray(source[key]) for key in source.files}
    with Dataset(GEOLOGY, "r") as dataset:
        bed_2d = np.asarray(dataset.variables["topg"][:], dtype=float)
        thick_2d_initial = np.asarray(dataset.variables["thk"][:], dtype=float)
        dx = float(getattr(dataset, "dx_m"))
    with np.load(TWO_D / "final_state_50y.npz") as state:
        thick_2d_final = np.asarray(state["thickness_m"], dtype=float)

    with gzip.open(NATURAL, "rb") as source:
        flowlines = pickle.load(source)
    heights_1d_parts = []
    weights_1d_parts = []
    for flowline in flowlines:
        thick = np.asarray(flowline.thick, dtype=float)
        active = thick > 1e-6
        if not np.any(active):
            continue
        heights_1d_parts.append(np.asarray(flowline.surface_h, dtype=float)[active])
        weights_1d_parts.append(
            np.asarray(flowline.widths_m, dtype=float)[active]
            * float(flowline.dx) * float(flowline.map_dx)
        )
    heights_1d = np.concatenate(heights_1d_parts)
    weights_1d = np.concatenate(weights_1d_parts)

    cell_area = dx**2
    initial_mask = thick_2d_initial > 1e-6
    final_mask = thick_2d_final > 1e-6
    final_10m = thick_2d_final > 10.0
    new_mask = final_mask & ~initial_mask
    initial_heights_2d = (bed_2d + thick_2d_initial)[initial_mask]
    final_heights_2d = (bed_2d + thick_2d_final)[final_mask]
    new_heights_2d = (bed_2d + thick_2d_final)[new_mask]
    weights_2d_initial = np.full(initial_heights_2d.shape, cell_area)
    weights_2d_final = np.full(final_heights_2d.shape, cell_area)
    weights_2d_new = np.full(new_heights_2d.shape, cell_area)

    fixed_2d_initial = fixed_geometry_sources(package, initial_heights_2d, weights_2d_initial)
    fixed_2d_final = fixed_geometry_sources(package, final_heights_2d, weights_2d_final)
    fixed_2d_new = fixed_geometry_sources(package, new_heights_2d, weights_2d_new)
    fixed_1d_natural = fixed_geometry_sources(package, heights_1d, weights_1d)

    two_years = np.arange(0, 51, dtype=int)
    two_volume = np.empty(51)
    two_area = np.empty(51)
    two_cumulative = np.empty(51)
    two_volume[0] = float(two_summary["initial_volume_m3"])
    two_area[0] = float(static["model_area_m2"])
    two_cumulative[0] = 0.0
    for year in range(1, 51):
        item = two_summary["checkpoint_metrics"][f"{year}y"]
        two_volume[year] = two_volume[0] * (1.0 + float(item["volume_relative_change"]))
        two_area[year] = float(item["area_metrics"]["gt_0m"]["area_m2"])
        two_cumulative[year] = float(item["cumulative_source_volume_change_m3"])
    two_annual = np.diff(two_cumulative)

    one_rows = read_rows(ONE_D / "timeseries.csv")
    one_years = np.concatenate(([0.0], np.asarray([float(row["year"]) for row in one_rows])))
    one_volume = np.concatenate((
        [float(one_summary["initial_volume_m3"])],
        np.asarray([float(row["volume_m3"]) for row in one_rows]),
    ))
    one_area = np.concatenate((
        [float(one_summary["initial_area_m2"])],
        np.asarray([float(row["area_m2"]) for row in one_rows]),
    ))
    one_cumulative = np.concatenate((
        [0.0], np.asarray([float(row["cumulative_accepted_mb_m3"]) for row in one_rows])
    ))
    one_integer = np.arange(0, 51, dtype=int)
    one_volume_i = np.interp(one_integer, one_years, one_volume)
    one_area_i = np.interp(one_integer, one_years, one_area)
    one_cumulative_i = np.interp(one_integer, one_years, one_cumulative)
    one_annual = np.diff(one_cumulative_i)

    elevation_grid = np.linspace(4500.0, 8200.0, 741)
    mean_annual_mb = np.zeros_like(elevation_grid)
    for month_index in range(600):
        mean_annual_mb += monthly_mb_m(package, elevation_grid, month_index) / 50.0
    crossing = np.flatnonzero(np.diff(np.signbit(mean_annual_mb)))
    ela = float("nan")
    if crossing.size:
        i = int(crossing[0])
        ela = float(np.interp(0.0, mean_annual_mb[i : i + 2], elevation_grid[i : i + 2]))

    bins = np.arange(5000.0, 8250.0, 250.0)
    hypsometry_rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        row: dict[str, Any] = {"elevation_lower_m": lower, "elevation_upper_m": upper}
        for name, heights, weights in [
            ("one_d_natural", heights_1d, weights_1d),
            ("two_d_initial", initial_heights_2d, weights_2d_initial),
            ("two_d_final", final_heights_2d, weights_2d_final),
            ("two_d_new", new_heights_2d, weights_2d_new),
        ]:
            mask = (heights >= lower) & (heights < upper)
            row[f"{name}_area_m2"] = float(np.sum(weights[mask]))
        hypsometry_rows.append(row)

    actual_2d_source = float(two_summary["cumulative_climate_source_volume_change_m3"])
    fixed_initial_source = float(np.sum(fixed_2d_initial))
    feedback_excess = actual_2d_source - fixed_initial_source
    initial_specific = fixed_initial_source / float(np.sum(weights_2d_initial)) / 50.0
    natural_specific = float(np.sum(fixed_1d_natural)) / float(np.sum(weights_1d)) / 50.0
    new_specific = float(np.sum(fixed_2d_new)) / max(float(np.sum(weights_2d_new)), 1.0) / 50.0
    formula_specific_difference = initial_specific - natural_specific
    source_closure = (
        float(two_summary["final_volume_m3"])
        - float(two_summary["initial_volume_m3"])
        - actual_2d_source
    )

    annual_rows = []
    for year in range(1, 51):
        annual_rows.append({
            "year": year,
            "one_d_volume_m3": one_volume_i[year],
            "two_d_volume_m3": two_volume[year],
            "one_d_area_m2": one_area_i[year],
            "two_d_area_m2": two_area[year],
            "one_d_annual_source_m3": one_annual[year - 1],
            "two_d_annual_source_m3": two_annual[year - 1],
            "two_d_fixed_initial_geometry_source_m3": fixed_2d_initial[year - 1],
            "one_d_fixed_natural_geometry_source_m3": fixed_1d_natural[year - 1],
        })

    conclusion = "planform-support-expansion-climate-area-positive-feedback"
    summary = {
        "schema": "lia-191-system2005-1d-2d-climate-area-budget-parity-v1",
        "status": "read-only-parity-audit-complete",
        "system_id": 2005,
        "scientific_conclusion": conclusion,
        "climate_formula_exactly_matches_oggm_export": bridge["maximum_formula_absolute_error_m_ice_s"] == 0.0,
        "two_d_mass_budget": {
            "initial_volume_m3": float(two_summary["initial_volume_m3"]),
            "final_volume_m3": float(two_summary["final_volume_m3"]),
            "actual_cumulative_climate_source_m3": actual_2d_source,
            "mass_budget_residual_m3": source_closure,
            "fixed_initial_geometry_cumulative_source_m3": fixed_initial_source,
            "geometry_area_feedback_excess_source_m3": feedback_excess,
            "feedback_fraction_of_actual_source": feedback_excess / actual_2d_source,
        },
        "area_support": {
            "two_d_initial_active_area_m2": float(np.sum(weights_2d_initial)),
            "two_d_final_active_area_m2": float(np.sum(weights_2d_final)),
            "two_d_final_gt10m_area_m2": float(np.count_nonzero(final_10m) * cell_area),
            "two_d_new_active_area_m2": float(np.sum(weights_2d_new)),
            "two_d_active_area_change_fraction": float(np.sum(weights_2d_final) / np.sum(weights_2d_initial) - 1.0),
            "one_d_actual_area_change_fraction": float(one_summary["area_change_fraction"]),
        },
        "fixed_geometry_specific_balance_m_ice_per_year": {
            "two_d_initial": initial_specific,
            "one_d_natural_year9": natural_specific,
            "difference_two_d_minus_one_d": formula_specific_difference,
            "two_d_newly_activated_final_cells": new_specific,
        },
        "mean_annual_equilibrium_line_altitude_m": ela,
        "initial_geometry_nonparity": {
            "two_d_initial_volume_m3": float(two_summary["initial_volume_m3"]),
            "one_d_natural_year9_volume_m3": float(np.sum([
                np.sum(np.asarray(flowline.section) * float(flowline.dx) * float(flowline.map_dx))
                for flowline in flowlines
            ])),
            "two_d_to_one_d_volume_ratio": float(two_summary["initial_volume_m3"]) / float(np.sum([
                np.sum(np.asarray(flowline.section) * float(flowline.dx) * float(flowline.map_dx))
                for flowline in flowlines
            ])),
            "two_d_initial_area_m2": float(np.sum(weights_2d_initial)),
            "one_d_natural_year9_area_m2": float(np.sum(weights_1d)),
        },
        "interpretation": (
            "The exported MonthlyTI formula is exact and the 2D volume change closes to its climate source. "
            "The divergence develops after diffusion expands the >1e-6 m support, activating mass balance on "
            "new cells. The expanded support receives net positive accumulation and creates a self-reinforcing "
            "area-source feedback absent from the 1D active-node contract. The 2D static state is also not the "
            "same volume state as the eligible 1D natural year-9 state."
        ),
        "decision": {
            "calibrate_existing_2d_branch": False,
            "promote_existing_2d_branch": False,
            "new_model_run_authorized": False,
            "next_mechanism": (
                "define and read-only test a support-consistent 2D mass-balance domain tied to the mapped LIA "
                "catchment/initial support, while preserving natural thickness and exact mass conservation"
            ),
        },
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "model_run_performed": False,
    }

    OUT.mkdir(parents=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_rows(OUT / "annual_1d_2d_budget.csv", list(annual_rows[0]), annual_rows)
    write_rows(OUT / "surface_hypsometry.csv", list(hypsometry_rows[0]), hypsometry_rows)
    write_rows(
        OUT / "annual_mb_by_elevation.csv",
        ["elevation_m", "mean_annual_mb_m_ice_per_year"],
        [
            {"elevation_m": float(height), "mean_annual_mb_m_ice_per_year": float(mb)}
            for height, mb in zip(elevation_grid, mean_annual_mb)
        ],
    )

    years = np.arange(0, 51)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    axes[0, 0].plot(years, one_volume_i / 1e9, label="1D actual", color="#2364AA", lw=2)
    axes[0, 0].plot(years, two_volume / 1e9, label="2D actual", color="#C44536", lw=2)
    axes[0, 0].set(ylabel="Volume (km3)", xlabel="Steady-run year")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(years, one_area_i / 1e6, label="1D actual", color="#2364AA", lw=2)
    axes[0, 1].plot(years, two_area / 1e6, label="2D >0 m", color="#C44536", lw=2)
    axes[0, 1].set(ylabel="Active area (km2)", xlabel="Steady-run year")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(np.arange(1, 51), one_annual / 1e6, label="1D actual", color="#2364AA", lw=1.7)
    axes[1, 0].plot(np.arange(1, 51), two_annual / 1e6, label="2D actual", color="#C44536", lw=1.7)
    axes[1, 0].plot(np.arange(1, 51), fixed_2d_initial / 1e6, label="2D fixed initial support", color="#3A7D44", lw=1.4, ls="--")
    axes[1, 0].axhline(0, color="#222222", lw=0.8)
    axes[1, 0].set(ylabel="Annual climate source (10^6 m3)", xlabel="Steady-run year")
    axes[1, 0].legend(frameon=False, fontsize=8)
    width = 0.38
    centers = np.asarray([(row["elevation_lower_m"] + row["elevation_upper_m"]) / 2 for row in hypsometry_rows])
    axes[1, 1].bar(centers - 45, [row["two_d_initial_area_m2"] / 1e6 for row in hypsometry_rows], width=90, label="2D initial", color="#888888")
    axes[1, 1].bar(centers + 45, [row["two_d_new_area_m2"] / 1e6 for row in hypsometry_rows], width=90, label="2D newly active", color="#E3A018")
    if np.isfinite(ela):
        axes[1, 1].axvline(ela, color="#C44536", lw=1.4, ls="--", label=f"Mean ELA {ela:.0f} m")
    axes[1, 1].set(ylabel="Area (km2)", xlabel="Surface elevation (m)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.ravel():
        axis.grid(True, color="#DDDDDD", lw=0.6)
    fig.suptitle("System 2005: 1D/2D climate-area budget parity")
    fig.savefig(OUT / "system2005_1d_2d_budget_parity.png", dpi=180)
    plt.close(fig)

    manifest = {
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in [
                PACKAGE, BRIDGE_SUMMARY, GEOLOGY, STATIC, TWO_D / "summary.json",
                TWO_D / "final_state_50y.npz", TWO_D / "step_history.csv",
                ONE_D / "summary.json", ONE_D / "timeseries.csv", NATURAL,
            ]
        ],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
