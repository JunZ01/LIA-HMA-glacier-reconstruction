"""Adjudicate the project LIA temperature curve with modern-validated ModE-RA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = tuple((start, start + 49) for start in range(1450, 1900, 50))
FAMILY_COLUMNS = {
    "local_enkf": "reference_local_enkf_hma_degC",
    "lmr": "reference_lmr_hma_degC",
    "neukom": "reference_neukom_hma_degC",
    "phyda": "reference_phyda_hma_degC",
}
PROJECT_COLUMN = "corrected_six_member_hma_degC"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=1)
    return np.nansum(values * weights[None], axis=1) / np.where(denominator == 0, np.nan, denominator)


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half : index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_70_mode_ra_lia_regional_adjudication_v1"
    round69 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v3"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round23 = loop / "round_23_corrected_reconstruction_reference_envelope_v1"
    paths = {
        "mode_ra_temperature": round69 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "round69_summary": round69 / "summary.json",
        "round68_summary": round68 / "summary.json",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "project_reference_curves": round23 / "corrected_reconstruction_reference_curves.csv",
        "round23_summary": round23 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    round69_summary = json.loads(paths["round69_summary"].read_text(encoding="utf-8"))
    round68_summary = json.loads(paths["round68_summary"].read_text(encoding="utf-8"))
    if not round69_summary["passes_acquisition_preflight"]:
        raise RuntimeError("round69-acquisition-did-not-pass")
    if not round68_summary["variables"]["temperature"]["passes_spatial_field"]:
        raise RuntimeError("round68-mode-ra-temperature-modern-validation-did-not-pass")

    support = pd.read_csv(paths["coarse_support"])
    with h5py.File(paths["mode_ra_temperature"], "r") as handle:
        mode_years = np.asarray(handle["time"], dtype=int)
        mode_lat = np.asarray(handle["lat"], dtype=float)
        mode_lon = np.asarray(handle["lon"], dtype=float)
        mode_field = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    if not np.array_equal(mode_years, np.arange(1421, 1921)):
        raise RuntimeError("mode-ra-paleo-year-axis-failed")
    if np.any((mode_years >= 1921) & (mode_years <= 1939)):
        raise RuntimeError("locked-period-in-mode-ra-field")
    mode_cells = []
    weights = []
    for row in support.itertuples(index=False):
        i = np.flatnonzero(np.isclose(mode_lat, row.latitude, atol=1e-8))
        j = np.flatnonzero(np.isclose(mode_lon, row.longitude, atol=1e-8))
        if len(i) != 1 or len(j) != 1:
            raise RuntimeError(f"coarse-support-grid-match-failed:{row.latitude}:{row.longitude}")
        mode_cells.append(mode_field[:, int(i[0]), int(j[0])])
        weights.append(np.cos(np.deg2rad(row.latitude)) * float(row.fine_hma_cell_count))
    mode_curve = weighted_series(np.stack(mode_cells, axis=1), np.asarray(weights))

    curves = pd.read_csv(paths["project_reference_curves"])
    required = {"year", PROJECT_COLUMN, *FAMILY_COLUMNS.values()}
    if not required.issubset(curves.columns):
        raise RuntimeError(f"round23-columns-missing:{sorted(required - set(curves.columns))}")
    curve_years = curves.year.to_numpy(int)
    if np.any((curve_years >= 1921) & (curve_years <= 1939)):
        raise RuntimeError("locked-period-in-round23-curves")
    selected = (curve_years >= 1421) & (curve_years <= 1920)
    if not np.array_equal(curve_years[selected], mode_years):
        raise RuntimeError("mode-ra-round23-common-year-axis-failed")
    common = curves.loc[selected].reset_index(drop=True)
    common["mode_ra_hma_degC"] = mode_curve
    family_values = common[list(FAMILY_COLUMNS.values())].to_numpy(float)
    common["reference_four_family_central_degC"] = np.mean(family_values, axis=1)
    common["reference_four_family_min_degC"] = np.min(family_values, axis=1)
    common["reference_four_family_max_degC"] = np.max(family_values, axis=1)
    common.to_csv(output / "mode_ra_project_reference_annual_curves.csv", index=False)

    rows: list[dict[str, Any]] = []
    for start, end in WINDOWS:
        window = common[(common.year >= start) & (common.year <= end)]
        if len(window) != 50:
            raise RuntimeError(f"incomplete-fifty-year-window:{start}:{end}:{len(window)}")
        family_means = np.asarray(
            [float(window[column].mean()) for column in FAMILY_COLUMNS.values()]
        )
        mode_mean = float(window.mode_ra_hma_degC.mean())
        project_mean = float(window[PROJECT_COLUMN].mean())
        reference_central = float(np.mean(family_means))
        expanded = np.append(family_means, mode_mean)
        expanded_central = float(np.mean(expanded))
        expanded_std = float(np.std(expanded, ddof=1))
        rows.append(
            {
                "start_year": start,
                "end_year": end,
                "mode_ra_mean_degC": mode_mean,
                "project_corrected_mean_degC": project_mean,
                "reference_four_family_central_degC": reference_central,
                "reference_four_family_min_degC": float(np.min(family_means)),
                "reference_four_family_max_degC": float(np.max(family_means)),
                "reference_four_family_std_degC": float(np.std(family_means, ddof=1)),
                "mode_ra_minus_reference_central_degC": mode_mean - reference_central,
                "project_minus_reference_central_degC": project_mean - reference_central,
                "mode_ra_closer_to_reference_than_project": abs(mode_mean - reference_central)
                < abs(project_mean - reference_central),
                "expanded_five_family_central_degC": expanded_central,
                "expanded_five_family_min_degC": float(np.min(expanded)),
                "expanded_five_family_max_degC": float(np.max(expanded)),
                "expanded_five_family_std_degC": expanded_std,
                "project_minus_expanded_central_degC": project_mean - expanded_central,
                "project_expanded_standardized_difference": (
                    (project_mean - expanded_central) / expanded_std if expanded_std > 0 else np.nan
                ),
                "project_inside_expanded_five_family_range": float(np.min(expanded))
                <= project_mean
                <= float(np.max(expanded)),
            }
        )
    windows = pd.DataFrame(rows)
    windows.to_csv(output / "mode_ra_lia_50year_adjudication.csv", index=False)
    closer_count = int(windows.mode_ra_closer_to_reference_than_project.sum())
    outside_count = int((~windows.project_inside_expanded_five_family_range).sum())
    median_mode_distance = float(np.median(np.abs(windows.mode_ra_minus_reference_central_degC)))
    median_project_distance = float(np.median(np.abs(windows.project_minus_reference_central_degC)))
    median_expanded_z = float(np.median(windows.project_expanded_standardized_difference))
    gates = {
        "mode_ra_is_cold_in_all_nine_complete_lia_windows": bool(np.all(windows.mode_ra_mean_degC < 0)),
        "mode_ra_is_closer_to_reference_in_at_least_7_of_9_windows": closer_count >= 7,
        "median_mode_ra_reference_distance_below_project_distance": median_mode_distance
        < median_project_distance,
        "project_is_warmer_than_expanded_central_in_all_windows": bool(
            np.all(windows.project_minus_expanded_central_degC > 0)
        ),
        "project_remains_outside_expanded_range_in_at_least_7_of_9_windows": outside_count >= 7,
        "median_project_expanded_standardized_warm_gap_at_least_1": median_expanded_z >= 1.0,
    }
    summary = {
        "status": "complete",
        "years": [1421, 1920],
        "complete_fifty_year_windows": [list(window) for window in WINDOWS],
        "locked_1921_1939_accessed": False,
        "mode_ra_modern_temperature_spatial_validation_passed": True,
        "mode_ra_closer_to_reference_window_count": closer_count,
        "project_outside_expanded_range_window_count": outside_count,
        "median_abs_mode_ra_minus_reference_central_degC": median_mode_distance,
        "median_abs_project_minus_reference_central_degC": median_project_distance,
        "median_project_minus_expanded_central_degC": float(
            np.median(windows.project_minus_expanded_central_degC)
        ),
        "median_project_expanded_standardized_difference": median_expanded_z,
        "gates": gates,
        "supports_existing_project_warm_bias_closure": bool(all(gates.values())),
        "decision": (
            "retain_project_temperature_warm_bias_closure_and_advance_spatial_comparison"
            if all(gates.values())
            else "reopen_temperature_reference_conflict_before_spatial_comparison"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(13, 8.2), constrained_layout=True)
    years = common.year.to_numpy(int)
    axes[0].fill_between(
        years,
        common.reference_four_family_min_degC,
        common.reference_four_family_max_degC,
        color="#B8B8B8",
        alpha=0.45,
        label="Four-family range",
    )
    axes[0].plot(years, smooth(common.reference_four_family_central_degC.to_numpy(float)), color="#0072B2", linewidth=2.0, label="Four-family central, 15y")
    axes[0].plot(years, smooth(common.mode_ra_hma_degC.to_numpy(float)), color="#009E73", linewidth=2.0, label="ModE-RA, 15y")
    axes[0].plot(years, smooth(common[PROJECT_COLUMN].to_numpy(float)), color="#D55E00", linewidth=2.0, label="Corrected project, 15y")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set(title="HMA temperature evidence, allowed years", ylabel="Anomaly (degC)")
    axes[0].legend(frameon=False, ncol=4)

    x = np.arange(len(windows))
    axes[1].fill_between(
        x,
        windows.expanded_five_family_min_degC,
        windows.expanded_five_family_max_degC,
        color="#B8B8B8",
        alpha=0.5,
        label="Five-family range",
    )
    axes[1].plot(x, windows.expanded_five_family_central_degC, color="#0072B2", marker="o", label="Five-family central")
    axes[1].plot(x, windows.mode_ra_mean_degC, color="#009E73", marker="s", label="ModE-RA")
    axes[1].plot(x, windows.project_corrected_mean_degC, color="#D55E00", marker="o", label="Corrected project")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, [f"{row.start_year}-{row.end_year}" for row in windows.itertuples()], rotation=30, ha="right")
    axes[1].set(xlabel="Complete 50-year window", ylabel="Mean anomaly (degC)")
    axes[1].legend(frameon=False, ncol=4)
    figure.savefig(output / "mode_ra_lia_regional_adjudication_curves.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-RA LIA Regional Adjudication",
                "",
                f"- ModE-RA closer to four-family central: `{closer_count}/9` windows.",
                f"- Median absolute ModE-RA/project distance from the four-family central: `{median_mode_distance:.3f}` / `{median_project_distance:.3f}` degC.",
                f"- Corrected project outside the expanded five-family range: `{outside_count}/9` windows.",
                f"- Median project warm gap from expanded central: `{np.median(windows.project_minus_expanded_central_degC):.3f}` degC (`{median_expanded_z:.2f}` family SD).",
                f"- All gates pass: `{str(summary['supports_existing_project_warm_bias_closure']).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
