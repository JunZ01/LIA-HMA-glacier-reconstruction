"""Test paleo spatial consensus between ModE-RA and the approved reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = tuple((start, start + 49) for start in range(1450, 1900, 50))


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


def cell_edges(centers: np.ndarray) -> np.ndarray:
    if not np.all(np.diff(centers) > 0):
        raise RuntimeError("cell-centers-not-increasing")
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 4:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    center = weighted_mean(values, weights)
    valid = np.isfinite(values)
    return float(np.sqrt(np.sum(weights[valid] * (values[valid] - center) ** 2) / np.sum(weights[valid])))


def aggregate_reference(
    values: np.ndarray,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    hma_mask: np.ndarray,
    support: pd.DataFrame,
    mode_lat: np.ndarray,
    mode_lon: np.ndarray,
) -> np.ndarray:
    lat_edges = cell_edges(mode_lat)
    lon_edges = cell_edges(mode_lon)
    result = np.full((values.shape[0], len(support)), np.nan)
    for cell_index, row in enumerate(support.itertuples(index=False)):
        i = int(np.flatnonzero(np.isclose(mode_lat, row.latitude, atol=1e-8))[0])
        j = int(np.flatnonzero(np.isclose(mode_lon, row.longitude, atol=1e-8))[0])
        fine_i = np.flatnonzero((fine_lat >= lat_edges[i]) & (fine_lat < lat_edges[i + 1]))
        fine_j = np.flatnonzero((fine_lon >= lon_edges[j]) & (fine_lon < lon_edges[j + 1]))
        local_mask = hma_mask[np.ix_(fine_i, fine_j)]
        ii, jj = np.where(local_mask)
        selected_i = fine_i[ii]
        selected_j = fine_j[jj]
        if len(selected_i) != int(row.fine_hma_cell_count):
            raise RuntimeError(
                f"coarse-support-count-conflict:{cell_index}:{len(selected_i)}:{row.fine_hma_cell_count}"
            )
        subset = values[:, selected_i, selected_j]
        weights = np.cos(np.deg2rad(fine_lat[selected_i]))
        valid = np.isfinite(subset)
        denominator = np.sum(valid * weights[None], axis=1)
        result[:, cell_index] = np.nansum(subset * weights[None], axis=1) / np.where(
            denominator == 0, np.nan, denominator
        )
    return result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_71_mode_ra_lia_spatial_consensus_v1"
    round69 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v3"
    round70 = loop / "round_70_mode_ra_lia_regional_adjudication_v2"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round21 = loop / "round_21_temperature_reference_field_v1"
    paths = {
        "mode_ra_temperature": round69 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "round69_summary": round69 / "summary.json",
        "round70_summary": round70 / "summary.json",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "reference_field": round21 / "temperature_reference_family_balanced_allowed_v1.nc",
        "round21_summary": round21 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round69_summary"].read_text(encoding="utf-8"))["passes_acquisition_preflight"]:
        raise RuntimeError("round69-did-not-pass")
    if not json.loads(paths["round70_summary"].read_text(encoding="utf-8"))[
        "supports_existing_project_warm_bias_closure"
    ]:
        raise RuntimeError("round70-regional-adjudication-did-not-pass")

    import h5py

    support = pd.read_csv(paths["coarse_support"])
    with h5py.File(paths["mode_ra_temperature"], "r") as handle:
        mode_years = np.asarray(handle["time"], dtype=int)
        mode_lat = np.asarray(handle["lat"], dtype=float)
        mode_lon = np.asarray(handle["lon"], dtype=float)
        mode_field = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    selected_mode = (mode_years >= 1450) & (mode_years <= 1899)
    if not np.array_equal(mode_years[selected_mode], np.arange(1450, 1900)):
        raise RuntimeError("mode-ra-comparison-year-axis-failed")
    mode_cells = []
    for row in support.itertuples(index=False):
        i = np.flatnonzero(np.isclose(mode_lat, row.latitude, atol=1e-8))
        j = np.flatnonzero(np.isclose(mode_lon, row.longitude, atol=1e-8))
        if len(i) != 1 or len(j) != 1:
            raise RuntimeError("mode-ra-support-grid-match-failed")
        mode_cells.append(mode_field[selected_mode, int(i[0]), int(j[0])])
    mode_coarse = np.stack(mode_cells, axis=1)

    with netcdf_file(paths["reference_field"], "r", mmap=False) as handle:
        years = np.asarray(handle.variables["year"][:], dtype=int)
        indices = np.flatnonzero((years >= 1450) & (years <= 1899))
        if not np.array_equal(years[indices], np.arange(1450, 1900)):
            raise RuntimeError("reference-comparison-year-axis-failed")
        if np.any((years[indices] >= 1921) & (years[indices] <= 1939)):
            raise RuntimeError("locked-period-selected-from-reference")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise RuntimeError("reference-selection-not-contiguous")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        fine_lat = np.asarray(handle.variables["lat"][:], dtype=float)
        fine_lon = np.asarray(handle.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(handle.variables["hma_mask"][:], dtype=int).astype(bool)
        reference = np.asarray(handle.variables["tas_reference_family_balanced"][selection], dtype=float)
        spread = np.asarray(handle.variables["tas_reference_family_spread_std"][selection], dtype=float)
        fill_reference = float(handle.variables["tas_reference_family_balanced"]._FillValue)
        fill_spread = float(handle.variables["tas_reference_family_spread_std"]._FillValue)
    reference[np.isclose(reference, fill_reference) | (reference < -1e20)] = np.nan
    spread[np.isclose(spread, fill_spread) | (spread < -1e20)] = np.nan
    reference_coarse = aggregate_reference(
        reference, fine_lat, fine_lon, hma_mask, support, mode_lat, mode_lon
    )
    spread_coarse = aggregate_reference(
        spread, fine_lat, fine_lon, hma_mask, support, mode_lat, mode_lon
    )
    if not np.all(np.isfinite(mode_coarse)) or not np.all(np.isfinite(reference_coarse)):
        raise RuntimeError("nonfinite-mode-ra-or-reference-coarse-field")
    weights = (
        np.cos(np.deg2rad(support.latitude.to_numpy(float)))
        * support.fine_hma_cell_count.to_numpy(float)
    )

    rows: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    years_common = np.arange(1450, 1900)
    for start, end in WINDOWS:
        selected = (years_common >= start) & (years_common <= end)
        mode_window = np.mean(mode_coarse[selected], axis=0)
        reference_window = np.mean(reference_coarse[selected], axis=0)
        spread_window = np.mean(spread_coarse[selected], axis=0)
        mode_center = weighted_mean(mode_window, weights)
        reference_center = weighted_mean(reference_window, weights)
        mode_residual = mode_window - mode_center
        reference_residual = reference_window - reference_center
        difference = mode_residual - reference_residual
        residual_rmse = float(np.sqrt(np.sum(weights * difference**2) / np.sum(weights)))
        zero_rmse = float(np.sqrt(np.sum(weights * reference_residual**2) / np.sum(weights)))
        amplitude_ratio = weighted_std(mode_residual, weights) / weighted_std(reference_residual, weights)
        rows.append(
            {
                "start_year": start,
                "end_year": end,
                "mode_ra_regional_mean_degC": mode_center,
                "reference_regional_mean_degC": reference_center,
                "mode_ra_minus_reference_regional_degC": mode_center - reference_center,
                "spatial_residual_correlation": correlation(mode_residual, reference_residual),
                "spatial_residual_rmse_degC": residual_rmse,
                "zero_spatial_residual_rmse_degC": zero_rmse,
                "spatial_residual_skill": float(1.0 - residual_rmse / zero_rmse),
                "mode_ra_to_reference_residual_std_ratio": float(amplitude_ratio),
                "same_sign_residual_fraction": float(np.mean(np.sign(mode_residual) == np.sign(reference_residual))),
                "median_reference_family_spread_degC": float(np.median(spread_window)),
            }
        )
        maps[f"{start}_{end}_mode"] = mode_window
        maps[f"{start}_{end}_reference"] = reference_window
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "mode_ra_reference_lia_spatial_window_scores.csv", index=False)
    median_correlation = float(scores.spatial_residual_correlation.median())
    positive_corr_count = int((scores.spatial_residual_correlation > 0).sum())
    median_skill = float(scores.spatial_residual_skill.median())
    positive_skill_count = int((scores.spatial_residual_skill > 0).sum())
    median_ratio = float(scores.mode_ra_to_reference_residual_std_ratio.median())
    gates = {
        "median_spatial_residual_correlation_at_least_0_30": median_correlation >= 0.30,
        "positive_spatial_correlation_in_at_least_7_of_9_windows": positive_corr_count >= 7,
        "median_spatial_residual_skill_positive": median_skill > 0,
        "positive_spatial_skill_in_at_least_7_of_9_windows": positive_skill_count >= 7,
        "median_residual_amplitude_ratio_between_0_5_and_1_5": 0.5 <= median_ratio <= 1.5,
        "worst_spatial_residual_skill_not_below_minus_0_25": float(scores.spatial_residual_skill.min()) >= -0.25,
    }
    summary = {
        "status": "complete",
        "years": [1450, 1899],
        "locked_1921_1939_accessed": False,
        "coarse_hma_cell_count": len(support),
        "window_count": len(scores),
        "median_spatial_residual_correlation": median_correlation,
        "positive_spatial_correlation_window_count": positive_corr_count,
        "median_spatial_residual_skill": median_skill,
        "positive_spatial_skill_window_count": positive_skill_count,
        "median_mode_ra_to_reference_residual_std_ratio": median_ratio,
        "worst_spatial_residual_skill": float(scores.spatial_residual_skill.min()),
        "gates": gates,
        "passes_lia_spatial_consensus": bool(all(gates.values())),
        "decision": (
            "permit_coarse_grid_mode_ra_expanded_temperature_spatial_reference"
            if all(gates.values())
            else "retain_mode_ra_as_regional_lia_evidence_only"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    x = np.arange(len(scores))
    labels = [f"{row.start_year}-{row.end_year}" for row in scores.itertuples()]
    axes[0, 0].plot(x, scores.spatial_residual_correlation, marker="o", color="#0072B2")
    axes[0, 0].axhline(0.30, color="#777777", linestyle="--", linewidth=0.9)
    axes[0, 0].set(title="LIA spatial residual correlation", ylabel="Correlation")
    axes[0, 1].plot(x, scores.spatial_residual_skill * 100, marker="o", color="#D55E00")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.9)
    axes[0, 1].set(title="LIA spatial residual RMSE skill", ylabel="Skill (%)")
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    selected_label = "1800_1849"
    mode_map = maps[f"{selected_label}_mode"]
    reference_map = maps[f"{selected_label}_reference"]
    vmin = float(min(mode_map.min(), reference_map.min()))
    vmax = float(max(mode_map.max(), reference_map.max()))
    first = axes[1, 0].scatter(
        support.longitude,
        support.latitude,
        c=mode_map,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=42,
    )
    axes[1, 0].set(title="ModE-RA 1800-1849", xlabel="Longitude", ylabel="Latitude")
    second = axes[1, 1].scatter(
        support.longitude,
        support.latitude,
        c=reference_map,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=42,
    )
    axes[1, 1].set(title="Four-family reference 1800-1849", xlabel="Longitude", ylabel="Latitude")
    figure.colorbar(second, ax=axes[1], label="Temperature anomaly (degC)")
    figure.savefig(output / "mode_ra_lia_spatial_consensus.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-RA LIA Spatial Consensus",
                "",
                f"- Median residual correlation: `{median_correlation:.3f}`; positive windows `{positive_corr_count}/9`.",
                f"- Median residual skill: `{median_skill:.2%}`; positive windows `{positive_skill_count}/9`.",
                f"- Median ModE-RA/reference residual-amplitude ratio: `{median_ratio:.3f}`.",
                f"- Worst residual skill: `{scores.spatial_residual_skill.min():.2%}`.",
                f"- All gates pass: `{str(summary['passes_lia_spatial_consensus']).lower()}`.",
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
