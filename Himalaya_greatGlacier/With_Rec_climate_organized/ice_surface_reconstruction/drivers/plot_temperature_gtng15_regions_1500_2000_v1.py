"""Plot project and reference HMA temperature curves for all 15 GTN-G regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

import run_climate_reconstruction_loop_v2_round23_corrected_reconstruction_reference_envelope_v1 as round23


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
INVENTORY_ROOT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1"
GRID_ASSIGNMENT = INVENTORY_ROOT / "hma_grid_gtng15_assignment.csv"
SUPPORT = INVENTORY_ROOT / "gtng15_region_support_inventory.csv"
REFERENCE = RESULTS / "climate_reconstruction_improvement_loop_v2/round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
LINEAGE = RESULTS / "climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
OUTPUT = RESULTS / "climate_reconstruction_temperature_gtng15_curves_1500_2000_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trend(year: np.ndarray, value: np.ndarray) -> float:
    valid = np.isfinite(year) & np.isfinite(value)
    return float(np.polyfit(year[valid], value[valid], 1)[0] * 100.0) if valid.sum() >= 3 else float("nan")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def smooth(value: np.ndarray) -> np.ndarray:
    return pd.Series(value).rolling(21, center=True, min_periods=11).mean().to_numpy()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lineage = pd.read_csv(LINEAGE)
    temperature = lineage[lineage["variable"].eq("tas")].copy()
    expected = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if len(temperature) != 6 or set(temperature["member"]) != expected:
        raise RuntimeError("Temperature lineage contract failed")
    member_paths = {row.member: Path(row.source_tif) for row in temperature.itertuples(index=False)}
    sources = {"grid_assignment": GRID_ASSIGNMENT, "support": SUPPORT, "reference": REFERENCE, "lineage": LINEAGE, **member_paths}
    before = {name: sha256(path) for name, path in sources.items()}
    for row in temperature.itertuples(index=False):
        if before[row.member] != row.source_tif_sha256:
            raise RuntimeError(f"Source hash mismatch: {row.member}")

    years = np.arange(1000, 2001)
    selected = (years >= 1500) & (years <= 2000)
    selected_year = years[selected]
    latitude, longitude = round23.read_reference_grid(REFERENCE)
    members: dict[str, np.ndarray] = {}
    for name, path in member_paths.items():
        members[name], _ = round23.read_allowed_tif(path, years, latitude, longitude)
    with netcdf_file(REFERENCE, "r", mmap=False) as dataset:
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"].data, dtype=float).copy()
    reference[np.isclose(reference, -9999.0) | (reference < -9000)] = np.nan

    assignment = pd.read_csv(GRID_ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    panel_data: dict[str, dict[str, np.ndarray]] = {}

    for region in support.itertuples(index=False):
        cells = assignment[assignment["o2region"].eq(region.o2region)]
        mask = np.zeros((len(latitude), len(longitude)), dtype=bool)
        mask[cells["grid_row"].to_numpy(int), cells["grid_col"].to_numpy(int)] = True
        if int(mask.sum()) != int(region.hma_grid_cell_count):
            raise RuntimeError(f"Region mask closure failed: {region.o2region}")
        member_series = {name: round23.weighted_series(field, mask, latitude)[selected] for name, field in members.items()}
        member_stack = np.stack(list(member_series.values()))
        project = np.nanmean(member_stack, axis=0)
        reference_series = round23.weighted_series(reference, mask, latitude)[selected]
        project_smooth = smooth(project)
        reference_smooth = smooth(reference_series)
        member_smooth = np.stack([smooth(series) for series in member_series.values()])
        member_min = np.full(len(selected_year), np.nan)
        member_max = np.full(len(selected_year), np.nan)
        finite = np.isfinite(member_smooth).any(axis=0)
        member_min[finite] = np.nanmin(member_smooth[:, finite], axis=0)
        member_max[finite] = np.nanmax(member_smooth[:, finite], axis=0)
        panel_data[region.o2region] = {
            "project": project,
            "reference": reference_series,
            "project_smooth": project_smooth,
            "reference_smooth": reference_smooth,
            "member_min": member_min,
            "member_max": member_max,
        }
        for index, year in enumerate(selected_year):
            rows.append({
                "year": int(year),
                "o2region": region.o2region,
                "full_name": region.full_name,
                "project_six_member_degC": project[index],
                "reference_family_central_degC": reference_series[index],
                "project_smooth21_degC": project_smooth[index],
                "reference_smooth21_degC": reference_smooth[index],
                "member_smooth21_min_degC": member_min[index],
                "member_smooth21_max_degC": member_max[index],
            })
        early = (selected_year >= 1901) & (selected_year <= 1920)
        lia = (selected_year >= 1500) & (selected_year <= 1899)
        modern = (selected_year >= 1940) & (selected_year <= 2000)
        project_trend = trend(selected_year[early], project[early])
        reference_trend = trend(selected_year[early], reference_series[early])
        metrics.append({
            "o2region": region.o2region,
            "full_name": region.full_name,
            "hma_grid_cell_count": int(region.hma_grid_cell_count),
            "temperature_proxy_unique_site_count": int(region.temperature_proxy_unique_site_count),
            "moisture_proxy_unique_site_count": int(region.moisture_proxy_unique_site_count),
            "core191_system_count": int(region.core191_system_count),
            "project_reference_correlation_1901_1920": correlation(project[early], reference_series[early]),
            "project_trend_1901_1920_degC_per_century": project_trend,
            "reference_trend_1901_1920_degC_per_century": reference_trend,
            "trend_sign_agreement_1901_1920": bool(np.sign(project_trend) == np.sign(reference_trend)),
            "project_reference_correlation_1940_2000": correlation(project[modern], reference_series[modern]),
            "mean_project_minus_reference_1500_1899_degC": float(np.nanmean(project[lia] - reference_series[lia])),
        })

    OUTPUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "temperature_gtng15_curves_1500_2000.csv", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUTPUT / "temperature_gtng15_region_metrics.csv", index=False)

    fig, axes = plt.subplots(5, 3, figsize=(17, 19), sharex=True, sharey=True, constrained_layout=True)
    flat = axes.ravel()
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(support)))
    for axis, color, region in zip(flat, colors, support.itertuples(index=False)):
        data = panel_data[region.o2region]
        axis.fill_between(selected_year, data["member_min"], data["member_max"], color=color, alpha=0.20)
        axis.plot(selected_year, data["project_smooth"], color=color, linewidth=2.0, label="Project six-member")
        axis.plot(selected_year, data["reference_smooth"], color="#222222", linewidth=1.7, linestyle="--", label="Reference central")
        axis.axvspan(1901, 1920, color="#888888", alpha=0.13)
        axis.axhline(0.0, color="#444444", linewidth=0.6)
        axis.set_title(f"{region.o2region}  {region.full_name}\nT sites={region.temperature_proxy_unique_site_count}; Core-191={region.core191_system_count}", fontsize=10)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(1500, 2000)
        axis.set_xticks([1500, 1700, 1900, 2000])
    for row_axes in axes:
        row_axes[0].set_ylabel("Temperature anomaly (degC)")
    for axis in axes[-1]:
        axis.set_xlabel("Year")
    flat[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("HMA temperature by GTN-G 2023 subregion, 1500-2000 (21-year means)", fontsize=16)
    figure_path = OUTPUT / "temperature_gtng15_regions_1500_2000.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    reversed_regions = metrics_frame.loc[~metrics_frame["trend_sign_agreement_1901_1920"], ["o2region", "full_name"]]
    summary = {
        "status": "complete",
        "region_count": int(len(support)),
        "period": [1500, 2000],
        "smoothing": "21-year centered mean with minimum 11 years at endpoints",
        "opposite_project_reference_trend_sign_regions_1901_1920": reversed_regions.to_dict(orient="records"),
        "opposite_trend_sign_region_count": int(len(reversed_regions)),
        "locked_placeholder_gap_retained": [1921, 1939],
        "interpretation_limit": (
            "Project-reference phase disagreement is diagnostic, not independent validation. "
            "Proxy support and direct-station evidence must govern any region-specific PSM decision."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {name: sha256(path) for name, path in sources.items()}
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "sources_unchanged": before == after}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(metrics_frame.to_string(index=False))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
