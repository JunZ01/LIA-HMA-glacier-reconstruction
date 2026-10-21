"""Plot 1500-2000 HMA temperature curves in frozen longitude proxy regions."""

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
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
LINEAGE = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
REFERENCE = LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
HMA_MASK = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_circulation_proxy_regions_1500_2000_v1"

REGIONS = {
    "west_lt80": {
        "bounds": (-np.inf, 80.0),
        "label": "Western proxy zone (<80 E; westerly-influenced)",
        "color": "#0072B2",
    },
    "central_80_90": {
        "bounds": (80.0, 90.0),
        "label": "Central transition proxy zone (80-90 E)",
        "color": "#E69F00",
    },
    "east_ge90": {
        "bounds": (90.0, np.inf),
        "label": "Eastern proxy zone (>=90 E; monsoon-influenced)",
        "color": "#009E73",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_trend(year: np.ndarray, value: np.ndarray) -> float:
    valid = np.isfinite(year) & np.isfinite(value)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(year[valid], value[valid], 1)[0] * 100.0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    lineage = pd.read_csv(LINEAGE)
    temperature = lineage[lineage["variable"].eq("tas")].copy()
    expected = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if len(temperature) != 6 or set(temperature["member"]) != expected:
        raise RuntimeError("Temperature lineage contract failed")
    member_paths = {row.member: Path(row.source_tif) for row in temperature.itertuples(index=False)}
    source_paths = {"lineage": LINEAGE, "reference": REFERENCE, "hma_mask": HMA_MASK, **member_paths}
    before = {name: sha256(path) for name, path in source_paths.items()}
    for row in temperature.itertuples(index=False):
        if before[row.member] != row.source_tif_sha256:
            raise RuntimeError(f"Source hash mismatch: {row.member}")

    years = np.arange(1000, 2001)
    latitude, longitude = round23.read_reference_grid(REFERENCE)
    hma_mask = np.load(HMA_MASK).astype(bool)
    longitude_grid = np.broadcast_to(longitude[None, :], hma_mask.shape)

    members: dict[str, np.ndarray] = {}
    for name, path in member_paths.items():
        members[name], _ = round23.read_allowed_tif(path, years, latitude, longitude)

    with netcdf_file(REFERENCE, "r", mmap=False) as dataset:
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"].data, dtype=float).copy()
        reference_year = np.asarray(dataset.variables["year"].data, dtype=int).copy()
    reference[np.isclose(reference, -9999.0) | (reference < -9000)] = np.nan
    if not np.array_equal(reference_year, years):
        raise RuntimeError("Reference year contract failed")

    selected = (years >= 1500) & (years <= 2000)
    selected_year = years[selected]
    output = pd.DataFrame({"year": selected_year})
    summary_regions: dict[str, dict[str, float | int | bool]] = {}

    for region, config in REGIONS.items():
        lower, upper = config["bounds"]
        region_mask = hma_mask & (longitude_grid >= lower) & (longitude_grid < upper)
        if region_mask.sum() == 0:
            raise RuntimeError(f"Empty region: {region}")
        member_series = {
            name: round23.weighted_series(field, region_mask, latitude)[selected]
            for name, field in members.items()
        }
        member_stack = np.stack(list(member_series.values()))
        project = np.nanmean(member_stack, axis=0)
        reference_series = round23.weighted_series(reference, region_mask, latitude)[selected]

        output[f"{region}_project_six_member_degC"] = project
        output[f"{region}_reference_family_central_degC"] = reference_series
        for name, series in member_series.items():
            output[f"{region}_{name}_degC"] = series
        output[f"{region}_project_smooth21_degC"] = pd.Series(project).rolling(21, center=True, min_periods=11).mean()
        output[f"{region}_reference_smooth21_degC"] = pd.Series(reference_series).rolling(21, center=True, min_periods=11).mean()
        smoothed_members = np.stack(
            [pd.Series(series).rolling(21, center=True, min_periods=11).mean().to_numpy() for series in member_series.values()]
        )
        output[f"{region}_member_smooth21_min_degC"] = np.nanmin(smoothed_members, axis=0)
        output[f"{region}_member_smooth21_max_degC"] = np.nanmax(smoothed_members, axis=0)

        early = (selected_year >= 1901) & (selected_year <= 1920)
        lia = (selected_year >= 1500) & (selected_year <= 1899)
        summary_regions[region] = {
            "cell_count": int(region_mask.sum()),
            "project_reference_correlation_1901_1920": correlation(project[early], reference_series[early]),
            "project_trend_1901_1920_degC_per_century": linear_trend(selected_year[early], project[early]),
            "reference_trend_1901_1920_degC_per_century": linear_trend(selected_year[early], reference_series[early]),
            "trend_sign_agreement_1901_1920": bool(
                np.sign(linear_trend(selected_year[early], project[early]))
                == np.sign(linear_trend(selected_year[early], reference_series[early]))
            ),
            "mean_project_minus_reference_1500_1899_degC": float(np.nanmean(project[lia] - reference_series[lia])),
        }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    output.to_csv(OUTPUT / "temperature_circulation_proxy_region_curves_1500_2000.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for axis, (region, config) in zip(axes, REGIONS.items()):
        color = config["color"]
        axis.fill_between(
            output["year"],
            output[f"{region}_member_smooth21_min_degC"],
            output[f"{region}_member_smooth21_max_degC"],
            color=color,
            alpha=0.18,
            label="Six-branch 21-year range",
        )
        axis.plot(
            output["year"],
            output[f"{region}_project_six_member_degC"],
            color=color,
            linewidth=0.65,
            alpha=0.28,
            label="Project annual mean",
        )
        axis.plot(
            output["year"],
            output[f"{region}_project_smooth21_degC"],
            color=color,
            linewidth=2.3,
            label="Project six-member mean (21-year)",
        )
        axis.plot(
            output["year"],
            output[f"{region}_reference_smooth21_degC"],
            color="#333333",
            linewidth=2.0,
            linestyle="--",
            label="Public-reference central (21-year)",
        )
        axis.axvspan(1901, 1920, color="#999999", alpha=0.14)
        axis.axhline(0.0, color="#333333", linewidth=0.7)
        axis.set_title(config["label"])
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(1500, 2000)
    axes[-1].set_xticks([1500, 1600, 1700, 1800, 1900, 1950, 2000])
    fig.suptitle("HMA temperature by frozen circulation-proxy region, 1500-2000", fontsize=15)
    figure_path = OUTPUT / "temperature_circulation_proxy_regions_1500_2000.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "period": [1500, 2000],
        "regions": summary_regions,
        "region_definition": {
            name: {"longitude_bounds": list(config["bounds"]), "label": config["label"]}
            for name, config in REGIONS.items()
        },
        "classification_limit": (
            "Frozen longitude bands are used as first-pass circulation proxies. They are not a formal "
            "monsoon-westerly climatology or a data-driven cluster classification."
        ),
        "locked_placeholder_gap_retained": [1921, 1939],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {name: sha256(path) for name, path in source_paths.items()}
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "sources_unchanged": before == after,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("Source file changed")

    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
