from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import extract_schlagintweit_1850s_temperature_v2 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_expanded_rows(rows: list[dict]) -> None:
    darjeeling = {
        "1850": [42.4, 41.3, 48.4, 56.7, 58.9, 62.8, 63.5, 63.2, 62.3, 61.8, 54.5, 45.9],
        "1852": [42.9, 47.1, 46.7, 53.9, 57.9, 62.1, 63.1, 63.6, 61.8, 56.0, 58.0, 42.7],
        "1855 partial": [39.4, 41.9, np.nan, np.nan, np.nan, 60.9, 62.2, 62.0, 60.4, np.nan, np.nan, np.nan],
    }
    for period, values in darjeeling.items():
        year = int(period[:4])
        base.add_series(
            rows,
            station="Darjeeling Observatory Hill",
            latitude=27 + 3 / 60,
            longitude=88 + 15.3 / 60,
            elevation_ft=7168,
            period=period,
            year_start=year,
            year_end=year,
            values_f=values,
            source_page=485,
            observer="Withcombe; Observatory Hill register; compiled by Schlagintweit",
            observation_family="Darjeeling Observatory Hill historical station register",
            independence_class="same_physical_station_family_as_later_IMD_GHCN",
            value_role="pre_GHCN_station_chain_anchor_not_independent_vote",
            notes="1855 used 6 and 9 a.m., 3 and 9 p.m.; earlier means approximate sunrise plus afternoon maximum. Missing cells are not filled.",
        )

    massuri_quality = ["two_series_station_composite"] * 12
    base.add_series(
        rows,
        station="Massuri",
        latitude=30 + 27.6 / 60,
        longitude=78 + 3.0 / 60,
        elevation_ft=6715,
        period="1856 composite",
        year_start=1856,
        year_end=1856,
        values_f=[45.2, 48.2, 57.0, 67.2, 69.2, 64.2, 66.4, 65.3, 64.9, 64.0, 53.0, 46.0],
        source_page=499,
        observer="Sir Andrew Waugh establishment at Mary Villa and Mr. Mackinnon",
        observation_family="Massuri two-observer historical composite",
        independence_class="conditional_two_series_station_composite",
        value_role="local_level_sensitivity_not_primary_anchor",
        quality_by_month=massuri_quality,
        notes="Monthly means combine two observation series; source reports sunrise, 9 a.m., 4 p.m. and variable additional hours.",
    )

    kardong_quality = [
        "filled_from_Mani_equal_elevation",
        "filled_from_Mani_equal_elevation",
        "filled_from_Mani_equal_elevation",
        "filled_from_Mani_equal_elevation",
        "direct_Kardong_9am_noon_sunset",
        "direct_Kardong_9am_noon_sunset",
        "direct_Kardong_9am_noon_sunset",
        "direct_Kardong_9am_noon_sunset",
        "direct_Kardong_9am_noon_sunset",
        "direct_Kardong_9am_noon_sunset",
        "filled_from_Mani_equal_elevation",
        "filled_from_Mani_equal_elevation",
    ]
    base.add_series(
        rows,
        station="Kardong",
        latitude=32 + 33.8 / 60,
        longitude=77 + 0.6 / 60,
        elevation_ft=10242,
        period="1855-1856 composite",
        year_start=1855,
        year_end=1856,
        values_f=[24.0, 36.0, 44.0, 47.0, 49.0, 54.0, 63.0, 60.0, 52.0, 46.0, 37.0, 27.0],
        source_page=513,
        observer="Missionaries Heyde and Pagell; winter values supplied from Mani",
        observation_family="Schlagintweit Kardong seasonal station composite",
        independence_class="mixed_direct_and_equal_elevation_fill",
        value_role="direct_May_Oct_seasonal_anchor_only",
        quality_by_month=kardong_quality,
        notes="The source station is Kardong in Lahol, not Kyelang. Only May-October 1855 are direct Kardong observations.",
    )


def correct_base_classifications(values: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    skardo = values.station.eq("Skardo")
    values.loc[skardo, "independence_class"] = "independent_local_establishment_not_single_fixed_station"
    values.loc[skardo, "value_role"] = "local_monthly_direction_context"
    values.loc[skardo, "notes"] = (
        "Compiled from Adolphe and an establishment in Skardo and its environs; sunrise/4 p.m. averaging and site mixing prevent primary fixed-station use."
    )
    return values


def make_figure(values: pd.DataFrame, output: Path) -> None:
    panels = [
        (
            "East and central-east",
            ["Kathmandu", "Darjeeling", "Darjeeling Observatory Hill"],
        ),
        (
            "Central-west and west",
            ["Simla", "Massuri", "Kardong", "Skardo", "Leh"],
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, (title, stations) in zip(axes, panels):
        subset = values[values.station.isin(stations)]
        for index, ((station, period), group) in enumerate(subset.groupby(["station", "period"], sort=False)):
            group = group.sort_values("month")
            label = f"{station} {period}"
            ax.plot(group.month, group.temperature_c, marker="o", linewidth=1.8, markersize=4, color=cmap(index % 10), label=label)
            direct_mask = group.observation_quality.str.startswith("direct_")
            if direct_mask.any():
                ax.scatter(
                    group.loc[direct_mask, "month"],
                    group.loc[direct_mask, "temperature_c"],
                    s=58,
                    facecolor="#f2c14e",
                    edgecolor="#111111",
                    zorder=5,
                )
        ax.set_title(title)
        ax.set_ylabel("Mean air temperature (deg C)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.legend(ncol=3, fontsize=7.8, frameon=False)
    axes[-1].set_xticks(range(1, 13), base.MONTHS)
    axes[-1].set_xlabel("Month")
    fig.suptitle("Visually verified 1850s High Asia station tables", fontsize=16)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def direct_month_count(series: pd.Series) -> int:
    return int(sum(value == "direct_or_register_derived" or value.startswith("direct_") for value in series))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    required_pages = [470, 480, 483, 484, 485, 488, 499, 500, 506, 513, 530, 531, 532]
    sources = [args.verified_pages / f"page_{page}.jpg" for page in required_pages]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing visually verified source pages: {missing}")
    before = {str(path.resolve()): sha256(path) for path in sources}

    rows = base.build_rows()
    add_expanded_rows(rows)
    values = correct_base_classifications(pd.DataFrame(rows))
    values = values[pd.notna(values.reported_temperature_f)].copy()
    values.to_csv(args.output / "verified_monthly_temperature_values.csv", index=False)

    annual = (
        values.groupby(["station", "period", "year_start", "year_end", "independence_class", "value_role"], as_index=False)
        .agg(
            available_month_count=("month", "count"),
            arithmetic_available_month_mean_f=("reported_temperature_f", "mean"),
            arithmetic_available_month_mean_c=("temperature_c", "mean"),
            direct_month_count=("observation_quality", direct_month_count),
        )
    )
    annual["is_complete_12_month_series"] = annual.available_month_count.eq(12)
    annual["non_direct_month_count"] = annual.available_month_count - annual.direct_month_count
    annual.to_csv(args.output / "series_completeness_and_quality.csv", index=False)

    station_roles = values[
        ["station", "latitude_n", "longitude_e", "elevation_ft", "observation_family", "independence_class", "value_role", "notes"]
    ].drop_duplicates()
    station_roles.to_csv(args.output / "station_role_and_independence_audit.csv", index=False)
    make_figure(values, args.output / "verified_1850s_monthly_temperature_curves.png")

    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Verified source page changed during extraction")
    summary = {
        "experiment": "temperature_schlagintweit_1850s_structured_extraction_v3",
        "supersedes": "temperature_schlagintweit_1850s_structured_extraction_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_expanded_visually_verified_extraction",
        "station_count": int(values.station.nunique()),
        "series_count": int(values[["station", "period"]].drop_duplicates().shape[0]),
        "numeric_month_count": int(len(values)),
        "new_direct_segments": [
            "Darjeeling Observatory Hill 1850 and 1852 full monthly series",
            "Darjeeling Observatory Hill 1855 six observed months",
            "Kardong May-October 1855",
        ],
        "conditional_segment": "Massuri 1856 two-series composite",
        "identity_correction": "preflight Kyelang label corrected to Kardong in Lahol",
        "Skardo_classification_correction": "local establishment/environs context, not a single fixed station",
        "scientific_limit": "local and seasonal anchors; no uniform HMA correction is authorized",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "source_hashes_unchanged": True,
        "manual_transcription": True,
        "visual_verification_completed": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = """# Schlagintweit 1850s structured temperature extraction v3

This package supersedes v2 after visual review of the fixed-station and seasonal tables.

## Concrete additions and corrections

- Darjeeling Observatory Hill now includes complete monthly means for 1850 and 1852 plus six observed months in 1855. It extends the same physical station family and is not counted as an independent second climate vote.
- The western seasonal station on printed page 513 is Kardong in Lahol, not Kyelang. Only May-October 1855 are direct Kardong observations; the remaining months were filled from Mani at similar elevation.
- Massuri 1856 is retained as conditional evidence because its monthly means combine two observer/site series.
- Skardo is reclassified as a local establishment/environs record, not a single fixed-station annual truth.
- Leh remains admissible only for its direct July-September 1856 segment.

The added records strengthen direct local and seasonal validation before 1875, but they still cannot prescribe a uniform HMA-wide correction.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 accessed or used: **no**
- Production reconstruction modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
