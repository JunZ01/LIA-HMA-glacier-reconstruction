from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


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


def fahrenheit_to_celsius(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def add_series(
    rows: list[dict],
    *,
    station: str,
    latitude: float,
    longitude: float,
    elevation_ft: int,
    period: str,
    year_start: int,
    year_end: int,
    values_f: list[float],
    source_page: int,
    observer: str,
    observation_family: str,
    independence_class: str,
    value_role: str,
    quality_by_month: list[str] | None = None,
    parenthetical_months: set[int] | None = None,
    notes: str = "",
) -> None:
    if len(values_f) != 12:
        raise ValueError(f"Expected 12 values for {station} {period}")
    quality_by_month = quality_by_month or ["direct_or_register_derived"] * 12
    parenthetical_months = parenthetical_months or set()
    for month, value_f, quality in zip(range(1, 13), values_f, quality_by_month):
        rows.append(
            {
                "station": station,
                "latitude_n": latitude,
                "longitude_e": longitude,
                "elevation_ft": elevation_ft,
                "elevation_m": elevation_ft * 0.3048,
                "period": period,
                "year_start": year_start,
                "year_end": year_end,
                "month": month,
                "month_name": MONTHS[month - 1],
                "reported_temperature_f": value_f,
                "temperature_c": fahrenheit_to_celsius(value_f),
                "source_parenthetical": month in parenthetical_months,
                "observation_quality": quality,
                "source_page": source_page,
                "source_url": f"https://digi.ub.uni-heidelberg.de/diglit/schlagintweit1866bd4/{source_page:04d}",
                "observer_or_compiler": observer,
                "observation_family": observation_family,
                "independence_class": independence_class,
                "value_role": value_role,
                "notes": notes,
            }
        )


def build_rows() -> list[dict]:
    rows: list[dict] = []

    kathmandu = {
        1854: [45.4, 50.0, 57.1, 61.5, 68.3, 73.0, 74.1, 73.6, 70.9, 65.6, 59.2, 49.9],
        1855: [44.4, 52.0, 56.6, 61.9, 69.6, 73.0, 73.4, 71.9, 70.2, 64.2, 55.9, 47.2],
        1856: [47.8, 50.9, 60.7, 65.1, 68.2, 73.7, 73.7, 72.6, 70.6, 71.0, 56.7, 50.4],
    }
    for year, values in kathmandu.items():
        add_series(
            rows,
            station="Kathmandu",
            latitude=27 + 42.1 / 60,
            longitude=85 + 12.2 / 60,
            elevation_ft=4354,
            period=str(year),
            year_start=year,
            year_end=year,
            values_f=values,
            source_page=488,
            observer="Residency surgeons; 1856 Dr. Brown; compiled by Schlagintweit",
            observation_family="Kathmandu historical register",
            independence_class="independent_pre_GHCN_historical_register",
            value_role="local_monthly_and_annual_anchor",
            parenthetical_months={1} if year == 1854 else set(),
            notes="Observation hours varied; compiler used repeated 24-hour observations to deduce true means.",
        )

    add_series(
        rows,
        station="Simla",
        latitude=31 + 6.2 / 60,
        longitude=77 + 9.4 / 60,
        elevation_ft=7057,
        period="1854",
        year_start=1854,
        year_end=1854,
        values_f=[50.3, 54.0, 52.5, 61.2, 67.3, 71.1, 65.6, 63.9, 62.8, 59.5, 52.6, 43.7],
        source_page=506,
        observer="Macannon and Martin, Staff Dispensary; compiled by Schlagintweit",
        observation_family="Simla official historical register",
        independence_class="same_physical_station_family_as_later_IMD_GHCN",
        value_role="pre_GHCN_station_chain_anchor_not_independent_vote",
        notes="Sunrise/noon/later 4 p.m. readings; noon series reduced to 4 p.m. using government-school daily curves.",
    )

    add_series(
        rows,
        station="Darjeeling",
        latitude=27 + 3 / 60,
        longitude=88 + 15 / 60,
        elevation_ft=7168,
        period="1857-1859 mean",
        year_start=1857,
        year_end=1859,
        values_f=[43.9, 44.8, 51.0, 53.9, 58.2, 60.8, 61.5, 61.5, 60.2, 56.6, 50.5, 44.1],
        source_page=470,
        observer="Darjeeling register; compiled by Schlagintweit",
        observation_family="Darjeeling historical station register",
        independence_class="same_physical_station_family_as_later_IMD_GHCN",
        value_role="pre_GHCN_station_chain_anchor_not_independent_vote",
        notes="Three-year monthly mean from register; not an annual sequence.",
    )

    add_series(
        rows,
        station="Skardo",
        latitude=35 + 20.2 / 60,
        longitude=75 + 44 / 60,
        elevation_ft=7255,
        period="1856",
        year_start=1856,
        year_end=1856,
        values_f=[32.0, 39.0, 45.0, 51.0, 58.0, 66.0, 69.0, 68.0, 59.0, 52.5, 43.0, 33.0],
        source_page=532,
        observer="Adolphe Schlagintweit and local establishment",
        observation_family="Schlagintweit Skardo expedition establishment",
        independence_class="independent_historical_observation",
        value_role="local_monthly_and_annual_anchor",
        notes="Monthly means compiled as the average of sunrise and 4 p.m. readings; possible observation-time bias remains.",
    )

    leh_quality = [
        "imputed_from_Kanam_curve",
        "imputed_from_Kanam_curve",
        "imputed_from_Kanam_curve",
        "imputed_from_Kanam_curve",
        "nearby_establishment_not_Leh_observatory",
        "nearby_establishment_not_Leh_observatory",
        "direct_Leh_observatory",
        "direct_Leh_observatory",
        "direct_Leh_observatory",
        "historical_isolated_month_not_direct_1856",
        "imputed_from_Kanam_curve",
        "imputed_from_Kanam_curve",
    ]
    add_series(
        rows,
        station="Leh",
        latitude=34 + 8.3 / 60,
        longitude=77 + 14.6 / 60,
        elevation_ft=11532,
        period="1856 composite",
        year_start=1856,
        year_end=1856,
        values_f=[20.0, 26.0, 36.0, 44.0, 50.1, 56.2, 66.4, 65.2, 56.0, 40.0, 34.0, 21.0],
        source_page=531,
        observer="Schlagintweit observatory, nearby establishments, and Kanam analogue",
        observation_family="Schlagintweit Leh composite",
        independence_class="mixed_direct_nearby_and_imputed",
        value_role="direct_Jul_Sep_seasonal_anchor_only",
        quality_by_month=leh_quality,
        notes="Only Jul-Sep are direct Leh observatory means. May-Jun use nearby establishments; Nov-Apr are calculated from Kanam; Oct follows isolated historical evidence.",
    )
    return rows


def make_figure(values: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    colors = {
        "Kathmandu 1854": "#c44e52",
        "Kathmandu 1855": "#dd8452",
        "Kathmandu 1856": "#937860",
        "Simla 1854": "#4c72b0",
        "Darjeeling 1857-1859 mean": "#55a868",
        "Skardo 1856": "#8172b3",
        "Leh 1856 composite": "#222222",
    }
    for (station, period), group in values.groupby(["station", "period"], sort=False):
        label = f"{station} {period}"
        ax.plot(
            group["month"],
            group["temperature_c"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=colors[label],
            label=label,
        )
    leh = values[values.station.eq("Leh")]
    direct = leh[leh.observation_quality.eq("direct_Leh_observatory")]
    ax.scatter(direct.month, direct.temperature_c, s=75, facecolor="#f2c14e", edgecolor="#111111", zorder=5, label="Leh direct months (Jul-Sep)")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Reported/derived mean air temperature (deg C)")
    ax.set_title("Verified 1850s High Asia temperature tables")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8.5, frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    required_pages = [470, 488, 506, 530, 531, 532]
    sources = [args.verified_pages / f"page_{page}.jpg" for page in required_pages]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing verified source pages: {missing}")
    before = {str(path.resolve()): sha256(path) for path in sources}

    values = pd.DataFrame(build_rows())
    values.to_csv(args.output / "verified_monthly_temperature_values.csv", index=False)

    annual = (
        values.groupby(["station", "period", "year_start", "year_end", "independence_class", "value_role"], as_index=False)
        .agg(
            arithmetic_annual_mean_f=("reported_temperature_f", "mean"),
            arithmetic_annual_mean_c=("temperature_c", "mean"),
            direct_month_count=("observation_quality", lambda x: int(sum("direct" in value and "not_direct" not in value for value in x))),
            imputed_or_nonlocal_month_count=("observation_quality", lambda x: int(sum(value != "direct_or_register_derived" and "direct_Leh" not in value for value in x))),
        )
    )
    published_means = {
        ("Kathmandu", "1854"): 62.4,
        ("Kathmandu", "1855"): 61.7,
        ("Kathmandu", "1856"): 63.5,
        ("Simla", "1854"): 58.7,
        ("Skardo", "1856"): 51.3,
        ("Leh", "1856 composite"): 42.9,
    }
    annual["published_annual_mean_f"] = [published_means.get((row.station, row.period), np.nan) for row in annual.itertuples()]
    annual["published_annual_mean_c"] = annual["published_annual_mean_f"].map(lambda x: fahrenheit_to_celsius(x) if pd.notna(x) else np.nan)
    annual["published_minus_recomputed_f"] = annual["published_annual_mean_f"] - annual["arithmetic_annual_mean_f"]
    annual.to_csv(args.output / "verified_annual_temperature_summary.csv", index=False)

    station_roles = values[
        ["station", "latitude_n", "longitude_e", "elevation_ft", "observation_family", "independence_class", "value_role", "notes"]
    ].drop_duplicates()
    station_roles.to_csv(args.output / "station_role_and_independence_audit.csv", index=False)
    make_figure(values, args.output / "verified_1850s_monthly_temperature_curves.png")

    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Verified source page changed during extraction")
    summary = {
        "experiment": "temperature_schlagintweit_1850s_structured_extraction_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_visually_verified_structured_extraction",
        "station_count": int(values.station.nunique()),
        "series_count": int(values[["station", "period"]].drop_duplicates().shape[0]),
        "monthly_value_count": int(len(values)),
        "independent_pre_GHCN_local_anchors": ["Kathmandu 1854-1856", "Skardo 1856"],
        "same_station_family_chain_anchors": ["Simla 1854", "Darjeeling 1857-1859 mean"],
        "mixed_or_imputed_series": ["Leh 1856 composite"],
        "scientific_limit": "local month/elevation and short annual anchors only; insufficient for a uniform HMA correction",
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
        "visual_verification_required_and_completed": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = """# Schlagintweit 1850s temperature extraction

The official Heidelberg scans were visually read and transcribed into a new, independent result package. Original scans and all production files remained unchanged.

## What is now numeric

- Kathmandu: monthly values for 1854, 1855, and 1856; independent pre-GHCN historical register.
- Skardo: monthly values for 1856; independent expedition/local-establishment observations.
- Simla: monthly values for 1854; useful as a pre-GHCN extension of the same physical station family, not a second independent climate vote.
- Darjeeling: 1857-1859 monthly climatology; same physical station family and not an annual sequence.
- Leh: 1856 composite. Only July-September are direct Leh observatory months. May-June are nearby-establishment values, November-April are derived from Kanam, and October follows isolated historical evidence.

## Scientific decision

These records materially improve the evidence base before 1875, but they do not justify a uniform HMA-wide temperature shift. They can test local sign, seasonality, elevation consistency, and short-period absolute level. Leh's composite annual mean must not be used as direct annual truth.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 accessed or used: **no**
- Production reconstruction modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
