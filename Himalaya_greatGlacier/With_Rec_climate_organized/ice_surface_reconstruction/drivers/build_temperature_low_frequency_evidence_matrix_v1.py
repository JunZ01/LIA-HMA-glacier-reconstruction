from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v1"

STATION = (
    LOOP
    / "temperature_1850_1950_residual_bias_final_synthesis_v1"
    / "regional_station_residual_summary.csv"
)
EAST = (
    LOOP
    / "round_217_temperature_bias_phase_three_record_synthesis_v1"
    / "three_record_historical_bias_evidence.csv"
)
WGMS = (
    LOOP
    / "wgms_hma_front_variation_adjudication_v2"
    / "valid_glacier_summary.csv"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_rows() -> list[dict]:
    common = {
        "sealed_period_values_used": False,
        "production_modified": False,
    }
    rows = [
        {
            "evidence_id": "GHCNM_QCF_15_STATION_PANEL",
            "region": "west_to_east_HMA_sparse",
            "longitude": "71.5-97.167",
            "coverage": "1880-1920;1940-1950",
            "target": "annual_Tmean_anomaly",
            "measurement_family": "direct_station",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "direct_observation_single_family",
            "spatial_relevance": "high_but_west_weighted",
            "season_match": "annual",
            "admitted_role": "quantitative_level_and_trend_gate",
            "finding": "west project-minus-qcf residual +1.418 C in 1880-1900, +0.847 C in 1901-1920, +0.177 C in 1940-1950",
            "next_action": "retain_as_primary_direct_gate",
            "source_url": "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly",
            **common,
        },
        {
            "evidence_id": "IMD_PRINTED_ANNUAL_TABLES",
            "region": "western_and_eastern_Himalayan_stations",
            "longitude": "73.38-88.27",
            "coverage": "isolated_years_1878-1890",
            "target": "printed_annual_air_temperature",
            "measurement_family": "direct_station_same_observations",
            "machine_readable": False,
            "numeric_now": True,
            "source_independence": "same_physical_observations_as_GHCN",
            "spatial_relevance": "high",
            "season_match": "annual",
            "admitted_role": "processing_and_transcription_audit_only",
            "finding": "six printed Year rows; qcf/qcu agreement differs by station and sign",
            "next_action": "do_not_count_as_second_climate_vote",
            "source_url": "https://archive.org/details/meteorologicalreporterindia",
            **common,
        },
        {
            "evidence_id": "HUANG_SETP_WINTER_TMIN",
            "region": "eastern_HMA",
            "longitude": "east_HMA",
            "coverage": "1500-1899_aggregate",
            "target": "winter_Tmin",
            "measurement_family": "independent_reconstruction",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_record",
            "spatial_relevance": "east_only",
            "season_match": "winter_minimum_not_annual_mean",
            "admitted_role": "historical_direction_gate",
            "finding": "external median +0.127 C; project local median +0.182 C",
            "next_action": "reject_uniform_HMA_cooling",
            "source_url": "local_admitted_round_217",
            **common,
        },
        {
            "evidence_id": "BAIYU_JUNE_JULY",
            "region": "eastern_HMA",
            "longitude": "east_HMA",
            "coverage": "1500-1899_aggregate",
            "target": "June-July_temperature",
            "measurement_family": "independent_reconstruction",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_record",
            "spatial_relevance": "east_only",
            "season_match": "summer_not_annual_mean",
            "admitted_role": "historical_direction_gate",
            "finding": "external median +0.191 C; project local median +0.152 C",
            "next_action": "reject_uniform_HMA_cooling",
            "source_url": "local_admitted_round_217",
            **common,
        },
        {
            "evidence_id": "HAIZI_ANNUAL",
            "region": "eastern_HMA",
            "longitude": "east_HMA",
            "coverage": "1500-1899_aggregate",
            "target": "annual_temperature",
            "measurement_family": "independent_reconstruction",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_record",
            "spatial_relevance": "east_only",
            "season_match": "annual",
            "admitted_role": "historical_direction_and_level_gate",
            "finding": "external median +0.260 C; project local median +0.122 C",
            "next_action": "reject_uniform_HMA_cooling",
            "source_url": "local_admitted_round_217",
            **common,
        },
        {
            "evidence_id": "DOKRIANI_TELA_TRW_2023",
            "region": "western_Himalaya_Din_Gad",
            "longitude": "78.83",
            "coverage": "1785-2015",
            "target": "March-June_temperature",
            "measurement_family": "tree_ring_reconstruction",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "new_sequence_regional_and_team_partial_overlap",
            "spatial_relevance": "high_western_Himalaya",
            "season_match": "spring_early_summer",
            "admitted_role": "published_direction_only_pending_numeric_identity",
            "finding": "published metadata identifies 1890-1910 as the warmest 21-year interval; exact sequence duplication not found",
            "next_action": "request_annual_series_chronology_and_sample_ids_from_jayendrasingh@wihg.res.in",
            "source_url": "https://doi.org/10.1016/j.quaint.2023.05.013",
            **common,
        },
        {
            "evidence_id": "UPPER_INDUS_MXD_2024",
            "region": "western_HMA_Astore_Rama",
            "longitude": "upper_Indus",
            "coverage": "1370-2020",
            "target": "March-September_temperature",
            "measurement_family": "MXD_reconstruction",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "identity_audit_pending",
            "spatial_relevance": "high_western_HMA",
            "season_match": "warm_season",
            "admitted_role": "candidate_pending_author_data",
            "finding": "57 percent instrumental variance reported for 1972-2020; no public annual table found yet",
            "next_action": "request_annual_series_site_and_sample_crosswalk",
            "source_url": "https://doi.org/10.1016/j.scitotenv.2024.177393",
            **common,
        },
        {
            "evidence_id": "BANERJEE_AZAM_GLACIER_LENGTH_2016",
            "region": "Himalaya_Karakoram_34_glaciers",
            "longitude": "regional",
            "coverage": "1860-2010",
            "target": "glacier_length_inverted_temperature",
            "measurement_family": "physical_glacier_inversion",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "non_tree_but_glacier_response_family",
            "spatial_relevance": "high_regional",
            "season_match": "effective_glacier_temperature",
            "admitted_role": "direction_and_broad_amplitude_pending_annual_table",
            "finding": "published total regional warming about 1.6 K over 1860-2010",
            "next_action": "request_annual_reconstruction_and_uncertainty_from_argha@iiserpune.ac.in",
            "source_url": "https://doi.org/10.3189/2016AoG71A047",
            **common,
        },
        {
            "evidence_id": "MALAN_GLACIER_BOREHOLE_2021",
            "region": "northern_Tibetan_Plateau",
            "longitude": "Malan",
            "coverage": "approximately_1900s-1990s",
            "target": "ground_surface_temperature_history",
            "measurement_family": "borehole_physical_inversion",
            "machine_readable": "profile_only",
            "numeric_now": False,
            "source_independence": "different_measurement_but_colocated_with_project_Malan_family",
            "spatial_relevance": "north_edge",
            "season_match": "low_frequency_GST_not_air_Tmean",
            "admitted_role": "direction_and_century_scale_context",
            "finding": "published century-scale GST increase about 1.1 C with clear warming since the 1950s",
            "next_action": "digitize_only_if_author_curve_unavailable_and_report_resolution_uncertainty",
            "source_url": "https://doi.org/10.1080/15230430.2021.1974667",
            **common,
        },
        {
            "evidence_id": "MULI_DK12_BOREHOLE_2020",
            "region": "Qilian_north_HMA_edge",
            "longitude": "Muli",
            "coverage": "approximately_1400_years",
            "target": "ground_surface_temperature_history",
            "measurement_family": "borehole_physical_inversion",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "independent_non_tree",
            "spatial_relevance": "north_edge_not_core_HMA",
            "season_match": "low_frequency_GST_not_air_Tmean",
            "admitted_role": "direction_only_pending_numeric_curve",
            "finding": "published major warming onset around 1500; insufficient spatial and temporal resolution for an HMA offset",
            "next_action": "seek_author_inversion_table_or_digitize_with_uncertainty",
            "source_url": "https://doi.org/10.1080/15230430.2020.1801149",
            **common,
        },
        {
            "evidence_id": "WGMS_WEST_KARAKORAM_FRONT",
            "region": "western_Karakoram",
            "longitude": "74.58-75.0",
            "coverage": "1889-1913",
            "target": "glacier_front_variation",
            "measurement_family": "physical_glacier_observation",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_physical_family",
            "spatial_relevance": "west_only_two_glaciers",
            "season_match": "integrated_mass_balance_and_dynamics",
            "admitted_role": "direction_only",
            "finding": "Chogo Lungma +205 m and Minapin +1670 m advance before 1921",
            "next_action": "use_as_spatial_heterogeneity_gate_not_temperature_offset",
            "source_url": "https://wgms.ch/data_databaseversions/",
            **common,
        },
        {
            "evidence_id": "REACHES_DOCUMENTARY",
            "region": "China_east_of_97.53E",
            "longitude": ">=97.53",
            "coverage": "1368-1911",
            "target": "documentary_temperature_indices",
            "measurement_family": "documentary_reconstruction",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_documentary",
            "spatial_relevance": "zero_overlap_with_70-90E_target",
            "season_match": "annual_and_seasonal",
            "admitted_role": "spatial_mismatch_context_only",
            "finding": "no polygon overlap with the western-central HMA target",
            "next_action": "do_not_use_to_set_correction",
            "source_url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=noaa-historical-37720",
            **common,
        },
        {
            "evidence_id": "CRU_BERKELEY_BACKGROUND_PRODUCTS",
            "region": "HMA_grids",
            "longitude": "regional",
            "coverage": "instrumental_era",
            "target": "gridded_air_temperature",
            "measurement_family": "station_derived_grids",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "shared_station_inputs_not_independent_votes",
            "spatial_relevance": "high_after_coverage_improves",
            "season_match": "monthly_to_annual",
            "admitted_role": "background_context_and_sensitivity_only",
            "finding": "cannot independently define a pre-1950 correction magnitude",
            "next_action": "retain_as_background_products",
            "source_url": "https://crudata.uea.ac.uk/cru/data/hrg/;https://berkeleyearth.org/data/",
            **common,
        },
        {
            "evidence_id": "TWENTIETH_CENTURY_REANALYSIS_V3",
            "region": "not_applicable",
            "longitude": "not_applicable",
            "coverage": "not_accessed",
            "target": "prohibited_source",
            "measurement_family": "prohibited",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "not_evaluated",
            "spatial_relevance": "not_evaluated",
            "season_match": "not_evaluated",
            "admitted_role": "PROHIBITED_DO_NOT_ACCESS_USE_OR_CITE",
            "finding": "excluded by explicit user decision",
            "next_action": "none",
            "source_url": "",
            **common,
        },
    ]
    return rows


def plot_quantitative_summary(station: pd.DataFrame, east: pd.DataFrame) -> None:
    west = station[(station["region_group"] == "west_lt80") & (station["target"] == "qcf")].copy()
    x = [1890, 1910.5, 1945]
    labels = ["1880-1900", "1901-1920", "1940-1950"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    axes[0].axhline(0, color="#3b4148", lw=1)
    axes[0].scatter(x, west["mean_project_minus_station_degC"], color="#c83f3f", s=58, zorder=3)
    for xx, yy, n in zip(x, west["mean_project_minus_station_degC"], west["station_count"]):
        axes[0].annotate(f"n={int(n)}", (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Project minus GHCN-M qcf (degC)")
    axes[0].set_title("Western direct-station residual")
    axes[0].grid(axis="y", color="#d9dde1", lw=0.7)

    idx = range(len(east))
    axes[1].axhline(0, color="#3b4148", lw=1)
    axes[1].scatter(idx, east["external_median_degC"], label="Independent record", color="#187a6b", s=56)
    axes[1].scatter(idx, east["project_median_degC"], label="Project local", color="#315a9b", marker="s", s=50)
    axes[1].scatter(idx, east["reference_median_degC"], label="Four-family reference", color="#8f6f36", marker="^", s=52)
    axes[1].set_xticks(list(idx), ["Huang winter Tmin", "Baiyu Jun-Jul", "Haizi annual"], rotation=12, ha="right")
    axes[1].set_ylabel("1500-1899 median anomaly (degC)")
    axes[1].set_title("Eastern independent historical gate")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#d9dde1", lw=0.7)

    fig.savefig(OUT / "quantitative_evidence_summary.png", dpi=220)
    plt.close(fig)


def plot_regional_map(wgms: pd.DataFrame) -> None:
    points = [
        (77.70, 34.20, "GHCN", "direct station"),
        (77.17, 31.10, "GHCN", "direct station"),
        (73.38, 33.92, "GHCN", "direct station"),
        (79.65, 29.47, "GHCN", "direct station"),
        (78.83, 30.83, "Dokriani", "numeric requested"),
        (91.13, 29.67, "GHCN", "direct station"),
        (97.17, 31.15, "GHCN", "direct station"),
    ]
    for row in wgms.itertuples():
        points.append((row.longitude, row.latitude, row.name, "direction only"))

    colors = {
        "direct station": "#c83f3f",
        "numeric requested": "#315a9b",
        "direction only": "#187a6b",
    }
    markers = {"direct station": "o", "numeric requested": "s", "direction only": "^"}

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    for role in colors:
        subset = [p for p in points if p[3] == role]
        ax.scatter([p[0] for p in subset], [p[1] for p in subset], c=colors[role], marker=markers[role], s=55, label=role)
    for lon, lat, name, role in points:
        if name not in {"GHCN"}:
            ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(70, 100)
    ax.set_ylim(27, 40)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Low-frequency evidence coverage: current western-central gap")
    ax.grid(color="#d9dde1", lw=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "regional_evidence_map.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    station = pd.read_csv(STATION)
    east = pd.read_csv(EAST)
    wgms = pd.read_csv(WGMS)

    assert not station["period"].str.contains("1921|1922|1923|1924|1925|1926|1927|1928|1929|1930|1931|1932|1933|1934|1935|1936|1937|1938|1939").any()
    assert wgms["last_survey_year"].max() <= 1920

    rows = evidence_rows()
    write_csv(OUT / "evidence_matrix.csv", rows)

    requests = [
        {
            "priority": 1,
            "record": "Dokriani Tela 1785-2015 March-June temperature",
            "contact": "jayendrasingh@wihg.res.in",
            "request": "annual reconstruction, uncertainty, reference baseline, chronology, tree/sample IDs, and instrumental target",
        },
        {
            "priority": 2,
            "record": "Upper Indus MXD 1370-2020 March-September temperature",
            "contact": "adam@ulm.edu.pk;feng653@163.com",
            "request": "annual reconstruction, uncertainty, chronology, site/sample crosswalk, and calibration target",
        },
        {
            "priority": 3,
            "record": "Himalaya-Karakoram glacier-length temperature 1860-2010",
            "contact": "argha@iiserpune.ac.in",
            "request": "annual or pentadal reconstruction, 95 percent interval, glacier inclusion by year, baseline, and inversion output",
        },
    ]
    write_csv(OUT / "author_data_request_queue.csv", requests)

    manifest = [
        {"path": str(path), "sha256": sha256(path), "role": "read_only_input"}
        for path in (STATION, EAST, WGMS)
    ]
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plot_quantitative_summary(station, east)
    plot_regional_map(wgms)

    summary = {
        "experiment": OUT.name,
        "decision": "no_uniform_HMA_cooling_western_early_residual_requires_regional_numeric_evidence",
        "evidence_rows": len(rows),
        "numeric_currently_available": sum(bool(row["numeric_now"]) for row in rows),
        "author_requests": len(requests),
        "new_fully_independent_west_central_numeric_temperature_series": 0,
        "uniform_historical_warm_bias_supported": False,
        "western_1880_1920_residual_supported": True,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# HMA low-frequency temperature evidence matrix v1

## Decision

`no_uniform_HMA_cooling_western_early_residual_requires_regional_numeric_evidence`

The evidence now separates two questions that had previously been conflated.

1. A uniform 1500-1899 HMA warm bias is not supported. Three admitted eastern records are all positive over 1500-1899 and the project local medians are substantially closer to them than the four-family reference.
2. A western early instrumental residual is supported. The project is warmer than the GHCN-M qcf station panel by +1.418 C in 1880-1900 and +0.847 C in 1901-1920, falling to +0.177 C in 1940-1950.

The new Dokriani record is highly relevant: its published metadata identifies 1890-1910 as its warmest 21-year interval. This means a genuinely warm western phase is plausible even while the project remains too warm relative to stations in absolute anomaly. The annual values are not public, so the record cannot yet define the correction magnitude. Exact sequence duplication was not found, but the record is regionally and institutionally related to the project's Gangotri evidence family; it remains a requested numeric source rather than a blind independent vote.

WGMS observations add an independent physical direction gate: Chogo Lungma and Minapin advanced before 1921. They establish western spatial heterogeneity, but glacier dynamics and precipitation prevent conversion to a temperature offset.

## Consequence

Do not apply a constant or HMA-wide cooling. The next defensible correction experiment requires at least one machine-readable western or central annual series with uncertainty and identity metadata. The first requests are Dokriani, Upper Indus MXD, and the Banerjee-Azam glacier-length inversion. If only direction evidence is obtained, retain the original production temperature. If numeric data arrive, test a regional and seasonal correction against the direct-station level and trend gates without using 1921-1939.

## Safety

- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No 1921-1939 annual value was read, saved, plotted, or used.
- Twentieth Century Reanalysis v3 was not accessed and is prohibited from use.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
