from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
BASE = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v5"
OUT = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update(rows: list[dict[str, str]], evidence_id: str, values: dict[str, str]) -> None:
    target = next(row for row in rows if row["evidence_id"] == evidence_id)
    target.update(values)


def new_row(template: dict[str, str], **values: str) -> dict[str, str]:
    row = {key: "" for key in template}
    row.update(values)
    row["sealed_period_values_used"] = "False"
    row["production_modified"] = "False"
    return row


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    inputs = [
        BASE / "evidence_matrix.csv",
        BASE / "author_data_request_queue.csv",
        LOOP / "temperature_schlagintweit_1850s_structured_extraction_v3/summary.json",
        LOOP / "temperature_schlagintweit_darjeeling_vs_ghcn_v2/summary.json",
        LOOP / "temperature_schlagintweit_simla_1854_project_anchor_v1/summary.json",
        LOOP / "temperature_india_weather_review_1899_1920_ocr_recovery_v2/summary.json",
        LOOP / "temperature_india_weather_review_1906_vs_ghcn_v2/summary.json",
        LOOP / "temperature_west_central_author_repository_search_v1/summary.json",
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    before = {str(path.resolve()): sha256(path) for path in inputs}
    rows = read_csv(inputs[0])

    update(
        rows,
        "GHCNM_QCF_15_STATION_PANEL",
        {
            "coverage": "1850-1920_sparse;1940-1950",
            "admitted_role": "qcf_primary_local_level_and_trend_gate_qcu_homogenization_sensitivity",
            "finding": (
                "western project-minus-qcf residual remains warm in 1880-1920; primary-scan comparison found two likely "
                "approximately 10 F leading-digit errors in early Darjeeling GHCN (1850-12 and 1852-11), while the other "
                "22 overlapping months reproduce with r=0.9988 and RMSE about 0.25 C"
            ),
            "next_action": "flag_the_two_months_from_primary_scan_and_do_not_treat_qcf_qcu_as_independent_votes",
        },
    )
    update(
        rows,
        "SCHLAGINTWEIT_HIGH_ASIA_1854_1858",
        {
            "coverage": "Darjeeling_1850_1852_1855;Kathmandu_1854-1856;Simla_1854;Massuri_Kardong_Leh_Skardo_segments",
            "target": "monthly_air_temperature_with_direct_filled_and_composite_status",
            "measurement_family": "historical_direct_station_register_and_expedition_observations",
            "machine_readable": "verified_manual_transcription_CSV",
            "numeric_now": "True",
            "source_independence": "mixed_independent_local_records_and_same_station_family_extensions",
            "spatial_relevance": "eight_local_stations_across_east_central_and_west_HMA",
            "season_match": "monthly_but_observation_time_and_fill_status_must_be_preserved",
            "admitted_role": "direct_local_annual_and_seasonal_anchor_not_regional_truth",
            "finding": (
                "138 verified monthly values in 12 series; Darjeeling 1850/1852 full; Kathmandu 1854-1856 full; Simla 1854 full; "
                "Kardong only May-Oct direct; Leh only Jul-Sep direct; Skardo is local establishment/environs rather than one fixed station. "
                "At Simla 1854 project-minus-official/qcf is +0.349 C, but the project lies inside the qcf/qcu homogenization envelope"
            ),
            "next_action": "use_station_specific_1961-1990_normals_where_eligible_and_retain_only_direct_months_for_seasonal_gates",
        },
    )
    update(
        rows,
        "INDIA_MONTHLY_WEATHER_REVIEW_1891_1920",
        {
            "coverage": "12_verified_safe_publication_years_1899-1920_with_gaps;1906_four_station_table",
            "machine_readable": "official_scans_indexed_OCR_plus_verified_manual_table_transcription",
            "numeric_now": "True",
            "admitted_role": "same_family_processing_transcription_and_gap_audit",
            "finding": (
                "13 verified sections and 1807 pages indexed; 1906 Table XI monthly departures for Leh Srinagar Simla and Darjeeling "
                "correlate 0.898-0.986 with GHCN after matched historical climatology; qcf/qcu have no universal winner"
            ),
            "next_action": "extract_only_year_station_tables_needed_to_resolve_GHCN_gaps_or_suspected_errors",
            "source_url": "https://archive.org/details/108906",
        },
    )
    update(
        rows,
        "RARA_CHOLANGPATI_MXD_2026",
        {
            "source_independence": "Rara_new_MXD_not_same_physical_sample_as_project_NEPA037;Cholangpati_exact_tree_overlap_unknown;composite_includes_old_sites",
            "finding": (
                "Rara and Cholangpati are new 2023 MXD collections, but the published regional composite also uses old Jumla/Kalinchowk sites; "
                "supplement contains no annual numeric series"
            ),
            "next_action": "request_Rara_and_Cholangpati_new_site_only_RCS_NegEx_combined_series_and_tree_ID_crosswalk",
        },
    )
    update(
        rows,
        "DOKRIANI_TELA_TRW_2023",
        {
            "source_independence": "new_regional_publication_but_likely_reuses_2019_Tela_Abies_cores",
            "finding": "published 1785-2015 reconstruction has no public annual table; same-team Tela core reuse is likely and exact sample crosswalk is unresolved",
            "next_action": "request_annual_series_and_core_ID_crosswalk_before_counting_as_independent",
        },
    )
    update(
        rows,
        "UPPER_INDUS_MXD_2024",
        {
            "source_independence": "MXD_variable_differs_from_project_TRW_but_exact_PAKI017_018_025_tree_crosswalk_missing",
            "finding": "public DOCX supplement contains no annual sequence; exact overlap with project Upper Indus sample families remains unresolved",
            "next_action": "request_annual_series_and_tree_core_crosswalk_before_independence_admission",
        },
    )

    template = rows[0]
    rows.append(
        new_row(
            template,
            evidence_id="WORLD_WEATHER_RECORDS_1800_1920",
            region="HMA_station_compilation",
            longitude="regional_station_tables",
            coverage="earliest_records_to_1920",
            target="monthly_station_temperature",
            measurement_family="official_historical_station_compilation",
            machine_readable="digitized_scan_OCR_route",
            numeric_now="False",
            source_independence="same_physical_observations_as_national_reports_and_GHCN",
            spatial_relevance="potentially_high_but_station_dependent",
            season_match="monthly",
            admitted_role="standardized_cross_check_and_fast_index_not_independent_vote",
            finding="official compilation can accelerate station discovery but cannot multiply evidence because it republishes national observations",
            next_action="use_only_to_locate_and_cross_check_values_then_return_to_primary_national_table_for_disputes",
            source_url="https://library.si.edu/digital-library/book/worldweatherreco",
        )
    )
    rows.append(
        new_row(
            template,
            evidence_id="CUOQIA_LAKE_BRGDGT_300Y",
            region="southeastern_Tibetan_Plateau_Hengduan",
            longitude="99.772",
            coverage="approximately_1700-2015",
            target="March-October_temperature",
            measurement_family="lake_sediment_brGDGT_quantitative_reconstruction",
            machine_readable="article_figures_and_supplement_route_not_yet_numeric",
            numeric_now="False",
            source_independence="independent_lake_sediment_physical_family",
            spatial_relevance="high_for_southeast_HMA_low_for_west_central",
            season_match="warm_season_March-October",
            admitted_role="independent_east_southeast_direction_and_low_frequency_shape",
            finding="published reconstruction declines before 1950 and rises rapidly after 1950; calibration reports R2=0.89 against nearby instrumental warm-season temperature",
            next_action="recover_numeric_CQ1_CQ2_temperature_series_and_age_uncertainty_before_amplitude_use",
            source_url="https://doi.org/10.3389/fevo.2023.1119869",
        )
    )
    rows.append(
        new_row(
            template,
            evidence_id="WUDAOLIANG_BOREHOLE_GST_2021",
            region="central_northern_Tibetan_Plateau",
            longitude="Wudaoliang",
            coverage="approximately_1705-2013_smoothed",
            target="ground_surface_temperature_history",
            measurement_family="borehole_temperature_inversion",
            machine_readable="published_curve_and_borehole_log_route",
            numeric_now="False",
            source_independence="independent_geothermal_physical_family",
            spatial_relevance="central_north_HMA_edge_single_site",
            season_match="multi_decadal_GST_not_annual_air_temperature",
            admitted_role="broad_century_scale_direction_and_amplitude_envelope",
            finding="Tikhonov inversion reports about +1.8 +/- 1.6 C GST change over 308 years; uncertainty is too large and temporal smoothing too strong for an 1850-1950 offset",
            next_action="recover_digitized_GST_history_and_resolution_kernel_then_use_only_as_low_frequency_envelope",
            source_url="https://doi.org/10.1080/17415977.2021.1975700",
        )
    )

    write_csv(OUT / "evidence_matrix.csv", rows)
    write_csv(OUT / "author_data_request_queue.csv", read_csv(inputs[1]))
    summary = {
        "experiment": OUT.name,
        "supersedes": BASE.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "no_uniform_HMA_correction_direct_local_evidence_now_numeric_targeted_residual_tests_continue",
        "evidence_rows": len(rows),
        "new_numeric_direct_package": "Schlagintweit_138_month_values_8_stations_12_series",
        "new_primary_archive_error_flags": ["Darjeeling_1850_12", "Darjeeling_1852_11"],
        "simla_1854_project_minus_qcf_c": 0.3488123674258094,
        "simla_1854_project_inside_qcf_qcu_envelope": True,
        "new_fully_independent_long_annual_west_central_series": 0,
        "new_short_independent_central_annual_chain": "Kathmandu_1854-1856",
        "uniform_historical_warm_bias_supported": False,
        "localized_qcf_primary_warm_residual_supported": True,
        "candidate_v2_authorized": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# HMA 1850-1950 low-frequency temperature evidence matrix v6

## What materially changed

The direct historical evidence is no longer only a search route. Official Heidelberg tables were visually transcribed into 138 numeric monthly values across eight stations and twelve series. Direct, filled, composite, and same-station-family values are explicitly separated.

The primary-scan audit also found two likely leading-digit errors in early GHCN Darjeeling: December 1850 behaves as 55.9 F in GHCN but is 45.9 F in the scan, while November 1852 behaves as 48.0 F in GHCN but is 58.0 F in the scan. The other 22 overlap months reproduce at r=0.9988 and about 0.25 C RMSE.

At Simla in 1854, the project six-member anomaly is +0.043 C. The official observation normalized with qcf is -0.306 C, supporting a local project warm residual of +0.349 C under the qcf-primary convention. The qcu normalization is +1.401 C, so the project still lies inside the qcf/qcu homogenization envelope. This one local year cannot set a regional correction.

India Weather Review is now an indexed and partly numeric same-family audit route. The 1906 four-station table reproduces GHCN month-to-month structure strongly, but qcf and qcu have no universal winner.

## Current decision

`no_uniform_HMA_correction_direct_local_evidence_now_numeric_targeted_residual_tests_continue`

There is now direct evidence for localized historical residuals and source-data errors. There is still no independent, spatially balanced evidence for one HMA-wide cooling constant, one fixed four-background mixture, or a production Candidate v2. The next useful work is station-specific anomaly construction where valid modern normals exist, plus recovery of numeric Cuoqia and Wudaoliang low-frequency curves as physically independent eastern/central envelopes.

## Safety

- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No value from 1921-1939 was used.
- The permanently prohibited source was not accessed or used.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    after = {str(path.resolve()): sha256(path) for path in inputs}
    manifest = {
        "read_only_inputs_before": before,
        "read_only_inputs_after": after,
        "all_inputs_unchanged": before == after,
        "external_primary_sources": [
            "https://doi.org/10.3389/fevo.2023.1119869",
            "https://doi.org/10.1080/17415977.2021.1975700",
            "https://library.si.edu/digital-library/book/worldweatherreco",
        ],
        "sealed_period_values_used": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
        "output_directory": str(OUT),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("A declared read-only input changed")


if __name__ == "__main__":
    main()
