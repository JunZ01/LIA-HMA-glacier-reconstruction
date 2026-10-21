from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
BASE = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v3"
REOPEN = LOOP / "temperature_low_frequency_reopen_boundary_v1"
OUT = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_row(**values) -> dict:
    row = {
        "evidence_id": "",
        "region": "",
        "longitude": "",
        "coverage": "",
        "target": "",
        "measurement_family": "",
        "machine_readable": False,
        "numeric_now": False,
        "source_independence": "",
        "spatial_relevance": "",
        "season_match": "",
        "admitted_role": "",
        "finding": "",
        "next_action": "",
        "source_url": "",
        "sealed_period_values_used": False,
        "production_modified": False,
    }
    row.update(values)
    return row


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    base_matrix = BASE / "evidence_matrix.csv"
    base_queue = BASE / "author_data_request_queue.csv"
    reopen_summary = REOPEN / "summary.json"
    sources = [base_matrix, base_queue, reopen_summary]
    before = {str(path): sha256(path) for path in sources}

    rows = [
        row
        for row in read_csv(base_matrix)
        if row["measurement_family"] != "prohibited"
    ]

    by_id = {row["evidence_id"]: row for row in rows}
    sano = by_id["SANO_HUMLA_TRW_MXD_2005"]
    sano.update(
        {
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "same_physical_46_cores_23_trees_as_project_proxy9_199",
            "admitted_role": "same_sample_different_variable_sensitivity_only",
            "finding": (
                "official repository audit found no public annual Sano temperature series; "
                "later Humla cellulose record maps to the same cores and trees as proxy9_199"
            ),
            "next_action": "do_not_count_as_independent_vote; author_request_only_if_samplewise_temperature_series_exists",
        }
    )

    rara = by_id["RARA_CHOLANGPATI_MXD_2026"]
    rara.update(
        {
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "two_new_2023_sites_plus_two_old_ITRDB_sites_mixed",
            "admitted_role": "chronology_mechanism_evidence_pending_new_site_numeric_data",
            "finding": (
                "RCS low-frequency plus NegEx high-frequency restores centennial warming and modern skill; "
                "public repository search exhausted without annual numeric release"
            ),
            "next_action": "request new-site-only RCS NegEx combined and uncertainty series from author",
        }
    )

    rows.extend(
        [
            new_row(
                evidence_id="CN_PX202_BOREHOLE_GST",
                region="eastern_HMA_edge_Sichuan",
                longitude="102.22",
                coverage="century_bins_pre1500_to_1984_measurement",
                target="ground_surface_temperature_change",
                measurement_family="borehole_physical_inversion",
                machine_readable=True,
                numeric_now=True,
                source_independence="independent_non_tree_physical_measurement",
                spatial_relevance="east_edge_single_site",
                season_match="century_scale_GST_not_annual_air_Tmean",
                admitted_role="pre1900_century_direction_gate_only",
                finding=(
                    "official standardized inversion gives nineteenth-century GST change +0.023 C "
                    "relative to pre-1500 and eighteenth-century +0.511 C; broad smoothing and one site "
                    "cannot set an HMA-wide offset"
                ),
                next_action="compare only allowed pre1900 century direction with local project grid; exclude all later bins",
                source_url="https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=noaa-borehole-1001030",
            ),
            new_row(
                evidence_id="SCHLAGINTWEIT_HIGH_ASIA_1854_1858",
                region="Bhutan_to_Kashmir_High_Asia_transect",
                longitude="multi_station",
                coverage="1854-1858_short_campaigns",
                target="direct_air_temperature_and_meteorological_observations",
                measurement_family="historical_direct_observation",
                machine_readable=False,
                numeric_now=False,
                source_independence="independent_expedition_observations",
                spatial_relevance="high_but_noncontinuous_transect",
                season_match="observation_time_and_month_specific",
                admitted_role="early_level_anchor_pending_table_extraction_and_instrument_audit",
                finding="digitized primary volumes contain High Asia observations but no admitted numeric panel yet",
                next_action="map table pages and extract only auditable station-month values with instrument and local-time metadata",
                source_url="https://digi.ub.uni-heidelberg.de/diglit/schlagintweit1866bd4",
            ),
            new_row(
                evidence_id="INDIA_MONTHLY_WEATHER_REVIEW_1891_1920",
                region="British_India_Himalayan_station_network",
                longitude="western_central_eastern_Himalaya",
                coverage="1891-1920",
                target="monthly_and_annual_station_temperature",
                measurement_family="official_historical_station_tables",
                machine_readable="digitized_scans_OCR_availability_pending",
                numeric_now=False,
                source_independence="same_physical_station_family_as_GHCN_when_overlapping",
                spatial_relevance="potentially_high_multi_region",
                season_match="monthly_enables_annual_and_seasonal",
                admitted_role="primary_processing_audit_and_gap_filling_pending_extraction",
                finding="new official publication route extends beyond the previously audited 1875-1890 IMD volumes",
                next_action="resolve volume manifests OCR and HMA station index; adjudicate printed values before comparison",
                source_url="https://catalog.crl.edu/",
            ),
            new_row(
                evidence_id="CHONGCE_ICE_CORE_ALREADY_ASSESSED",
                region="western_HMA",
                longitude="western_HMA",
                coverage="previous_blind_validation",
                target="temperature_reconstruction",
                measurement_family="independent_reconstruction",
                machine_readable=True,
                numeric_now=True,
                source_independence="previously_adjudicated",
                spatial_relevance="west",
                season_match="record_specific",
                admitted_role="closed_existing_evidence_not_new",
                finding="candidate improved RMSE by 16.6 percent but retained +1.153 C mean bias and failed all acceptance gates",
                next_action="do_not_recount_or_rerun_without_new_information",
                source_url="local_prior_audit",
            ),
            new_row(
                evidence_id="NGRAMRING_TIANCAI_AWENG_ALREADY_ASSESSED",
                region="Tibetan_Plateau_HMA_edges",
                longitude="multi_site",
                coverage="sparse_prelock_points",
                target="external_low_frequency_temperature",
                measurement_family="independent_reconstruction",
                machine_readable=True,
                numeric_now=True,
                source_independence="previously_adjudicated",
                spatial_relevance="site_specific",
                season_match="record_specific",
                admitted_role="closed_existing_sparse_evidence_not_new",
                finding="Ngamring 4, Tiancai 2, and Aweng 5 usable prelock points are too sparse to define a correction",
                next_action="do_not_recount_as_new_low_frequency_evidence",
                source_url="local_round_175_176_audit",
            ),
        ]
    )

    write_csv(OUT / "evidence_matrix.csv", rows)

    queue = read_csv(base_queue)
    queue_ids = {row["record"] for row in queue}
    request = "Central Himalayan new-site MXD low-frequency chronology"
    if request not in queue_ids:
        queue.append(
            {
                "priority": "1",
                "record": request,
                "contact": "nahtz@geo.uni-mainz.de",
                "request": (
                    "new Rara and Cholangpati site-only annual NegEx, RCS and combined series, "
                    "uncertainty and sample crosswalk; omit 1921-1939 values"
                ),
            }
        )
    for index, row in enumerate(queue, start=1):
        row["priority"] = str(index)
    write_csv(OUT / "author_data_request_queue.csv", queue)

    summary = {
        "experiment": OUT.name,
        "decision": "no_uniform_HMA_correction_new_direct_and_physical_routes_prioritized",
        "evidence_rows": len(rows),
        "new_actionable_routes": [
            "India_Monthly_Weather_Review_1891_1920",
            "Schlagintweit_1854_1858",
            "CN_PX202_pre1900_century_gate",
            "new_site_RCS_data_request",
        ],
        "public_new_site_RCS_numeric_data_found": False,
        "new_fully_independent_annual_west_central_temperature_series": 0,
        "uniform_historical_warm_bias_supported": False,
        "western_1880_1920_station_residual_supported": True,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "sealed_overlapping_metadata_aggregate_excluded": True,
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# HMA 1850-1950 low-frequency temperature evidence matrix v4

## Decision

`no_uniform_HMA_correction_new_direct_and_physical_routes_prioritized`

The search has produced a mechanism and three genuinely new acquisition routes, but it has not yet produced an independent annual western-central temperature series capable of defining a correction magnitude. The production reconstruction therefore remains unchanged.

## Substantive findings

1. The central-Himalayan MXD study demonstrates that conventional chronology standardization can remove centennial variability before PSM calibration or assimilation. This mechanism was not tested by the old common-mode experiment, which only averaged already-standardized project proxies. Public searches found no annual new-site series, so this route is conditionally reopened for acquisition, not claimed as a successful correction.
2. Sano Humla is not an independent validation series. The later official record uses the same 46 cores from 23 trees represented by project proxy9_199; it can only test within-sample measurement sensitivity.
3. CN-PX202 is an independent borehole constraint at the eastern HMA edge. Its published nineteenth-century GST change is +0.023 C relative to pre-1500, versus +0.511 C for the eighteenth century. Because this is a century-bin ground-temperature inversion at one site, it constrains broad direction only and cannot set an annual or HMA-wide offset.
4. India / Monthly Weather Review 1891-1920 is the strongest actionable direct-observation route. It may extend the official printed station audit beyond the six accepted 1878-1890 rows. Overlapping stations remain the same physical observation family as GHCN and will be treated as processing checks, not extra climate votes.
5. The 1854-1858 Schlagintweit volumes provide a second, earlier direct-observation route. Their short campaigns can anchor local levels only after instrument, observation-time, elevation and table identity are audited.
6. Chongce, Ngamring, Tiancai and Aweng were already assessed. They are recorded here to prevent them from being repeatedly presented as new evidence.

## Next work

Extract and adjudicate the 1891-1920 official station tables first, in parallel with the 1854-1858 expedition table map. Compare the allowed pre-1900 CN-PX202 century direction to the frozen local project curve only if the target and baseline can be matched without accessing sealed values. Keep the raw-RCS route open solely for genuinely new numeric data from the authors.

## Safety

- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No value from 1921-1939 was read, saved, plotted or used.
- A metadata preview exposed a later aggregate overlapping the sealed interval; that aggregate was excluded from every calculation and judgment.
- The permanently prohibited source was not accessed, used, or cited.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {str(path): sha256(path) for path in sources}
    manifest = {
        "read_only_inputs_before": before,
        "read_only_inputs_after": after,
        "all_inputs_unchanged": before == after,
        "output_directory": str(OUT),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("A declared read-only input changed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
