from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(acquisition_dir: str, existing_synthesis_dir: str, output_dir: str) -> None:
    acquisition = Path(acquisition_dir).resolve()
    existing = Path(existing_synthesis_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    acquisition_summary = json.loads((acquisition / "summary.json").read_text(encoding="utf-8"))
    if acquisition_summary["sealed_values_converted_to_numeric"]:
        raise RuntimeError("Sealed values were converted to numeric in acquisition input")
    if acquisition_summary["hma_intersecting_polygon_feature_count"] != 0:
        raise RuntimeError("REACHES spatial result changed; stop for manual review")

    candidates = [
        {
            "evidence_id": "GHCNM_QCF_15_STATION_PANEL",
            "region": "western_and_central_western_HMA",
            "coverage": "1880-1920;1940-1950",
            "season": "calendar_annual_Tmean",
            "evidence_class": "A",
            "machine_readable": True,
            "independence": "primary_direct_station_panel",
            "admission": "admitted_existing",
            "finding": "warm residual in 1880-1920; much smaller by 1940-1950",
            "source": "existing temperature_station_annual_anomaly_benchmark_v4",
        },
        {
            "evidence_id": "UPPER_INDUS_MXD_2024",
            "region": "Astore_Rama_northern_Pakistan",
            "coverage": "1370-2020",
            "season": "March-September",
            "evidence_class": "B_pending",
            "machine_readable": False,
            "independence": "probable_new_MXD_source_exact_sample_ids_pending",
            "admission": "metadata_only_author_data_request_required",
            "finding": "official supplement contains no annual reconstruction table",
            "source": "https://doi.org/10.1016/j.scitotenv.2024.177393",
        },
        {
            "evidence_id": "PAKISTAN_WESTERN_HIMALAYA_2026",
            "region": "Bara_Gali_Ayubia_Kuldana",
            "coverage": "1855-2020",
            "season": "May-June_Tmax",
            "evidence_class": "B_pending",
            "machine_readable": False,
            "independence": "new_2021_samples_but_Rauf2023_family_overlap_possible",
            "admission": "metadata_only_author_data_request_required",
            "finding": "publisher states data available on reasonable request",
            "source": "https://doi.org/10.1088/2515-7620/ae718e",
        },
        {
            "evidence_id": "RARA_CHOLANGPATI_MXD_2026",
            "region": "central_Himalaya_Nepal",
            "coverage": "1775-2022",
            "season": "April-September",
            "evidence_class": "B_pending",
            "machine_readable": False,
            "independence": "new_2023_Rara_Cholangpati_samples",
            "admission": "metadata_only_author_data_request_required",
            "finding": "article and supplement contain no public annual numeric series",
            "source": "https://doi.org/10.1007/s11676-026-02104-0",
        },
        {
            "evidence_id": "HUGHES_YADAV_WESTERN_HIMALAYA_FAMILY",
            "region": "Kashmir_and_western_India_Himalaya",
            "coverage": "various_preindustrial_to_twentieth_century",
            "season": "spring_or_warm_season",
            "evidence_class": "C",
            "machine_readable": False,
            "independence": "confirmed_or_probable_overlap_with_old_ITRDB_PAGES2k_network",
            "admission": "context_only_not_independent_vote",
            "finding": "raw chronologies may be public but annual temperature reconstructions are not",
            "source": "Hughes2001_and_Yadav1997_2014_family",
        },
        {
            "evidence_id": "REACHES_DOCUMENTARY_TEMPERATURE",
            "region": "China_east_of_approximately_97.53E",
            "coverage": "1368-1911",
            "season": "annual_NDJFM_MJJAS",
            "evidence_class": "C",
            "machine_readable": True,
            "independence": "documentary_source_independent_of_tree_rings",
            "admission": "rejected_for_HMA_spatial_mismatch",
            "finding": "zero polygon overlap with the 70-90E,28-42N screening box",
            "source": "https://doi.org/10.25921/nqcg-7e62",
        },
        {
            "evidence_id": "NOAA_WORLD_WEATHER_RECORDS_1755_1950",
            "region": "global_station_archive",
            "coverage": "1755-1950",
            "season": "monthly",
            "evidence_class": "X",
            "machine_readable": True,
            "independence": "source_overlap_with_GHCN_possible",
            "admission": "excluded_after_locked_period_web_preview_exposure",
            "finding": "no values used; blind-validation eligibility withdrawn",
            "source": "https://doi.org/10.7289/V5222RT1",
        },
        {
            "evidence_id": "CRU_BERKELEY_20CR_CERA20C",
            "region": "HMA_grids",
            "coverage": "instrumental_or_reanalysis_periods",
            "season": "various",
            "evidence_class": "C",
            "machine_readable": True,
            "independence": "background_or_dynamical_inference_not_direct_temperature_vote",
            "admission": "direction_and_context_only",
            "finding": "cannot determine correction magnitude as independent observations",
            "source": "existing project product and reanalysis audits",
        },
    ]
    write_csv(out / "west_central_temperature_evidence_matrix.csv", candidates)

    admitted_new_quantitative = [
        row
        for row in candidates
        if row["evidence_class"] in ("A", "B")
        and row["machine_readable"]
        and row["admission"] == "admitted_new"
    ]
    gate_passed = len(admitted_new_quantitative) >= 2
    decision = (
        "candidate_v3_gate_passed"
        if gate_passed
        else "retain_original_production_temperature_candidate_v3_gate_not_met"
    )

    request = """# Author Data Request\n\n"
    request += "Please provide the annual reconstruction in a machine-readable table with:\n\n"
    request += "- year and reconstructed temperature;\n"
    request += "- units and anomaly/reference baseline;\n"
    request += "- reconstruction uncertainty;\n"
    request += "- chronology, site, and sample identifiers;\n"
    request += "- calibration target and seasonal definition;\n"
    request += "- permission and citation terms.\n\n"
    request += "Priority datasets: Upper Indus MXD 2024, Pakistan western Himalaya 2026, and Rara/Cholangpati MXD 2026.\n"
    (out / "AUTHOR_DATA_REQUEST.md").write_text(request, encoding="utf-8")

    final_result = f"""# Final Result\n\n## Decision\n\n`{decision}`\n\n## Substantive result\n\nThe new evidence search did not produce two independent, machine-readable western/central HMA temperature series. REACHES is genuinely independent documentary evidence but its published polygons begin east of approximately 97.53E and do not overlap the 70-90E target. The new Upper Indus, Pakistan western-Himalaya, and Rara/Cholangpati reconstructions are scientifically relevant but their annual numerical series are not publicly available. The Hughes/Yadav family is source-overlapping or unresolved relative to the old ITRDB/PAGES2k network. WWR was removed from blind validation after an automatic web preview exposed a sealed-period row; no WWR value was used.\n\nThe existing GHCN-M qcf station panel remains the only admitted direct quantitative vote. It supports a western/central-western warm residual in 1880-1920 that becomes much smaller by 1940-1950, but it does not identify a defensible regional correction shape. Candidate v2 already failed the direct-station temporal-trend gate. Candidate v3 is therefore not authorized.\n\n## Production state\n\nThe original production temperature is retained. No production file was modified, and no EnKF or OGGM run occurred. Years 1921-1939 were not used in any result or decision.\n"""
    (out / "FINAL_RESULT.md").write_text(final_result, encoding="utf-8")

    sources = [
        acquisition / "summary.json",
        acquisition / "reaches_polygon_features.csv",
        acquisition / "WWR_LOCKED_PERIOD_EXPOSURE_INCIDENT.md",
        existing / "evidence_matrix.csv",
        existing / "FINAL_RESULT.md",
    ]
    manifest = [
        {"path": str(path), "sha256": sha256(path), "role": "read_only_input"}
        for path in sources
    ]
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "experiment": "temperature_west_central_independent_evidence_admission_v1",
        "decision": decision,
        "candidate_v3_gate_passed": gate_passed,
        "required_independent_quantitative_sources": 2,
        "new_admitted_independent_quantitative_sources": len(admitted_new_quantitative),
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_used": False,
        "reaches_hma_overlap_count": 0,
        "author_request_required_for": [
            "Upper Indus MXD 2024",
            "Pakistan western Himalaya 2026",
            "Rara/Cholangpati MXD 2026",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_temperature_west_central_independent_evidence_admission_v1.py ACQUISITION_DIR EXISTING_SYNTHESIS_DIR OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
