from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main(acquisition_dir: str, synthesis_dir: str, output_dir: str) -> None:
    acquisition = Path(acquisition_dir).resolve()
    synthesis = Path(synthesis_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    acquired = json.loads((acquisition / "summary.json").read_text(encoding="utf-8"))
    assert acquired["sealed_values_converted_to_numeric"] is False
    assert acquired["hma_intersecting_polygon_feature_count"] == 0

    rows = [
        ["GHCNM_QCF_15_STATION_PANEL", "west_and_central_west_HMA", "1880-1920;1940-1950", "annual_Tmean", "A", "yes", "admitted_existing", "direct station panel", "warm residual early; much smaller by 1940-1950"],
        ["UPPER_INDUS_MXD_2024", "Astore_Rama", "1370-2020", "March-September", "B_pending", "no", "author_request_required", "probable new source", "official supplement has no annual series"],
        ["PAKISTAN_WESTERN_HIMALAYA_2026", "Bara_Gali_Ayubia_Kuldana", "1855-2020", "May-June_Tmax", "B_pending", "no", "author_request_required", "new samples; Rauf2023 family overlap possible", "publisher says data available on reasonable request"],
        ["RARA_CHOLANGPATI_MXD_2026", "central_Himalaya_Nepal", "1775-2022", "April-September", "B_pending", "no", "author_request_required", "new Rara/Chola mixed with old Jumla/Kalin", "article and supplement have no annual numeric table"],
        ["SANO_HUMLA_2005", "western_Nepal", "1752-2000", "March-September", "B_pending", "no", "author_request_required", "identity relative to production unresolved", "only published curve; no public annual reconstruction"],
        ["HUGHES_YADAV_FAMILY", "Kashmir_and_western_India", "various", "spring_or_warm_season", "C", "no", "context_only", "confirmed or probable old ITRDB/PAGES2k overlap", "not an independent quantitative vote"],
        ["REACHES_DOCUMENTARY", "China_east_of_97.53E", "1368-1911", "annual_NDJFM_MJJAS", "C", "yes", "spatial_mismatch", "independent documentary source", "zero overlap with 70-90E target"],
        ["WWR_1755_1950", "global_stations", "1755-1950", "monthly", "X", "yes", "locked_preview_exposure", "GHCN overlap possible", "no values used; blind eligibility withdrawn"],
        ["CRU_BERKELEY_20CR_CERA20C", "HMA_grids", "various", "various", "C", "yes", "context_only", "background or dynamical inference", "cannot set correction magnitude"],
    ]
    header = ["evidence_id", "region", "coverage", "season", "class", "machine_readable", "admission", "independence", "finding"]
    with (output / "west_central_temperature_evidence_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    new_votes = 0
    decision = "retain_original_production_temperature_candidate_v3_gate_not_met"
    summary = {
        "experiment": "temperature_west_central_independent_evidence_admission_v2",
        "supersedes": "admission_v1_unexecuted_syntax_preflight_failure",
        "decision": decision,
        "candidate_v3_gate_passed": False,
        "required_new_independent_quantitative_sources": 2,
        "new_admitted_independent_quantitative_sources": new_votes,
        "reaches_hma_overlap_count": 0,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final = f"""# Final Result

## Decision

`{decision}`

The search produced no new source that is simultaneously independent,
machine-readable, and spatially relevant to western/central HMA. REACHES is an
independent documentary reconstruction, but its published domain begins east of
approximately 97.53E and has zero overlap with the 70-90E target. Upper Indus
2024, Pakistan western Himalaya 2026, Rara/Cholangpati 2026, and Sano Humla 2005
are scientifically relevant, but no public annual numerical reconstruction was
found. Hughes/Yadav and the downloadable Nepal chronologies overlap or probably
overlap the old ITRDB/PAGES2k source family.

WWR was withdrawn from blind validation after its web preview automatically
displayed a sealed-period row. No WWR value was extracted or used.

The existing GHCN-M qcf panel remains the only admitted direct quantitative
vote. It supports an 1880-1920 warm residual that becomes much smaller by
1940-1950, but it does not determine a defensible regional correction shape.
Candidate v2 already failed the direct-station temporal-trend gate, and Candidate
v3 is not authorized.

The original production temperature is retained. No production file was
modified, and no EnKF or OGGM run occurred. Years 1921-1939 were not used in any
result or decision.
"""
    (output / "FINAL_RESULT.md").write_text(final, encoding="utf-8")

    request = """# Author Data Request

Please provide the annual reconstruction as a machine-readable table containing
year, reconstructed temperature, units/reference baseline, reconstruction
uncertainty, chronology/site/sample identifiers, calibration target, seasonal
definition, and reuse/citation terms.

Priority: Upper Indus MXD 2024, Pakistan western Himalaya 2026,
Rara/Cholangpati MXD 2026, and Sano Humla 2005.
"""
    (output / "AUTHOR_DATA_REQUEST.md").write_text(request, encoding="utf-8")

    inputs = [
        acquisition / "summary.json",
        acquisition / "reaches_polygon_features.csv",
        acquisition / "WWR_LOCKED_PERIOD_EXPOSURE_INCIDENT.md",
        synthesis / "evidence_matrix.csv",
        synthesis / "FINAL_RESULT.md",
    ]
    manifest = [{"path": str(path), "sha256": digest(path), "role": "read_only_input"} for path in inputs]
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_temperature_west_central_independent_evidence_admission_v2.py ACQUISITION_DIR SYNTHESIS_DIR OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
