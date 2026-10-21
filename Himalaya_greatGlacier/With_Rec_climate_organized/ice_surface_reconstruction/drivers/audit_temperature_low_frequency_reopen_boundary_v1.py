from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = BASE / "temperature_low_frequency_reopen_boundary_v1"

ROUND16_SUMMARY = BASE / "round_16_purged_low_frequency_common_mode_v1" / "summary.json"
ROUND16_CONTRACT = BASE / "round_16_purged_low_frequency_common_mode_v1" / "ROUND_CONTRACT.md"
ROUND101_RESULT = BASE / "round_101_temperature_evidence_boundary_router_v1" / "FINAL_RESULT.md"
ROUND119_RESULT = BASE / "round_119_candidate_boundary_synthesis_v2" / "FINAL_RESULT.md"

NAHTZ_DOI = "https://doi.org/10.1007/s11676-026-02104-0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)

    round16 = json.loads(ROUND16_SUMMARY.read_text(encoding="utf-8"))
    required_text = {
        "round16_contract": ROUND16_CONTRACT.read_text(encoding="utf-8"),
        "round101_result": ROUND101_RESULT.read_text(encoding="utf-8"),
        "round119_result": ROUND119_RESULT.read_text(encoding="utf-8"),
    }

    assert round16["decision"] == "purged_low_frequency_common_mode_failed"
    assert "11-year" in required_text["round16_contract"]
    assert "additional low-frequency-preserving" in required_text["round119_result"]
    assert "Existing-data repair families closed" in required_text["round101_result"]

    rows = [
        {
            "route": "closed_existing_network_common_mode",
            "input_level": "already-standardized existing project proxy records",
            "operation": "mean formally positive proxies; 11-year centered smoothing; one fitted scale",
            "target_or_period": "1940-2000 held-out blocks; Berkeley and CRU products",
            "low_frequency_mechanism": "tests whether an existing aggregate common mode predicts modern low-frequency variation",
            "source_independence": "none; same existing proxy network",
            "status": "closed",
            "reason": "all temperature branches failed frozen out-of-fold gates",
        },
        {
            "route": "new_raw_chronology_frequency_recovery",
            "input_level": "individual cross-dated MXD series before chronology construction",
            "operation": "RCS-derived 101-year low-frequency component plus NegEx high-frequency residuals",
            "target_or_period": "chronology construction mechanism; no project target fitted in this audit",
            "low_frequency_mechanism": "preserves centennial signal otherwise weakened by conventional detrending",
            "source_independence": "partial: Rara and Chola are new; Jumla and Kalin are older sources",
            "status": "conditionally_reopened_for_data_acquisition",
            "reason": "mechanistically distinct from the closed common-mode test and matches the Round 119 reopen condition only if new-site numeric data are obtained",
        },
    ]
    write_csv(OUT / "mechanism_boundary_matrix.csv", rows)

    summary = {
        "schema": "temperature-low-frequency-reopen-boundary-v1",
        "status": "complete",
        "decision": "new_raw_frequency_preserving_chronology_route_is_distinct_but_requires_new_site_numeric_data",
        "old_route_closed": True,
        "new_route_is_duplicate": False,
        "new_route_scope": "data acquisition and identity audit only",
        "production_correction_authorized": False,
        "enkf_authorized": False,
        "oggm_authorized": False,
        "locked_period_values_accessed": False,
        "prohibited_source_accessed": False,
        "round16_temperature_family_pass_count": sum(
            int(item["passes"])
            for key, item in round16["branch_summary"].items()
            if key.startswith("temperature_")
        ),
        "required_next_data": [
            "Rara raw or standardized MXD series",
            "Chola raw or standardized MXD series",
            "new-site-only RCS chronology",
            "new-site-only NegEx chronology",
            "new-site-only frequency-combined reconstruction or sufficient metadata to reproduce it",
        ],
        "scientific_test_after_acquisition": [
            "verify sample and sequence independence from the production proxy ledger",
            "compare NegEx versus RCS low-frequency retention without using the sealed interval",
            "test whether the new-site low-frequency component explains western, central, or eastern residual structure",
            "reject a uniform HMA correction unless spatially independent evidence supports it",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = """# HMA temperature low-frequency route boundary

## Decision

The newly identified raw-chronology route is not a repeat of the closed existing-network common-mode experiment. It is therefore conditionally reopened for data acquisition and identity auditing only.

No temperature correction, EnKF run, OGGM run, or production change is authorized by this result.

## What the closed experiment tested

The old test started from already standardized project proxies, averaged all formally positive records into a common-mode composite, applied an 11-year centered mean, and tested one fitted scale against 1940-2000 held-out blocks. All temperature branches failed their frozen out-of-fold gates. This closes attempts to recover the missing signal by simply aggregating or rescaling the existing proxy network.

## What the new mechanism changes

Nahtz et al. (2026) operate one level earlier, on individual cross-dated MXD series. They use Regional Curve Standardization to preserve a 101-year low-frequency component and add the high-frequency residuals from a conventional negative-exponential chronology. Their conventional chronology has no significant low-frequency trend, while the frequency-combined chronology restores a significant warming trend and improves calibration statistics.

This is chronology-construction evidence: it indicates that centennial information can be lost before PSM fitting or assimilation. It is not a background blend, a curve translation, or an existing-network common-mode adjustment.

## Independence boundary

The published four-site composite is not yet a clean independent validation record because it combines two newly sampled sites, Rara and Chola, with older Jumla and Kalin material. The route satisfies the prior reopen condition only after new-site-only numeric data are obtained and their identities are checked against the production proxy ledger.

## Decisive next test

Obtain Rara and Chola raw MXD or their site chronologies, then construct or request matched NegEx, RCS, and frequency-combined series. Compare only allowed periods and report the two new sites separately before forming any regional composite. If the RCS component restores the disputed low-frequency behavior at the new sites, chronology standardization becomes a supported cause. If it does not, this mechanism closes without changing the production reconstruction.

## Safety

No sealed-period values were accessed. No prohibited source was accessed. Original and production files were not modified.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    sources = [ROUND16_SUMMARY, ROUND16_CONTRACT, ROUND101_RESULT, ROUND119_RESULT]
    manifest = {
        "local_sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in sources
        ],
        "external_primary_source": NAHTZ_DOI,
        "external_values_downloaded": False,
        "locked_period_values_accessed": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
