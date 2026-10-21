from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUTPUT = BASE / "temperature_west_central_author_repository_hit_adjudication_v2"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
V1_HITS = BASE / "temperature_west_central_author_repository_search_v1" / "plausible_repository_hits.csv"
ROUND168 = BASE / "round_168_central_hma_mxd_public_data_recovery_v1" / "summary.json"
ADMISSION = BASE / "temperature_west_central_independent_evidence_admission_v2" / "FINAL_RESULT.md"
DRIVER = Path(__file__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": FACTS,
        "v1_plausible_hits": V1_HITS,
        "round168_supplement_audit": ROUND168,
        "prior_admission": ADMISSION,
        "driver": DRIVER,
    }
    before = {name: sha256(path) for name, path in sources.items()}

    rows = [
        {
            "candidate_id": "upper_indus_mxd_2024",
            "doi": "10.1016/j.scitotenv.2024.177393",
            "v1_hit": "PubMed",
            "adjudicated_resource": "bibliographic_index",
            "machine_readable_annual_series": False,
            "independent_numeric_evidence_admitted": False,
            "reason": "PubMed is an article index; it is not a data repository attachment.",
            "next_action": "author_request",
        },
        {
            "candidate_id": "pakistan_western_himalaya_2026",
            "doi": "10.1088/2515-7620/ae718e",
            "v1_hit": "DOAJ",
            "adjudicated_resource": "bibliographic_index",
            "machine_readable_annual_series": False,
            "independent_numeric_evidence_admitted": False,
            "reason": "DOAJ is an article index; the article states that data are available on request.",
            "next_action": "author_request",
        },
        {
            "candidate_id": "rara_cholangpati_mxd_2026",
            "doi": "10.1007/s11676-026-02104-0",
            "v1_hit": "OPUS_FAU",
            "adjudicated_resource": "institutional_article_copy",
            "machine_readable_annual_series": False,
            "independent_numeric_evidence_admitted": False,
            "reason": (
                "The repository location resolves the article DOI, not a dataset DOI. "
                "Round 168 audited the public 12-page supplement: zero embedded files, "
                "zero public numeric links, and no annual numeric reconstruction."
            ),
            "next_action": "author_request",
        },
        {
            "candidate_id": "sano_humla_2005",
            "doi": "10.1016/j.dendro.2005.08.003",
            "v1_hit": "none",
            "adjudicated_resource": "no_public_repository_record",
            "machine_readable_annual_series": False,
            "independent_numeric_evidence_admitted": False,
            "reason": "No public machine-readable annual reconstruction was found.",
            "next_action": "author_request_or_archive_search",
        },
    ]
    write_csv(OUTPUT / "repository_hit_adjudication.csv", rows)

    request_rows = [
        {
            "priority": 1,
            "candidate_id": "rara_cholangpati_mxd_2026",
            "region": "central_HMA_82E_to_86E",
            "period": "1775_to_2022",
            "season": "April_to_September_mean_temperature",
            "contact": "nahtz@geo.uni-mainz.de",
            "request_reason": "Best spatial match and independent modern Rara/Cholangpati MXD sites.",
        },
        {
            "priority": 2,
            "candidate_id": "upper_indus_mxd_2024",
            "region": "western_HMA_upper_Indus",
            "period": "1370_to_2020",
            "season": "March_to_September_temperature",
            "contact": "adam@ulm.edu.pk; feng653@163.com",
            "request_reason": "Longest western-HMA record and direct low-frequency relevance.",
        },
        {
            "priority": 3,
            "candidate_id": "pakistan_western_himalaya_2026",
            "region": "western_Himalaya_Pakistan",
            "period": "1855_to_2020",
            "season": "May_to_June_Tmax",
            "contact": "adam@ulm.edu.pk",
            "request_reason": "Directly covers the 1855-1950 residual interval but has a narrow season.",
        },
        {
            "priority": 4,
            "candidate_id": "sano_humla_2005",
            "region": "western_Nepal_Humla",
            "period": "1752_to_2000",
            "season": "March_to_September_temperature",
            "contact": "contact_not_yet_verified",
            "request_reason": "Scientifically relevant legacy curve; lower priority because provenance overlap must be audited.",
        },
    ]
    write_csv(OUTPUT / "author_request_priority.csv", request_rows)

    request_text = """# Author Data Request Package

Requester: Jun Zhou, Yunnan University, China (`junzhounbu1@gmail.com`).

## Required fields

Please provide the annual reconstruction as a machine-readable table with year,
reconstructed temperature, units and reference baseline, uncertainty, site and
chronology identifiers, calibration target, exact seasonal definition, and
reuse/citation terms. Site-level Rara and Cholangpati series are preferred over a
composite that includes legacy sites.

## Priority 1: Rara and Cholangpati

To: nahtz@geo.uni-mainz.de
Subject: Request for annual Rara/Cholangpati MXD temperature reconstruction data

Dear Mr. Nahtz,

I am reconstructing historical temperature variability across High Mountain
Asia at Yunnan University. Your 2026 Central Himalayan MXD study is especially
valuable for an independent assessment of low-frequency temperature behavior in
western and central Nepal. Could you please share the annual reconstructed
temperature series and, if available, the separate Rara and Cholangpati site
chronologies in a machine-readable format?

For reproducible comparison, it would be very helpful to include the temperature
units and reference baseline, exact April-September target definition,
uncertainty estimates, chronology/site identifiers, and reuse/citation terms.
The data will be used for scientific validation with full attribution.

Thank you for considering this request.

Best regards,
Jun Zhou
Yunnan University, China
junzhounbu1@gmail.com

## Priority 2: Upper Indus Basin

To: adam@ulm.edu.pk
Cc: feng653@163.com
Subject: Request for annual Upper Indus MXD summer-temperature reconstruction data

Dear Dr. Khan and Dr. Chen,

I am reconstructing historical temperature variability across High Mountain
Asia at Yunnan University. Your 2024 Upper Indus Basin MXD reconstruction is a
key independent record for evaluating western-HMA low-frequency temperature
behavior. Could you please share the annual 1370-2020 reconstructed temperature
series in a machine-readable format?

Please include, if available, the temperature units and reference baseline,
March-September target definition, uncertainty estimates, chronology/site
identifiers, and reuse/citation terms. The data will be used for scientific
validation with full attribution.

Thank you for your help.

Best regards,
Jun Zhou
Yunnan University, China
junzhounbu1@gmail.com

## Priority 3: Western Himalaya Pakistan

To: adam@ulm.edu.pk
Subject: Request for annual May-June Tmax reconstruction data since 1855 CE

Dear Dr. Khan,

I am reconstructing historical temperature variability across High Mountain
Asia at Yunnan University. Your 2026 western-Himalaya reconstruction directly
covers the 1855-1950 interval that we are independently evaluating. Could you
please share the annual reconstructed May-June maximum-temperature series in a
machine-readable format?

Please include the temperature units and reference baseline, uncertainty
estimates, chronology/site identifiers, calibration target, and reuse/citation
terms. The data will be used for scientific validation with full attribution.

Thank you for considering this request.

Best regards,
Jun Zhou
Yunnan University, China
junzhounbu1@gmail.com
"""
    (OUTPUT / "AUTHOR_REQUEST_PACKAGE.md").write_text(request_text, encoding="utf-8")

    summary = {
        "experiment": "temperature_west_central_author_repository_hit_adjudication_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes": "temperature_west_central_author_repository_search_v1 repository-hit interpretation",
        "candidate_count": len(rows),
        "public_machine_readable_annual_series_found": 0,
        "false_or_nondata_repository_hits": 3,
        "no_hit_candidates": 1,
        "decision": "public_repository_route_closed_author_request_required",
        "sealed_1921_1939_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final = """# Final Result

The permissive v1 repository classification was adjudicated. PubMed and DOAJ are
bibliographic indexes, while OPUS FAU is an institutional copy of the Rara/
Cholangpati article. Round 168 already audited the only public supplement and
found no embedded file, numeric link, or annual machine-readable reconstruction.

No new independent annual temperature series was admitted. The public repository
route is closed for all four priority candidates. Targeted author requests are
now the shortest credible route to new low-frequency evidence; until data arrive,
the original production temperature remains unchanged.

Decision: `public_repository_route_closed_author_request_required`.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(final, encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    manifest = {
        name: {"path": str(sources[name]), "sha256_before": before[name], "sha256_after": after[name], "unchanged": before[name] == after[name]}
        for name in sources
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not all(item["unchanged"] for item in manifest.values()):
        raise RuntimeError("A read-only source changed during adjudication")


if __name__ == "__main__":
    main()
