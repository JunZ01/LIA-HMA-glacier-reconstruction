from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_prohibited_source_subagent_isolation_v1"
)
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "temperature_prohibited_source_subagent_isolation_v1",
        "status": "isolated_closed_no_scientific_use",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "incident": {
            "agent_id": "01a03fde-846c-7d00-ae7e-927199f7c0bf",
            "task": "Kusai_Lake_and_Yamzhog_Yumco_original_numeric_source_search",
            "event": "an_initial_broad_local_filename_scan_returned_one_unrelated_station_index_line_from_a_prohibited_directory",
            "prohibited_scientific_file_opened": False,
            "prohibited_scientific_values_parsed": False,
            "prohibited_source_used_in_conclusion": False,
            "project_files_modified_by_agent": False,
        },
        "isolation": {
            "agent_closed": True,
            "entire_agent_result_excluded_from_evidence_matrix": True,
            "agent_urls_and_findings_admitted": False,
            "no_scientific_output_depends_on_agent_result": True,
            "preincident_packages_unaffected": [
                "temperature_original_lake_author_reconstructions_v2",
                "temperature_1850_1950_low_frequency_evidence_matrix_v12",
                "temperature_region_season_low_frequency_preflight_v2",
            ],
        },
        "future_guard": {
            "broad_local_filesystem_search_for_external_sources_allowed": False,
            "prohibited_source_access_download_parse_citation_or_use_allowed": False,
            "allowed_search_scope": "official_web_and_explicit_nonprohibited_project_paths_only",
        },
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed_in_quarantined_agent_branch": True,
        "prohibited_source_accessed_in_scientific_pipeline": False,
        "sealed_period_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = """# Prohibited-source subagent isolation

A delegated source-search branch performed an over-broad local filename scan.
The scan returned one unrelated station-index line from a directory whose
source is prohibited by the project. No scientific file or value from that
source was opened, parsed, cited, or used, but the directory access itself
violated the literal no-access rule.

The branch was stopped and closed. Its entire result, including otherwise
unrelated web findings, is excluded from every evidence matrix and scientific
decision. The three scientific packages completed before this incident do not
depend on the branch and remain unchanged.

Future external-source searches are restricted to official web endpoints and
explicit nonprohibited project paths. Broad local filesystem searches are not
allowed. The sealed period was not used, production was not modified, and no
EnKF or OGGM run occurred.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    manifest = {
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "external_scientific_sources_admitted": [],
        "quarantined_agent_result_persisted": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed_in_quarantined_agent_branch": True,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
