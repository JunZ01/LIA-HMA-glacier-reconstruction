from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = BASE / "temperature_original_numeric_source_availability_v1"
DRIVER = Path(__file__).resolve()
INPUTS = {
    "evidence_matrix": BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v12/evidence_matrix.csv",
    "original_lakes": BASE / "temperature_original_lake_author_reconstructions_v2/summary.json",
    "bamu_preflight": BASE / "temperature_bamu_author_supplement_preflight_v1/summary.json",
    "incident": BASE / "temperature_prohibited_source_subagent_isolation_v1/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    bamu = json.loads(INPUTS["bamu_preflight"].read_text(encoding="utf-8"))
    rows = [
        {
            "record": "Ngamring Co brGDGT MAAT",
            "source_status": "public_original_numeric",
            "original_author_data": True,
            "machine_readable": True,
            "current_safe_use": "primary_direction_1597_1778_no_1850_1920_bridge",
            "direct_url": "https://www.ncei.noaa.gov/pub/data/paleo/paleolimnology/asia/china/sun2022/sun2022-ngamring-co.txt",
            "request_contact": "",
        },
        {
            "record": "Bangong Co brGDGT RF MAF",
            "source_status": "public_original_numeric_this_study",
            "original_author_data": True,
            "machine_readable": True,
            "current_safe_use": "original_archive_and_calibration_evidence_no_1850_1920_or_1940_1950_points",
            "direct_url": "https://ndownloader.figshare.com/files/50361615",
            "request_contact": "jie.liang@awi.de",
        },
        {
            "record": "Tengchongqinghai brGDGT MAAT",
            "source_status": "public_original_numeric",
            "original_author_data": True,
            "machine_readable": True,
            "current_safe_use": "primary_direction_1560_1899_one_bridge_point_not_a_bridge_trend",
            "direct_url": "https://ars.els-cdn.com/content/image/1-s2.0-S2095927320307222-mmc1.xlsx",
            "request_contact": "",
        },
        {
            "record": "Lugu Lake brGDGT MAAT",
            "source_status": "public_original_numeric",
            "original_author_data": True,
            "machine_readable": True,
            "current_safe_use": "primary_full_window_and_four_point_1850_1914_bridge",
            "direct_url": "https://ars.els-cdn.com/content/image/1-s2.0-S2095927320307222-mmc1.xlsx",
            "request_contact": "",
        },
        {
            "record": "Qinghai Lake salinity-corrected brGDGT MAAT",
            "source_status": "public_original_numeric",
            "original_author_data": True,
            "machine_readable": True,
            "current_safe_use": "method_identity_one_safe_point_no_direction_trend",
            "direct_url": "https://ars.els-cdn.com/content/image/1-s2.0-S0016703721002672-mmc2.xlsx",
            "request_contact": "",
        },
        {
            "record": "Gahai Lake Bayesian MAF",
            "source_status": "original_numeric_not_public_author_request_required",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "no_primary_numeric_vote_secondary_common_RF_only",
            "direct_url": "https://cp.copernicus.org/articles/20/335/2024/cp-20-335-2024-supplement.pdf",
            "request_contact": "houjz@itpcas.ac.cn",
        },
        {
            "record": "Xiada Co brGDGT temperature",
            "source_status": "data_statement_requests_author_no_public_numeric_table",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "none_pending_author_table",
            "direct_url": "https://ars.els-cdn.com/content/image/1-s2.0-S1040618218309765-mmc1.xml",
            "request_contact": "houjz@itpcas.ac.cn",
        },
        {
            "record": "Bamu Co brGDGT MAAT",
            "source_status": "official_author_DOCX_downloaded_two_OLE_objects_numeric_identity_not_decoded",
            "original_author_data": True,
            "machine_readable": "container_only_pending_safe_schema",
            "current_safe_use": "none_until_age_first_parser_is_possible",
            "direct_url": bamu["source_url"],
            "request_contact": "houjz@itpcas.ac.cn",
        },
        {
            "record": "Dagze Co UK37 summer temperature",
            "source_status": "no_public_original_numeric_table_found",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "none_pending_author_table",
            "direct_url": "https://doi.org/10.1177/0959683615572852",
            "request_contact": "houjz@itpcas.ac.cn",
        },
        {
            "record": "Kusai Lake JJA temperature",
            "source_status": "unresolved_search_branch_quarantined_no_findings_admitted",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "none_requires_fresh_web_only_audit",
            "direct_url": "https://doi.org/10.1038/srep06672",
            "request_contact": "",
        },
        {
            "record": "Yamzhog Yumco pollen temperature",
            "source_status": "unresolved_search_branch_quarantined_no_findings_admitted",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "none_requires_fresh_web_only_audit",
            "direct_url": "https://doi.org/10.1016/j.palaeo.2018.04.005",
            "request_contact": "",
        },
        {
            "record": "Cuoqia Lake brGDGT March-October temperature",
            "source_status": "official_figure_only_approximate_digitization_exists_no_author_numeric_table",
            "original_author_data": False,
            "machine_readable": "figure_derived_only",
            "current_safe_use": "shape_context_not_primary_numeric_table",
            "direct_url": "https://doi.org/10.3389/fevo.2023.1119869",
            "request_contact": "",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "original_numeric_source_availability.csv", index=False)
    counts = frame.source_status.value_counts().to_dict()
    summary = {
        "experiment": "temperature_original_numeric_source_availability_v1",
        "status": "complete_with_explicit_unresolved_records",
        "record_count": int(len(frame)),
        "status_counts": counts,
        "public_original_numeric_records": frame.loc[
            frame.source_status.str.startswith("public_original_numeric"), "record"
        ].tolist(),
        "author_request_required": frame.loc[
            frame.source_status.str.contains("author", case=False) & ~frame.source_status.str.startswith("public"), "record"
        ].tolist(),
        "quarantined_search_records": ["Kusai Lake JJA temperature", "Yamzhog Yumco pollen temperature"],
        "quarantined_result_used": False,
        "linked_incident": str(INPUTS["incident"]),
        "decision": "close_public_source_search_for_all_nonquarantined_records;do_not_promote_secondary_or_figure_derived_series_to_original_data",
        "sealed_period_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in INPUTS.values()],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "quarantined_agent_result_used": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Original numeric temperature-source availability",
        "",
        "| Record | Status | Current safe use |",
        "|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.record} | {row.source_status} | {row.current_safe_use} |")
    lines.extend(
        [
            "",
            "Kusai and Yamzhog remain unresolved because their delegated search branch was quarantined in full; none of that branch's findings are admitted. All other rows are based on official public sources, downloaded source files, or explicit publisher data statements.",
            "",
            "No sealed-period value was used. Production was not changed, EnKF and OGGM were not run, and this package did not access 20CRv3.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
