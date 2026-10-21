from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V12 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v12"
AVAILABILITY = BASE / "temperature_original_numeric_source_availability_v2"
BAMU = BASE / "temperature_bamu_author_supplement_preflight_v1"
INCIDENT = BASE / "temperature_prohibited_source_subagent_isolation_v1"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v13"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_row(matrix: pd.DataFrame, evidence_id: str, values: dict[str, object]) -> None:
    index = matrix.index[matrix.evidence_id == evidence_id]
    if len(index) != 1:
        raise RuntimeError(f"row-identity-failed:{evidence_id}:{len(index)}")
    for key, value in values.items():
        matrix.loc[index[0], key] = value


def add_row(matrix: pd.DataFrame, values: dict[str, object]) -> pd.DataFrame:
    unknown = set(values) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-columns:{sorted(unknown)}")
    return pd.concat(
        [matrix, pd.DataFrame([{column: values.get(column, "") for column in matrix.columns}])],
        ignore_index=True,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matrix = pd.read_csv(V12 / "evidence_matrix.csv")
    update_row(
        matrix,
        "XIADA_CO_BRGDGT_PENDING_NUMERIC",
        {
            "machine_readable": False,
            "numeric_now": False,
            "admitted_role": "published_record_author_request_required_no_public_numeric_metric",
            "finding": "Publisher data statement says data are available from the author; no public original age-temperature table was found.",
            "next_action": "do_not_repeat_public_search;request_original_table_from_author_if_needed",
            "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S1040618218309765-mmc1.xml",
        },
    )
    update_row(
        matrix,
        "BAMU_CO_BRGDGT_PENDING_NUMERIC",
        {
            "machine_readable": "official_DOCX_container_with_two_OLE_objects",
            "numeric_now": False,
            "admitted_role": "official_author_supplement_acquired_numeric_identity_not_safely_decoded",
            "finding": "The official 2.84 MB DOCX was downloaded to E. It has two OLE embedded objects but no direct XLS/XLSX/CSV member; document and numeric content remain unopened to protect the sealed interval.",
            "next_action": "retain_download;build_age_first_parser_only_if_container_structure_allows_predecode_lock_filter",
            "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S0031018223004418-mmc1.docx",
        },
    )
    update_row(
        matrix,
        "DAGZE_CO_UK37_PENDING_NUMERIC",
        {
            "machine_readable": False,
            "numeric_now": False,
            "admitted_role": "published_record_no_public_original_numeric_table",
            "finding": "No public original age-summer-temperature table or linked dataset was found in the publisher and formal repository routes.",
            "next_action": "do_not_repeat_public_search;request_original_table_from_author_if_needed",
        },
    )
    update_row(
        matrix,
        "KUSAI_LAKE_SUMMER_TEMPERATURE_PENDING_NUMERIC",
        {
            "machine_readable": False,
            "numeric_now": False,
            "admitted_role": "published_JJA_direction_context_no_public_numeric_series",
            "finding": "The official article gives the LT-to-JJA calibration and a plotted 350-2010 reconstruction; the official DOC supplement is tables and figures, with no public machine-readable year-temperature table.",
            "next_action": "retain_qualitative_direction_only;no_numeric_metric_without_author_series",
            "source_url": "https://www.nature.com/articles/srep06672",
        },
    )
    update_row(
        matrix,
        "YAMZHOG_YUMCO_POLLEN_TEMPERATURE_PENDING_NUMERIC",
        {
            "machine_readable": False,
            "numeric_now": False,
            "admitted_role": "published_cold_warm_stage_context_no_public_numeric_series",
            "finding": "The official article reports pollen-derived cold/warm stages but provides no public machine-readable calibrated temperature sequence.",
            "next_action": "retain_qualitative_stage_context_only;no_numeric_metric_without_author_series",
            "source_url": "https://www.sciencedirect.com/science/article/pii/S0031018217311057",
        },
    )
    update_row(
        matrix,
        "LIANG2024_BANGONG_BRGDGT_RF_MAF",
        {
            "source_independence": "Bangong_core_is_this_study_original_in_Liang2024;calibration_branches_same_archive_one_vote",
            "admitted_role": "original_Bangong_archive_pre1850_direction_and_calibration_sensitivity_no_bridge_points",
            "finding": "Bangong is an original core in Liang et al. 2024 and the official Figshare workbook is primary numeric data for this site. Safe points end in 1831, so it cannot constrain the 1850-1920 or 1940-1950 bridge.",
            "next_action": "retain_pre1850_site_context;do_not_claim_bridge_constraint",
        },
    )
    matrix = add_row(
        matrix,
        {
            "evidence_id": "HOU2024_GAHAI_AUTHOR_BAYESIAN_MAF_UNAVAILABLE",
            "region": "eastern_Tibetan_Plateau",
            "longitude": 102.18,
            "coverage": "published_record_but_original_numeric_table_not_public",
            "target": "Bayesian_MAF_ice_free_month_temperature",
            "measurement_family": "lake_sediment_brGDGT_author_Bayesian_calibration",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "same_archive_as_Liang2024_Gahai_do_not_double_count",
            "spatial_relevance": "eastern_HMA_high",
            "season_match": "MAF_not_annual_temperature",
            "admitted_role": "published_original_record_author_request_required_no_primary_numeric_metric",
            "finding": "The official Copernicus supplement contains figures only. The paper data statement directs requests to the author; Liang2024 Gahai is a secondary common-RF calibration, not the original Bayesian MAF table.",
            "next_action": "do_not_repeat_public_search;request_original_table_from_houjz_at_itpcas_ac_cn_if_needed",
            "source_url": "https://cp.copernicus.org/articles/20/335/2024/cp-20-335-2024-supplement.pdf",
            "sealed_period_values_used": False,
            "production_modified": False,
        },
    )
    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    previous = json.loads((V12 / "summary.json").read_text(encoding="utf-8"))
    previous.update(
        {
            "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v13",
            "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v12",
            "row_count": int(len(matrix)),
            "public_numeric_source_search_closed": True,
            "source_search_outcomes": {
                "public_original_numeric": ["Ngamring", "Bangong", "Tengchongqinghai", "Lugu", "Qinghai_brGDGT"],
                "author_request_or_future_deposit": ["Gahai", "Xiada", "Dagze"],
                "official_container_pending_safe_decode": ["Bamu"],
                "qualitative_only_no_public_numeric": ["Kusai", "Yamzhog", "Cuoqia_figure_derived"],
            },
            "quarantined_agent_result_used": False,
            "decision": "source_search_closed;proceed_only_with_primary_numeric_and_explicit_qualitative_roles",
            "candidate_v2_authorized": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(V12 / "evidence_matrix.csv"), "sha256": sha256(V12 / "evidence_matrix.csv")},
            {"path": str(V12 / "summary.json"), "sha256": sha256(V12 / "summary.json")},
            {"path": str(AVAILABILITY / "summary.json"), "sha256": sha256(AVAILABILITY / "summary.json")},
            {"path": str(BAMU / "summary.json"), "sha256": sha256(BAMU / "summary.json")},
            {"path": str(INCIDENT / "summary.json"), "sha256": sha256(INCIDENT / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "quarantined_agent_result_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = """# HMA low-frequency temperature evidence matrix v13

The public original-numeric source search is closed. Ngamring, Bangong,
Tengchongqinghai, Lugu, and Qinghai brGDGT have official original numeric
files. Gahai, Xiada, and Dagze require author delivery or a future repository
deposit. Bamu has an official downloaded DOCX with two OLE objects, but its
numeric content remains unopened until a sealed-period-safe parser is possible.
Kusai and Yamzhog have official published qualitative temperature information
but no public machine-readable temperature series.

The quarantined search branch was not used. This matrix does not authorize a
uniform HMA correction or Candidate v2.

No sealed-period value was used. Production was not changed, EnKF and OGGM were
not run, and this package did not access 20CRv3.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
