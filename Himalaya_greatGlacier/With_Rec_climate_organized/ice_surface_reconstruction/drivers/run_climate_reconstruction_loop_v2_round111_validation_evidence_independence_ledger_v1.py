from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUND_ID = "round_111_validation_evidence_independence_ledger_v1"
FROZEN_FACTS_PREFIX_LINE_COUNT = 5472
FROZEN_FACTS_PREFIX_SHA256 = "aa84aa86c02f6044cce20eeaac1f363d424cda07242474f0336791b6ad6bd8d2"


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def truth(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none"} else text


def slug(value: object) -> str:
    text = clean(value).casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown_site"


def verify_facts_prefix(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if len(lines) < FROZEN_FACTS_PREFIX_LINE_COUNT:
        raise RuntimeError("facts-file-shorter-than-frozen-prefix")
    prefix_hash = hashlib.sha256(b"".join(lines[:FROZEN_FACTS_PREFIX_LINE_COUNT])).hexdigest()
    if prefix_hash != FROZEN_FACTS_PREFIX_SHA256:
        raise RuntimeError("frozen-facts-prefix-changed-stop-and-reread")
    return {
        "line_count": len(lines),
        "full_sha256": hashlib.sha256(content).hexdigest(),
        "frozen_prefix_line_count": FROZEN_FACTS_PREFIX_LINE_COUNT,
        "frozen_prefix_sha256": prefix_hash,
    }


def main() -> int:
    project = project_root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    output = loop / ROUND_ID
    output.mkdir(parents=True, exist_ok=False)

    paths = {
        "facts": project / "LIA_PROJECT_FACTS.md",
        "hydro_coverage": loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1" / "dod2k_hma_hydroclimate_coverage.csv",
        "hydro_identity": loop / "round_36_dod2k_hma_identity_chronology_admissibility_v1" / "dod2k_identity_chronology_admissibility.csv",
        "temperature_coverage": loop / "round_52_dod2k_temperature_allowed_period_coverage_v1" / "dod2k_hma_temperature_coverage.csv",
        "temperature_identity": loop / "round_53_dod2k_temperature_identity_chronology_v1" / "dod2k_temperature_identity_chronology.csv",
        "temperature_lia_only": loop / "round_55_lia_only_icecore_spatial_identity_v1" / "lia_only_icecore_spatial_identity.csv",
        "master_groups": loop / "round_110_unresolved_proxy_metadata_recovery_v4" / "evidence_groups_metadata_recovered.csv",
        "round73": loop / "round_73_mode_ra_hma_source_overlap_v1" / "summary.json",
        "round81": loop / "round_81_cpc_fixed_candidate_adjudication_v1" / "summary.json",
        "round86": loop / "round_86_frozen_temperature_final_test_v1" / "summary.json",
        "round93": loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4" / "summary.json",
        "round95": loop / "round_95_era20c_ghcnm_temperature_value_validation_v4" / "summary.json",
        "round102": loop / "round_102_tree_ring_hydroclimate_evidence_v4" / "summary.json",
        "round103": loop / "round_103_hydroclimate_proxy_sequence_overlap_v1" / "summary.json",
        "round106": loop / "round_106_climate_candidate_boundary_synthesis_v1" / "summary.json",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing-input:{name}:{path}")

    facts = verify_facts_prefix(paths["facts"])
    before = {name: file_state(path) for name, path in paths.items()}

    hydro = pd.read_csv(paths["hydro_coverage"])
    hydro_identity = pd.read_csv(paths["hydro_identity"])
    temperature = pd.read_csv(paths["temperature_coverage"])
    temperature_identity = pd.read_csv(paths["temperature_identity"])
    temperature_lia_only = pd.read_csv(paths["temperature_lia_only"])
    master_groups = pd.read_csv(paths["master_groups"])

    if len(master_groups) != 297:
        raise RuntimeError("unexpected-master-evidence-group-count")
    if int(hydro["lia_evidence_eligible"].map(truth).sum()) != 44:
        raise RuntimeError("unexpected-hydro-lia-eligible-count")
    if int(temperature["both_lia_and_modern_capable"].map(truth).sum()) != 155:
        raise RuntimeError("unexpected-temperature-both-capable-count")

    hydro["lia_eligible"] = hydro["lia_evidence_eligible"].map(truth)
    hydro["modern_capable"] = hydro["modern_calibration_capable"].map(truth)
    hydro["new_site"] = hydro["candidate_site_status"].eq("new_site_candidate")
    hydro["site_cluster_id"] = hydro["geo_siteName"].map(slug)
    hydro["round36_route"] = hydro["datasetId"].map(
        hydro_identity.set_index("datasetId")["route"].to_dict()
    ).fillna("")

    reserve = hydro[hydro["lia_eligible"] & ~hydro["modern_capable"] & hydro["new_site"]].copy()
    if len(reserve) != 16 or reserve["site_cluster_id"].nunique() != 12:
        raise RuntimeError("unexpected-hydro-blind-reserve-size")
    if reserve["round36_route"].ne("").any():
        raise RuntimeError("blind-reserve-was-used-in-round36")

    reserve["evidence_class"] = "A_source_and_method_independent_frozen_blind_reserve"
    reserve["development_status"] = "metadata_and_coverage_only_values_not_interpreted"
    reserve["allowed_future_use"] = "paleo_low_frequency_phase_spatial_validation_only"
    reserve["value_access_rule"] = "keep_values_blind_until_candidate_and_interpretation_are_frozen"
    reserve["direction_ready"] = reserve["interpretation_direction"].map(clean).ne("")
    reserve["seasonality_ready"] = reserve["interpretation_seasonality"].map(clean).ne("")
    reserve["operator_ready"] = reserve["direction_ready"] & reserve["seasonality_ready"]
    reserve["amplitude_target_allowed"] = False
    reserve["reason"] = (
        "new HMA site outside the reconstruction proxy network; LIA coverage; no 1940-2000 calibration route; "
        "not entered into Round 36-41 PSM development"
    )

    cluster_rows = []
    for cluster_id, group in reserve.groupby("site_cluster_id", sort=True):
        cluster_rows.append({
            "site_cluster_id": cluster_id,
            "site_name": clean(group.iloc[0]["geo_siteName"]),
            "record_count": int(len(group)),
            "dataset_ids": ";".join(sorted(group["datasetId"].astype(str))),
            "archive_types": ";".join(sorted(set(group["archiveType"].map(clean)))),
            "source_databases": ";".join(sorted(set(group["originalDatabase"].map(clean)))),
            "latitude": float(pd.to_numeric(group["geo_meanLat"], errors="coerce").mean()),
            "longitude": float(pd.to_numeric(group["geo_meanLon"], errors="coerce").mean()),
            "maximum_lia_50year_windows": int(pd.to_numeric(group["lia_occupied_50year_window_count"]).max()),
            "direction_ready_any": bool(group["direction_ready"].any()),
            "seasonality_ready_any": bool(group["seasonality_ready"].any()),
            "operator_ready_any": bool(group["operator_ready"].any()),
            "freeze_status": "frozen_blind_validation_cluster",
            "evidence_class": "A",
            "permitted_role": "phase_spatial_low_frequency_validation",
            "amplitude_target_allowed": False,
        })
    reserve_clusters = pd.DataFrame(cluster_rows)

    hydro_all_rows = []
    for _, row in hydro.iterrows():
        dataset_id = str(row["datasetId"])
        if dataset_id in set(reserve["datasetId"].astype(str)):
            evidence_class = "A"
            status = "frozen_blind_paleo_validation_reserve"
        elif row["round36_route"] in {
            "annual_moisture_psm_candidate",
            "independent_low_frequency_validation",
            "source_identity_confirmation_required",
        }:
            evidence_class = "B"
            status = "source_independent_but_development_inspected"
        elif row["round36_route"] in {
            "exclude_existing_project_identity",
            "exclude_current_network_duplicate",
        } or row["candidate_site_status"] == "existing_site_requires_identity_review":
            evidence_class = "D"
            status = "reconstruction_source_or_site_overlap"
        else:
            evidence_class = "U"
            status = "not_operational_for_paleo_validation"
        hydro_all_rows.append({
            "dataset_id": dataset_id,
            "site_cluster_id": row["site_cluster_id"],
            "site_name": clean(row["geo_siteName"]),
            "archive_type": clean(row["archiveType"]),
            "source_database": clean(row["originalDatabase"]),
            "lia_eligible": bool(row["lia_eligible"]),
            "modern_capable": bool(row["modern_capable"]),
            "round36_route": clean(row["round36_route"]),
            "evidence_class": evidence_class,
            "status": status,
        })
    hydro_ledger = pd.DataFrame(hydro_all_rows)

    temperature["lia_eligible"] = temperature["lia_evidence_eligible"].map(truth)
    temperature["modern_capable"] = temperature["modern_calibration_capable"].map(truth)
    temperature["route"] = temperature["datasetId"].map(
        temperature_identity.set_index("datasetId")["route"].to_dict()
    ).fillna("")
    known_lia_only = set(temperature_lia_only["datasetId"].astype(str))
    temp_rows = []
    for _, row in temperature.iterrows():
        dataset_id = str(row["datasetId"])
        route = clean(row["route"])
        if route == "annual_temperature_psm_candidate":
            evidence_class = "B"
            status = "Malan_new_site_but_development_inspected_and_PSM_failed"
        elif route == "exclude_current_proxy_sequence_duplicate" or dataset_id in known_lia_only:
            evidence_class = "D"
            status = "current_proxy_sequence_or_site_overlap"
        else:
            evidence_class = "U"
            status = "not_operational_or_identity_unresolved"
        temp_rows.append({
            "dataset_id": dataset_id,
            "site_name": clean(row["geo_siteName"]),
            "archive_type": clean(row["archiveType"]),
            "source_database": clean(row["originalDatabase"]),
            "lia_eligible": bool(row["lia_eligible"]),
            "modern_capable": bool(row["modern_capable"]),
            "route": route,
            "evidence_class": evidence_class,
            "status": status,
        })
    temperature_ledger = pd.DataFrame(temp_rows)

    if int((temperature_ledger["evidence_class"].eq("A") & temperature_ledger["lia_eligible"]).sum()) != 0:
        raise RuntimeError("unexpected-independent-temperature-paleo-reserve")

    modern_rows = [
        {
            "component": "temperature", "evidence_family": "CRU_TS_gridded_temperature",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "shared_station_inputs_and_already_used_in_candidate_development",
            "allowed_role": "blocked_spatiotemporal_OOF_reference_only",
        },
        {
            "component": "temperature", "evidence_family": "GHCN_M_monthly_stations",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "station_source_overlaps_gridded_products_and_prior_station_audits",
            "allowed_role": "station_cluster_holdout_if_cluster_excluded_from_all_training_products",
        },
        {
            "component": "temperature", "evidence_family": "Mode_RA_five_family_references",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "proxy_evidence_overlap_confirmed_by_Round73",
            "allowed_role": "method_independent_consistency_not_source_independent_validation",
        },
        {
            "component": "temperature", "evidence_family": "20CRv3",
            "class": "U", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "rejected_as_external_HMA_temperature_change_evidence_Round93",
            "allowed_role": "none_for_acceptance",
        },
        {
            "component": "temperature", "evidence_family": "ERA20C",
            "class": "U", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "rejected_as_external_HMA_temperature_change_evidence_Round95",
            "allowed_role": "none_for_acceptance",
        },
        {
            "component": "precipitation", "evidence_family": "CPC_global_precipitation",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "gauge_independence_not_claimed_and_candidates_already_adjudicated_Round81",
            "allowed_role": "blocked_spatiotemporal_OOF_reference_only",
        },
        {
            "component": "precipitation", "evidence_family": "GPCC_and_CRU_precipitation",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "shared_gauge_networks_and_already_used_in_development",
            "allowed_role": "product_family_sensitivity_not_independent_acceptance",
        },
        {
            "component": "precipitation", "evidence_family": "GHCN_D_precipitation_stations",
            "class": "U", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "early_HMA_completeness_insufficient_Rounds82_83",
            "allowed_role": "none_until_new_station_completeness_evidence",
        },
        {
            "component": "precipitation", "evidence_family": "APHRODITE_or_equivalent",
            "class": "U", "source_independent": False, "method_independent": True,
            "development_untouched": True, "status": "currently_unavailable_or_access_blocked",
            "allowed_role": "future_product_family_cross_check_not_assumed_source_independent",
        },
        {
            "component": "precipitation", "evidence_family": "MADA_and_tree_ring_hydroclimate_atlases",
            "class": "C", "source_independent": False, "method_independent": True,
            "development_untouched": False, "status": "tree_ring_source_overlap_and_sequence_matches_Rounds102_103",
            "allowed_role": "phase_and_spatial_consistency_only_no_amplitude_target",
        },
    ]
    modern_ledger = pd.DataFrame(modern_rows)

    protocol = pd.DataFrame([
        {
            "gate_id": "MODERN_T_BLOCKED_OOF", "component": "temperature", "period": "1940-2000",
            "independence_unit": "station_cluster_x_product_family_x_time_block",
            "rule": "exclude each validation station cluster from every training/calibration/reference product where traceable; score 10-year blocked folds and regional spatial folds",
            "minimum_claim": "method-independent OOF; source-independent only where station exclusion is documented",
            "locked_period_access": "1921-1939 prohibited",
        },
        {
            "gate_id": "MODERN_P_BLOCKED_OOF", "component": "precipitation", "period": "1940-2000",
            "independence_unit": "gauge_cluster_x_product_family_x_time_block",
            "rule": "use disjoint gauge/product families where possible; score wet/dry phase, trend, variance and spatial rank in blocked folds",
            "minimum_claim": "method-independent OOF; do not call shared-gauge products source-independent",
            "locked_period_access": "1921-1939 prohibited",
        },
        {
            "gate_id": "PALEO_P_BLIND_RESERVE", "component": "precipitation", "period": "1000-1900",
            "independence_unit": "whole site/study cluster",
            "rule": "freeze the 12 Round111 clusters outside reconstruction; recover interpretation without viewing values; open values only after precipitation candidate freeze",
            "minimum_claim": "source-and-method-independent low-frequency phase/spatial validation; no amplitude calibration",
            "locked_period_access": "not applicable",
        },
        {
            "gate_id": "PALEO_T_EXTERNAL_REQUIRED", "component": "temperature", "period": "1000-1900",
            "independence_unit": "whole external site/study/archive cluster",
            "rule": "acquire genuinely new temperature-sensitive evidence or use predeclared event/spatial constraints; current local DoD2k pool has no independent reserve",
            "minimum_claim": "blocked until new evidence is frozen",
            "locked_period_access": "not applicable",
        },
        {
            "gate_id": "FINAL_LOCKBOX", "component": "joint", "period": "1921-1939",
            "independence_unit": "single final untouched interval",
            "rule": "open exactly once only after temperature and precipitation candidates and every threshold are frozen",
            "minimum_claim": "final temporal generalization test",
            "locked_period_access": "prohibited in Round111",
        },
    ])

    reserve_columns = [
        "datasetId", "site_cluster_id", "geo_siteName", "geo_meanLat", "geo_meanLon", "archiveType",
        "originalDatabase", "originalDataURL", "allowed_first_year_ce", "allowed_last_year_ce",
        "lia_occupied_50year_window_count", "interpretation_variable", "interpretation_direction",
        "interpretation_seasonality", "direction_ready", "seasonality_ready", "operator_ready",
        "evidence_class", "development_status", "allowed_future_use", "value_access_rule",
        "amplitude_target_allowed", "reason",
    ]
    reserve[reserve_columns].to_csv(output / "frozen_blind_hydroclimate_paleo_reserve_records.csv", index=False)
    reserve_clusters.to_csv(output / "frozen_blind_hydroclimate_paleo_reserve_clusters.csv", index=False)
    hydro_ledger.to_csv(output / "hydroclimate_evidence_independence_ledger.csv", index=False)
    temperature_ledger.to_csv(output / "temperature_evidence_independence_ledger.csv", index=False)
    modern_ledger.to_csv(output / "modern_evidence_family_ledger.csv", index=False)
    protocol.to_csv(output / "validation_protocol_frozen_v1.csv", index=False)

    evidence_counts = {
        "reconstruction_master_groups": int(len(master_groups)),
        "hydro_lia_eligible_records": int(hydro["lia_eligible"].sum()),
        "hydro_blind_reserve_records": int(len(reserve)),
        "hydro_blind_reserve_site_clusters": int(reserve_clusters["site_cluster_id"].nunique()),
        "hydro_blind_reserve_operator_ready_records": int(reserve["operator_ready"].sum()),
        "temperature_lia_eligible_records": int(temperature["lia_eligible"].sum()),
        "temperature_independent_blind_reserve_records": 0,
        "temperature_independent_blind_reserve_site_clusters": 0,
        "temperature_lia_only_overlap_records": int(len(temperature_lia_only)),
    }
    decision = {
        "round_id": ROUND_ID,
        "decision": "freeze_hydroclimate_blind_reserve_and_block_temperature_paleo_independence_claim",
        "evidence_counts": evidence_counts,
        "facts_contract": facts,
        "locked_period_1921_1939_accessed": False,
        "key_interpretation": [
            "1901-1920 is a local diagnostic only and is not an acceptance center.",
            "Twelve whole hydroclimate site clusters are now frozen before value interpretation.",
            "The local temperature pool has no source-independent paleo reserve; new external evidence is required.",
            "Shared-station or shared-gauge gridded products can support blocked method-OOF tests but are not automatically source-independent.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    contract = """# Round 111 Contract

- Purpose: freeze independent validation evidence before new temperature or precipitation candidate fitting.
- Inputs are prior metadata/audit outputs only. No original proxy value file is opened by this round.
- The 1921-1939 final lockbox is neither read nor summarized.
- Whole site/study clusters are the paleo independence unit; database labels alone do not establish independence.
- Shared station/gauge products may provide method-independent blocked OOF evidence but are not called source-independent without documented observation exclusion.
- Frozen reserve records cannot enter PSM fitting, hyperparameter selection, prior tuning, operator tuning, or candidate ranking.
- Original and prior result inputs are immutable and source hashes are checked before and after the round.
"""
    (output / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")

    final_result = f"""# Validation Evidence Independence Ledger

## Result

- Reconstruction proxy master: **{evidence_counts['reconstruction_master_groups']} unique evidence groups**.
- Hydroclimate LIA-eligible pool: **{evidence_counts['hydro_lia_eligible_records']} records**.
- Frozen blind hydroclimate reserve: **{evidence_counts['hydro_blind_reserve_records']} records in {evidence_counts['hydro_blind_reserve_site_clusters']} whole site clusters**.
- Immediately operator-ready reserve records: **{evidence_counts['hydro_blind_reserve_operator_ready_records']}**; the rest require publication-based direction/season metadata recovery while values remain blind.
- Temperature source-independent paleo reserve in the current local pool: **0 records / 0 sites**.
- The seven temperature LIA-only ice-core records are all existing-site overlaps with the reconstruction network.

## Decision

Freeze the 12 hydroclimate clusters as Class A source-and-method-independent low-frequency validation evidence. They may not be used for fitting or candidate selection and may not be treated as amplitude targets.

Temperature paleo acceptance remains blocked on a genuinely new external site/study/archive cluster or a separately predeclared event/spatial validation route. Mode-RA, MADA, CRU, CPC, GPCC and station-derived products retain useful roles, but shared proxy/station/gauge inputs prevent a blanket source-independence claim.

Modern validation will use blocked 1940-2000 folds. The final 1921-1939 interval remains sealed. The 1901-1920 anomaly is retained only as one local diagnostic, not the reconstruction objective.
"""
    (output / "FINAL_RESULT.md").write_text(final_result, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    axes[0].scatter(
        reserve_clusters["longitude"], reserve_clusters["latitude"],
        s=45 + 15 * reserve_clusters["record_count"], c=reserve_clusters["maximum_lia_50year_windows"],
        cmap="viridis", edgecolor="black", linewidth=0.5,
    )
    for _, row in reserve_clusters.iterrows():
        axes[0].annotate(row["site_name"], (row["longitude"], row["latitude"]), xytext=(3, 3),
                         textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Frozen hydroclimate paleo reserve (12 site clusters)")
    axes[0].grid(alpha=0.2)

    labels = ["Hydro\nreserve\nrecords", "Hydro\nreserve\nsites", "Temp\nreserve\nrecords", "Temp LIA-only\noverlap\nrecords"]
    values = [len(reserve), reserve_clusters["site_cluster_id"].nunique(), 0, len(temperature_lia_only)]
    colors = ["#2a9d8f", "#457b9d", "#c1121f", "#9c6644"]
    bars = axes[1].bar(labels, values, color=colors)
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Independent paleo evidence boundary")
    axes[1].set_ylim(0, max(values) * 1.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "validation_evidence_independence.png", dpi=180)
    plt.close(fig)

    after = {name: file_state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-input-mutated")
    manifest = {
        "round_id": ROUND_ID,
        "sources_unchanged": True,
        "locked_period_1921_1939_accessed": False,
        "inputs": before,
        "outputs": {
            path.name: file_state(path)
            for path in sorted(output.iterdir())
            if path.name != "source_manifest.json"
        },
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
