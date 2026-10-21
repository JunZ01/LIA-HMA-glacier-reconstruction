"""Build the unified proxy6/proxy9 evidence and metadata ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FACTS_SHA256 = "2e6765a63aa5d3b91b14d0e8768bafb53cb2a9fbc8f8810033c7f9ac8ddebde9"
ROUND_ID = "round_107_proxy_evidence_master_ledger_v1"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def normalize_proxy6(p6: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    table = p6.merge(mapping, on="proxy_index", how="left", validate="one_to_one")
    result = pd.DataFrame(
        {
            "network": "proxy6",
            "network_record_1based": table.proxy_index.astype(int),
            "record_id": "proxy6_" + table.proxy_index.astype(int).astype(str).str.zfill(3),
            "latitude": table.latitude,
            "longitude": table.longitude,
            "archive_type": table.proxy_type,
            "psm_method": table.psm_method,
            "psm_b0": table.coefficient_b0,
            "psm_temperature_coefficient": table.coefficient_b1,
            "psm_precipitation_coefficient": table.coefficient_b2,
            "psm_correlation": table.psm_correlation,
            "psm_pvalue": table.psm_pvalue,
            "first_year": table.first_observed_year,
            "last_year": table.last_observed_year,
            "observed_year_count": table.observed_year_count,
            "source_flag": table.source_flag,
            "data_name": table.data_name,
            "source_family": table.source_family,
            "archive_column_1based": table.archive_column_1based,
            "archive_mapping_correlation": table.archive_mapping_correlation,
            "archive_mapping_max_abs_residual": table.archive_mapping_max_abs_residual,
            "climate_variable": np.nan,
            "climate_relation": np.nan,
            "seasonality": np.nan,
            "site_name": np.nan,
            "measurement": np.nan,
            "interpretation_direction": np.nan,
            "publication_author": np.nan,
            "publication_cite_key": np.nan,
            "publication_year": np.nan,
            "metadata_coefficient_sign_match": np.nan,
            "seasonality_matches_archived_annual": np.nan,
            "public_identity_id": np.nan,
            "public_identity_citation": np.nan,
        }
    )
    exact = result.archive_mapping_correlation.ge(1 - 1e-12) & result.archive_mapping_max_abs_residual.le(1e-10)
    result["exact_archive_identity"] = exact
    return result


def normalize_proxy9(p9: pd.DataFrame, study: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    study_keep = study[["proxy_index", "interpretation_direction", "publication_author", "publication_cite_key", "publication_year"]].drop_duplicates("proxy_index")
    table = p9.merge(study_keep, left_on="proxy9_record_1based", right_on="proxy_index", how="left", validate="one_to_one")
    public_keep = public[["proxy9_record", "public_id", "citation"]].drop_duplicates("proxy9_record")
    table = table.merge(public_keep, left_on="proxy9_record_1based", right_on="proxy9_record", how="left", validate="one_to_one")
    result = pd.DataFrame(
        {
            "network": "proxy9",
            "network_record_1based": table.proxy9_record_1based.astype(int),
            "record_id": "proxy9_" + table.proxy9_record_1based.astype(int).astype(str).str.zfill(3),
            "latitude": table.latitude,
            "longitude": table.longitude,
            "archive_type": table.proxy_type,
            "psm_method": table.psm_method,
            "psm_b0": table.psm_b0,
            "psm_temperature_coefficient": table.psm_temp_coefficient_b1,
            "psm_precipitation_coefficient": table.psm_prcp_coefficient_b2,
            "psm_correlation": table.psm_correlation,
            "psm_pvalue": table.psm_p_value,
            "first_year": table.first_finite_year,
            "last_year": table.last_finite_year,
            "observed_year_count": np.nan,
            "source_flag": table.combined_source_flag,
            "data_name": table.combined_name,
            "source_family": np.where(table.combined_source_flag.astype(str).eq("1"), "PAGES2k2017", "other_or_unknown"),
            "archive_column_1based": table.combined_column_1based,
            "archive_mapping_correlation": table.identity_correlation,
            "archive_mapping_max_abs_residual": table.identity_maximum_abs_residual,
            "climate_variable": table.lmr_climate_variable,
            "climate_relation": table.lmr_relation,
            "seasonality": table.lmr_seasonality,
            "site_name": table.lmr_site_name,
            "measurement": table.lmr_measurement,
            "interpretation_direction": table.interpretation_direction,
            "publication_author": table.publication_author,
            "publication_cite_key": table.publication_cite_key,
            "publication_year": table.publication_year,
            "metadata_coefficient_sign_match": table.metadata_coefficient_sign_match,
            "seasonality_matches_archived_annual": table.seasonality_matches_archived_annual,
            "public_identity_id": table.public_id,
            "public_identity_citation": table.citation,
        }
    )
    result["exact_archive_identity"] = result.archive_mapping_correlation.ge(1 - 1e-12) & result.archive_mapping_max_abs_residual.le(1e-10)
    return result


def add_evidence_groups(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    result["evidence_group_id"] = result.record_id
    p6 = result[result.network.eq("proxy6") & result.exact_archive_identity & result.archive_column_1based.notna()]
    p9 = result[result.network.eq("proxy9") & result.exact_archive_identity & result.archive_column_1based.notna()]
    shared = set(p6.archive_column_1based.astype(int)) & set(p9.archive_column_1based.astype(int))
    for column in sorted(shared):
        rows = result.archive_column_1based.eq(column) & result.exact_archive_identity
        coords = result.loc[rows, ["latitude", "longitude"]].to_numpy(float)
        if len(coords) != 2 or np.max(np.abs(coords - coords[0])) > 1e-8:
            continue
        result.loc[rows, "evidence_group_id"] = f"archive_{column:04d}"
    result["cross_network_duplicate"] = result.groupby("evidence_group_id").record_id.transform("size").gt(1)
    return result


def add_completeness(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    result["source_identity_known"] = present(result.source_flag) & present(result.data_name)
    result["chronology_bounds_known"] = result.first_year.notna() & result.last_year.notna()
    result["seasonality_known"] = present(result.seasonality)
    result["climate_target_known"] = present(result.climate_variable)
    result["publication_known"] = present(result.publication_cite_key) | present(result.public_identity_citation)
    result["psm_significant_p05"] = result.psm_pvalue.le(0.05)
    gaps = []
    for row in result.itertuples(index=False):
        missing = []
        if not row.source_identity_known: missing.append("source_identity")
        if not row.chronology_bounds_known: missing.append("chronology_bounds")
        if not row.seasonality_known: missing.append("seasonality")
        if not row.climate_target_known: missing.append("climate_target")
        if not row.publication_known: missing.append("publication")
        if not row.exact_archive_identity: missing.append("archive_identity")
        gaps.append(";".join(missing))
    result["metadata_gaps"] = gaps
    result["validation_metadata_ready"] = result.metadata_gaps.eq("")
    result["review_priority"] = np.select(
        [result.psm_significant_p05 & ~result.validation_metadata_ready, ~result.source_identity_known, ~result.validation_metadata_ready],
        [1, 1, 2],
        default=3,
    )
    return result.sort_values(["review_priority", "network", "network_record_1based"]).reset_index(drop=True)


def main() -> int:
    root = project_root()
    base = root / "validation_data/results/paleo_surface_validation_v0"
    output = base / "climate_reconstruction_improvement_loop_v2" / ROUND_ID
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "proxy6_metadata": base / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy6_mapping": base / "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_mapping.csv",
        "proxy9_metadata": base / "climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "proxy9_study": base / "climate_reconstruction_pages2k_fixed_network_study_batch_v1/fixed_network_study_metadata.csv",
        "proxy9_public": base / "climate_reconstruction_proxy9_record_provenance_v1/record_public_identity.csv",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if digest(paths["facts"]) != FACTS_SHA256: raise RuntimeError("facts-hash-conflict")
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    p6 = normalize_proxy6(pd.read_csv(paths["proxy6_metadata"]), pd.read_csv(paths["proxy6_mapping"]))
    p9 = normalize_proxy9(pd.read_csv(paths["proxy9_metadata"]), pd.read_csv(paths["proxy9_study"]), pd.read_csv(paths["proxy9_public"]))
    ledger = add_completeness(add_evidence_groups(pd.concat([p6, p9], ignore_index=True)))
    if len(ledger) != 499 or ledger.record_id.nunique() != 499: raise RuntimeError("ledger-record-contract-failed")
    ledger.to_csv(output / "proxy_evidence_master_ledger.csv", index=False)
    gaps = ledger[~ledger.validation_metadata_ready].copy()
    gaps.to_csv(output / "metadata_gap_queue.csv", index=False)
    groups = ledger.groupby("evidence_group_id", as_index=False).agg(
        network_record_count=("record_id", "size"), networks=("network", lambda x: ";".join(sorted(set(x)))),
        latitude=("latitude", "first"), longitude=("longitude", "first"), review_priority=("review_priority", "min")
    )
    groups.to_csv(output / "evidence_groups.csv", index=False)

    summary = {
        "status": "complete", "locked_1921_1939_accessed": False, "time_series_values_read": False,
        "network_record_count": int(len(ledger)), "proxy6_count": int((ledger.network == "proxy6").sum()),
        "proxy9_count": int((ledger.network == "proxy9").sum()), "unique_evidence_group_count": int(ledger.evidence_group_id.nunique()),
        "cross_network_duplicate_record_count": int(ledger.cross_network_duplicate.sum()),
        "cross_network_shared_evidence_group_count": int(groups.network_record_count.eq(2).sum()),
        "validation_metadata_ready_count": int(ledger.validation_metadata_ready.sum()),
        "priority_1_review_count": int(ledger.review_priority.eq(1).sum()),
        "source_identity_missing_count": int((~ledger.source_identity_known).sum()),
        "seasonality_missing_count": int((~ledger.seasonality_known).sum()),
        "climate_target_missing_count": int((~ledger.climate_target_known).sum()),
        "publication_missing_count": int((~ledger.publication_known).sum()),
        "decision": "freeze_master_ledger_and_begin_priority_1_metadata_repair",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = ledger.groupby(["network", "review_priority"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#C44E52", "#DD8452", "#4C72B0"])
    ax.set(xlabel="Proxy network", ylabel="Record count", title="Proxy metadata review queue")
    ax.legend(title="Priority", frameon=False)
    ax.figure.tight_layout(); ax.figure.savefig(output / "proxy_metadata_review_queue.png", dpi=220); plt.close(ax.figure)
    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Proxy Evidence Master Ledger", "",
            f"- Network records: `{len(ledger)}` (`{len(p6)}` proxy6, `{len(p9)}` proxy9).",
            f"- Unique evidence groups after exact cross-network deduplication: `{ledger.evidence_group_id.nunique()}`.",
            f"- Shared proxy6/proxy9 evidence groups: `{groups.network_record_count.eq(2).sum()}`; duplicated network records: `{ledger.cross_network_duplicate.sum()}`.",
            f"- Records ready for validation metadata: `{ledger.validation_metadata_ready.sum()}`; priority-1 repairs: `{ledger.review_priority.eq(1).sum()}`.",
            f"- Missing source identity / seasonality / climate target / publication: `{(~ledger.source_identity_known).sum()}` / `{(~ledger.seasonality_known).sum()}` / `{(~ledger.climate_target_known).sum()}` / `{(~ledger.publication_known).sum()}`.",
            "", "Decision: `freeze_master_ledger_and_begin_priority_1_metadata_repair`.",
        ]) + "\n", encoding="utf-8"
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"local_sources_before": before, "local_sources_after": after, "local_source_states_unchanged": before == after, "locked_1921_1939_accessed": False, "time_series_values_read": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]: raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
