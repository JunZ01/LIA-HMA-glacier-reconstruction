from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUND_ID = "round_108_proxy_metadata_reconciliation_v1"
FACTS_SHA256 = "2e6765a63aa5d3b91b14d0e8768bafb53cb2a9fbc8f8810033c7f9ac8ddebdE9".lower()


def root() -> Path:
    return Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<missing>", "unresolved"} else text


def fill_empty(frame: pd.DataFrame, target: str, source: str) -> None:
    current = frame[target].map(clean)
    candidate = frame[source].map(clean)
    frame[target] = current.where(current.ne(""), candidate)


def conflict(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    a = frame[left].map(clean)
    b = frame[right].map(clean)
    return a.ne("") & b.ne("") & a.str.casefold().ne(b.str.casefold())


def source_known(row: pd.Series) -> bool:
    return any(clean(row.get(field, "")) for field in ("formal_metadata_identity", "data_name", "public_identity_id"))


def main() -> int:
    project = root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    output = loop / ROUND_ID
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "ledger": loop / "round_107_proxy_evidence_master_ledger_v1" / "proxy_evidence_master_ledger.csv",
        "round07": loop / "round_07_proxy_network_suitability_inventory_v2" / "proxy9_suitability_inventory.csv",
        "fixed_publication": base / "climate_reconstruction_pages2k_fixed_network_study_batch_v1" / "fixed_network_study_metadata.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    before = {name: state(path) for name, path in paths.items()}
    if before["facts"]["sha256"].lower() != FACTS_SHA256:
        raise RuntimeError("facts-hash-changed-stop-and-reread")

    ledger = pd.read_csv(paths["ledger"])
    r07 = pd.read_csv(paths["round07"])
    publications = pd.read_csv(paths["fixed_publication"])
    if len(ledger) != 499 or len(r07) != 297:
        raise RuntimeError("unexpected-input-row-count")

    r07_fields = [
        "record_index_1based",
        "metadata_sources",
        "metadata_identities",
        "archive_type",
        "measurement",
        "documented_climate_variable",
        "documented_climate_detail",
        "documented_seasonality",
        "documented_direction",
        "source_roles",
        "network_role",
        "formal_metadata_entry_count",
    ]
    r07 = r07[r07_fields].rename(
        columns={
            "record_index_1based": "network_record_1based",
            "archive_type": "r07_archive_type",
            "measurement": "r07_measurement",
            "documented_climate_variable": "r07_climate_variable",
            "documented_climate_detail": "r07_climate_detail",
            "documented_seasonality": "r07_seasonality",
            "documented_direction": "r07_direction",
        }
    )
    proxy9 = ledger[ledger["network"].eq("proxy9")].merge(
        r07, on="network_record_1based", how="left", validate="one_to_one"
    )
    proxy6 = ledger[ledger["network"].eq("proxy6")].copy()
    records = pd.concat([proxy6, proxy9], ignore_index=True, sort=False)

    field_pairs = {
        "archive_type": "r07_archive_type",
        "measurement": "r07_measurement",
        "climate_variable": "r07_climate_variable",
        "seasonality": "r07_seasonality",
        "interpretation_direction": "r07_direction",
    }
    conflict_rows: list[dict[str, object]] = []
    for target, source in field_pairs.items():
        mask = records["network"].eq("proxy9") & conflict(records, target, source)
        for row in records.loc[mask, ["network", "network_record_1based", "evidence_group_id", target, source]].itertuples(index=False):
            conflict_rows.append({
                "network": row[0],
                "network_record_1based": int(row[1]),
                "evidence_group_id": row[2],
                "field": target,
                "round107_value": row[3],
                "round07_value": row[4],
            })
        fill_empty(records, target, source)

    records["formal_metadata_identity"] = records.get("metadata_identities", pd.Series("", index=records.index)).map(clean)
    records["formal_metadata_source"] = records.get("metadata_sources", pd.Series("", index=records.index)).map(clean)
    records["documented_climate_detail"] = records.get("r07_climate_detail", pd.Series("", index=records.index)).map(clean)
    records["documented_source_roles"] = records.get("source_roles", pd.Series("", index=records.index)).map(clean)
    records["documented_network_role"] = records.get("network_role_y", records.get("network_role", pd.Series("", index=records.index))).map(clean)

    # Propagate only through exact shared groups already frozen in Round 107.
    propagation_fields = [
        "formal_metadata_identity",
        "formal_metadata_source",
        "archive_type",
        "measurement",
        "climate_variable",
        "documented_climate_detail",
        "seasonality",
        "interpretation_direction",
        "documented_source_roles",
        "documented_network_role",
    ]
    proxy9_by_group = records[records["network"].eq("proxy9")].set_index("evidence_group_id")
    propagated_cells = 0
    for index in records.index[records["network"].eq("proxy6") & records["cross_network_duplicate"].astype(bool)]:
        group = records.at[index, "evidence_group_id"]
        source_row = proxy9_by_group.loc[group]
        if isinstance(source_row, pd.DataFrame):
            raise RuntimeError(f"ambiguous-proxy9-evidence-group:{group}")
        for field in propagation_fields:
            if not clean(records.at[index, field]) and clean(source_row[field]):
                records.at[index, field] = clean(source_row[field])
                propagated_cells += 1

    # Existing publication metadata are retained and propagated within exact groups.
    for field in ("publication_author", "publication_cite_key", "publication_year"):
        records[field] = records[field].map(clean)
    proxy9_by_group = records[records["network"].eq("proxy9")].set_index("evidence_group_id")
    for index in records.index[records["network"].eq("proxy6") & records["cross_network_duplicate"].astype(bool)]:
        source_row = proxy9_by_group.loc[records.at[index, "evidence_group_id"]]
        for field in ("publication_author", "publication_cite_key", "publication_year"):
            if not clean(records.at[index, field]) and clean(source_row[field]):
                records.at[index, field] = clean(source_row[field])
                propagated_cells += 1

    records["source_identity_known_v2"] = records.apply(source_known, axis=1)
    records["seasonality_known_v2"] = records["seasonality"].map(clean).ne("")
    records["climate_target_known_v2"] = records["climate_variable"].map(clean).ne("")
    records["publication_known_v2"] = records[["publication_author", "publication_cite_key"]].apply(
        lambda row: any(clean(value) for value in row), axis=1
    )
    records["operator_metadata_ready"] = (
        records["source_identity_known_v2"]
        & records["chronology_bounds_known"].astype(bool)
        & records["seasonality_known_v2"]
        & records["climate_target_known_v2"]
        & records["exact_archive_identity"].astype(bool)
    )
    records["release_citation_ready"] = records["operator_metadata_ready"] & records["publication_known_v2"]

    gap_fields = [
        ("source_identity_known_v2", "source_identity"),
        ("chronology_bounds_known", "chronology"),
        ("seasonality_known_v2", "seasonality"),
        ("climate_target_known_v2", "climate_target"),
        ("exact_archive_identity", "archive_identity"),
    ]
    records["operator_metadata_gaps"] = records.apply(
        lambda row: ";".join(label for field, label in gap_fields if not bool(row[field])), axis=1
    )
    records["operator_review_priority"] = np.select(
        [records["psm_significant_p05"].astype(bool) & ~records["operator_metadata_ready"], ~records["operator_metadata_ready"]],
        [1, 2],
        default=3,
    ).astype(int)

    conflicts = pd.DataFrame(conflict_rows, columns=[
        "network", "network_record_1based", "evidence_group_id", "field", "round107_value", "round07_value"
    ])
    group_rows = []
    for group, subset in records.groupby("evidence_group_id", sort=True):
        group_rows.append({
            "evidence_group_id": group,
            "network_record_count": int(len(subset)),
            "networks": ";".join(sorted(subset["network"].unique())),
            "latitude": float(subset["latitude"].iloc[0]),
            "longitude": float(subset["longitude"].iloc[0]),
            "documented_network_role": ";".join(sorted({clean(value) for value in subset["documented_network_role"] if clean(value)})),
            "operator_metadata_ready": bool(subset["operator_metadata_ready"].all()),
            "release_citation_ready": bool(subset["release_citation_ready"].all()),
            "psm_significant_any": bool(subset["psm_significant_p05"].astype(bool).any()),
            "operator_metadata_gaps": ";".join(sorted({clean(value) for value in subset["operator_metadata_gaps"] if clean(value)})),
        })
    groups = pd.DataFrame(group_rows)
    operator_queue = records[~records["operator_metadata_ready"]].sort_values(
        ["operator_review_priority", "network", "network_record_1based"]
    )
    publication_queue = records[records["operator_metadata_ready"] & ~records["publication_known_v2"]].sort_values(
        ["formal_metadata_source", "formal_metadata_identity", "network", "network_record_1based"]
    )

    records = records.sort_values(["network", "network_record_1based"]).reset_index(drop=True)
    records.to_csv(output / "proxy_evidence_master_ledger_reconciled.csv", index=False)
    groups.to_csv(output / "evidence_groups_reconciled.csv", index=False)
    operator_queue.to_csv(output / "operator_metadata_repair_queue.csv", index=False)
    publication_queue.to_csv(output / "publication_identity_queue.csv", index=False)
    conflicts.to_csv(output / "metadata_conflicts.csv", index=False)

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "time_series_values_read": False,
        "network_record_count": int(len(records)),
        "unique_evidence_group_count": int(len(groups)),
        "round07_formal_metadata_proxy9_count": int((records["network"].eq("proxy9") & records["formal_metadata_identity"].map(clean).ne("")).sum()),
        "propagated_metadata_cell_count": int(propagated_cells),
        "metadata_conflict_count": int(len(conflicts)),
        "operator_metadata_ready_record_count": int(records["operator_metadata_ready"].sum()),
        "operator_metadata_ready_evidence_group_count": int(groups["operator_metadata_ready"].sum()),
        "release_citation_ready_record_count": int(records["release_citation_ready"].sum()),
        "publication_queue_record_count": int(len(publication_queue)),
        "priority_1_operator_repair_record_count": int((records["operator_review_priority"].eq(1)).sum()),
        "operator_gap_record_count": int(len(operator_queue)),
        "documented_role_evidence_group_counts": groups["documented_network_role"].replace("", "unresolved").value_counts().to_dict(),
        "decision": "freeze_reconciled_operator_metadata_and_resolve_remaining_unique_evidence_groups",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    role_counts = groups["documented_network_role"].replace("", "unresolved").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].bar(["Round 107", "Round 108"], [1, summary["operator_metadata_ready_record_count"]], color=["#9ca3af", "#147d64"])
    axes[0].set_ylabel("Network records")
    axes[0].set_title("Operator metadata readiness")
    axes[0].grid(axis="y", alpha=0.25)
    role_counts.sort_values().plot.barh(ax=axes[1], color="#4472a8")
    axes[1].set_xlabel("Unique evidence groups")
    axes[1].set_title("Documented climate role")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("HMA proxy evidence metadata reconciliation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "proxy_metadata_reconciliation.png", dpi=180)
    plt.close(fig)

    final = (
        "# Proxy Metadata Reconciliation\n\n"
        f"- Reconciled `{len(records)}` network records in `{len(groups)}` unique evidence groups.\n"
        f"- Operator-metadata ready: `{summary['operator_metadata_ready_record_count']}` records / "
        f"`{summary['operator_metadata_ready_evidence_group_count']}` evidence groups.\n"
        f"- Release-citation ready: `{summary['release_citation_ready_record_count']}` records; "
        f"publication queue: `{len(publication_queue)}` records.\n"
        f"- Remaining priority-1 operator repairs: `{summary['priority_1_operator_repair_record_count']}`; "
        f"metadata conflicts: `{len(conflicts)}`.\n\n"
        f"Decision: `{summary['decision']}`.\n"
    )
    (output / "FINAL_RESULT.md").write_text(final, encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
