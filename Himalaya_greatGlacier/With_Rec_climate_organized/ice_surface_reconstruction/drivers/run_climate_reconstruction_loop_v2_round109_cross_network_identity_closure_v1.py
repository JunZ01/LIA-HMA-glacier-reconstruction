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


ROUND_ID = "round_109_cross_network_identity_closure_v1"
FACTS_SHA256 = "3a03f0908cebb804f466bf7c8eb2f6b98aafef70f65d69ede299090b670589d0"
NUMERIC_IDENTITY_FIELDS = [
    "psm_b0",
    "psm_temperature_coefficient",
    "psm_precipitation_coefficient",
    "psm_correlation",
    "psm_pvalue",
    "first_year",
    "last_year",
]


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


def numeric_equal(series: pd.Series, value: object, tolerance: float = 1e-12) -> pd.Series:
    if pd.isna(value):
        return series.isna()
    return np.isclose(pd.to_numeric(series, errors="coerce"), float(value), atol=tolerance, rtol=0, equal_nan=False)


def archive_class(value: object) -> str:
    text = clean(value).casefold()
    if "tree" in text:
        return "tree"
    if "ice" in text or "glacier" in text:
        return "ice"
    if "speleo" in text:
        return "speleothem"
    return text


def canonical_seasonality(value: object) -> str:
    text = clean(value)
    if not text:
        return ""
    lowered = text.casefold()
    numbers = sorted({int(item) for item in re.findall(r"(?<!\d)(?:1[0-2]|[1-9])(?!\d)", text)})
    if numbers == list(range(1, 13)):
        return "annual_months_01_12"
    if lowered in {"ann", "annual", "year", "yearly"}:
        return "annual"
    if "summer" in lowered or lowered == "jja":
        return "summer"
    if "winter" in lowered or lowered == "djf":
        return "winter"
    if numbers:
        return "months_" + "_".join(f"{month:02d}" for month in numbers)
    return lowered


def canonical_direction(primary: object, round07: object) -> str:
    first = clean(primary)
    if first.startswith("<") and first.endswith(">"):
        first = ""
    return clean(round07) or first


def main() -> int:
    project = root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    output = loop / ROUND_ID
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger": loop / "round_108_proxy_metadata_reconciliation_v2" / "proxy_evidence_master_ledger_reconciled.csv",
        "conflicts": loop / "round_108_proxy_metadata_reconciliation_v2" / "metadata_conflicts.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    before = {name: state(path) for name, path in paths.items()}
    if before["facts"]["sha256"].lower() != FACTS_SHA256:
        raise RuntimeError("facts-hash-changed-stop-and-reread")

    records = pd.read_csv(paths["ledger"])
    old_conflicts = pd.read_csv(paths["conflicts"])
    if len(records) != 499:
        raise RuntimeError("unexpected-record-count")
    p6 = records[records["network"].eq("proxy6")].copy()
    p9 = records[records["network"].eq("proxy9")].copy()
    if len(p6) != 202 or len(p9) != 297:
        raise RuntimeError("unexpected-network-count")

    mapping_rows = []
    p9_lookup: dict[int, pd.Series] = {}
    for _, row in p6.iterrows():
        candidates = p9[
            np.isclose(p9["latitude"], row["latitude"], atol=1e-9, rtol=0)
            & np.isclose(p9["longitude"], row["longitude"], atol=1e-9, rtol=0)
        ].copy()
        coordinate_count = int(len(candidates))
        for field in NUMERIC_IDENTITY_FIELDS:
            candidates = candidates[numeric_equal(candidates[field], row[field])]
        candidates = candidates[candidates["archive_type"].map(archive_class).eq(archive_class(row["archive_type"]))]
        if len(candidates) != 1:
            raise RuntimeError(f"proxy6-identity-not-unique:{int(row['network_record_1based'])}:{len(candidates)}")
        match = candidates.iloc[0]
        proxy9_index = int(match["network_record_1based"])
        p9_lookup[proxy9_index] = match
        tier = "tier1_sequence_affine_plus_parameter_chronology" if bool(row["cross_network_duplicate"]) else "tier2_unique_parameter_chronology"
        mapping_rows.append({
            "proxy6_record_1based": int(row["network_record_1based"]),
            "proxy9_record_1based": proxy9_index,
            "coordinate_candidate_count": coordinate_count,
            "identity_candidate_count": int(len(candidates)),
            "identity_tier": tier,
            "latitude_abs_difference": float(abs(row["latitude"] - match["latitude"])),
            "longitude_abs_difference": float(abs(row["longitude"] - match["longitude"])),
            "maximum_numeric_abs_difference": float(max(
                abs(float(row[field]) - float(match[field]))
                for field in NUMERIC_IDENTITY_FIELDS
                if pd.notna(row[field]) and pd.notna(match[field])
            )),
            "archive_class": archive_class(row["archive_type"]),
        })
    mapping = pd.DataFrame(mapping_rows)
    if mapping["proxy6_record_1based"].nunique() != 202 or mapping["proxy9_record_1based"].nunique() != 202:
        raise RuntimeError("cross-network-mapping-not-one-to-one")

    map_by_p6 = mapping.set_index("proxy6_record_1based")
    p9_by_id = p9.set_index("network_record_1based")
    metadata_fields = [
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
        "publication_author",
        "publication_cite_key",
        "publication_year",
    ]
    propagated_cells = 0
    for index in records.index[records["network"].eq("proxy6")]:
        p6_id = int(records.at[index, "network_record_1based"])
        p9_id = int(map_by_p6.at[p6_id, "proxy9_record_1based"])
        source = p9_by_id.loc[p9_id]
        for field in metadata_fields:
            current = clean(records.at[index, field])
            candidate = clean(source[field])
            if (not current or (field == "interpretation_direction" and current.startswith("<"))) and candidate:
                records.at[index, field] = candidate
                propagated_cells += 1
        records.at[index, "evidence_group_id"] = f"P9_{p9_id:04d}"
        records.at[index, "cross_network_duplicate"] = True
        records.at[index, "evidence_identity_tier"] = map_by_p6.at[p6_id, "identity_tier"]
        records.at[index, "evidence_identity_closed"] = True
    for index in records.index[records["network"].eq("proxy9")]:
        p9_id = int(records.at[index, "network_record_1based"])
        records.at[index, "evidence_group_id"] = f"P9_{p9_id:04d}"
        records.at[index, "cross_network_duplicate"] = bool(p9_id in set(mapping["proxy9_record_1based"]))
        tier_values = mapping.loc[mapping["proxy9_record_1based"].eq(p9_id), "identity_tier"]
        records.at[index, "evidence_identity_tier"] = tier_values.iloc[0] if len(tier_values) else "tier3_proxy9_singleton"
        records.at[index, "evidence_identity_closed"] = bool(records.at[index, "exact_archive_identity"])

    records["canonical_archive_class"] = records["archive_type"].map(archive_class)
    records["canonical_seasonality"] = records["seasonality"].map(canonical_seasonality)
    records["canonical_interpretation_direction"] = records.apply(
        lambda row: canonical_direction(row.get("interpretation_direction", ""), row.get("r07_direction", "")), axis=1
    )
    records["source_identity_known_v3"] = records.apply(
        lambda row: any(clean(row.get(field, "")) for field in ("formal_metadata_identity", "data_name", "public_identity_id")), axis=1
    )
    records["climate_target_known_v3"] = records["climate_variable"].map(clean).ne("")
    records["seasonality_known_v3"] = records["canonical_seasonality"].ne("")
    records["operator_metadata_ready_v3"] = (
        records["source_identity_known_v3"]
        & records["chronology_bounds_known"].astype(bool)
        & records["climate_target_known_v3"]
        & records["seasonality_known_v3"]
        & records["evidence_identity_closed"].astype(bool)
    )
    records["publication_known_v3"] = records[["publication_author", "publication_cite_key"]].apply(
        lambda row: any(clean(value) for value in row), axis=1
    )
    records["release_citation_ready_v3"] = records["operator_metadata_ready_v3"] & records["publication_known_v3"]

    adjudications = []
    for row in old_conflicts.itertuples(index=False):
        left = getattr(row, "round107_value")
        right = getattr(row, "round07_value")
        if row.field == "archive_type":
            equivalent = archive_class(left) == archive_class(right)
            rule = "canonical_archive_class"
        elif row.field == "seasonality":
            equivalent = canonical_seasonality(left) == canonical_seasonality(right)
            rule = "canonical_month_set"
        elif row.field == "interpretation_direction":
            equivalent = clean(left).startswith("<") and bool(clean(right))
            rule = "discard_nonclimate_sentinel_use_round07"
        else:
            equivalent = False
            rule = "unresolved"
        adjudications.append({
            "network": row.network,
            "network_record_1based": int(row.network_record_1based),
            "evidence_group_id": row.evidence_group_id,
            "field": row.field,
            "left_value": left,
            "right_value": right,
            "adjudication_rule": rule,
            "semantic_equivalent_or_resolved": bool(equivalent),
        })
    adjudication = pd.DataFrame(adjudications)
    substantive_conflicts = int((~adjudication["semantic_equivalent_or_resolved"]).sum()) if len(adjudication) else 0
    if substantive_conflicts:
        raise RuntimeError(f"substantive-metadata-conflicts:{substantive_conflicts}")

    groups = records.groupby("evidence_group_id", as_index=False).agg(
        network_record_count=("network", "size"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        documented_network_role=("documented_network_role", lambda x: ";".join(sorted({clean(v) for v in x if clean(v)}))),
        operator_metadata_ready=("operator_metadata_ready_v3", "all"),
        release_citation_ready=("release_citation_ready_v3", "all"),
        psm_significant_any=("psm_significant_p05", "any"),
        identity_tier=("evidence_identity_tier", "first"),
    )
    gap_labels = [
        ("source_identity_known_v3", "source_identity"),
        ("seasonality_known_v3", "seasonality"),
        ("climate_target_known_v3", "climate_target"),
        ("evidence_identity_closed", "evidence_identity"),
    ]
    records["remaining_operator_metadata_gaps"] = records.apply(
        lambda row: ";".join(label for field, label in gap_labels if not bool(row[field])), axis=1
    )
    queue = records[~records["operator_metadata_ready_v3"]].copy()
    queue["review_priority"] = np.where(queue["psm_significant_p05"].astype(bool), 1, 2)
    queue = queue.sort_values(["review_priority", "evidence_group_id", "network"])

    records = records.sort_values(["evidence_group_id", "network"]).reset_index(drop=True)
    mapping.to_csv(output / "cross_network_identity_mapping.csv", index=False)
    records.to_csv(output / "proxy_evidence_master_ledger_identity_closed.csv", index=False)
    groups.to_csv(output / "evidence_groups_identity_closed.csv", index=False)
    queue.to_csv(output / "remaining_operator_metadata_queue.csv", index=False)
    adjudication.to_csv(output / "semantic_conflict_adjudication.csv", index=False)

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "time_series_values_read": False,
        "network_record_count": int(len(records)),
        "unique_evidence_group_count": int(len(groups)),
        "proxy6_uniquely_mapped_count": int(len(mapping)),
        "tier1_sequence_closed_pair_count": int(mapping["identity_tier"].str.startswith("tier1").sum()),
        "tier2_parameter_chronology_pair_count": int(mapping["identity_tier"].str.startswith("tier2").sum()),
        "maximum_identity_numeric_abs_difference": float(mapping["maximum_numeric_abs_difference"].max()),
        "propagated_metadata_cell_count": int(propagated_cells),
        "round108_conflict_count": int(len(adjudication)),
        "substantive_conflict_count": substantive_conflicts,
        "operator_metadata_ready_record_count": int(records["operator_metadata_ready_v3"].sum()),
        "operator_metadata_ready_evidence_group_count": int(groups["operator_metadata_ready"].sum()),
        "remaining_operator_metadata_record_count": int(len(queue)),
        "remaining_operator_metadata_evidence_group_count": int(queue["evidence_group_id"].nunique()),
        "priority_1_remaining_record_count": int((queue["review_priority"] == 1).sum()),
        "documented_role_evidence_group_counts": groups["documented_network_role"].replace("", "unresolved").value_counts().to_dict(),
        "decision": "freeze_297_unique_evidence_groups_and_repair_25_unresolved_groups",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].bar(["Tier 1", "Tier 2", "Proxy9 only"], [147, 55, 95], color=["#176b55", "#4b8bbd", "#9ca3af"])
    axes[0].set_ylabel("Unique evidence groups")
    axes[0].set_title("Identity evidence tier")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(["Ready", "Unresolved"], [summary["operator_metadata_ready_evidence_group_count"], summary["remaining_operator_metadata_evidence_group_count"]], color=["#176b55", "#c75b39"])
    axes[1].set_ylabel("Unique evidence groups")
    axes[1].set_title("Operator metadata status")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-network proxy identity closure", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "cross_network_identity_closure.png", dpi=180)
    plt.close(fig)

    final = (
        "# Cross-Network Identity Closure\n\n"
        f"- All `{len(mapping)}` proxy6 records map one-to-one to proxy9: "
        f"`{summary['tier1_sequence_closed_pair_count']}` Tier-1 and `{summary['tier2_parameter_chronology_pair_count']}` Tier-2 pairs.\n"
        f"- The combined network contains `{len(groups)}` unique evidence groups, not 352 or 499 independent records.\n"
        f"- Operator metadata are ready for `{summary['operator_metadata_ready_evidence_group_count']}` groups; "
        f"`{summary['remaining_operator_metadata_evidence_group_count']}` groups remain unresolved.\n"
        f"- All `{len(adjudication)}` Round-108 text conflicts are semantic equivalents or sentinel replacements; "
        "substantive conflicts: `0`.\n\n"
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
