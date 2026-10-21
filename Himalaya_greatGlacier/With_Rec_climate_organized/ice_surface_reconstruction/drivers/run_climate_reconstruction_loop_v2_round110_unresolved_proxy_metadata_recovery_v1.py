from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat


ROUND_ID = "round_110_unresolved_proxy_metadata_recovery_v1"
FACTS_SHA256 = "3a03f0908cebb804f466bf7c8eb2f6b98aafef70f65d69ede299090b670589d0"
MANN_METADATA_VARIABLES = [
    "paleoData_pages2kID",
    "dataSetName",
    "archiveType",
    "geo_meanElev",
    "geo_meanLat",
    "geo_meanLon",
    "paleoData_variableName",
    "paleoData_units",
    "paleoData_proxy",
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
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def stat_only(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<missing>", "unresolved"} else text


def scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    return clean(item)


def set_if_empty(frame: pd.DataFrame, index: int, field: str, value: object) -> bool:
    candidate = clean(value)
    if not clean(frame.at[index, field]) and candidate:
        frame.at[index, field] = candidate
        return True
    return False


def canonical_season(value: object) -> str:
    text = clean(value).casefold()
    if not text:
        return ""
    if text in {"[6, 7, 8]", "jja", "summer"}:
        return "JJA"
    if "jj" in text and "summer" in text:
        return "JJAS_boreal_summer"
    if "boreal_summer" in text or "bsummer" in text:
        return "boreal_summer"
    if all(str(month) in text for month in range(1, 13)):
        return "annual_months_01_12"
    return clean(value)


def main() -> int:
    project = root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    output = loop / ROUND_ID
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger": loop / "round_109_cross_network_identity_closure_v1" / "proxy_evidence_master_ledger_identity_closed.csv",
        "lmr_metadata": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0\lmr_meta.csv"),
        "dod2k_metadata": loop / "round_34_sisalv3_hma_hydroclimate_source_audit_v1" / "dod2k_v2.0_compact_metadata.csv",
        "round73_iso": loop / "round_73_mode_ra_hma_source_overlap_v1" / "hma_unique_source_overlap.csv",
        "record_provenance": base / "climate_reconstruction_proxy9_record_provenance_v1" / "record_public_identity.csv",
        "mann_metadata_mat": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\past2000_proxy_data1\proxy_PAGES2k_MannleTS005.mat"),
        "external_evidence": output / "external_metadata_evidence.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    hashed_names = [name for name in paths if name != "mann_metadata_mat"]
    before = {name: state(paths[name]) for name in hashed_names}
    before["mann_metadata_mat"] = stat_only(paths["mann_metadata_mat"])
    if before["facts"]["sha256"].lower() != FACTS_SHA256:
        raise RuntimeError("facts-hash-changed-stop-and-reread")

    records = pd.read_csv(paths["ledger"])
    lmr = pd.read_csv(paths["lmr_metadata"])
    dod = pd.read_csv(paths["dod2k_metadata"])
    round73 = pd.read_csv(paths["round73_iso"])
    provenance = pd.read_csv(paths["record_provenance"])
    external = pd.read_csv(paths["external_evidence"])
    unresolved = records[(records["network"].eq("proxy9")) & (~records["operator_metadata_ready_v3"].astype(bool))]
    if len(unresolved) != 25:
        raise RuntimeError(f"unexpected-unresolved-count:{len(unresolved)}")

    recovery_rows = []
    recovered_cells = 0
    lmr_ids = set(range(186, 198)) | {292}
    for index, row in records.iterrows():
        record_id = int(row["network_record_1based"])
        if row["network"] != "proxy9" or record_id not in lmr_ids:
            continue
        candidates = lmr[
            np.isclose(pd.to_numeric(lmr["Lat (N)"], errors="coerce"), row["latitude"], atol=0.005, rtol=0)
            & np.isclose(pd.to_numeric(lmr["Lon (E)"], errors="coerce"), row["longitude"], atol=0.005, rtol=0)
            & pd.to_numeric(lmr["Oldest (C.E.)"], errors="coerce").eq(int(row["first_year"]))
            & pd.to_numeric(lmr["Youngest (C.E.)"], errors="coerce").eq(int(row["last_year"]))
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"lmr-metadata-match-not-unique:{record_id}:{len(candidates)}")
        match = candidates.iloc[0]
        values = {
            "formal_metadata_identity": match["NCDC ID"],
            "formal_metadata_source": "LMR_coordinate_and_year_range",
            "archive_type": match["Archive type"],
            "measurement": match["Proxy measurement"],
            "seasonality": match["Seasonality"],
            "climate_variable": match["climateVariable"],
            "interpretation_direction": match["Relation_to_climateVariable"],
            "site_name": match["Site name"],
            "publication_author": match["Investigators"],
            "publication_cite_key": match["Study name"],
        }
        for field, value in values.items():
            recovered_cells += int(set_if_empty(records, index, field, value))
        recovery_rows.append({"proxy9_record_1based": record_id, "route": "LMR_coordinate_and_year_range", "candidate_count": 1, "formal_identity": clean(match["NCDC ID"]), "climate_variable": clean(match["climateVariable"]), "seasonality": clean(match["Seasonality"]), "status": "metadata_recovered"})

    iso_ids = {198, 199, 200, 201}
    for index, row in records.iterrows():
        record_id = int(row["network_record_1based"])
        if row["network"] != "proxy9" or record_id not in iso_ids:
            continue
        candidates = dod[
            dod["originalDatabase"].astype(str).str.contains("Iso2k", case=False, na=False)
            & np.isclose(pd.to_numeric(dod["geo_meanLat"], errors="coerce"), row["latitude"], atol=0.005, rtol=0)
            & np.isclose(pd.to_numeric(dod["geo_meanLon"], errors="coerce"), row["longitude"], atol=0.005, rtol=0)
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"iso2k-metadata-match-not-unique:{record_id}:{len(candidates)}")
        match = candidates.iloc[0]
        values = {
            "formal_metadata_identity": match["datasetId"],
            "formal_metadata_source": "DoD2k_exact_coordinate_Iso2k",
            "archive_type": match["archiveType"],
            "measurement": match["paleoData_proxy"],
            "seasonality": match["interpretation_seasonality"],
            "climate_variable": match["interpretation_variable"],
            "documented_climate_detail": match["interpretation_variableDetail"],
            "interpretation_direction": match["interpretation_direction"],
            "site_name": match["geo_siteName"],
            "publication_cite_key": match["originalDataURL"],
        }
        for field, value in values.items():
            recovered_cells += int(set_if_empty(records, index, field, value))
        recovery_rows.append({"proxy9_record_1based": record_id, "route": "DoD2k_exact_coordinate_Iso2k", "candidate_count": 1, "formal_identity": clean(match["datasetId"]), "climate_variable": clean(match["interpretation_variable"]), "seasonality": clean(match["interpretation_seasonality"]), "status": "metadata_recovered"})

    for _, evidence in external.iterrows():
        record_id = int(evidence["proxy9_record_1based"])
        indices = records.index[(records["network"].eq("proxy9")) & records["network_record_1based"].eq(record_id)]
        if len(indices) != 1:
            raise RuntimeError(f"external-record-not-unique:{record_id}")
        index = int(indices[0])
        for field in ["formal_metadata_identity", "formal_metadata_source", "site_name", "archive_type", "measurement", "climate_variable", "seasonality", "interpretation_direction", "publication_author", "publication_cite_key", "publication_year"]:
            recovered_cells += int(set_if_empty(records, index, field, evidence[field]))
        recovery_rows.append({"proxy9_record_1based": record_id, "route": "NOAA_WDS_plus_primary_publication", "candidate_count": 1, "formal_identity": clean(evidence["formal_identity"]), "climate_variable": clean(evidence["climate_variable"]), "seasonality": clean(evidence["seasonality"]), "status": "metadata_recovered"})

    variable_directory = {name for name, _, _ in whosmat(paths["mann_metadata_mat"])}
    if "year" not in variable_directory or "paleoData_values" not in variable_directory:
        raise RuntimeError("unexpected-mann-variable-directory")
    mann = loadmat(paths["mann_metadata_mat"], squeeze_me=True, struct_as_record=False, variable_names=MANN_METADATA_VARIABLES)
    if "year" in mann or "paleoData_values" in mann:
        raise RuntimeError("forbidden-mann-timeseries-variable-loaded")
    mann_arrays = {name: np.asarray(mann[name]).reshape(-1) for name in MANN_METADATA_VARIABLES}
    mann_lat = np.asarray([float(np.asarray(value).reshape(-1)[0]) for value in mann_arrays["geo_meanLat"]])
    mann_lon = np.asarray([float(np.asarray(value).reshape(-1)[0]) for value in mann_arrays["geo_meanLon"]])
    for record_id in (203, 204, 205):
        index = int(records.index[(records["network"].eq("proxy9")) & records["network_record_1based"].eq(record_id)][0])
        row = records.loc[index]
        candidates = np.flatnonzero(np.isclose(mann_lat, row["latitude"], atol=1e-6) & np.isclose(mann_lon, row["longitude"], atol=1e-6))
        if len(candidates) != 1:
            raise RuntimeError(f"mann-metadata-match-not-unique:{record_id}:{len(candidates)}")
        candidate = int(candidates[0])
        identity = scalar_text(mann_arrays["paleoData_pages2kID"][candidate])
        values = {
            "formal_metadata_identity": identity,
            "formal_metadata_source": "Mann_2008_metadata_variables_only",
            "archive_type": scalar_text(mann_arrays["archiveType"][candidate]),
            "measurement": scalar_text(mann_arrays["paleoData_proxy"][candidate]),
        }
        for field, value in values.items():
            recovered_cells += int(set_if_empty(records, index, field, value))
        public = provenance[provenance["proxy9_record"].eq(record_id)]
        if len(public) == 1:
            recovered_cells += int(set_if_empty(records, index, "site_name", public.iloc[0]["site_name"]))
            recovered_cells += int(set_if_empty(records, index, "publication_cite_key", public.iloc[0]["citation"]))
            recovered_cells += int(set_if_empty(records, index, "publication_author", public.iloc[0]["primary_investigators"]))
        recovery_rows.append({"proxy9_record_1based": record_id, "route": "Mann_2008_metadata_variables_only", "candidate_count": 1, "formal_identity": identity, "climate_variable": "", "seasonality": "", "status": "identity_recovered_interpretation_unencoded"})

    # Propagate recovered metadata to proxy6 records in the same evidence group.
    propagation_fields = ["formal_metadata_identity", "formal_metadata_source", "archive_type", "measurement", "climate_variable", "documented_climate_detail", "seasonality", "interpretation_direction", "site_name", "publication_author", "publication_cite_key", "publication_year"]
    proxy9_by_group = records[records["network"].eq("proxy9")].set_index("evidence_group_id")
    for index in records.index[records["network"].eq("proxy6")]:
        source = proxy9_by_group.loc[records.at[index, "evidence_group_id"]]
        for field in propagation_fields:
            recovered_cells += int(set_if_empty(records, int(index), field, source[field]))

    records["canonical_seasonality_v4"] = records["seasonality"].map(canonical_season)
    records["source_identity_known_v4"] = records.apply(lambda row: any(clean(row.get(field, "")) for field in ("formal_metadata_identity", "data_name", "public_identity_id")), axis=1)
    records["climate_target_known_v4"] = records["climate_variable"].map(clean).ne("")
    records["seasonality_known_v4"] = records["canonical_seasonality_v4"].ne("")
    records["operator_metadata_ready_v4"] = records["source_identity_known_v4"] & records["chronology_bounds_known"].astype(bool) & records["climate_target_known_v4"] & records["seasonality_known_v4"] & records["evidence_identity_closed"].astype(bool)
    records["publication_known_v4"] = records[["publication_author", "publication_cite_key"]].apply(lambda row: any(clean(value) for value in row), axis=1)
    records["release_citation_ready_v4"] = records["operator_metadata_ready_v4"] & records["publication_known_v4"]
    records["documented_network_role_v4"] = np.select(
        [records["climate_variable"].astype(str).str.contains("temperature", case=False, na=False) & ~records["climate_variable"].astype(str).str.contains("moisture|precip", case=False, na=False), records["climate_variable"].astype(str).str.contains("moisture|precip", case=False, na=False)],
        ["documented_temperature", "documented_moisture"], default=records["documented_network_role"].fillna("unresolved")
    )

    groups = records.groupby("evidence_group_id", as_index=False).agg(
        network_record_count=("network", "size"), latitude=("latitude", "first"), longitude=("longitude", "first"),
        documented_network_role=("documented_network_role_v4", "first"), operator_metadata_ready=("operator_metadata_ready_v4", "all"),
        release_citation_ready=("release_citation_ready_v4", "all"), psm_significant_any=("psm_significant_p05", "any")
    )
    remaining = records[(records["network"].eq("proxy9")) & (~records["operator_metadata_ready_v4"])].copy()
    remaining["review_priority"] = np.where(remaining["psm_significant_p05"].astype(bool), 1, 2)
    remaining["remaining_boundary"] = np.select(
        [~remaining["climate_target_known_v4"] & ~remaining["seasonality_known_v4"], ~remaining["climate_target_known_v4"], ~remaining["seasonality_known_v4"]],
        ["climate_target_and_seasonality_unencoded", "climate_target_unencoded", "seasonality_unencoded"], default="other"
    )
    remaining = remaining.sort_values(["review_priority", "network_record_1based"])
    recovery = pd.DataFrame(recovery_rows).sort_values("proxy9_record_1based")

    records.to_csv(output / "proxy_evidence_master_ledger_metadata_recovered.csv", index=False)
    groups.to_csv(output / "evidence_groups_metadata_recovered.csv", index=False)
    recovery.to_csv(output / "metadata_recovery_audit.csv", index=False)
    remaining.to_csv(output / "remaining_unresolved_proxy_groups.csv", index=False)

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "time_series_values_read": False,
        "mann_loaded_variables": sorted(name for name in mann if not name.startswith("__")),
        "mann_forbidden_variables_loaded": False,
        "network_record_count": int(len(records)),
        "unique_evidence_group_count": int(len(groups)),
        "metadata_recovery_route_count": int(len(recovery)),
        "recovered_metadata_cell_count": int(recovered_cells),
        "operator_metadata_ready_record_count": int(records["operator_metadata_ready_v4"].sum()),
        "operator_metadata_ready_evidence_group_count": int(groups["operator_metadata_ready"].sum()),
        "remaining_unresolved_evidence_group_count": int(len(remaining)),
        "priority_1_remaining_group_count": int((remaining["review_priority"] == 1).sum()),
        "remaining_boundary_counts": remaining["remaining_boundary"].value_counts().to_dict(),
        "documented_role_evidence_group_counts": groups["documented_network_role"].replace("", "unresolved").value_counts().to_dict(),
        "decision": "freeze_recovered_master_ledger_and_quarantine_unencoded_interpretations",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].bar(["Round 109", "Round 110"], [272, summary["operator_metadata_ready_evidence_group_count"]], color=["#7a8796", "#16745b"])
    axes[0].set_ylabel("Unique evidence groups")
    axes[0].set_title("Operator metadata readiness")
    axes[0].grid(axis="y", alpha=0.25)
    boundary = remaining["remaining_boundary"].value_counts()
    boundary.sort_values().plot.barh(ax=axes[1], color="#bd5d3a")
    axes[1].set_xlabel("Evidence groups")
    axes[1].set_title("Remaining documented boundaries")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Unresolved proxy metadata recovery", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "unresolved_proxy_metadata_recovery.png", dpi=180)
    plt.close(fig)

    final = (
        "# Unresolved Proxy Metadata Recovery\n\n"
        f"- Operator-metadata-ready evidence groups increased from `272` to `{summary['operator_metadata_ready_evidence_group_count']}` of `297`.\n"
        f"- Remaining unresolved groups: `{len(remaining)}`; significant-PSM priority-1 groups: `{summary['priority_1_remaining_group_count']}`.\n"
        "- Mann metadata extraction loaded no `year` or `paleoData_values` variable.\n"
        f"- Remaining boundaries: `{json.dumps(summary['remaining_boundary_counts'], sort_keys=True)}`.\n\n"
        f"Decision: `{summary['decision']}`.\n"
    )
    (output / "FINAL_RESULT.md").write_text(final, encoding="utf-8")

    after = {name: state(paths[name]) for name in hashed_names}
    after["mann_metadata_mat"] = stat_only(paths["mann_metadata_mat"])
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": before == after,
        "mann_source_hash_inherited_from_prior_verified_provenance": "4ce408b14d1c7a654e0cfd1bb31b786f8d25cf56c74f50759527e1d7a1b10aba",
        "external_primary_sources": ["https://www.ncei.noaa.gov/access/paleo-search/study/22550", "https://doi.org/10.5194/cp-14-653-2018"],
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
