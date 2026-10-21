"""Reconcile official PAGES2k/LMR metadata and classify proxy9 network roles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOISTURE_TERMS = ("precipitation", "moisture", "drought", "pdsi", "hydroclimate", "hydro", "water", "streamflow", "runoff")
TEMPERATURE_TERMS = ("temperature", "air temperature", "sea surface temperature")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("round07_pages_tools_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("pages-tools-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<missing>"} else text


def role(variable: str, detail: str) -> str:
    text = f"{variable} {detail}".lower()
    moisture = any(term in text for term in MOISTURE_TERMS)
    temperature = any(term in text for term in TEMPERATURE_TERMS)
    if moisture and temperature:
        return "documented_mixed"
    if moisture:
        return "documented_moisture"
    if temperature:
        return "documented_temperature"
    if clean(variable) or clean(detail):
        return "documented_other"
    return "unresolved"


def joined(entries: list[dict[str, str]], key: str) -> str:
    values = sorted({clean(entry.get(key, "")) for entry in entries if clean(entry.get(key, ""))})
    return " || ".join(values)


def reconciled_role(entries: list[dict[str, str]]) -> str:
    roles = {entry["role"] for entry in entries if entry["role"] not in {"unresolved", "documented_other"}}
    if len(roles) > 1:
        return "metadata_conflict"
    if len(roles) == 1:
        return next(iter(roles))
    other = {entry["role"] for entry in entries}
    return "documented_other" if "documented_other" in other else "unresolved"


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_07_proxy_network_suitability_inventory_v2"
    pages_root = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\PAGES2k_v2\proxy-data\pages2k_proxy")
    paths = {
        "identity": base / "round_07_proxy_network_identity_alias_audit_v3/proxy9_identity_alias_inventory.csv",
        "combined": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"),
        "pages_reconstruction": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat"),
        "pages_annual": pages_root / "proxy_data_1_13_1.mat",
        "pages_metadata": pages_root / "metadata_1_13_1.txt",
        "pages_ts": pages_root / "Pages2kTSv1_13_1.mat",
        "lmr_metadata": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0\lmr_meta.csv"),
        "round06_skill": base / "round_06_incremental_precipitation_skill_v2/record_incremental_skill.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "pages_tools": drivers / "audit_pages2k_fixed_network_metadata_family_v1.py",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    tools = load_module(paths["pages_tools"])
    identity = pd.read_csv(paths["identity"])
    if len(identity) != 297 or identity["identity_key"].nunique() != 296:
        raise RuntimeError("identity-contract-failed")

    combined = loadmat(paths["combined"], squeeze_me=True, struct_as_record=False)
    combined_years = np.asarray(combined["year"], dtype=int).reshape(-1)
    combined_matrix = np.asarray(combined["combined6_data"], dtype=float)
    combined_lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    combined_lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    combined_names = np.char.strip(np.asarray(combined["dataName"]).astype(str)).reshape(-1)
    pages = loadmat(paths["pages_reconstruction"], squeeze_me=True, struct_as_record=False)
    pages_years = np.asarray(pages["year"], dtype=int).reshape(-1)
    pages_matrix = np.asarray(pages["pages2k_data"], dtype=float)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    official = loadmat(paths["pages_annual"], squeeze_me=True, struct_as_record=False)
    official_years = np.asarray(official["year"], dtype=int).reshape(-1)
    official_matrix = np.asarray(official["proxy_ann"], dtype=float)
    official_meta = tools.metadata_table(paths["pages_metadata"])
    official_lat = official_meta["geo_latitude"].to_numpy(float)
    official_lon = official_meta["geo_longitude"].to_numpy(float)
    ts_values = np.asarray(loadmat(paths["pages_ts"], squeeze_me=True, struct_as_record=False, variable_names=["TS"])["TS"]).reshape(-1)
    ts_index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        key = (tools.clean_group(getattr(item, "dataSetName", "")), tools.clean_group(getattr(item, "paleoData_TSid", "")))
        ts_index[key].append(item)
    lmr = pd.read_csv(paths["lmr_metadata"])
    lmr["NCDC ID"] = lmr["NCDC ID"].astype(str).str.strip()
    lmr_index = lmr.set_index("NCDC ID", drop=False)

    metadata_rows = []
    for record in identity.itertuples(index=False):
        alias_columns = [int(value) - 1 for value in str(record.alias_combined_columns_1based).split(";")]
        entries: list[dict[str, str]] = []
        seen = set()
        for combined_column in alias_columns:
            page_candidates = np.flatnonzero(
                np.isclose(pages_lat, record.latitude, atol=1e-6)
                & np.isclose(pages_lon, record.longitude, atol=1e-6)
            )
            page_passed = tools.passing_matches(
                combined_years, combined_matrix[:, combined_column], pages_years, pages_matrix, page_candidates
            )
            if len(page_passed) == 1:
                page_column = int(page_passed[0]["column_0based"])
                official_candidates = np.flatnonzero(
                    np.isclose(official_lat, pages_lat[page_column], atol=1e-6)
                    & np.isclose(official_lon, pages_lon[page_column], atol=1e-6)
                )
                official_passed = tools.passing_matches(
                    pages_years, pages_matrix[:, page_column], official_years, official_matrix, official_candidates
                )
                if len(official_passed) == 1:
                    official_column = int(official_passed[0]["column_0based"])
                    meta = official_meta.iloc[official_column]
                    key = (tools.clean_group(meta["dataSetName"]), tools.clean_group(meta["paleoData_TSid"]))
                    matches = ts_index.get(key, [])
                    if len(matches) == 1 and ("PAGES2k", key) not in seen:
                        item = matches[0]
                        variable = tools.clean_group(getattr(item, "climateInterpretation_variable", ""))
                        detail = tools.clean_group(getattr(item, "climateInterpretation_variableDetail", ""))
                        entries.append({
                            "source": "PAGES2k_unique_sequence", "identity": f"{key[0]}::{key[1]}",
                            "archive": tools.clean_group(getattr(item, "archiveType", "")),
                            "measurement": tools.clean_group(getattr(item, "paleoData_proxy", "")),
                            "variable": variable, "detail": detail,
                            "seasonality": tools.clean_group(getattr(item, "climateInterpretation_seasonality", "")),
                            "direction": tools.clean_group(getattr(item, "climateInterpretation_interpDirection", "")),
                            "role": role(variable, detail),
                        })
                        seen.add(("PAGES2k", key))
            alias_name = combined_names[combined_column]
            if alias_name in lmr_index.index and ("LMR", alias_name) not in seen:
                meta = lmr_index.loc[alias_name]
                if isinstance(meta, pd.DataFrame):
                    raise RuntimeError(f"ambiguous-lmr-id:{alias_name}")
                variable = clean(meta["climateVariable"])
                entries.append({
                    "source": "LMR_exact_NCDC_ID", "identity": alias_name,
                    "archive": clean(meta["Archive type"]), "measurement": clean(meta["Proxy measurement"]),
                    "variable": variable, "detail": "", "seasonality": clean(meta["Seasonality"]),
                    "direction": clean(meta["Relation_to_climateVariable"]), "role": role(variable, ""),
                })
                seen.add(("LMR", alias_name))
        final_role = reconciled_role(entries)
        metadata_rows.append({
            "record_index_1based": int(record.record_index_1based),
            "formal_metadata_entry_count": int(len(entries)),
            "metadata_sources": joined(entries, "source") or "unresolved",
            "metadata_identities": joined(entries, "identity"),
            "archive_type": joined(entries, "archive"), "measurement": joined(entries, "measurement"),
            "documented_climate_variable": joined(entries, "variable"),
            "documented_climate_detail": joined(entries, "detail"),
            "documented_seasonality": joined(entries, "seasonality"),
            "documented_direction": joined(entries, "direction"),
            "source_roles": joined(entries, "role"), "network_role": final_role,
        })
    metadata = pd.DataFrame(metadata_rows)
    inventory = identity.merge(metadata, on="record_index_1based", validate="one_to_one")
    skill = pd.read_csv(paths["round06_skill"])
    skill_columns = [column for column in skill.columns if column.endswith("_median_improvement")]
    inventory = inventory.merge(skill[["record_index_1based", "positive_all_four_paths", *skill_columns]], on="record_index_1based", how="left", validate="one_to_one")
    inventory.to_csv(output / "proxy9_suitability_inventory.csv", index=False)

    unique = inventory.sort_values("record_index_1based").drop_duplicates("identity_key")
    role_rows = []
    for name, group in unique.groupby("network_role"):
        weakest = group[skill_columns].min(axis=1, skipna=True)
        role_rows.append({
            "network_role": name, "independent_identity_count": int(len(group)),
            "raw_record_count": int(inventory[inventory["network_role"] == name].shape[0]),
            "unique_site_count": int(group.groupby(["latitude", "longitude"]).ngroups),
            "round06_positive_all_four_count": int(group["positive_all_four_paths"].fillna(False).astype(bool).sum()),
            "round06_positive_all_four_fraction": float(group["positive_all_four_paths"].fillna(False).astype(bool).mean()),
            "median_weakest_round06_improvement": float(np.nanmedian(weakest)),
        })
    role_summary = pd.DataFrame(role_rows).sort_values("independent_identity_count", ascending=False)
    role_summary.to_csv(output / "network_role_summary.csv", index=False)
    inventory.groupby("metadata_sources", as_index=False).agg(record_count=("record_index_1based", "size")).to_csv(output / "metadata_source_summary.csv", index=False)

    moisture = unique["network_role"].isin(["documented_moisture", "documented_mixed"])
    moisture_count = int(moisture.sum())
    moisture_sites = int(unique.loc[moisture].groupby(["latitude", "longitude"]).ngroups)
    eligible = bool(moisture_count >= 20 and moisture_sites >= 10)
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False,
        "raw_proxy9_records": int(len(inventory)), "independent_identity_count": int(len(unique)),
        "formal_metadata_resolved_identities": int((unique["metadata_sources"] != "unresolved").sum()),
        "formal_metadata_unresolved_identities": int((unique["metadata_sources"] == "unresolved").sum()),
        "network_role_counts_independent": unique["network_role"].value_counts().to_dict(),
        "metadata_conflict_identities": int((unique["network_role"] == "metadata_conflict").sum()),
        "documented_moisture_plus_mixed_identities": moisture_count,
        "documented_moisture_plus_mixed_unique_sites": moisture_sites,
        "eligible_for_nested_moisture_validation": eligible,
        "decision": "advance_documented_moisture_network_to_nested_validation" if eligible else "close_proxy9_as_standalone_precipitation_network",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    plotted = role_summary.sort_values("independent_identity_count")
    axes[0].barh(plotted["network_role"], plotted["independent_identity_count"], color="#005f73")
    axes[0].set(title="Formal proxy-network roles", xlabel="Independent identities")
    axes[1].barh(plotted["network_role"], plotted["round06_positive_all_four_fraction"] * 100, color="#ee9b00")
    axes[1].axvline(50, color="#9b2226", linestyle="--", linewidth=1.4)
    axes[1].set(title="Round 06 four-path improvement", xlabel="Identities positive on all paths (%)", xlim=(0, 100))
    fig.suptitle("Proxy9 suitability inventory", fontsize=14)
    fig.savefig(output / "proxy9_suitability_inventory.png", dpi=180)
    plt.close(fig)
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Suitability Inventory v2\n\n"
        f"- Independent identities: `{len(unique)}`; formal metadata resolved: `{summary['formal_metadata_resolved_identities']}`.\n"
        f"- Role counts: `{json.dumps(summary['network_role_counts_independent'], sort_keys=True)}`.\n"
        f"- Documented moisture/mixed: `{moisture_count}` identities on `{moisture_sites}` sites.\n"
        f"- Eligible for nested moisture validation: `{str(eligible).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
