"""Audit omitted records in the complete HMA proxy-source envelope."""

from __future__ import annotations

import csv
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
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
    return "documented_other" if clean(variable) or clean(detail) else "unresolved"


def reconcile(entries: list[dict[str, str]]) -> str:
    roles = {entry["role"] for entry in entries if entry["role"] not in {"unresolved", "documented_other"}}
    if len(roles) > 1:
        return "metadata_conflict"
    if len(roles) == 1:
        return next(iter(roles))
    return "documented_other" if any(entry["role"] == "documented_other" for entry in entries) else "unresolved"


def join(entries: list[dict[str, str]], key: str) -> str:
    return " || ".join(sorted({clean(entry.get(key, "")) for entry in entries if clean(entry.get(key, ""))}))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_10_complete_hma_proxy_pool_audit_v1"
    pages_root = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\PAGES2k_v2\proxy-data\pages2k_proxy")
    paths = {
        "identity": base / "round_07_proxy_network_identity_alias_audit_v3/proxy9_identity_alias_inventory.csv",
        "inventory": base / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "combined": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"),
        "pages_reconstruction": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat"),
        "pages_annual": pages_root / "proxy_data_1_13_1.mat",
        "pages_metadata": pages_root / "metadata_1_13_1.txt",
        "pages_ts": pages_root / "Pages2kTSv1_13_1.mat",
        "lmr_metadata": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0\lmr_meta.csv"),
        "pages_tools": drivers / "audit_pages2k_fixed_network_metadata_family_v1.py",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    tools = load_module(paths["pages_tools"], "round10_pages_tools")

    identity = pd.read_csv(paths["identity"])
    inventory = pd.read_csv(paths["inventory"])
    represented: set[int] = set()
    for value in identity["alias_combined_columns_1based"].astype(str):
        represented.update(int(item) - 1 for item in value.split(";") if item)

    combined = loadmat(paths["combined"], squeeze_me=True, struct_as_record=False)
    combined_years = np.asarray(combined["year"], dtype=int).reshape(-1)
    combined_matrix = np.asarray(combined["combined6_data"], dtype=float)
    combined_lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    combined_lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    combined_names = np.char.strip(np.asarray(combined["dataName"]).astype(str)).reshape(-1)
    if combined_matrix.shape[1] != len(combined_lat):
        raise RuntimeError("combined-array-contract-failed")

    lat_min, lat_max = float(inventory["latitude"].min()), float(inventory["latitude"].max())
    lon_min, lon_max = float(inventory["longitude"].min()), float(inventory["longitude"].max())
    in_envelope = (combined_lat >= lat_min) & (combined_lat <= lat_max) & (combined_lon >= lon_min) & (combined_lon <= lon_max)
    omitted = [int(column) for column in np.flatnonzero(in_envelope) if int(column) not in represented]

    unlocked_combined = ~((combined_years >= 1921) & (combined_years <= 1939))
    calibration = (combined_years >= 1940) & (combined_years <= 2000)
    paleoclimate = (combined_years >= 1500) & (combined_years <= 1920)
    if np.any(calibration & ~unlocked_combined) or np.any(paleoclimate & ~unlocked_combined):
        raise RuntimeError("locked-period-mask-failed")

    pages = loadmat(paths["pages_reconstruction"], squeeze_me=True, struct_as_record=False)
    pages_years = np.asarray(pages["year"], dtype=int).reshape(-1)
    pages_matrix = np.asarray(pages["pages2k_data"], dtype=float)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    pages_unlocked = ~((pages_years >= 1921) & (pages_years <= 1939))

    official = loadmat(paths["pages_annual"], squeeze_me=True, struct_as_record=False)
    official_years = np.asarray(official["year"], dtype=int).reshape(-1)
    official_matrix = np.asarray(official["proxy_ann"], dtype=float)
    official_unlocked = ~((official_years >= 1921) & (official_years <= 1939))
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

    existing_ids: set[str] = set()
    for value in inventory["metadata_identities"].dropna().astype(str):
        existing_ids.update(item.strip() for item in value.split("||") if item.strip())

    rows: list[dict[str, object]] = []
    seen_new_ids: set[str] = set()
    for column in omitted:
        entries: list[dict[str, str]] = []
        page_candidates = np.flatnonzero(
            np.isclose(pages_lat, combined_lat[column], atol=1e-6)
            & np.isclose(pages_lon, combined_lon[column], atol=1e-6)
        )
        page_passed = tools.passing_matches(
            combined_years[unlocked_combined], combined_matrix[unlocked_combined, column],
            pages_years[pages_unlocked], pages_matrix[pages_unlocked], page_candidates,
        )
        page_match_count = len(page_passed)
        if page_match_count == 1:
            page_column = int(page_passed[0]["column_0based"])
            official_candidates = np.flatnonzero(
                np.isclose(official_lat, pages_lat[page_column], atol=1e-6)
                & np.isclose(official_lon, pages_lon[page_column], atol=1e-6)
            )
            official_passed = tools.passing_matches(
                pages_years[pages_unlocked], pages_matrix[pages_unlocked, page_column],
                official_years[official_unlocked], official_matrix[official_unlocked], official_candidates,
            )
            if len(official_passed) == 1:
                meta = official_meta.iloc[int(official_passed[0]["column_0based"])]
                key = (tools.clean_group(meta["dataSetName"]), tools.clean_group(meta["paleoData_TSid"]))
                matches = ts_index.get(key, [])
                if len(matches) == 1:
                    item = matches[0]
                    variable = tools.clean_group(getattr(item, "climateInterpretation_variable", ""))
                    detail = tools.clean_group(getattr(item, "climateInterpretation_variableDetail", ""))
                    entries.append({
                        "source": "PAGES2k_unique_unlocked_sequence",
                        "identity": f"{key[0]}::{key[1]}",
                        "archive": tools.clean_group(getattr(item, "archiveType", "")),
                        "measurement": tools.clean_group(getattr(item, "paleoData_proxy", "")),
                        "variable": variable,
                        "detail": detail,
                        "seasonality": tools.clean_group(getattr(item, "climateInterpretation_seasonality", "")),
                        "direction": tools.clean_group(getattr(item, "climateInterpretation_interpDirection", "")),
                        "role": role(variable, detail),
                    })
        alias_name = combined_names[column]
        if alias_name in lmr_index.index:
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
        metadata_ids = sorted({entry["identity"] for entry in entries})
        already_present = any(identity_id in existing_ids for identity_id in metadata_ids)
        new_key = " || ".join(metadata_ids)
        duplicate_new = bool(new_key and new_key in seen_new_ids)
        if new_key:
            seen_new_ids.add(new_key)
        final_role = reconcile(entries)
        calibration_count = int(np.isfinite(combined_matrix[calibration, column]).sum())
        paleoclimate_count = int(np.isfinite(combined_matrix[paleoclimate, column]).sum())
        eligible = bool(entries and not already_present and not duplicate_new and calibration_count >= 20 and final_role != "metadata_conflict")
        rows.append({
            "combined_column_1based": column + 1,
            "data_name": alias_name,
            "latitude": float(combined_lat[column]),
            "longitude": float(combined_lon[column]),
            "calibration_1940_2000_finite_count": calibration_count,
            "paleoclimate_1500_1920_finite_count": paleoclimate_count,
            "pages_coordinate_candidates": int(len(page_candidates)),
            "pages_unlocked_sequence_matches": page_match_count,
            "metadata_entry_count": int(len(entries)),
            "metadata_sources": join(entries, "source") or "unresolved",
            "metadata_identities": new_key,
            "archive_type": join(entries, "archive"),
            "measurement": join(entries, "measurement"),
            "documented_climate_variable": join(entries, "variable"),
            "documented_climate_detail": join(entries, "detail"),
            "documented_seasonality": join(entries, "seasonality"),
            "documented_direction": join(entries, "direction"),
            "network_role": final_role,
            "metadata_identity_already_present": already_present,
            "duplicate_among_omitted": duplicate_new,
            "eligible_new_independent_identity": eligible,
        })
    if not rows:
        raise RuntimeError("no-omitted-envelope-records")
    write_csv(output / "omitted_hma_envelope_proxy_inventory.csv", rows)
    frame = pd.DataFrame(rows)

    eligible = frame[frame["eligible_new_independent_identity"]]
    additional_moisture = eligible[eligible["network_role"].isin(["documented_moisture", "documented_mixed"])]
    additional_temperature = eligible[eligible["network_role"] == "documented_temperature"]
    current_unique = inventory.sort_values("record_index_1based").drop_duplicates("identity_key")
    current_moisture = current_unique[current_unique["network_role"].isin(["documented_moisture", "documented_mixed"])]
    revised_moisture_count = int(len(current_moisture) + len(additional_moisture))
    sites = {(round(float(row.latitude), 6), round(float(row.longitude), 6)) for row in current_moisture.itertuples(index=False)}
    sites.update((round(float(row.latitude), 6), round(float(row.longitude), 6)) for row in additional_moisture.itertuples(index=False))
    revised_moisture_sites = len(sites)
    moisture_eligible = bool(revised_moisture_count >= 20 and revised_moisture_sites >= 10)
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "combined_source_columns": int(combined_matrix.shape[1]),
        "proxy9_envelope": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "combined_columns_in_envelope": int(in_envelope.sum()),
        "represented_combined_columns": int(len(represented)),
        "omitted_envelope_columns": int(len(frame)),
        "omitted_formal_metadata_resolved": int((frame["metadata_sources"] != "unresolved").sum()),
        "omitted_calibration_capable": int((frame["calibration_1940_2000_finite_count"] >= 20).sum()),
        "eligible_new_independent_identities": int(len(eligible)),
        "eligible_new_temperature_identities": int(len(additional_temperature)),
        "eligible_new_moisture_or_mixed_identities": int(len(additional_moisture)),
        "revised_documented_moisture_identity_count": revised_moisture_count,
        "revised_documented_moisture_unique_site_count": revised_moisture_sites,
        "expanded_moisture_network_eligible": moisture_eligible,
        "decision": "advance_expanded_moisture_network_to_nested_validation" if moisture_eligible else "complete_source_pool_does_not_rescue_current_moisture_network",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    role_counts = frame["network_role"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].bar(role_counts.index, role_counts.values, color="#005f73")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(title="Omitted source records", ylabel="Columns")
    axes[1].scatter(frame["calibration_1940_2000_finite_count"], frame["paleoclimate_1500_1920_finite_count"], color="#ee9b00", s=45)
    axes[1].axvline(20, color="#9b2226", linestyle="--")
    axes[1].set(title="Coverage of omitted records", xlabel="Finite years, 1940-2000", ylabel="Finite years, 1500-1920")
    fig.suptitle("Complete HMA proxy-pool audit", fontsize=14)
    fig.savefig(output / "complete_hma_proxy_pool_audit.png", dpi=180)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Complete HMA Proxy-Pool Audit\n\n"
        f"- Complete source columns: `{combined_matrix.shape[1]}`; inside proxy9 envelope: `{int(in_envelope.sum())}`.\n"
        f"- Omitted envelope columns: `{len(frame)}`; formal metadata resolved: `{summary['omitted_formal_metadata_resolved']}`; calibration-capable: `{summary['omitted_calibration_capable']}`.\n"
        f"- Eligible new independent identities: `{len(eligible)}`; temperature: `{len(additional_temperature)}`; moisture/mixed: `{len(additional_moisture)}`.\n"
        f"- Revised moisture network: `{revised_moisture_count}` identities on `{revised_moisture_sites}` sites; eligible: `{str(moisture_eligible).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
