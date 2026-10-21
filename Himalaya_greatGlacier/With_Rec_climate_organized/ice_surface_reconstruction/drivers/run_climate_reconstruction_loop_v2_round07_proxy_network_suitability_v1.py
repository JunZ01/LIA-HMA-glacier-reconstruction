"""Build an evidence-only temperature/moisture suitability inventory for proxy9."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_CORRELATION = 0.999999999
MAX_AFFINE_RESIDUAL = 1e-10
COORDINATE_TOLERANCE = 1e-6
MOISTURE_TERMS = (
    "precipitation", "moisture", "drought", "pdsi", "hydroclimate",
    "hydro", "water", "streamflow", "runoff",
)
TEMPERATURE_TERMS = ("temperature", "air temperature", "sea surface temperature")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "sha256": value.hexdigest(),
    }


def decode(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def scalar(group: h5py.Group, name: str) -> float:
    if name not in group:
        return float("nan")
    return float(np.asarray(group[name]).reshape(-1)[0])


def affine_closure(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < 20:
        return np.nan, np.nan, np.nan, np.nan, count
    x = right[valid]
    y = left[valid]
    matrix = np.column_stack([x, np.ones(count)])
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ np.asarray([slope, intercept])
    return (
        float(np.corrcoef(x, y)[0, 1]), float(slope), float(intercept),
        float(np.max(np.abs(residual))), count,
    )


def combined_series(
    years: np.ndarray, values: np.ndarray, requested_years: np.ndarray
) -> np.ndarray:
    lookup = {int(year): float(value) for year, value in zip(years, values)}
    return np.asarray([lookup.get(int(year), np.nan) for year in requested_years], dtype=float)


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<missing>"} else text


def classify(variable: str, detail: str) -> str:
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


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    pages_tools_path = drivers / "audit_pages2k_fixed_network_metadata_family_v1.py"
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_07_proxy_network_suitability_inventory_v1"
    )
    pages_root = Path(
        r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\PAGES2k_v2\proxy-data\pages2k_proxy"
    )
    paths = {
        "proxy9_archive": Path(
            r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all\python_data\proxy_data_9.mat"
        ),
        "combined": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"),
        "pages_reconstruction": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat"),
        "pages_annual": pages_root / "proxy_data_1_13_1.mat",
        "pages_metadata": pages_root / "metadata_1_13_1.txt",
        "pages_ts": pages_root / "Pages2kTSv1_13_1.mat",
        "lmr_metadata": Path(
            r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0\lmr_meta.csv"
        ),
        "round06_skill": project / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v2/round_06_incremental_precipitation_skill_v2/"
            "record_incremental_skill.csv"
        ),
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "pages_tools": pages_tools_path,
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    pages_tools = load_module(pages_tools_path, "round07_pages_tools")

    combined = loadmat(paths["combined"], squeeze_me=True, struct_as_record=False)
    combined_years = np.asarray(combined["year"], dtype=int).reshape(-1)
    combined_matrix = np.asarray(combined["combined6_data"], dtype=float)
    combined_names = np.char.strip(np.asarray(combined["dataName"]).astype(str)).reshape(-1)
    combined_lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    combined_lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    combined_flags = np.asarray(combined["source_flag"], dtype=int).reshape(-1)

    identity_rows: list[dict[str, object]] = []
    with h5py.File(paths["proxy9_archive"], "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        if len(references) != 297:
            raise RuntimeError("proxy9-record-count-failed")
        for position, reference in enumerate(references, start=1):
            proxy = handle[reference]
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            proxy_years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            proxy_values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            candidates = np.flatnonzero(
                np.isclose(combined_lat, location[0], atol=COORDINATE_TOLERANCE)
                & np.isclose(combined_lon, location[1], atol=COORDINATE_TOLERANCE)
            )
            matches = []
            for column in candidates:
                target = combined_series(combined_years, combined_matrix[:, column], proxy_years)
                correlation, slope, intercept, residual, count = affine_closure(proxy_values, target)
                if np.isfinite(correlation) and correlation >= MIN_CORRELATION and residual <= MAX_AFFINE_RESIDUAL:
                    matches.append((column, correlation, slope, intercept, residual, count))
            if len(matches) != 1:
                raise RuntimeError(
                    f"proxy9-combined-identity-not-unique:{position}:candidates={len(candidates)}:passing={len(matches)}"
                )
            column, correlation, slope, intercept, residual, count = matches[0]
            identity_rows.append({
                "record_index_1based": position,
                "combined_column_1based": int(column + 1),
                "combined_name": combined_names[column],
                "source_flag": int(combined_flags[column]),
                "latitude": float(location[0]),
                "longitude": float(location[1]),
                "proxy_type": decode(proxy["ptype"]),
                "psm_method": decode(proxy["psm_mthd"]),
                "psm_b0": float(coefficients[0]) if len(coefficients) > 0 else np.nan,
                "psm_b1_temperature": float(coefficients[1]) if len(coefficients) > 1 else np.nan,
                "psm_b2_precipitation": float(coefficients[2]) if len(coefficients) > 2 else np.nan,
                "psm_correlation": scalar(proxy, "pcr"),
                "psm_temperature_correlation": scalar(proxy, "pcr_t"),
                "psm_precipitation_correlation": scalar(proxy, "pcr_p"),
                "psm_p_value": scalar(proxy, "pval"),
                "identity_coordinate_candidate_count": int(len(candidates)),
                "identity_correlation": correlation,
                "identity_affine_slope": slope,
                "identity_affine_intercept": intercept,
                "identity_maximum_abs_residual": residual,
                "identity_finite_overlap": count,
            })
    identity = pd.DataFrame(identity_rows)

    pages = loadmat(paths["pages_reconstruction"], squeeze_me=True, struct_as_record=False)
    pages_years = np.asarray(pages["year"], dtype=int).reshape(-1)
    pages_matrix = np.asarray(pages["pages2k_data"], dtype=float)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    pages_names = np.asarray(pages["pages2k_dataName"], dtype=object).reshape(-1)
    official = loadmat(paths["pages_annual"], squeeze_me=True, struct_as_record=False)
    official_years = np.asarray(official["year"], dtype=int).reshape(-1)
    official_matrix = np.asarray(official["proxy_ann"], dtype=float)
    official_metadata = pages_tools.metadata_table(paths["pages_metadata"])
    official_lat = official_metadata["geo_latitude"].to_numpy(float)
    official_lon = official_metadata["geo_longitude"].to_numpy(float)
    ts_values = np.asarray(loadmat(
        paths["pages_ts"], squeeze_me=True, struct_as_record=False, variable_names=["TS"]
    )["TS"]).reshape(-1)
    ts_index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        key = (
            pages_tools.clean_group(getattr(item, "dataSetName", "")),
            pages_tools.clean_group(getattr(item, "paleoData_TSid", "")),
        )
        ts_index[key].append(item)

    lmr = pd.read_csv(paths["lmr_metadata"])
    lmr["NCDC ID"] = lmr["NCDC ID"].astype(str).str.strip()
    lmr_index = lmr.set_index("NCDC ID", drop=False)
    metadata_rows: list[dict[str, object]] = []
    for row in identity.itertuples(index=False):
        combined_column = int(row.combined_column_1based) - 1
        page_candidates = np.flatnonzero(
            np.isclose(pages_lat, row.latitude, atol=COORDINATE_TOLERANCE)
            & np.isclose(pages_lon, row.longitude, atol=COORDINATE_TOLERANCE)
        )
        page_passed = pages_tools.passing_matches(
            combined_years, combined_matrix[:, combined_column],
            pages_years, pages_matrix, page_candidates,
        )
        metadata_source = ""
        metadata_identity = ""
        archive_type = ""
        measurement = ""
        climate_variable = ""
        climate_detail = ""
        seasonality = ""
        direction = ""
        pages_column = np.nan
        official_column = np.nan
        if len(page_passed) == 1:
            page_column = int(page_passed[0]["column_0based"])
            official_candidates = np.flatnonzero(
                np.isclose(official_lat, pages_lat[page_column], atol=COORDINATE_TOLERANCE)
                & np.isclose(official_lon, pages_lon[page_column], atol=COORDINATE_TOLERANCE)
            )
            official_passed = pages_tools.passing_matches(
                pages_years, pages_matrix[:, page_column],
                official_years, official_matrix, official_candidates,
            )
            if len(official_passed) == 1:
                official_index = int(official_passed[0]["column_0based"])
                official_row = official_metadata.iloc[official_index]
                key = (
                    pages_tools.clean_group(official_row["dataSetName"]),
                    pages_tools.clean_group(official_row["paleoData_TSid"]),
                )
                matches = ts_index.get(key, [])
                if len(matches) == 1:
                    item = matches[0]
                    metadata_source = "PAGES2k_unique_sequence"
                    metadata_identity = f"{key[0]}::{key[1]}"
                    archive_type = pages_tools.clean_group(getattr(item, "archiveType", ""))
                    measurement = pages_tools.clean_group(getattr(item, "paleoData_proxy", ""))
                    climate_variable = pages_tools.clean_group(getattr(item, "climateInterpretation_variable", ""))
                    climate_detail = pages_tools.clean_group(getattr(item, "climateInterpretation_variableDetail", ""))
                    seasonality = pages_tools.clean_group(getattr(item, "climateInterpretation_seasonality", ""))
                    direction = pages_tools.clean_group(getattr(item, "climateInterpretation_interpDirection", ""))
                    pages_column = page_column + 1
                    official_column = official_index + 1
        if not metadata_source and row.combined_name in lmr_index.index:
            meta = lmr_index.loc[row.combined_name]
            if isinstance(meta, pd.DataFrame):
                raise RuntimeError(f"ambiguous-lmr-id:{row.combined_name}")
            metadata_source = "LMR_exact_NCDC_ID"
            metadata_identity = row.combined_name
            archive_type = clean(meta["Archive type"])
            measurement = clean(meta["Proxy measurement"])
            climate_variable = clean(meta["climateVariable"])
            climate_detail = ""
            seasonality = clean(meta["Seasonality"])
            direction = clean(meta["Relation_to_climateVariable"])
        role = classify(climate_variable, climate_detail)
        metadata_rows.append({
            "record_index_1based": int(row.record_index_1based),
            "metadata_source": metadata_source or "unresolved",
            "metadata_identity": metadata_identity,
            "pages_reconstruction_column_1based": pages_column,
            "official_pages_column_1based": official_column,
            "archive_type": archive_type,
            "measurement": measurement,
            "documented_climate_variable": climate_variable,
            "documented_climate_detail": climate_detail,
            "documented_seasonality": seasonality,
            "documented_direction": direction,
            "network_role": role,
        })
    metadata = pd.DataFrame(metadata_rows)
    inventory = identity.merge(metadata, on="record_index_1based", validate="one_to_one")
    inventory["site_id"] = inventory["latitude"].round(6).astype(str) + ":" + inventory["longitude"].round(6).astype(str)

    skill = pd.read_csv(paths["round06_skill"])
    skill_columns = [column for column in skill.columns if column.endswith("_median_improvement")]
    inventory = inventory.merge(
        skill[["record_index_1based", "positive_all_four_paths", *skill_columns]],
        on="record_index_1based", how="left", validate="one_to_one",
    )
    inventory.to_csv(output / "proxy9_suitability_inventory.csv", index=False)

    role_rows: list[dict[str, object]] = []
    for role, group in inventory.groupby("network_role", dropna=False):
        weakest = group[skill_columns].min(axis=1, skipna=True)
        role_rows.append({
            "network_role": role,
            "record_count": int(len(group)),
            "unique_site_count": int(group["site_id"].nunique()),
            "metadata_resolved_count": int((group["metadata_source"] != "unresolved").sum()),
            "round06_positive_all_four_count": int(group["positive_all_four_paths"].fillna(False).astype(bool).sum()),
            "round06_positive_all_four_fraction": float(group["positive_all_four_paths"].fillna(False).astype(bool).mean()),
            "median_weakest_round06_improvement": float(np.nanmedian(weakest)),
        })
    role_summary = pd.DataFrame(role_rows).sort_values("record_count", ascending=False)
    role_summary.to_csv(output / "network_role_summary.csv", index=False)

    source_summary = inventory.groupby("metadata_source", as_index=False).agg(
        record_count=("record_index_1based", "size"), unique_site_count=("site_id", "nunique")
    )
    source_summary.to_csv(output / "metadata_source_summary.csv", index=False)
    moisture = inventory["network_role"].isin(["documented_moisture", "documented_mixed"])
    moisture_records = int(moisture.sum())
    moisture_sites = int(inventory.loc[moisture, "site_id"].nunique())
    eligible = bool(moisture_records >= 20 and moisture_sites >= 10)
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "proxy9_records": int(len(inventory)),
        "identity_minimum_correlation": float(inventory["identity_correlation"].min()),
        "identity_maximum_affine_residual": float(inventory["identity_maximum_abs_residual"].max()),
        "metadata_resolved_records": int((inventory["metadata_source"] != "unresolved").sum()),
        "metadata_unresolved_records": int((inventory["metadata_source"] == "unresolved").sum()),
        "network_role_counts": inventory["network_role"].value_counts().to_dict(),
        "documented_moisture_plus_mixed_records": moisture_records,
        "documented_moisture_plus_mixed_unique_sites": moisture_sites,
        "eligible_for_nested_moisture_validation": eligible,
        "decision": "advance_documented_moisture_network_to_nested_validation" if eligible else "close_proxy9_as_standalone_precipitation_network",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    role_plot = role_summary.sort_values("record_count")
    axes[0].barh(role_plot["network_role"], role_plot["record_count"], color="#005f73")
    axes[0].set(title="Documented proxy-network roles", xlabel="Proxy records")
    axes[1].barh(
        role_plot["network_role"], role_plot["round06_positive_all_four_fraction"] * 100,
        color="#ee9b00",
    )
    axes[1].axvline(50, color="#9b2226", linestyle="--", linewidth=1.4)
    axes[1].set(title="Round 06 four-path improvement", xlabel="Records positive on all paths (%)", xlim=(0, 100))
    fig.suptitle("Proxy9 suitability inventory", fontsize=14)
    fig.savefig(output / "proxy9_suitability_inventory.png", dpi=180)
    plt.close(fig)

    report = (
        "# Proxy9 Suitability Inventory\n\n"
        f"- All `{len(inventory)}` records close uniquely to combined_proxydata6; minimum correlation "
        f"`{summary['identity_minimum_correlation']:.12f}`.\n"
        f"- Official metadata resolved: `{summary['metadata_resolved_records']}`; unresolved: "
        f"`{summary['metadata_unresolved_records']}`.\n"
        f"- Role counts: `{json.dumps(summary['network_role_counts'], sort_keys=True)}`.\n"
        f"- Documented moisture plus mixed network: `{moisture_records}` records on `{moisture_sites}` sites.\n"
        f"- Eligible for nested moisture validation: `{str(eligible).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n\n"
        "Round 06 skill was merged only after evidence-based classification and did not select records. "
        "No climate source, EnKF, OGGM model, or 1921-1939 value was accessed.\n"
    )
    (output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
