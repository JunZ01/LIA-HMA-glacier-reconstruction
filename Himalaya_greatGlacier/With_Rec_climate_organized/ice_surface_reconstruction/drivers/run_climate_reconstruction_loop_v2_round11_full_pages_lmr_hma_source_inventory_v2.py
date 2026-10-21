"""Inventory new calibration-capable HMA records in full PAGES2k and LMR."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
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
    spec = importlib.util.spec_from_file_location("round11_pages_tools", path)
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


def climate_role(variable: str, detail: str) -> str:
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def affine_match(years_a: np.ndarray, values_a: np.ndarray, years_b: np.ndarray, values_b: np.ndarray) -> tuple[bool, int, float, float]:
    years = np.intersect1d(years_a, years_b)
    years = years[~((years >= 1921) & (years <= 1939))]
    lookup_a = {int(year): value for year, value in zip(years_a, values_a)}
    lookup_b = {int(year): value for year, value in zip(years_b, values_b)}
    a = np.asarray([lookup_a[int(year)] for year in years], dtype=float)
    b = np.asarray([lookup_b[int(year)] for year in years], dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    count = int(valid.sum())
    if count < 20:
        return False, count, np.nan, np.nan
    matrix = np.column_stack([b[valid], np.ones(count)])
    coefficient = np.linalg.lstsq(matrix, a[valid], rcond=None)[0]
    residual = float(np.max(np.abs(a[valid] - matrix @ coefficient)))
    correlation = float(np.corrcoef(a[valid], b[valid])[0, 1])
    return bool(correlation >= 0.999999 and residual <= 1e-8), count, correlation, residual


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_11_full_pages_lmr_hma_source_inventory_v2"
    proxy_root = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data")
    pages_root = proxy_root / "PAGES2k_v2/proxy-data/pages2k_proxy"
    lmr_root = proxy_root / "LMR_NCDC_database_v.0.2.0"
    paths = {
        "inventory": base / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "pages_annual": pages_root / "proxy_data_1_13_1.mat",
        "pages_metadata": pages_root / "metadata_1_13_1.txt",
        "pages_ts": pages_root / "Pages2kTSv1_13_1.mat",
        "lmr_data": lmr_root / "LMR_data.mat",
        "lmr_metadata": lmr_root / "lmr_meta.csv",
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
    tools = load_module(paths["pages_tools"])
    inventory = pd.read_csv(paths["inventory"])
    lat_min, lat_max = float(inventory["latitude"].min()), float(inventory["latitude"].max())
    lon_min, lon_max = float(inventory["longitude"].min()), float(inventory["longitude"].max())
    existing_ids: set[str] = set()
    for value in inventory["metadata_identities"].dropna().astype(str):
        existing_ids.update(item.strip() for item in value.split("||") if item.strip())

    candidates: list[dict[str, object]] = []
    series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    pages = loadmat(paths["pages_annual"], squeeze_me=True, struct_as_record=False)
    pages_years = np.asarray(pages["year"], dtype=int).reshape(-1)
    pages_matrix = np.asarray(pages["proxy_ann"], dtype=float)
    pages_meta = tools.metadata_table(paths["pages_metadata"])
    ts_values = np.asarray(loadmat(paths["pages_ts"], squeeze_me=True, struct_as_record=False, variable_names=["TS"])["TS"]).reshape(-1)
    ts_index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        key = (tools.clean_group(getattr(item, "dataSetName", "")), tools.clean_group(getattr(item, "paleoData_TSid", "")))
        ts_index[key].append(item)
    pages_calibration = (pages_years >= 1940) & (pages_years <= 2000)
    if np.any(pages_calibration & (pages_years >= 1921) & (pages_years <= 1939)):
        raise RuntimeError("pages-locked-period-mask-failed")
    for index, meta in pages_meta.iterrows():
        latitude, longitude = float(meta["geo_latitude"]), float(meta["geo_longitude"])
        if not (lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max):
            continue
        key = (tools.clean_group(meta["dataSetName"]), tools.clean_group(meta["paleoData_TSid"]))
        matches = ts_index.get(key, [])
        item = matches[0] if len(matches) == 1 else None
        variable = tools.clean_group(getattr(item, "climateInterpretation_variable", "")) if item is not None else ""
        detail = tools.clean_group(getattr(item, "climateInterpretation_variableDetail", "")) if item is not None else ""
        identity_id = f"{key[0]}::{key[1]}"
        count = int(np.isfinite(pages_matrix[pages_calibration, index]).sum())
        candidates.append({
            "source_database": "PAGES2k_1_13_1", "source_column_1based": int(index + 1),
            "formal_identity": identity_id, "latitude": latitude, "longitude": longitude,
            "archive_type": tools.clean_group(getattr(item, "archiveType", "")) if item is not None else "",
            "measurement": tools.clean_group(getattr(item, "paleoData_proxy", "")) if item is not None else "",
            "documented_climate_variable": variable, "documented_climate_detail": detail,
            "documented_seasonality": tools.clean_group(getattr(item, "climateInterpretation_seasonality", "")) if item is not None else "",
            "documented_direction": tools.clean_group(getattr(item, "climateInterpretation_interpDirection", "")) if item is not None else "",
            "network_role": climate_role(variable, detail), "calibration_1940_2000_finite_count": count,
            "formal_identity_already_present": identity_id in existing_ids,
        })
        series[("PAGES2k_1_13_1", int(index))] = (pages_years, pages_matrix[:, index])

    lmr_meta = pd.read_csv(paths["lmr_metadata"])
    lmr_meta["NCDC ID"] = lmr_meta["NCDC ID"].astype(str).str.strip()
    if lmr_meta["NCDC ID"].duplicated().any():
        raise RuntimeError("duplicate-lmr-metadata-id")
    lmr_meta = lmr_meta.set_index("NCDC ID", drop=False)
    lmr = loadmat(paths["lmr_data"], squeeze_me=True, struct_as_record=False, variable_names=["lmr2k_data", "lmr_data_names", "year"])
    lmr_years = np.asarray(lmr["year"], dtype=int).reshape(-1)
    lmr_matrix = np.asarray(lmr["lmr2k_data"], dtype=float)
    lmr_names = [tools.text_value(value).strip() for value in np.asarray(lmr["lmr_data_names"]).reshape(-1)]
    if lmr_matrix.shape != (len(lmr_years), len(lmr_names)):
        raise RuntimeError("lmr-array-shape-contract-failed")
    missing_metadata_ids = sorted(set(lmr_meta.index) - set(lmr_names))
    unknown_data_names = sorted(name for name in set(lmr_names) - set(lmr_meta.index))
    blank_name_resolution = "none"
    if len(missing_metadata_ids) == 1 and unknown_data_names == ["nan"] and lmr_names.count("nan") == 1:
        blank_index = lmr_names.index("nan")
        lmr_names[blank_index] = missing_metadata_ids[0]
        blank_name_resolution = f"column_{blank_index + 1}:{missing_metadata_ids[0]}"
    if set(lmr_names) != set(lmr_meta.index):
        raise RuntimeError("lmr-array-metadata-contract-failed")
    lmr_calibration = (lmr_years >= 1940) & (lmr_years <= 2000)
    if np.any(lmr_calibration & (lmr_years >= 1921) & (lmr_years <= 1939)):
        raise RuntimeError("lmr-locked-period-mask-failed")
    for index, identity_id in enumerate(lmr_names):
        meta = lmr_meta.loc[identity_id]
        latitude, longitude = float(meta["Lat (N)"]), float(meta["Lon (E)"])
        if not (lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max):
            continue
        variable = clean(meta["climateVariable"])
        detail = clean(meta["Realm"])
        count = int(np.isfinite(lmr_matrix[lmr_calibration, index]).sum())
        candidates.append({
            "source_database": "LMR_v0_2_0", "source_column_1based": int(index + 1),
            "formal_identity": identity_id, "latitude": latitude, "longitude": longitude,
            "archive_type": clean(meta["Archive type"]), "measurement": clean(meta["Proxy measurement"]),
            "documented_climate_variable": variable, "documented_climate_detail": detail,
            "documented_seasonality": clean(meta["Seasonality"]), "documented_direction": clean(meta["Relation_to_climateVariable"]),
            "network_role": climate_role(variable, detail), "calibration_1940_2000_finite_count": count,
            "formal_identity_already_present": identity_id in existing_ids,
        })
        series[("LMR_v0_2_0", int(index))] = (lmr_years, lmr_matrix[:, index])

    frame = pd.DataFrame(candidates)
    frame["eligible_before_sequence_dedup"] = (
        (~frame["formal_identity_already_present"])
        & (frame["calibration_1940_2000_finite_count"] >= 20)
        & frame["network_role"].isin(["documented_temperature", "documented_moisture", "documented_mixed"])
    )
    eligible_indices = frame.index[frame["eligible_before_sequence_dedup"]].tolist()
    union = UnionFind(len(eligible_indices))
    match_rows: list[dict[str, object]] = []
    for left_position, left_index in enumerate(eligible_indices):
        left = frame.loc[left_index]
        for right_position in range(left_position + 1, len(eligible_indices)):
            right_index = eligible_indices[right_position]
            right = frame.loc[right_index]
            if abs(float(left.latitude) - float(right.latitude)) > 1e-6 or abs(float(left.longitude) - float(right.longitude)) > 1e-6:
                continue
            left_key = (str(left.source_database), int(left.source_column_1based) - 1)
            right_key = (str(right.source_database), int(right.source_column_1based) - 1)
            matched, count, correlation, residual = affine_match(*series[left_key], *series[right_key])
            if matched:
                union.union(left_position, right_position)
                match_rows.append({
                    "left_formal_identity": left.formal_identity, "right_formal_identity": right.formal_identity,
                    "finite_unlocked_overlap": count, "correlation": correlation, "maximum_abs_affine_residual": residual,
                })
    groups: dict[int, list[int]] = defaultdict(list)
    for position, frame_index in enumerate(eligible_indices):
        groups[union.find(position)].append(frame_index)
    canonical_rows: list[dict[str, object]] = []
    for number, members in enumerate(groups.values(), start=1):
        group = frame.loc[members]
        roles = sorted(set(group["network_role"]))
        final_role = roles[0] if len(roles) == 1 else "metadata_conflict"
        canonical_rows.append({
            "canonical_group": number, "member_count": int(len(group)),
            "formal_identities": " || ".join(sorted(group["formal_identity"])),
            "source_databases": " || ".join(sorted(set(group["source_database"]))),
            "latitude": float(group.iloc[0]["latitude"]), "longitude": float(group.iloc[0]["longitude"]),
            "network_role": final_role,
            "archive_types": " || ".join(sorted({clean(value) for value in group["archive_type"] if clean(value)})),
            "measurements": " || ".join(sorted({clean(value) for value in group["measurement"] if clean(value)})),
            "maximum_calibration_count": int(group["calibration_1940_2000_finite_count"].max()),
        })
        frame.loc[members, "canonical_group"] = number
    frame["canonical_group"] = frame["canonical_group"].astype("Int64")
    frame.to_csv(output / "full_pages_lmr_hma_candidate_inventory.csv", index=False)
    if match_rows:
        write_csv(output / "cross_source_sequence_aliases.csv", match_rows)
    canonical = pd.DataFrame(canonical_rows)
    canonical.to_csv(output / "eligible_new_canonical_identities.csv", index=False)

    role_counts = Counter(canonical["network_role"]) if len(canonical) else Counter()
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False,
        "proxy9_envelope": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "pages_full_records": int(pages_matrix.shape[1]), "lmr_full_records": int(lmr_matrix.shape[1]),
        "pages_hma_envelope_records": int((frame["source_database"] == "PAGES2k_1_13_1").sum()),
        "lmr_hma_envelope_records": int((frame["source_database"] == "LMR_v0_2_0").sum()),
        "new_eligible_records_before_sequence_dedup": int(frame["eligible_before_sequence_dedup"].sum()),
        "new_eligible_canonical_identities": int(len(canonical)),
        "new_canonical_role_counts": dict(role_counts),
        "cross_source_or_within_source_alias_pairs": int(len(match_rows)),
        "lmr_unique_blank_name_resolution": blank_name_resolution,
        "decision": "design_source_expanded_network_without_skill_selection" if len(canonical) else "full_pages_lmr_sources_add_no_calibration_capable_hma_identity",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    source_counts = frame.groupby("source_database").size()
    new_role_counts = canonical["network_role"].value_counts() if len(canonical) else pd.Series(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].bar(source_counts.index, source_counts.values, color="#005f73")
    axes[0].set(title="Full-source HMA-envelope records", ylabel="Records")
    axes[1].bar(new_role_counts.index, new_role_counts.values, color="#ee9b00")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(title="New eligible canonical identities", ylabel="Identities")
    fig.suptitle("Full PAGES2k and LMR source inventory", fontsize=14)
    fig.savefig(output / "full_pages_lmr_hma_source_inventory.png", dpi=180)
    plt.close(fig)
    (output / "FINAL_RESULT.md").write_text(
        "# Full PAGES2k and LMR HMA Source Inventory\n\n"
        f"- Full databases: PAGES2k `{pages_matrix.shape[1]}` records; LMR `{lmr_matrix.shape[1]}` records.\n"
        f"- Inside proxy9 envelope: PAGES2k `{summary['pages_hma_envelope_records']}`; LMR `{summary['lmr_hma_envelope_records']}`.\n"
        f"- New eligible records before sequence deduplication: `{summary['new_eligible_records_before_sequence_dedup']}`.\n"
        f"- New eligible canonical identities: `{len(canonical)}`; roles: `{json.dumps(dict(role_counts), sort_keys=True)}`.\n\n"
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
