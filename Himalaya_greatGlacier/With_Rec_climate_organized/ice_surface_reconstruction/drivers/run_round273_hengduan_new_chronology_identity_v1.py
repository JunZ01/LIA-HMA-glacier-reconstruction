"""Audit modern sequence identity of the new Hengduan chronologies."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy.io import loadmat


YEARS = np.arange(1940, 2001, dtype=int)
NEW_CLUSTERS = {
    "HLS": "north",
    "JYC": "north",
    "YZC": "north",
    "BLX": "south",
    "CYS": "south",
    "FNPTS": "south",
    "GDCTS": "south",
    "GDCYS": "south",
    "LGQ": "south",
}
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 100_000_000:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def workbook_sheet_paths(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{NS_PACKAGE_REL}}}Relationship")
    }
    result = {}
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        name = sheet.attrib["name"]
        relationship_id = sheet.attrib[f"{{{NS_REL}}}id"]
        target = targets[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = posixpath.normpath(posixpath.join("xl", target))
        result[name] = target
    return result


def _cell_raw_value(row: ET.Element, column: str) -> str | None:
    for cell in row.findall(f"{{{NS_MAIN}}}c"):
        reference = cell.attrib.get("r", "")
        if reference.rstrip("0123456789") != column:
            continue
        value = cell.find(f"{{{NS_MAIN}}}v")
        return None if value is None else value.text
    return None


def parse_modern_chronologies(path: Path):
    sheet_paths = workbook_sheet_paths(path)
    series: dict[str, np.ndarray] = {}
    audit: dict[str, dict[str, int | bool]] = {}
    with zipfile.ZipFile(path) as archive:
        for name, member in sheet_paths.items():
            values: dict[int, float] = {}
            locked = 0
            outside = 0
            decoded = 0
            with archive.open(member) as handle:
                for _, row in ET.iterparse(handle, events=("end",)):
                    if row.tag != f"{{{NS_MAIN}}}row":
                        continue
                    raw_year = _cell_raw_value(row, "A")
                    if raw_year is None:
                        row.clear()
                        continue
                    try:
                        year = int(float(raw_year))
                    except ValueError:
                        row.clear()
                        continue
                    if 1921 <= year <= 1939:
                        locked += 1
                        row.clear()
                        continue
                    if year < 1940 or year > 2000:
                        outside += 1
                        row.clear()
                        continue
                    raw_value = _cell_raw_value(row, "B")
                    if raw_value is not None:
                        values[year] = float(raw_value)
                        decoded += 1
                    row.clear()
            series[name] = np.asarray([values.get(int(year), np.nan) for year in YEARS], dtype=float)
            audit[name] = {
                "decoded_value_count": decoded,
                "locked_rows_skipped_before_value_access": locked,
                "outside_rows_skipped_before_value_access": outside,
                "locked_numeric_values_accessed": False,
            }
    return series, audit


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    count = int(valid.sum())
    if count < 3:
        return {
            "overlap_count": count,
            "correlation": np.nan,
            "absolute_correlation": np.nan,
            "affine_rmse_fraction": np.nan,
            "exact_affine_alias": False,
            "near_alias": False,
        }
    left = candidate[valid]
    right = reference[valid]
    left_sd = float(np.std(left))
    right_sd = float(np.std(right))
    if left_sd <= 1e-12 or right_sd <= 1e-12:
        correlation = np.nan
        fraction = np.nan
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
        design = np.column_stack((np.ones(count), right))
        fitted = design @ np.linalg.lstsq(design, left, rcond=None)[0]
        fraction = float(np.sqrt(np.mean((left - fitted) ** 2)) / left_sd)
    absolute = abs(correlation) if np.isfinite(correlation) else np.nan
    return {
        "overlap_count": count,
        "correlation": correlation,
        "absolute_correlation": absolute,
        "affine_rmse_fraction": fraction,
        "exact_affine_alias": bool(count >= 50 and absolute >= 0.999999 and fraction <= 1e-6),
        "near_alias": bool(count >= 50 and absolute >= 0.995),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-modern-only", action="store_true")
    parser.add_argument("--chronologies", type=Path, required=True)
    parser.add_argument("--expected-chronology-sha256", required=True)
    parser.add_argument("--proxy6", type=Path, required=True)
    parser.add_argument("--proxy9", type=Path, required=True)
    parser.add_argument("--proxy6-metadata", type=Path, required=True)
    parser.add_argument("--proxy9-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_modern_only:
        raise RuntimeError("explicit-modern-only-flag-required")
    if args.output.exists():
        raise FileExistsError(args.output)

    contract = Path(__file__).with_name("round273_hengduan_new_chronology_identity_v1_contract.md")
    paths = {
        "chronologies": args.chronologies,
        "proxy6": args.proxy6,
        "proxy9": args.proxy9,
        "proxy6_metadata": args.proxy6_metadata,
        "proxy9_metadata": args.proxy9_metadata,
        "contract": contract,
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["chronologies"].get("sha256", "").lower() != args.expected_chronology_sha256.lower():
        raise RuntimeError("chronology-hash-conflict")

    chronologies, parse_audit = parse_modern_chronologies(args.chronologies)
    if len(chronologies) != 30 or set(NEW_CLUSTERS) - set(chronologies):
        raise RuntimeError(f"unexpected-workbook-inventory:{len(chronologies)}")
    metadata = {
        "proxy6": pd.read_csv(args.proxy6_metadata),
        "proxy9": pd.read_csv(args.proxy9_metadata),
    }
    rows: list[dict[str, object]] = []
    for network, path in (("proxy6", args.proxy6), ("proxy9", args.proxy9)):
        observations = np.asarray(loadmat(path, variable_names=["observations"])["observations"], dtype=float)
        if observations.shape[0] != len(YEARS):
            raise RuntimeError(f"unexpected-network-shape:{network}:{observations.shape}")
        frame = metadata[network]
        for chronology, candidate in chronologies.items():
            for column in range(observations.shape[1]):
                metric = metrics(candidate, observations[:, column])
                identity = frame.iloc[column].to_dict() if column < len(frame) else {}
                rows.append({
                    "chronology": chronology,
                    "source_class": "new_2018" if chronology in NEW_CLUSTERS else "ITRDB",
                    "cluster": NEW_CLUSTERS.get(chronology, "published_ITRDB"),
                    "network": network,
                    "network_column": column,
                    "network_identity": identity.get("formal_identity", identity.get("proxy_name", identity.get("proxy_id", ""))),
                    **metric,
                })
    comparison = pd.DataFrame(rows)

    chronology_rows = []
    for name, values in chronologies.items():
        subset = comparison[comparison.chronology == name]
        chronology_rows.append({
            "chronology": name,
            "source_class": "new_2018" if name in NEW_CLUSTERS else "ITRDB",
            "cluster": NEW_CLUSTERS.get(name, "published_ITRDB"),
            "modern_value_count": int(np.isfinite(values).sum()),
            "exact_alias_count": int(subset.exact_affine_alias.sum()),
            "near_alias_count": int(subset.near_alias.sum()),
            "maximum_absolute_correlation": float(subset.absolute_correlation.max()),
        })
    chronology_summary = pd.DataFrame(chronology_rows)
    new = chronology_summary[chronology_summary.source_class == "new_2018"].copy()
    new["admitted_independent"] = (
        (new.modern_value_count >= 55) & (new.exact_alias_count == 0) & (new.near_alias_count == 0)
    )
    chronology_summary = chronology_summary.merge(
        new[["chronology", "admitted_independent"]], on="chronology", how="left"
    )
    chronology_summary["admitted_independent"] = chronology_summary.admitted_independent.fillna(False).astype(bool)

    admitted_count = int(new.admitted_independent.sum())
    admitted_north = int(new.loc[new.cluster == "north", "admitted_independent"].sum())
    admitted_south = int(new.loc[new.cluster == "south", "admitted_independent"].sum())
    itrdb_alias_count = int(
        ((chronology_summary.source_class == "ITRDB") & (chronology_summary.near_alias_count > 0)).sum()
    )
    gates = {
        "at_least_six_new_chronologies_independent": admitted_count >= 6,
        "at_least_two_independent_north": admitted_north >= 2,
        "at_least_two_independent_south": admitted_south >= 2,
        "at_least_one_itrdb_positive_control_alias": itrdb_alias_count >= 1,
    }
    passes = bool(all(gates.values()))
    decision = (
        "advance_independent_hengduan_clusters_to_modern_scpdsi_oof"
        if passes
        else "close_hengduan_new_chronologies_on_identity_or_coverage"
    )

    args.output.mkdir(parents=True)
    comparison.to_csv(args.output / "all_sequence_comparisons.csv", index=False)
    chronology_summary.to_csv(args.output / "chronology_identity_summary.csv", index=False)
    (args.output / "parse_audit.json").write_text(json.dumps(parse_audit, indent=2), encoding="utf-8")
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": "round_273_hengduan_new_chronology_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "workbook_chronology_count": len(chronologies),
        "new_chronology_count": len(NEW_CLUSTERS),
        "admitted_independent_count": admitted_count,
        "admitted_north_count": admitted_north,
        "admitted_south_count": admitted_south,
        "itrdb_chronologies_with_alias": itrdb_alias_count,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_values_accessed": False,
        "published_paleo_reconstruction_read": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (args.output / "FINAL_RESULT.md").write_text(
        "# Round 273 Hengduan new chronology identity\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Independent new chronologies: `{admitted_count}/9` "
        f"(north `{admitted_north}/3`, south `{admitted_south}/6`).\n"
        f"- ITRDB positive-control chronologies with aliases: `{itrdb_alias_count}/21`.\n"
        f"- Passes all gates: `{passes}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

