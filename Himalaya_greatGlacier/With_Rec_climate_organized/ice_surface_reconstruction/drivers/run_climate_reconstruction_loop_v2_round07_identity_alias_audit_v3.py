"""Resolve proxy9 identities with exact-value, nested-coverage aliases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


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


def decode(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def scalar(group: h5py.Group, name: str) -> float:
    return float(np.asarray(group[name]).reshape(-1)[0]) if name in group else float("nan")


def affine(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < 20:
        return np.nan, np.nan, np.nan, np.nan, count
    x, y = right[valid], left[valid]
    design = np.column_stack([x, np.ones(count)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.asarray([slope, intercept])
    return float(np.corrcoef(x, y)[0, 1]), float(slope), float(intercept), float(np.max(np.abs(residual))), count


def alias_relation(matrix: np.ndarray, columns: list[int]) -> tuple[bool, str]:
    if len(columns) == 1:
        return True, "single_identity"
    reference = matrix[:, columns[0]]
    reference_finite = np.isfinite(reference)
    for column in columns[1:]:
        candidate = matrix[:, column]
        candidate_finite = np.isfinite(candidate)
        common = reference_finite & candidate_finite
        if int(common.sum()) < 20 or not np.array_equal(reference[common], candidate[common]):
            return False, "different_common_values"
        nested = bool(np.all(~reference_finite | candidate_finite) or np.all(~candidate_finite | reference_finite))
        if not nested:
            return False, "nonnested_coverage"
    return True, "shared_values_nested_coverage"


def main() -> int:
    project = root()
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_07_proxy_network_identity_alias_audit_v3"
    paths = {
        "proxy9": Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all\python_data\proxy_data_9.mat"),
        "combined": Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"),
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    source = loadmat(paths["combined"], squeeze_me=True, struct_as_record=False)
    years = np.asarray(source["year"], dtype=int).reshape(-1)
    matrix = np.asarray(source["combined6_data"], dtype=float)
    names = np.char.strip(np.asarray(source["dataName"]).astype(str)).reshape(-1)
    lat = np.asarray(source["p_lat"], dtype=float).reshape(-1)
    lon = np.asarray(source["p_lon"], dtype=float).reshape(-1)
    flags = np.asarray(source["source_flag"], dtype=int).reshape(-1)
    year_lookup = {int(year): index for index, year in enumerate(years)}
    rows = []
    with h5py.File(paths["proxy9"], "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for position, reference in enumerate(references, start=1):
            proxy = handle[reference]
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            proxy_years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            proxy_values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            coordinate = np.flatnonzero(np.isclose(lat, location[0], atol=1e-6) & np.isclose(lon, location[1], atol=1e-6))
            passing, metrics = [], {}
            for column in coordinate:
                target = np.asarray([matrix[year_lookup[int(year)], column] if int(year) in year_lookup else np.nan for year in proxy_years])
                result = affine(proxy_values, target)
                if np.isfinite(result[0]) and result[0] >= 0.999999999 and result[3] <= 1e-10:
                    passing.append(int(column))
                    metrics[int(column)] = result
            if not passing:
                raise RuntimeError(f"no-passing-identity:{position}")
            compatible, relation = alias_relation(matrix, passing)
            if not compatible:
                raise RuntimeError(f"incompatible-identities:{position}:{relation}:{passing}")
            canonical = max(passing, key=lambda column: int(np.isfinite(matrix[:, column]).sum()))
            correlation, slope, intercept, residual, overlap = metrics[canonical]
            aliases = ";".join(str(column + 1) for column in passing)
            identity_key = f"{location[0]:.6f}:{location[1]:.6f}:{aliases}"
            rows.append({
                "record_index_1based": position, "latitude": float(location[0]), "longitude": float(location[1]),
                "proxy_type": decode(proxy["ptype"]), "psm_method": decode(proxy["psm_mthd"]),
                "psm_b0": float(coefficients[0]) if len(coefficients) else np.nan,
                "psm_b1_temperature": float(coefficients[1]) if len(coefficients) > 1 else np.nan,
                "psm_b2_precipitation": float(coefficients[2]) if len(coefficients) > 2 else np.nan,
                "psm_correlation": scalar(proxy, "pcr"), "psm_temperature_correlation": scalar(proxy, "pcr_t"),
                "psm_precipitation_correlation": scalar(proxy, "pcr_p"), "psm_p_value": scalar(proxy, "pval"),
                "coordinate_candidate_count": int(len(coordinate)), "passing_identity_count": int(len(passing)),
                "alias_relation": relation, "canonical_combined_column_1based": canonical + 1,
                "canonical_combined_name": names[canonical], "canonical_source_flag": int(flags[canonical]),
                "alias_combined_columns_1based": aliases,
                "alias_combined_names": ";".join(names[column] for column in passing),
                "alias_source_flags": ";".join(str(int(flags[column])) for column in passing),
                "identity_key": identity_key, "identity_correlation": correlation,
                "identity_affine_slope": slope, "identity_affine_intercept": intercept,
                "identity_maximum_abs_residual": residual, "identity_finite_overlap": overlap,
            })
    frame = pd.DataFrame(rows)
    counts = frame.groupby("identity_key")["record_index_1based"].transform("size")
    frame["proxy9_duplicate_representation_count"] = counts.astype(int)
    frame["proxy9_duplicate_representation"] = counts > 1
    frame.to_csv(output / "proxy9_identity_alias_inventory.csv", index=False)
    frame[frame["proxy9_duplicate_representation"]].to_csv(output / "proxy9_duplicate_representation_records.csv", index=False)
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False,
        "proxy9_records": int(len(frame)), "unique_identity_keys": int(frame["identity_key"].nunique()),
        "multi_alias_records": int((frame["passing_identity_count"] > 1).sum()),
        "duplicate_representation_records": int(frame["proxy9_duplicate_representation"].sum()),
        "duplicate_representation_groups": int(frame.loc[frame["proxy9_duplicate_representation"], "identity_key"].nunique()),
        "minimum_identity_correlation": float(frame["identity_correlation"].min()),
        "maximum_identity_affine_residual": float(frame["identity_maximum_abs_residual"].max()),
        "decision": "retain_nested_coverage_aliases_and_reconcile_metadata",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Identity Alias Audit v3\n\n"
        f"- Records: `{len(frame)}`; unique identities: `{summary['unique_identity_keys']}`.\n"
        f"- Multi-alias records: `{summary['multi_alias_records']}`.\n"
        f"- Duplicate representations: `{summary['duplicate_representation_records']}` records in `{summary['duplicate_representation_groups']}` group.\n\n"
        "All aliases share exact common values with nested year coverage. No record was deleted.\n",
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
