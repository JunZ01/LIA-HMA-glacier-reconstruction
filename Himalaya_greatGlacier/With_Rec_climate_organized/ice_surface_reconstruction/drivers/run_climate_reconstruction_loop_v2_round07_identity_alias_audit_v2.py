"""Resolve proxy9 identities while retaining exact cross-source aliases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

MIN_CORRELATION = 0.999999999
MAX_AFFINE_RESIDUAL = 1e-10


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
    return float(np.asarray(group[name]).reshape(-1)[0]) if name in group else float("nan")


def affine(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < 20:
        return np.nan, np.nan, np.nan, np.nan, count
    x = right[valid]
    y = left[valid]
    design = np.column_stack([x, np.ones(count)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.asarray([slope, intercept])
    return (
        float(np.corrcoef(x, y)[0, 1]), float(slope), float(intercept),
        float(np.max(np.abs(residual))), count,
    )


def exact_aliases(matrix: np.ndarray, columns: list[int]) -> bool:
    if len(columns) < 2:
        return True
    reference = matrix[:, columns[0]]
    for column in columns[1:]:
        candidate = matrix[:, column]
        finite = np.isfinite(reference) & np.isfinite(candidate)
        if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
            return False
        if not np.array_equal(reference[finite], candidate[finite]):
            return False
    return True


def main() -> int:
    project = root()
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_07_proxy_network_identity_alias_audit_v2"
    )
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

    combined = loadmat(paths["combined"], squeeze_me=True, struct_as_record=False)
    years = np.asarray(combined["year"], dtype=int).reshape(-1)
    matrix = np.asarray(combined["combined6_data"], dtype=float)
    names = np.char.strip(np.asarray(combined["dataName"]).astype(str)).reshape(-1)
    latitude = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    longitude = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    flags = np.asarray(combined["source_flag"], dtype=int).reshape(-1)
    year_lookup = {int(year): index for index, year in enumerate(years)}

    rows = []
    with h5py.File(paths["proxy9"], "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        if len(references) != 297:
            raise RuntimeError("proxy9-count-failed")
        for position, reference in enumerate(references, start=1):
            proxy = handle[reference]
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            proxy_years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            proxy_values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            coordinate_candidates = np.flatnonzero(
                np.isclose(latitude, location[0], atol=1e-6)
                & np.isclose(longitude, location[1], atol=1e-6)
            )
            passing = []
            metrics = []
            for column in coordinate_candidates:
                target = np.asarray([
                    matrix[year_lookup[int(year)], column] if int(year) in year_lookup else np.nan
                    for year in proxy_years
                ])
                result = affine(proxy_values, target)
                correlation, slope, intercept, residual, count = result
                if np.isfinite(correlation) and correlation >= MIN_CORRELATION and residual <= MAX_AFFINE_RESIDUAL:
                    passing.append(int(column))
                    metrics.append(result)
            if not passing:
                raise RuntimeError(f"no-passing-identity:{position}")
            alias_exact = exact_aliases(matrix, passing)
            if len(passing) > 1 and not alias_exact:
                raise RuntimeError(f"nonidentical-multiple-identities:{position}:{passing}")
            best = int(np.argmax([metric[0] for metric in metrics]))
            canonical = passing[best]
            correlation, slope, intercept, residual, count = metrics[best]
            alias_columns = ";".join(str(column + 1) for column in passing)
            alias_names = ";".join(names[column] for column in passing)
            alias_flags = ";".join(str(int(flags[column])) for column in passing)
            identity_key = f"{location[0]:.6f}:{location[1]:.6f}:{alias_columns}"
            rows.append({
                "record_index_1based": position,
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
                "coordinate_candidate_count": int(len(coordinate_candidates)),
                "passing_identity_count": int(len(passing)),
                "aliases_are_exactly_identical": bool(alias_exact),
                "canonical_combined_column_1based": canonical + 1,
                "canonical_combined_name": names[canonical],
                "canonical_source_flag": int(flags[canonical]),
                "alias_combined_columns_1based": alias_columns,
                "alias_combined_names": alias_names,
                "alias_source_flags": alias_flags,
                "identity_key": identity_key,
                "identity_correlation": correlation,
                "identity_affine_slope": slope,
                "identity_affine_intercept": intercept,
                "identity_maximum_abs_residual": residual,
                "identity_finite_overlap": count,
            })
    frame = pd.DataFrame(rows)
    group_size = frame.groupby("identity_key")["record_index_1based"].transform("size")
    frame["proxy9_duplicate_representation_count"] = group_size.astype(int)
    frame["proxy9_duplicate_representation"] = group_size > 1
    frame.to_csv(output / "proxy9_identity_alias_inventory.csv", index=False)
    duplicate = frame[frame["proxy9_duplicate_representation"]].copy()
    duplicate.to_csv(output / "proxy9_duplicate_representation_records.csv", index=False)

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "proxy9_records": int(len(frame)),
        "unique_identity_keys": int(frame["identity_key"].nunique()),
        "multi_alias_records": int((frame["passing_identity_count"] > 1).sum()),
        "multi_alias_all_exact": bool(frame.loc[frame["passing_identity_count"] > 1, "aliases_are_exactly_identical"].all()),
        "duplicate_representation_records": int(frame["proxy9_duplicate_representation"].sum()),
        "duplicate_representation_groups": int(frame.loc[frame["proxy9_duplicate_representation"], "identity_key"].nunique()),
        "minimum_identity_correlation": float(frame["identity_correlation"].min()),
        "maximum_identity_affine_residual": float(frame["identity_maximum_abs_residual"].max()),
        "decision": "retain_aliases_and_reconcile_metadata_without_deletion",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Identity Alias Audit\n\n"
        f"- Records: `{len(frame)}`; unique identity keys: `{summary['unique_identity_keys']}`.\n"
        f"- Multi-alias records: `{summary['multi_alias_records']}`; all aliases exactly identical: "
        f"`{str(summary['multi_alias_all_exact']).lower()}`.\n"
        f"- Duplicate proxy9 representations: `{summary['duplicate_representation_records']}` records in "
        f"`{summary['duplicate_representation_groups']}` group.\n\n"
        "Aliases and repeated records are retained explicitly for metadata reconciliation; no record was deleted.\n",
        encoding="utf-8",
    )

    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({
        "sources_before": before, "sources_after": after, "source_states_unchanged": before == after
    }, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
