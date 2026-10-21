"""Boundary-safe identity audit for two acquired NOAA temperature products."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = np.concatenate((np.arange(1000, 1921), np.arange(1940, 2001))).astype(int)
YEAR_RE = re.compile(r"^\s*([+-]?\d{3,4})(?:\s+|$)")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def parse_allowed(raw: Path, export: Path, header_export: Path) -> dict[str, object]:
    headers: list[str] = []
    rows: list[list[float]] = []
    width = None
    source_first = None
    source_last = None
    locked = 0
    outside = 0
    with raw.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            match = YEAR_RE.match(line)
            if match is None:
                headers.append(line)
                continue
            year = int(match.group(1))
            if not 500 <= year <= 2500:
                headers.append(line)
                continue
            source_first = year if source_first is None else min(source_first, year)
            source_last = year if source_last is None else max(source_last, year)
            if 1921 <= year <= 1939:
                locked += 1
                continue
            if not (1000 <= year <= 1920 or 1940 <= year <= 2000):
                outside += 1
                continue
            tokens = line[match.end():].split()
            if width is None:
                width = len(tokens)
            if len(tokens) != width:
                raise RuntimeError(f"data-width-changed:{raw.name}:{line_number}:{len(tokens)}:{width}")
            values = []
            for column, token in enumerate(tokens, start=2):
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise RuntimeError(f"nonnumeric-allowed-field:{raw.name}:{line_number}:{column}") from exc
            rows.append([year, *values])
    if not rows or width is None:
        raise RuntimeError(f"no-allowed-data:{raw.name}")
    columns = ["year", *[f"value_{index}" for index in range(1, width + 1)]]
    with export.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    header_export.write_text("\n".join(headers) + "\n", encoding="utf-8")
    return {
        "source_first_year": source_first,
        "source_last_year": source_last,
        "allowed_first_year": int(rows[0][0]),
        "allowed_last_year": int(rows[-1][0]),
        "allowed_row_count": len(rows),
        "numeric_value_column_count": width,
        "locked_rows_skipped_before_value_decode": locked,
        "outside_allowed_rows_skipped": outside,
    }


def aligned(frame: pd.DataFrame, column: str) -> np.ndarray:
    lookup = dict(zip(frame.year.astype(int), frame[column].astype(float)))
    return np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)


def metrics(source: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(source) & np.isfinite(candidate)
    count = int(valid.sum())
    result = {
        "overlap_count": count,
        "correlation": float("nan"),
        "affine_intercept": float("nan"),
        "affine_slope": float("nan"),
        "maximum_abs_residual": float("nan"),
        "normalized_rmse": float("nan"),
    }
    if count < 20:
        return result
    left = source[valid]
    right = candidate[valid]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return result
    design = np.column_stack((np.ones(count), right))
    coefficients = np.linalg.lstsq(design, left, rcond=None)[0]
    residual = left - design @ coefficients
    result.update({
        "correlation": float(np.corrcoef(left, right)[0, 1]),
        "affine_intercept": float(coefficients[0]),
        "affine_slope": float(coefficients[1]),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / np.std(left, ddof=1)),
    })
    return result


def classify(item: dict[str, object]) -> tuple[bool, bool]:
    exact = bool(
        item["overlap_count"] >= 20
        and np.isfinite(item["correlation"])
        and abs(item["correlation"]) >= 0.999999
        and item["maximum_abs_residual"] <= 1e-8
    )
    near = bool(
        item["overlap_count"] >= 20
        and np.isfinite(item["correlation"])
        and abs(item["correlation"]) >= 0.995
        and item["normalized_rmse"] <= 0.05
    )
    return exact, near


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_149_noaa_temperature_candidate_identity_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round149_noaa_temperature_candidate_identity_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    acquisition = root / "validation_data/external_sources/hma_temperature_public_acquisition_20260819_v1"
    manifest_path = acquisition / "acquisition_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_files = {
        "tibet2015": acquisition / "raw_unopened/tibet2015temp.txt",
        "bhutan2015": acquisition / "raw_unopened/bhutan2015temp.txt",
    }

    base = load_module("round149_base", driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py")
    archive = load_module("round149_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    round25 = load_module("round149_round25", driver / "run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1.py")
    manifest27 = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    prior_manifest = json.loads(manifest27.read_text(encoding="utf-8"))
    proxy9 = Path(prior_manifest["sources_before"]["proxy9"]["path"])
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "acquisition_manifest": manifest_path,
        "proxy6": archive.archive_proxy_path(),
        "proxy6_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "proxy9": proxy9,
        "proxy9_metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy_network_psm_audit_v2/proxy9_identity_and_psm_metadata.csv",
        "manifest27": manifest27,
        **{f"raw_{name}": path for name, path in raw_files.items()},
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    for candidate_id, name in (("T_TIBET2015", "tibet2015"), ("T_BHUTAN2015", "bhutan2015")):
        item = next(record for record in manifest["files"] if record["candidate_id"] == candidate_id)
        if before[f"raw_{name}"]["sha256"].upper() != item["sha256"].upper() or item["values_opened"]:
            raise RuntimeError(f"acquisition-manifest-mismatch:{name}")

    parse_summaries = {}
    candidate_series = {}
    for name, raw in raw_files.items():
        export = output / f"{name}_allowed.csv"
        parse_summaries[name] = parse_allowed(raw, export, output / f"{name}_header_metadata.txt")
        frame = pd.read_csv(export)
        for column in frame.columns[1:]:
            candidate_series[f"{name}_{column}"] = aligned(frame, column)
    if sum(item["locked_rows_skipped_before_value_decode"] for item in parse_summaries.values()) != 38:
        raise RuntimeError(f"unexpected-locked-row-count:{parse_summaries}")

    proxy6_meta = pd.read_csv(paths["proxy6_metadata"])
    proxy6_ids = tuple(proxy6_meta.proxy_index.astype(int))
    proxy6_values = base.read_proxy_values(paths["proxy6"], proxy6_ids, YEARS)
    _, proxy9_values, locked = round25.read_proxy_allowed(paths["proxy9"], YEARS)
    if locked != 0:
        raise RuntimeError("proxy9-locked-read")
    proxy9_meta = pd.read_csv(paths["proxy9_metadata"])
    if proxy9_values.shape[0] != len(proxy9_meta):
        raise RuntimeError("proxy9-metadata-count-mismatch")

    rows = []
    exact_counts = {}
    near_counts = {}
    for candidate_name, candidate_values in candidate_series.items():
        exact_total = 0
        near_total = 0
        for network, matrix, metadata in (
            ("proxy6", proxy6_values.T, proxy6_meta),
            ("proxy9", proxy9_values, proxy9_meta),
        ):
            for index in range(matrix.shape[0]):
                item = metrics(candidate_values, matrix[index])
                exact, near = classify(item)
                exact_total += int(exact)
                near_total += int(near)
                meta = metadata.iloc[index]
                rows.append({
                    "candidate": candidate_name,
                    "network": network,
                    "proxy_record_index_1based": index + 1,
                    "proxy_latitude": float(meta.latitude),
                    "proxy_longitude": float(meta.longitude),
                    **item,
                    "absolute_correlation": abs(item["correlation"]) if np.isfinite(item["correlation"]) else np.nan,
                    "exact_affine_alias": exact,
                    "near_exact_standardized_sequence": near,
                })
        exact_counts[candidate_name] = exact_total
        near_counts[candidate_name] = near_total
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output / "all_candidate_proxy_sequence_comparisons.csv", index=False)
    comparisons.sort_values(
        ["candidate", "exact_affine_alias", "near_exact_standardized_sequence", "absolute_correlation"],
        ascending=[True, False, False, False],
    ).groupby("candidate", as_index=False).head(20).to_csv(output / "top_candidate_proxy_sequence_matches.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_149_noaa_temperature_candidate_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "parse_summaries": parse_summaries,
        "candidate_series": list(candidate_series),
        "proxy6_record_count": int(proxy6_values.shape[1]),
        "proxy9_record_count": int(proxy9_values.shape[0]),
        "exact_affine_alias_counts": exact_counts,
        "near_exact_standardized_sequence_counts": near_counts,
        "scientific_role": "derived_external_consistency_benchmark_not_raw_proxy_or_independent_validation",
        "decision": "retain_as_external_trajectory_benchmarks_after_source_overlap_review",
        "locked_1921_1939_values_decoded": 0,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 149 NOAA Temperature Candidate Identity\n\n"
        f"- Candidate numeric series: `{list(candidate_series)}`.\n"
        f"- Exact affine aliases: `{exact_counts}`.\n"
        f"- Near-exact standardized sequences: `{near_counts}`.\n"
        "- Locked 1921--1939 values decoded: `0`.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
