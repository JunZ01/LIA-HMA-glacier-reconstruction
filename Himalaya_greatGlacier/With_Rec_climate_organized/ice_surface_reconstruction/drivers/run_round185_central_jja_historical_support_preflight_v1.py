"""Audit historical coverage for the frozen Round 156 central-JJA network."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd


PALEO_WINDOWS = [(start, start + 49) for start in range(1550, 1900, 50)]
LOCKED_START = 1921
LOCKED_END = 1939


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_year(token: str) -> int:
    value = float(token)
    year = int(round(value))
    if not math.isfinite(value) or abs(value - year) > 1e-6:
        raise ValueError(f"invalid-year-token:{token!r}")
    return year


def token_is_finite(token: str) -> bool:
    text = token.strip()
    if not text:
        return False
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def longest_consecutive_run(years: set[int]) -> int:
    longest = 0
    current = 0
    previous = None
    for year in sorted(years):
        if previous is not None and year == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = year
    return longest


def main() -> None:
    root = project_root()
    base = root / "validation_data/results/paleo_surface_validation_v0"
    drivers = (
        root
        / "Himalaya_greatGlacier/With_Rec_climate_organized"
        / "ice_surface_reconstruction/drivers"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=base
        / "climate_reconstruction_improvement_loop_v2"
        / "round_13_new_lmr_temperature_network_validation_v2"
        / "network_manifest.csv",
    )
    parser.add_argument(
        "--lmr-csv",
        type=Path,
        default=Path(
            "D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master/"
            "proxy-data/LMR_NCDC_database_v.0.2.0/lmr_data.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base
        / "climate_reconstruction_improvement_loop_v2"
        / "round_185_central_jja_historical_support_preflight_v1",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    shutil.copyfile(
        drivers / "round185_central_jja_historical_support_preflight_v1_contract.md",
        args.output / "ROUND_CONTRACT.md",
    )

    inputs = {"network_manifest": args.manifest, "lmr_csv": args.lmr_csv}
    before = {name: sha256(path) for name, path in inputs.items()}

    metadata = pd.read_csv(args.manifest)
    selected = metadata.loc[
        (metadata["season_code"] == "JJA")
        & (metadata["longitude"] >= 80.0)
        & (metadata["longitude"] < 90.0)
    ].copy()
    if len(selected) != 41:
        raise RuntimeError(f"unexpected-central-jja-count:{len(selected)}")

    with args.lmr_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: index for index, name in enumerate(header)}
        identities = selected["formal_identity"].tolist()
        missing = [identity for identity in identities if identity not in positions]
        if missing:
            raise RuntimeError(f"unmatched-formal-identities:{missing}")
        columns = [positions[identity] for identity in identities]
        if len(set(columns)) != len(columns):
            raise RuntimeError("non-unique-csv-column-match")

        finite_years = {identity: set() for identity in identities}
        locked_rows_skipped = 0
        relevant_rows = 0
        for row in reader:
            if not row:
                continue
            year = decode_year(row[0])
            if LOCKED_START <= year <= LOCKED_END:
                locked_rows_skipped += 1
                continue
            if not (1550 <= year <= 1899 or 1940 <= year <= 2000):
                continue
            relevant_rows += 1
            for identity, column in zip(identities, columns):
                if column < len(row) and token_is_finite(row[column]):
                    finite_years[identity].add(year)

    rows = []
    for _, item in selected.iterrows():
        identity = item["formal_identity"]
        years = finite_years[identity]
        paleo_counts = {
            f"finite_{start}_{end}": sum(start <= year <= end for year in years)
            for start, end in PALEO_WINDOWS
        }
        modern_count = sum(1940 <= year <= 2000 for year in years)
        long_core = all(value >= 40 for value in paleo_counts.values()) and modern_count >= 40
        rows.append(
            {
                "network_index_1based": int(item["network_index_1based"]),
                "formal_identity": identity,
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "measurement": item["measurement"],
                "csv_index_0based": int(positions[identity]),
                "first_allowed_finite_year": min(years) if years else None,
                "last_allowed_finite_year": max(years) if years else None,
                "longest_allowed_consecutive_run": longest_consecutive_run(years),
                **paleo_counts,
                "finite_1940_2000": modern_count,
                "coverage_only_long_core": long_core,
            }
        )

    coverage = pd.DataFrame(rows)
    long_core = coverage.loc[coverage["coverage_only_long_core"]].copy()
    window_summary = []
    for start, end in PALEO_WINDOWS:
        column = f"finite_{start}_{end}"
        window_summary.append(
            {
                "window_start": start,
                "window_end": end,
                "full_network_records_with_any_value": int((coverage[column] > 0).sum()),
                "full_network_records_with_at_least_40_years": int((coverage[column] >= 40).sum()),
                "long_core_records_with_any_value": int((long_core[column] > 0).sum()),
                "long_core_records_with_at_least_40_years": int((long_core[column] >= 40).sum()),
            }
        )

    support_feasible = bool(
        len(long_core) >= 3
        and all(row["long_core_records_with_any_value"] >= 3 for row in window_summary)
    )
    after = {name: sha256(path) for name, path in inputs.items()}
    source_unchanged = before == after
    if not source_unchanged:
        raise RuntimeError("source-fingerprint-changed")

    summary = {
        "status": "complete",
        "decision": (
            "historical_support_feasible_for_prospective_chongce_shape_test"
            if support_feasible
            else "historical_support_inadequate_close_central_jja_branch"
        ),
        "selected_record_count": len(selected),
        "exact_header_match_count": len(columns),
        "csv_column_count": len(header),
        "relevant_allowed_rows_decoded": relevant_rows,
        "locked_rows_skipped_before_proxy_conversion": locked_rows_skipped,
        "locked_1921_1939_proxy_value_accessed": False,
        "long_core_record_count": len(long_core),
        "support_feasible": support_feasible,
        "long_core_rule": "at least 40 finite years in every 1550-1899 50-year window and at least 40 finite years in 1940-2000",
        "window_support": window_summary,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "sources_unchanged": source_unchanged,
    }

    coverage.to_csv(args.output / "record_coverage.csv", index=False)
    long_core.to_csv(args.output / "coverage_only_long_core.csv", index=False)
    pd.DataFrame(window_summary).to_csv(args.output / "window_support.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {name: str(path) for name, path in inputs.items()},
                "sha256_before": before,
                "sha256_after": after,
                "sources_unchanged": source_unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

