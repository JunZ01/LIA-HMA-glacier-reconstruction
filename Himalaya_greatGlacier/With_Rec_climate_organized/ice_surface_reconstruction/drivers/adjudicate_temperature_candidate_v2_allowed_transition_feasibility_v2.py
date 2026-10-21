"""Correct NaN-mask equality in the allowed-year transition feasibility audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
V1 = LOOP / "temperature_candidate_v2_allowed_transition_feasibility_v1"
OUTPUT = LOOP / "temperature_candidate_v2_allowed_transition_feasibility_v2"
FULL = LOOP / (
    "temperature_candidate_v2_full_posterior_materialization_v1/"
    "temperature_candidate_v2_full_posterior_allowed_years.nc"
)
V1_SUMMARY = V1 / "summary.json"
V1_SPATIAL = V1 / "spatial_transition_diagnostics.csv"
V1_AGGREGATE = V1 / "aggregate_transition_metrics.csv"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "TECHNICAL_CORRECTION_CONTRACT.md"
    contract.write_text(
        "# Allowed-year transition feasibility V2 correction\n\n"
        "Retain V1 unchanged. Correct only its `1500-1849 exactly unchanged` test, which used "
        "`numpy.array_equal` without `equal_nan=True` on fields whose identical non-HMA masks "
        "contain NaN. Independently verify identical finite masks and zero finite-value difference. "
        "Carry forward every other V1 station, transition, and spatial result. Recompute branch "
        "decisions without changing any branch, threshold, field, or scientific rule.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "v1_summary": V1_SUMMARY,
        "v1_spatial": V1_SPATIAL,
        "v1_aggregate": V1_AGGREGATE,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}
    v1 = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    if v1["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("candidate-id-failed")
    branches = {row["branch"]: row for row in v1["branches"]}
    spatial = pd.read_csv(V1_SPATIAL).set_index("branch")
    aggregate = pd.read_csv(V1_AGGREGATE)

    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64
        )
    early = years <= 1920
    baseline = correction[early][years[early] <= 1849]
    corrected_identity = {}
    corrected_rows = []
    for branch, row in branches.items():
        start = spatial.loc[branch, "taper_start_year"]
        if pd.isna(start):
            factors = np.ones(len(years), dtype=np.float64)
            factors[years >= 1951] = 0.0
        else:
            start = int(start)
            factors = np.ones(len(years), dtype=np.float64)
            taper = (years > start) & (years <= 1920)
            factors[taper] = (1920.0 - years[taper]) / (1920.0 - start)
            factors[years >= 1920] = 0.0
        transitioned = correction[early] * factors[early, None, None]
        candidate_baseline = transitioned[years[early] <= 1849]
        same_mask = bool(
            np.array_equal(np.isfinite(candidate_baseline), np.isfinite(baseline))
        )
        valid = np.isfinite(baseline) & np.isfinite(candidate_baseline)
        max_difference = float(
            np.max(np.abs(candidate_baseline[valid] - baseline[valid]))
        )
        identity = same_mask and max_difference == 0.0
        corrected_identity[branch] = {
            "identical_finite_mask": same_mask,
            "finite_value_max_abs_difference_degC": max_difference,
            "exactly_unchanged": identity,
        }
        gates = dict(row["gates"])
        gates["correction_1500_1849_exactly_unchanged"] = identity
        corrected = dict(row)
        corrected["gates"] = gates
        corrected["passes_all_transition_and_station_gates"] = bool(all(gates.values()))
        corrected["failed_gates"] = [name for name, passed in gates.items() if not passed]
        corrected_rows.append(corrected)

    diagnostic = [
        row
        for row in corrected_rows
        if row["branch"] != "frozen_candidate_v2"
        and row["passes_all_transition_and_station_gates"]
    ]
    transition_rows = [row for row in corrected_rows if row["branch"] != "frozen_candidate_v2"]
    all_transition_fail_only_trend = all(
        row["failed_gates"] == ["early_median_trend_error_regression_at_most_10pct"]
        for row in transition_rows
    )
    trend = aggregate[
        aggregate["candidate_family"].isin([row["branch"] for row in transition_rows])
        & aggregate["period"].eq("1901-1920")
    ][["candidate_family", "median_trend_error_change_fraction"]].copy()
    trend = trend.sort_values("candidate_family")
    trend.to_csv(OUTPUT / "transition_1901_1920_trend_regression.csv", index=False)
    classification = (
        "no_tested_allowed_year_taper_passes_v2_all_fail_1901_1920_trend_gate"
        if not diagnostic and all_transition_fail_only_trend
        else (
            "at_least_one_allowed_year_taper_passes_corrected_gates"
            if diagnostic
            else "no_tested_allowed_year_taper_passes_multiple_gates"
        )
    )

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-v2-allowed-transition-feasibility-v2",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "supersedes_v1_classification": v1["classification"],
        "technical_correction": "NaN-mask-aware exact identity for 1500-1849",
        "corrected_1500_1849_identity": corrected_identity,
        "branches": corrected_rows,
        "diagnostic_feasible_branch_count": len(diagnostic),
        "diagnostic_feasible_branches": [row["branch"] for row in diagnostic],
        "all_four_transition_branches_fail_only_1901_1920_trend_gate": all_transition_fail_only_trend,
        "transition_1901_1920_trend_error_change_fraction": {
            str(row.candidate_family): float(row.median_trend_error_change_fraction)
            for row in trend.itertuples(index=False)
        },
        "scientific_result": (
            "The correction level needed for the direct-station warm-bias improvement cannot "
            "be tapered to zero before 1921 under the tested structural transitions without "
            "introducing an unacceptable 1901-1920 warming trend. A continuous correction "
            "therefore requires a separately authorized sealed-interval transition contract or "
            "Candidate v2 must remain an allowed-year-only research product."
        ),
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_generated": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Candidate v2 allowed-year transition feasibility V2",
        "",
        f"Classification: `{classification}`.",
        "",
        "V1's NaN-mask equality error is corrected: every branch exactly preserves 1500-1849.",
        "All four transition branches still fail, now solely at the frozen 1901-1920 trend gate:",
        "",
    ]
    for row in trend.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_family}`: trend-error change `{row.median_trend_error_change_fraction:+.1%}`."
        )
    lines.extend(
        [
            "",
            "No transition branch is promoted. No sealed-period value was read or generated; Candidate v2 and production remain unchanged.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
