"""Test necessary linear conditions for any monotone allowed-year transition."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import linprog


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
FULL = LOOP / (
    "temperature_candidate_v2_full_posterior_materialization_v1/"
    "temperature_candidate_v2_full_posterior_allowed_years.nc"
)
STATION = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "station_year_candidate_values.csv"
)
SEARCH = LOOP / "temperature_candidate_monotone_allowed_transition_search_v1/summary.json"
BOUNDARY = LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1/summary.json"
RELEASE = LOOP / (
    "temperature_candidate_v2_full_posterior_independent_verification_v2/"
    "RELEASE_ADDENDUM.json"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_monotone_transition_lp_infeasibility_v1"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
YEARS = np.arange(1880, 1921)
PERIODS = ((1880, 1900), (1901, 1920))
HMA_MEAN_DEPARTURE_LIMIT = 0.05


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


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def station_groups(stations: pd.DataFrame) -> list[dict[str, object]]:
    groups = []
    for period_index, (start, end) in enumerate(PERIODS):
        segment = stations[stations["year"].between(start, end)]
        for station_id, group in segment.groupby("station_id"):
            valid = (
                np.isfinite(group["qcf_station_degC"])
                & np.isfinite(group["project_six_member_degC"])
                & np.isfinite(group["candidate_background_correction_degC"])
            )
            group = group.loc[valid].sort_values("year")
            if len(group) < 5:
                continue
            years = group["year"].to_numpy(int)
            indices = years - YEARS[0]
            observed = group["qcf_station_degC"].to_numpy(float)
            project = group["project_six_member_degC"].to_numpy(float)
            correction = group["candidate_background_correction_degC"].to_numpy(float)
            error = project - observed
            centered = years.astype(float) - float(np.mean(years))
            weights = centered / float(np.sum(centered * centered)) * 100.0
            bias_coeff = np.zeros(len(YEARS))
            bias_coeff[indices] = correction / len(indices)
            trend_coeff = np.zeros(len(YEARS))
            trend_coeff[indices] = correction * weights
            groups.append(
                {
                    "period_index": period_index,
                    "station_id": str(station_id),
                    "bias_constant": float(np.mean(error)),
                    "bias_coeff": bias_coeff,
                    "trend_constant": float(weights @ error),
                    "trend_coeff": trend_coeff,
                    "original_abs_trend_error": abs(float(weights @ error)),
                }
            )
    return groups


def add_absolute_constraint(
    rows: list[np.ndarray], limits: list[float], coeff: np.ndarray, constant: float, bound: float
) -> None:
    rows.append(coeff.copy())
    limits.append(float(bound - constant))
    rows.append(-coeff.copy())
    limits.append(float(bound + constant))


def solve(rows: list[np.ndarray], limits: list[float]) -> object:
    bounds = [(1.0, 1.0), *([(0.0, 1.0)] * (len(YEARS) - 2)), (0.0, 0.0)]
    return linprog(
        np.zeros(len(YEARS)),
        A_ub=np.asarray(rows, dtype=float),
        b_ub=np.asarray(limits, dtype=float),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "LP_NECESSARY_CONDITION_CONTRACT.md"
    contract.write_text(
        "# Monotone transition necessary-condition LP\n\n"
        "Represent every annual factor f(1880..1920) directly, with f1880=1, f1920=0, and "
        "1>=f(t)>=f(t+1)>=0. Retain only necessary linear conditions from the frozen gates: "
        "equal-station mean bias must not regress in either period; HMA correction annual changes "
        "must not exceed Candidate v2's own allowed maximum; 1850-1899 HMA mean departure must be "
        "at most 0.05 degC; and enough individual station trend errors must be below 1.1 times the "
        "original period-median threshold for the candidate median to possibly pass. Enumerate all "
        "2-of-3 and 3-of-6 trend-station subsets (60 joint LPs). Deliberately omit MAE, majority-"
        "improvement, and spatial-RMS gates, so infeasibility is a proof for the stricter full "
        "problem. Verify relaxed fixtures. No sealed value is read or generated.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "station": STATION,
        "search": SEARCH,
        "boundary": BOUNDARY,
        "release": RELEASE,
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
    search = json.loads(SEARCH.read_text(encoding="utf-8"))
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if any(item["candidate_id"] != CANDIDATE_ID for item in [search, boundary, release]):
        raise RuntimeError("candidate-id-chain-failed")
    if search["search"]["feasible_before_spatial_rms_count"] != 0:
        raise RuntimeError("search-already-found-feasible-path")

    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station-input")
    groups = station_groups(stations)
    period_groups = [
        [group for group in groups if group["period_index"] == period]
        for period in range(2)
    ]
    if [len(group) for group in period_groups] != [3, 6]:
        raise RuntimeError("station-count-contract-failed")
    with Dataset(FULL) as dataset:
        all_years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64
        )
    if np.any((all_years >= 1921) & (all_years <= 1939)):
        raise RuntimeError("locked-year-in-full-field")
    selected = np.isin(all_years, YEARS)
    hma = weighted_mean(correction[selected], common, lat)
    hma_internal_max = float(
        boundary["hma_allowed_internal_correction_change"]["annual_max_abs_degC"]
    )

    base_rows: list[np.ndarray] = []
    base_limits: list[float] = []
    for index in range(len(YEARS) - 1):
        row = np.zeros(len(YEARS))
        row[index + 1] = 1.0
        row[index] = -1.0
        base_rows.append(row)
        base_limits.append(0.0)
    for index in range(len(YEARS) - 1):
        row = np.zeros(len(YEARS))
        row[index + 1] = hma[index + 1]
        row[index] = -hma[index]
        base_rows.append(row)
        base_limits.append(hma_internal_max)
        base_rows.append(-row)
        base_limits.append(hma_internal_max)
    mean_coeff = np.zeros(len(YEARS))
    mean_coeff[:20] = hma[:20] / 50.0
    mean_constant = -float(np.sum(hma[:20]) / 50.0)
    add_absolute_constraint(
        base_rows,
        base_limits,
        mean_coeff,
        mean_constant,
        HMA_MEAN_DEPARTURE_LIMIT,
    )
    period_trend_thresholds = []
    period_candidate_v2_passing = []
    for current in period_groups:
        bias_coeff = np.mean(np.stack([group["bias_coeff"] for group in current]), axis=0)
        bias_constant = float(np.mean([group["bias_constant"] for group in current]))
        add_absolute_constraint(
            base_rows,
            base_limits,
            bias_coeff,
            bias_constant,
            abs(bias_constant),
        )
        original_median = float(
            np.median([group["original_abs_trend_error"] for group in current])
        )
        threshold = 1.10 * original_median
        period_trend_thresholds.append(threshold)
        candidate_errors = [
            abs(group["trend_constant"] + float(np.sum(group["trend_coeff"])))
            for group in current
        ]
        period_candidate_v2_passing.append(
            [index for index, value in enumerate(candidate_errors) if value <= threshold + 1e-12]
        )

    structural_rows = []
    structural_limits = []
    for index in range(len(YEARS) - 1):
        row = np.zeros(len(YEARS))
        row[index + 1] = 1.0
        row[index] = -1.0
        structural_rows.append(row)
        structural_limits.append(0.0)
    structural_fixture = solve(structural_rows, structural_limits)
    base_solution = solve(base_rows, base_limits)

    ones = np.ones(len(YEARS))
    candidate_v2_fixture = {
        "period_0_passing_station_count": len(period_candidate_v2_passing[0]),
        "period_1_passing_station_count": len(period_candidate_v2_passing[1]),
        "period_0_required_count": 2,
        "period_1_required_count": 3,
        "hma_internal_change_max_abs_degC": float(np.max(np.abs(np.diff(ones * hma)))),
        "hma_internal_limit_degC": hma_internal_max,
        "hma_mean_departure_degC": 0.0,
        "passes_relaxed_conditions_without_1920_zero_endpoint": bool(
            len(period_candidate_v2_passing[0]) >= 2
            and len(period_candidate_v2_passing[1]) >= 3
            and np.max(np.abs(np.diff(hma))) <= hma_internal_max + 1e-12
        ),
    }
    if not structural_fixture.success:
        raise RuntimeError("structural-lp-fixture-failed")
    if not candidate_v2_fixture["passes_relaxed_conditions_without_1920_zero_endpoint"]:
        raise RuntimeError("candidate-v2-relaxed-fixture-failed")

    subset_options = [
        list(itertools.combinations(range(len(period_groups[0])), 2)),
        list(itertools.combinations(range(len(period_groups[1])), 3)),
    ]

    def with_subset_constraints(
        selected_subsets: list[tuple[int, tuple[int, ...]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        rows = [row.copy() for row in base_rows]
        limits = list(base_limits)
        for period_index, subset in selected_subsets:
            threshold = period_trend_thresholds[period_index]
            for station_index in subset:
                group = period_groups[period_index][station_index]
                add_absolute_constraint(
                    rows,
                    limits,
                    group["trend_coeff"],
                    group["trend_constant"],
                    threshold,
                )
        return rows, limits

    joint_rows = []
    joint_feasible = []
    for subset0 in subset_options[0]:
        for subset1 in subset_options[1]:
            rows, limits = with_subset_constraints([(0, subset0), (1, subset1)])
            result = solve(rows, limits)
            record = {
                "period_1880_1900_subset": ";".join(
                    period_groups[0][index]["station_id"] for index in subset0
                ),
                "period_1901_1920_subset": ";".join(
                    period_groups[1][index]["station_id"] for index in subset1
                ),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
            }
            joint_rows.append(record)
            if result.success:
                joint_feasible.append(result.x)
    joint_table = pd.DataFrame(joint_rows)
    joint_table.to_csv(OUTPUT / "joint_trend_subset_lp_results.csv", index=False)

    separate_summary = []
    for period_index in range(2):
        feasible = 0
        for subset in subset_options[period_index]:
            rows, limits = with_subset_constraints([(period_index, subset)])
            feasible += int(solve(rows, limits).success)
        separate_summary.append(
            {
                "period": f"{PERIODS[period_index][0]}-{PERIODS[period_index][1]}",
                "subset_count": len(subset_options[period_index]),
                "feasible_subset_lp_count": feasible,
            }
        )
    pd.DataFrame(separate_summary).to_csv(
        OUTPUT / "separate_period_trend_subset_lp_summary.csv", index=False
    )

    all_joint_infeasible = len(joint_feasible) == 0
    classification = (
        "necessary_linear_conditions_infeasible_proves_no_monotone_allowed_year_transition_can_pass_full_gates"
        if all_joint_infeasible
        else "necessary_linear_conditions_feasible_full_nonlinear_problem_remains_unresolved"
    )
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-monotone-transition-lp-infeasibility-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "logic": (
            "Passing the full median trend gates requires at least 2 of 3 and 3 of 6 station "
            "trend errors to lie below the period threshold. Every such station subset is "
            "enumerated. The LP omits stricter MAE, station-fraction, and spatial-RMS gates; "
            "therefore infeasibility of all relaxed LPs implies infeasibility of the full problem."
        ),
        "fixtures": {
            "structural_monotone_endpoint_lp_feasible": bool(structural_fixture.success),
            "base_without_station_trend_subsets_feasible": bool(base_solution.success),
            "candidate_v2_without_zero_endpoint": candidate_v2_fixture,
        },
        "trend_thresholds_degC_per_century": {
            "1880-1900": period_trend_thresholds[0],
            "1901-1920": period_trend_thresholds[1],
        },
        "joint_lp_count": len(joint_rows),
        "joint_feasible_lp_count": len(joint_feasible),
        "separate_period_results": separate_summary,
        "search_crosscheck": {
            "evaluated_path_count": search["search"]["total_evaluated_path_count"],
            "feasible_path_count": search["search"]["feasible_before_spatial_rms_count"],
            "best_max_normalized_violation": search["best"]["score_max_normalized_violation"],
        },
        "decision": (
            "Close all monotone transitions confined to 1880-1920. A continuous corrected "
            "forcing requires a transition inside 1921-1939 under a new explicit contract, or "
            "Candidate v2 remains allowed-year-only."
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
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Monotone transition necessary-condition LP\n\n"
        f"Classification: `{classification}`.\n\n"
        f"- Joint station-subset LPs: `{len(joint_rows)}`; feasible: `{len(joint_feasible)}`.\n"
        f"- Base LP before station trend subsets is feasible: `{str(base_solution.success).lower()}`.\n"
        f"- Separate period feasible subset counts: `{separate_summary[0]['feasible_subset_lp_count']}/{separate_summary[0]['subset_count']}` and "
        f"`{separate_summary[1]['feasible_subset_lp_count']}/{separate_summary[1]['subset_count']}`.\n"
        "- The LP deliberately omits stricter nonlinear gates, so all-joint infeasibility closes the full monotone allowed-year class.\n"
        "- No sealed-period value was read or generated; Candidate v2 and production remain unchanged.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
