"""Resolve the two relaxed LP-feasible monotone transition polytopes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import prove_temperature_candidate_monotone_transition_lp_infeasibility_v1 as lpbase
import search_temperature_candidate_monotone_allowed_transition_v1 as searchbase


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
FULL = lpbase.FULL
STATION = lpbase.STATION
LP_V1 = LOOP / "temperature_candidate_monotone_transition_lp_infeasibility_v1"
LP_SUMMARY = LP_V1 / "summary.json"
LP_JOINT = LP_V1 / "joint_trend_subset_lp_results.csv"
SEARCH_SUMMARY = searchbase.OUTPUT / "summary.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_monotone_lp_feasible_polytopes_v2"

CANDIDATE_ID = lpbase.CANDIDATE_ID
RANDOM_SEED = 20260829
RANDOM_OBJECTIVES_PER_POLYTOPE = 500
INTERIOR_SAMPLES = 50000
TOP_KEEP = 200


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


def build_lp_system(
    groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    subset_ids: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    period_groups = [
        [group for group in groups if group["period_index"] == period]
        for period in range(2)
    ]
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for index in range(len(lpbase.YEARS) - 1):
        row = np.zeros(len(lpbase.YEARS))
        row[index + 1] = 1.0
        row[index] = -1.0
        rows.append(row)
        limits.append(0.0)
    for index in range(len(lpbase.YEARS) - 1):
        row = np.zeros(len(lpbase.YEARS))
        row[index + 1] = hma[index + 1]
        row[index] = -hma[index]
        rows.extend([row, -row])
        limits.extend([hma_internal_max, hma_internal_max])
    mean_coeff = np.zeros(len(lpbase.YEARS))
    mean_coeff[:20] = hma[:20] / 50.0
    mean_constant = -float(np.sum(hma[:20]) / 50.0)
    lpbase.add_absolute_constraint(
        rows,
        limits,
        mean_coeff,
        mean_constant,
        lpbase.HMA_MEAN_DEPARTURE_LIMIT,
    )
    for period_index, current in enumerate(period_groups):
        bias_coeff = np.mean(np.stack([group["bias_coeff"] for group in current]), axis=0)
        bias_constant = float(np.mean([group["bias_constant"] for group in current]))
        lpbase.add_absolute_constraint(
            rows, limits, bias_coeff, bias_constant, abs(bias_constant)
        )
        threshold = 1.10 * float(
            np.median([group["original_abs_trend_error"] for group in current])
        )
        selected = set(subset_ids[period_index])
        if len(selected) != (2 if period_index == 0 else 3):
            raise RuntimeError("subset-size-failed")
        matched = [group for group in current if group["station_id"] in selected]
        if len(matched) != len(selected):
            raise RuntimeError("subset-station-id-failed")
        for group in matched:
            lpbase.add_absolute_constraint(
                rows,
                limits,
                group["trend_coeff"],
                group["trend_constant"],
                threshold,
            )
    return np.asarray(rows, dtype=float), np.asarray(limits, dtype=float)


def lp_solve(objective: np.ndarray, matrix: np.ndarray, limits: np.ndarray) -> object:
    bounds = [(1.0, 1.0), *([(0.0, 1.0)] * (len(lpbase.YEARS) - 2)), (0.0, 0.0)]
    return linprog(
        objective,
        A_ub=matrix,
        b_ub=limits,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )


def spatial_rms_departure(
    factors: np.ndarray,
    correction_fields: np.ndarray,
    common: np.ndarray,
    lat: np.ndarray,
) -> float:
    mean_delta = np.sum(
        (factors[:20] - 1.0)[:, None, None] * correction_fields[:20], axis=0
    ) / 50.0
    return float(searchbase.weighted_rms(mean_delta[None], common, lat)[0])


def full_record(
    label: str,
    factors: np.ndarray,
    groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    correction_fields: np.ndarray,
    common: np.ndarray,
    lat: np.ndarray,
) -> dict[str, object]:
    result = searchbase.evaluate_factors(
        factors[None], groups, hma, hma_internal_max
    )
    base = searchbase.row_record(label, factors[None], result, 0)
    rms = spatial_rms_departure(factors, correction_fields, common, lat)
    violations = result["violations"][0]
    names = [
        "bias_1880_1900",
        "mae_1880_1900",
        "improving_fraction_1880_1900",
        "trend_1880_1900",
        "bias_1901_1920",
        "mae_1901_1920",
        "improving_fraction_1901_1920",
        "trend_1901_1920",
        "hma_annual_smoothness",
        "hma_1850_1899_mean_departure",
    ]
    positive = {name: float(value) for name, value in zip(names, violations) if value > 1e-10}
    rms_violation = max(rms / 0.05 - 1.0, 0.0)
    if rms_violation > 1e-10:
        positive["hma_1850_1899_spatial_rms_departure"] = rms_violation
    base.update(
        {
            "hma_1850_1899_spatial_rms_departure_degC": rms,
            "full_max_normalized_violation": max(
                float(result["score"][0]), rms_violation
            ),
            "full_sum_positive_normalized_violation": float(
                result["sum_violation"][0] + rms_violation
            ),
            "positive_normalized_gate_violations": positive,
            "passes_all_full_gates": not positive,
        }
    )
    return base


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "LP_POLYTOPE_RESOLUTION_CONTRACT.md"
    contract.write_text(
        "# Relaxed LP-feasible polytope resolution\n\n"
        "Supersede only the unsupported closure statement in LP V1. Reconstruct its two relaxed "
        "feasible station-subset polytopes. For each, solve 500 fixed-seed random linear objectives "
        "to sample boundary vertices, sample 50,000 convex interior paths from the unique vertices, "
        "evaluate every path against all nonlinear station gates and HMA temporal gates, evaluate "
        "full spatial RMS for finalists, and run multi-start 41-factor SLSQP under the exact LP "
        "constraints. A relaxed LP path is not feasible unless every original nonlinear and "
        "spatial gate passes. No sealed value is read or generated; no candidate is promoted.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "station": STATION,
        "lp_v1_summary": LP_SUMMARY,
        "lp_v1_joint": LP_JOINT,
        "search_summary": SEARCH_SUMMARY,
        "lp_v1_driver": Path(lpbase.__file__).resolve(),
        "search_driver": Path(searchbase.__file__).resolve(),
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
    lp_summary = json.loads(LP_SUMMARY.read_text(encoding="utf-8"))
    search_summary = json.loads(SEARCH_SUMMARY.read_text(encoding="utf-8"))
    if lp_summary["candidate_id"] != CANDIDATE_ID or search_summary["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("candidate-id-chain-failed")
    feasible_table = pd.read_csv(LP_JOINT)
    feasible_table = feasible_table[feasible_table["success"]].copy()
    if len(feasible_table) != 2:
        raise RuntimeError("expected-two-relaxed-feasible-polytopes")

    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = lpbase.station_groups(stations)
    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64
        )
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-year-in-candidate")
    transition = np.isin(years, lpbase.YEARS)
    correction_fields = correction[transition]
    hma = lpbase.weighted_mean(correction_fields, common, lat)
    hma_internal_max = float(
        json.loads(lpbase.BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )

    rng = np.random.default_rng(RANDOM_SEED)
    polytopes = []
    all_vertices = []
    vertex_labels = []
    systems = []
    for polytope_index, row in enumerate(feasible_table.itertuples(index=False)):
        subset_ids = (
            tuple(str(row.period_1880_1900_subset).split(";")),
            tuple(str(row.period_1901_1920_subset).split(";")),
        )
        matrix, limits = build_lp_system(groups, hma, hma_internal_max, subset_ids)
        systems.append((matrix, limits))
        vertices = []
        zero = lp_solve(np.zeros(len(lpbase.YEARS)), matrix, limits)
        if not zero.success:
            raise RuntimeError("relaxed-polytope-no-longer-feasible")
        vertices.append(zero.x)
        success_count = 1
        for _ in range(RANDOM_OBJECTIVES_PER_POLYTOPE):
            result = lp_solve(rng.normal(size=len(lpbase.YEARS)), matrix, limits)
            if result.success:
                vertices.append(result.x)
                success_count += 1
        unique = np.unique(np.round(np.asarray(vertices), 10), axis=0)
        polytopes.append(
            {
                "polytope_index": polytope_index,
                "period_1880_1900_subset": subset_ids[0],
                "period_1901_1920_subset": subset_ids[1],
                "random_objective_success_count": success_count,
                "unique_vertex_count": int(len(unique)),
            }
        )
        all_vertices.extend(unique)
        vertex_labels.extend(
            [f"polytope_{polytope_index}_vertex_{index}" for index in range(len(unique))]
        )
    all_vertices = np.asarray(all_vertices)

    candidates = [row.copy() for row in all_vertices]
    labels = list(vertex_labels)
    for sample_index in range(INTERIOR_SAMPLES):
        count = int(rng.choice([2, 3, 5, min(10, len(all_vertices))]))
        indices = rng.choice(len(all_vertices), size=count, replace=False)
        weights = rng.dirichlet(np.ones(count))
        candidates.append(weights @ all_vertices[indices])
        labels.append(f"convex_interior_{sample_index}")
    candidates = np.asarray(candidates)

    top = []
    full_feasible_count = 0
    for left in range(0, len(candidates), 5000):
        current = candidates[left : left + 5000]
        current_labels = labels[left : left + len(current)]
        result = searchbase.evaluate_factors(current, groups, hma, hma_internal_max)
        order = np.lexsort((result["sum_violation"], result["score"]))[:TOP_KEEP]
        for index in order:
            record = searchbase.row_record(
                current_labels[index], current, result, int(index)
            )
            top.append(
                (
                    record["score_max_normalized_violation"],
                    record["sum_positive_normalized_violation"],
                    current_labels[index],
                    current[index].copy(),
                    record,
                )
            )
        full_feasible_count += int(np.sum(result["score"] <= 1e-12))
        top = sorted(top, key=lambda item: (item[0], item[1]))[:TOP_KEEP]

    linear_constraints = [
        LinearConstraint(matrix, -np.inf, limits) for matrix, limits in systems
    ]
    lower = np.zeros(len(lpbase.YEARS))
    upper = np.ones(len(lpbase.YEARS))
    lower[0] = upper[0] = 1.0
    lower[-1] = upper[-1] = 0.0
    bounds = Bounds(lower, upper)

    def objective(values: np.ndarray) -> float:
        record = full_record(
            "objective",
            values,
            groups,
            hma,
            hma_internal_max,
            correction_fields,
            common,
            lat,
        )
        return float(
            record["full_max_normalized_violation"]
            + 0.02 * record["full_sum_positive_normalized_violation"]
        )

    optimization_rows = []
    optimized = []
    starts = top[:20]
    for start_index, item in enumerate(starts):
        best_result = None
        for polytope_index, constraint in enumerate(linear_constraints):
            if np.max(systems[polytope_index][0] @ item[3] - systems[polytope_index][1]) > 1e-7:
                continue
            result = minimize(
                objective,
                item[3],
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint],
                options={"maxiter": 1200, "ftol": 1e-12, "disp": False},
            )
            record = full_record(
                f"slsqp_{start_index:02d}_p{polytope_index}",
                result.x,
                groups,
                hma,
                hma_internal_max,
                correction_fields,
                common,
                lat,
            )
            optimization_rows.append(
                {
                    "start_index": start_index,
                    "polytope_index": polytope_index,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "objective": float(result.fun),
                    "full_max_normalized_violation": record["full_max_normalized_violation"],
                    "passes_all_full_gates": record["passes_all_full_gates"],
                }
            )
            optimized.append((record, result.x.copy()))
            if best_result is None or record["full_max_normalized_violation"] < best_result[0]["full_max_normalized_violation"]:
                best_result = (record, result.x.copy())
    pd.DataFrame(optimization_rows).to_csv(OUTPUT / "full_gate_constrained_optimization.csv", index=False)

    finalists = []
    for item in top[:TOP_KEEP]:
        finalists.append(
            (
                full_record(
                    item[2], item[3], groups, hma, hma_internal_max,
                    correction_fields, common, lat,
                ),
                item[3],
            )
        )
    finalists.extend(optimized)
    finalists.sort(
        key=lambda item: (
            item[0]["full_max_normalized_violation"],
            item[0]["full_sum_positive_normalized_violation"],
        )
    )
    full_pass = [item for item in finalists if item[0]["passes_all_full_gates"]]
    best_record, best_factors = finalists[0]
    pd.DataFrame([item[0] for item in finalists[:TOP_KEEP]]).to_csv(
        OUTPUT / "top_full_gate_candidates.csv", index=False
    )
    pd.DataFrame(
        {
            "year": lpbase.YEARS,
            "best_factor": best_factors,
        }
    ).to_csv(OUTPUT / "best_full_gate_factor.csv", index=False)
    pd.DataFrame(polytopes).to_csv(OUTPUT / "relaxed_polytope_inventory.csv", index=False)

    classification = (
        "full_gate_monotone_transition_found_inside_relaxed_lp_polytope_requires_independent_validation"
        if full_pass
        else "two_relaxed_lp_polytopes_fail_full_nonlinear_and_spatial_gates_after_targeted_resolution"
    )
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-monotone-lp-feasible-polytopes-v2",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "supersedes_lp_v1_decision": lp_summary["decision"],
        "relaxed_feasible_polytope_count": len(polytopes),
        "polytope_inventory": polytopes,
        "unique_vertex_count_total": int(len(all_vertices)),
        "convex_interior_sample_count": INTERIOR_SAMPLES,
        "nonlinear_temporal_gate_feasible_sample_count": full_feasible_count,
        "constrained_optimization_run_count": len(optimization_rows),
        "full_gate_feasible_finalist_count": len(full_pass),
        "best": best_record,
        "decision": (
            "Do not close the monotone class from LP V1 alone. "
            + (
                "Advance the discovered full-gate path only to independent non-promotion validation."
                if full_pass
                else "The only relaxed LP-feasible regions do not satisfy the omitted nonlinear/spatial gates; retain Candidate v2 as allowed-year-only unless a sealed-period transition is authorized."
            )
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
        "# Relaxed LP-feasible monotone polytopes V2\n\n"
        f"Classification: `{classification}`.\n\n"
        f"- Relaxed polytopes: `{len(polytopes)}`; unique sampled vertices: `{len(all_vertices)}`.\n"
        f"- Convex interior paths: `{INTERIOR_SAMPLES}`.\n"
        f"- Full temporal-gate feasible samples: `{full_feasible_count}`.\n"
        f"- Full-gate feasible finalists after spatial RMS and optimization: `{len(full_pass)}`.\n"
        f"- Best maximum normalized violation: `{best_record['full_max_normalized_violation']:.4f}`.\n"
        f"- Best remaining violations: `{json.dumps(best_record['positive_normalized_gate_violations'], sort_keys=True)}`.\n"
        "- LP V1's premature closure decision is superseded. No sealed value was read or generated.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
