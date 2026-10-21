"""Search arbitrary monotone allowed-year tapers for a feasible Candidate v2 transition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
LINEAR_V2 = LOOP / "temperature_candidate_v2_allowed_transition_feasibility_v2/summary.json"
BOUNDARY = LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1/summary.json"
RELEASE = LOOP / (
    "temperature_candidate_v2_full_posterior_independent_verification_v2/"
    "RELEASE_ADDENDUM.json"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_monotone_allowed_transition_search_v1"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
TRANSITION_YEARS = np.arange(1880, 1921)
PERIODS = ((1880, 1900), (1901, 1920))
RANDOM_SEED = 20260828
DIRICHLET_ALPHA = (0.1, 0.3, 1.0, 3.0, 10.0)
SAMPLES_PER_ALPHA = 40000
CHUNK_SIZE = 5000
TOP_KEEP = 100
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


def weighted_rms(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.sqrt(np.nansum(values * values * weights[None], axis=(1, 2)) / np.sum(weights))


def build_station_groups(stations: pd.DataFrame) -> list[dict[str, object]]:
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
            year_indices = years - TRANSITION_YEARS[0]
            observed = group["qcf_station_degC"].to_numpy(float)
            project = group["project_six_member_degC"].to_numpy(float)
            correction = group["candidate_background_correction_degC"].to_numpy(float)
            centered = years.astype(float) - float(np.mean(years))
            slope_weights = centered / float(np.sum(centered * centered)) * 100.0
            original_error = project - observed
            groups.append(
                {
                    "period_index": period_index,
                    "station_id": str(station_id),
                    "year_indices": year_indices,
                    "original_error": original_error,
                    "correction": correction,
                    "slope_weights": slope_weights,
                    "original_bias": float(np.mean(original_error)),
                    "original_mae": float(np.mean(np.abs(original_error))),
                    "original_abs_trend_error": abs(float(slope_weights @ original_error)),
                }
            )
    return groups


def evaluate_factors(
    factors: np.ndarray,
    groups: list[dict[str, object]],
    hma_correction: np.ndarray,
    hma_internal_max: float,
) -> dict[str, np.ndarray]:
    count = len(factors)
    period_results = []
    constraint_columns = []
    for period_index in range(2):
        current = [group for group in groups if group["period_index"] == period_index]
        biases = np.empty((count, len(current)), dtype=np.float64)
        maes = np.empty_like(biases)
        trends = np.empty_like(biases)
        original_biases = np.asarray([group["original_bias"] for group in current])
        original_maes = np.asarray([group["original_mae"] for group in current])
        original_trends = np.asarray([group["original_abs_trend_error"] for group in current])
        for column, group in enumerate(current):
            indices = group["year_indices"]
            candidate_error = (
                group["original_error"][None]
                + factors[:, indices] * group["correction"][None]
            )
            biases[:, column] = np.mean(candidate_error, axis=1)
            maes[:, column] = np.mean(np.abs(candidate_error), axis=1)
            trends[:, column] = np.abs(candidate_error @ group["slope_weights"])
        aggregate_bias = np.mean(biases, axis=1)
        aggregate_original_bias = float(np.mean(original_biases))
        median_mae = np.median(maes, axis=1)
        original_median_mae = float(np.median(original_maes))
        improving_fraction = np.mean(maes < original_maes[None], axis=1)
        median_trend = np.median(trends, axis=1)
        original_median_trend = float(np.median(original_trends))
        metrics = {
            "aggregate_bias": aggregate_bias,
            "absolute_bias_improvement": (
                abs(aggregate_original_bias) - np.abs(aggregate_bias)
            ) / abs(aggregate_original_bias),
            "median_mae": median_mae,
            "median_mae_improvement": (
                original_median_mae - median_mae
            ) / original_median_mae,
            "improving_fraction": improving_fraction,
            "median_trend": median_trend,
            "trend_error_change": (
                median_trend - original_median_trend
            ) / original_median_trend,
        }
        period_results.append(metrics)
        constraint_columns.extend(
            [
                np.abs(aggregate_bias) / abs(aggregate_original_bias) - 1.0,
                median_mae / original_median_mae - 1.0,
                (0.5 - improving_fraction) / 0.5,
                median_trend / (1.10 * original_median_trend) - 1.0,
            ]
        )
    transitioned_hma = factors * hma_correction[None]
    annual_max = np.max(np.abs(np.diff(transitioned_hma, axis=1)), axis=1)
    mean_departure = np.abs(
        np.sum(
            (factors[:, :20] - 1.0) * hma_correction[None, :20], axis=1
        )
        / 50.0
    )
    constraint_columns.extend(
        [
            annual_max / hma_internal_max - 1.0,
            mean_departure / HMA_MEAN_DEPARTURE_LIMIT - 1.0,
        ]
    )
    violations = np.column_stack(constraint_columns)
    positive = np.maximum(violations, 0.0)
    score = np.max(positive, axis=1)
    sum_violation = np.sum(positive, axis=1)
    return {
        "score": score,
        "sum_violation": sum_violation,
        "violations": violations,
        "period_0": period_results[0],
        "period_1": period_results[1],
        "hma_annual_max": annual_max,
        "hma_mean_departure": mean_departure,
    }


def deterministic_factors() -> tuple[np.ndarray, list[str]]:
    values = []
    labels = []
    for transition in range(1881, 1921):
        current = np.ones(len(TRANSITION_YEARS))
        current[TRANSITION_YEARS >= transition] = 0.0
        values.append(current)
        labels.append(f"step_at_{transition}")
    for start in range(1880, 1920):
        current = np.ones(len(TRANSITION_YEARS))
        selected = TRANSITION_YEARS > start
        current[selected] = (1920.0 - TRANSITION_YEARS[selected]) / (1920.0 - start)
        values.append(current)
        labels.append(f"linear_{start}_1920")
        x = np.clip((TRANSITION_YEARS - start) / (1920.0 - start), 0.0, 1.0)
        for power in (0.5, 2.0, 3.0):
            values.append(1.0 - x**power)
            labels.append(f"power_{power:g}_{start}_1920")
        values.append(1.0 - (3.0 * x * x - 2.0 * x * x * x))
        labels.append(f"smoothstep_{start}_1920")
    result = np.asarray(values, dtype=np.float64)
    result[:, 0] = 1.0
    result[:, -1] = 0.0
    return result, labels


def row_record(
    label: str, factors: np.ndarray, result: dict[str, np.ndarray], index: int
) -> dict[str, object]:
    p0 = result["period_0"]
    p1 = result["period_1"]
    return {
        "label": label,
        "score_max_normalized_violation": float(result["score"][index]),
        "sum_positive_normalized_violation": float(result["sum_violation"][index]),
        "bias_improvement_1880_1900": float(p0["absolute_bias_improvement"][index]),
        "bias_improvement_1901_1920": float(p1["absolute_bias_improvement"][index]),
        "mae_improvement_1880_1900": float(p0["median_mae_improvement"][index]),
        "mae_improvement_1901_1920": float(p1["median_mae_improvement"][index]),
        "improving_fraction_1880_1900": float(p0["improving_fraction"][index]),
        "improving_fraction_1901_1920": float(p1["improving_fraction"][index]),
        "trend_error_change_1880_1900": float(p0["trend_error_change"][index]),
        "trend_error_change_1901_1920": float(p1["trend_error_change"][index]),
        "hma_annual_change_max_abs_degC": float(result["hma_annual_max"][index]),
        "hma_1850_1899_mean_departure_from_v2_degC": float(result["hma_mean_departure"][index]),
        "factor_max_annual_drop": float(np.max(-np.diff(factors[index]))),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "MONOTONE_SEARCH_CONTRACT.md"
    contract.write_text(
        "# Arbitrary monotone allowed-year transition search\n\n"
        "Search correction factors f(1880..1920) satisfying f(1880)=1, f(1920)=0, "
        "and 1>=f(t)>=f(t+1)>=0. Correction is zero from 1940 onward. Evaluate all step "
        "and annual linear transition years, power and smoothstep families, 200,000 fixed-seed "
        "Dirichlet paths spanning five concentration regimes, and multi-start constrained "
        "optimization. Joint constraints are positive bias and median-MAE improvement, at least "
        "half of stations improving, at most 10% median trend-error regression in both direct-qcf "
        "periods, HMA annual correction change no larger than frozen Candidate v2's own allowed "
        "maximum, and 1850-1899 HMA mean departure from Candidate v2 at most 0.05 degC. No "
        "1921-1939 value is read or generated. This search diagnoses feasibility only and cannot "
        "promote a candidate or modify production.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "station": STATION,
        "linear_v2": LINEAR_V2,
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
    linear = json.loads(LINEAR_V2.read_text(encoding="utf-8"))
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if any(item["candidate_id"] != CANDIDATE_ID for item in [linear, boundary, release]):
        raise RuntimeError("candidate-id-chain-failed")

    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station-input")
    groups = build_station_groups(stations)
    if sum(group["period_index"] == 0 for group in groups) != 3:
        raise RuntimeError("1880-1900-station-count-failed")
    if sum(group["period_index"] == 1 for group in groups) != 6:
        raise RuntimeError("1901-1920-station-count-failed")
    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64
        )
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-year-in-full-candidate")
    selected = np.isin(years, TRANSITION_YEARS)
    correction_fields = correction[selected]
    hma_correction = weighted_mean(correction_fields, common, lat)
    hma_internal_max = float(
        boundary["hma_allowed_internal_correction_change"]["annual_max_abs_degC"]
    )

    top: list[tuple[float, float, str, np.ndarray, dict[str, object]]] = []
    feasible_count = 0
    evaluated_count = 0

    def consume(matrix: np.ndarray, labels: list[str]) -> None:
        nonlocal top, feasible_count, evaluated_count
        result = evaluate_factors(matrix, groups, hma_correction, hma_internal_max)
        feasible_count += int(np.sum(result["score"] <= 1e-12))
        evaluated_count += len(matrix)
        order = np.lexsort((result["sum_violation"], result["score"]))[:TOP_KEEP]
        candidates = []
        for index in order:
            record = row_record(labels[index], matrix, result, int(index))
            candidates.append(
                (
                    record["score_max_normalized_violation"],
                    record["sum_positive_normalized_violation"],
                    labels[index],
                    matrix[index].copy(),
                    record,
                )
            )
        top = sorted(top + candidates, key=lambda item: (item[0], item[1]))[:TOP_KEEP]

    deterministic, deterministic_labels = deterministic_factors()
    consume(deterministic, deterministic_labels)
    rng = np.random.default_rng(RANDOM_SEED)
    for alpha in DIRICHLET_ALPHA:
        remaining = SAMPLES_PER_ALPHA
        batch_index = 0
        while remaining > 0:
            size = min(CHUNK_SIZE, remaining)
            decrements = rng.dirichlet(np.full(40, alpha), size=size)
            factors = np.column_stack(
                [np.ones(size), 1.0 - np.cumsum(decrements, axis=1)]
            )
            factors[:, -1] = 0.0
            labels = [f"dirichlet_a{alpha:g}_b{batch_index}_{index}" for index in range(size)]
            consume(factors, labels)
            remaining -= size
            batch_index += 1

    knot_years = np.asarray([1880, 1885, 1890, 1895, 1900, 1905, 1910, 1915, 1920])
    optimization_rows = []
    optimized_factors = []
    optimized_labels = []

    def expand_knots(interior: np.ndarray) -> np.ndarray:
        values = np.r_[1.0, interior, 0.0]
        return np.interp(TRANSITION_YEARS, knot_years, values)

    def objective(interior: np.ndarray) -> float:
        current = expand_knots(interior)[None]
        result = evaluate_factors(current, groups, hma_correction, hma_internal_max)
        return float(result["score"][0] + 0.05 * result["sum_violation"][0])

    constraints = [
        {"type": "ineq", "fun": lambda x, index=index: x[index] - x[index + 1]}
        for index in range(6)
    ]
    starts = []
    for item in top[:12]:
        starts.append(np.interp(knot_years[1:-1], TRANSITION_YEARS, item[3]))
    for start in (1880, 1890, 1900, 1910):
        values = np.clip((1920.0 - knot_years[1:-1]) / (1920.0 - start), 0.0, 1.0)
        starts.append(values)
    for index, initial in enumerate(starts):
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 7,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
        factors = expand_knots(result.x)
        optimized_factors.append(factors)
        optimized_labels.append(f"slsqp_{index:02d}")
        optimization_rows.append(
            {
                "label": f"slsqp_{index:02d}",
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective": float(result.fun),
            }
        )
    consume(np.asarray(optimized_factors), optimized_labels)

    top_rows = []
    factor_rows = []
    for rank, item in enumerate(top, start=1):
        record = dict(item[4])
        record["rank"] = rank
        top_rows.append(record)
        factor_rows.extend(
            {
                "rank": rank,
                "label": item[2],
                "year": int(year),
                "factor": float(value),
            }
            for year, value in zip(TRANSITION_YEARS, item[3])
        )
    pd.DataFrame(top_rows).to_csv(OUTPUT / "top_monotone_transition_candidates.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(OUTPUT / "top_monotone_transition_factors.csv", index=False)
    pd.DataFrame(optimization_rows).to_csv(OUTPUT / "constrained_optimization_runs.csv", index=False)

    best = top[0]
    best_factors = best[3]
    delta_fields = (
        (best_factors[:20] - 1.0)[:, None, None] * correction_fields[:20]
    )
    mean_delta_field = np.sum(delta_fields, axis=0) / 50.0
    best_spatial_rms_departure = float(
        weighted_rms(mean_delta_field[None], common, lat)[0]
    )
    best_hma_mean_departure = abs(float(weighted_mean(mean_delta_field[None], common, lat)[0]))
    best_result = evaluate_factors(best_factors[None], groups, hma_correction, hma_internal_max)
    best_violations = best_result["violations"][0]
    constraint_names = [
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
    violation_map = {
        name: float(value) for name, value in zip(constraint_names, best_violations)
    }
    positive_best = {name: value for name, value in violation_map.items() if value > 1e-12}
    spatial_rms_gate = best_spatial_rms_departure <= 0.05
    any_feasible = feasible_count > 0 and spatial_rms_gate
    classification = (
        "monotone_allowed_transition_feasible_in_search_requires_independent_confirmation"
        if any_feasible
        else "no_feasible_monotone_allowed_transition_found_after_extreme_family_random_and_constrained_search"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for rank, item in enumerate(top[:10], start=1):
        axes[0].plot(
            TRANSITION_YEARS,
            item[3],
            lw=2.2 if rank == 1 else 0.9,
            alpha=1.0 if rank == 1 else 0.45,
            label="best" if rank == 1 else None,
        )
    axes[0].set_ylabel("Correction factor")
    axes[0].set_title("Best monotone allowed-year transition factors")
    axes[0].legend(frameon=False)
    names = list(positive_best) if positive_best else ["none"]
    values = [100 * positive_best[name] for name in names] if positive_best else [0]
    axes[1].bar(np.arange(len(names)), values, color="#D55E00")
    axes[1].set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axes[1].set_ylabel("Normalized gate violation (%)")
    axes[1].set_title("Remaining violations of the best searched path")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure = OUTPUT / "monotone_allowed_transition_search.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-monotone-allowed-transition-search-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "search": {
            "random_seed": RANDOM_SEED,
            "deterministic_path_count": int(len(deterministic)),
            "dirichlet_path_count": int(len(DIRICHLET_ALPHA) * SAMPLES_PER_ALPHA),
            "constrained_optimization_start_count": len(starts),
            "total_evaluated_path_count": evaluated_count,
            "feasible_before_spatial_rms_count": feasible_count,
        },
        "best": {
            **best[4],
            "positive_normalized_gate_violations": positive_best,
            "hma_1850_1899_mean_departure_from_v2_degC_recomputed": best_hma_mean_departure,
            "hma_1850_1899_spatial_rms_departure_from_v2_degC": best_spatial_rms_departure,
            "spatial_rms_departure_at_most_0p05_degC": spatial_rms_gate,
        },
        "interpretation": (
            "Every monotone path is a convex combination of annual step paths through its "
            "nonnegative decrement sequence. The search spans all step extremes, common smooth "
            "families, five broad random simplex regimes, and local constrained refinement. "
            "Failure is strong numerical evidence, not a formal analytic impossibility proof."
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
        "# Monotone allowed-year transition search\n\n"
        f"Classification: `{classification}`.\n\n"
        f"- Evaluated paths: `{evaluated_count}`.\n"
        f"- Feasible paths before final spatial-RMS check: `{feasible_count}`.\n"
        f"- Best maximum normalized violation: `{best[4]['score_max_normalized_violation']:.4f}`.\n"
        f"- Best remaining positive violations: `{json.dumps(positive_best, sort_keys=True)}`.\n"
        f"- Best 1850-1899 spatial-RMS departure: `{best_spatial_rms_departure:.4f} degC`.\n"
        "- No sealed-period value was read or generated; Candidate v2 and production remain unchanged.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
