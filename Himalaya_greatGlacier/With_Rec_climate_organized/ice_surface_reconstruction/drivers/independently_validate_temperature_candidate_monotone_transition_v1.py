"""Independently validate the best allowed-year monotone transition path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
POLYTOPE = LOOP / "temperature_candidate_monotone_lp_feasible_polytopes_v3"
BEST = POLYTOPE / "best_full_gate_factor.csv"
CLAIM = POLYTOPE / "summary.json"
STATION = LOOP / "temperature_background_candidate_v2_allowed_years_v4/station_year_candidate_values.csv"
FULL = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1/temperature_candidate_v2_full_posterior_allowed_years.nc"
BOUNDARY = LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1/summary.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_monotone_transition_independent_validation_v1"

YEARS = np.arange(1880, 1921)
PERIODS = ((1880, 1900), (1901, 1920))
MEAN_LIMIT = 0.05
RMS_LIMIT = 0.05
RNG_SEED = 20260830
NEIGHBOR_EPS = (0.001, 0.005, 0.01, 0.02)
NEIGHBORS_PER_EPS = 5000


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
    return np.sum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def station_groups(stations: pd.DataFrame) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
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
            years = group["year"].to_numpy(dtype=int)
            observed = group["qcf_station_degC"].to_numpy(dtype=float)
            project = group["project_six_member_degC"].to_numpy(dtype=float)
            correction = group["candidate_background_correction_degC"].to_numpy(dtype=float)
            centered = years.astype(float) - np.mean(years)
            slope_weights = centered / np.sum(centered * centered) * 100.0
            original_error = project - observed
            groups.append(
                {
                    "period_index": period_index,
                    "station_id": str(station_id),
                    "years": years,
                    "indices": years - YEARS[0],
                    "error": original_error,
                    "correction": correction,
                    "slope_weights": slope_weights,
                    "original_bias": float(np.mean(original_error)),
                    "original_mae": float(np.mean(np.abs(original_error))),
                    "original_trend": abs(float(slope_weights @ original_error)),
                }
            )
    return groups


def evaluate_temporal(
    factors: np.ndarray,
    groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    period_metrics: dict[str, np.ndarray] = {}
    violations: list[np.ndarray] = []
    for period_index in range(2):
        current = [group for group in groups if group["period_index"] == period_index]
        biases = []
        maes = []
        trends = []
        original_biases = []
        original_maes = []
        original_trends = []
        for group in current:
            candidate_error = (
                group["error"][None]
                + factors[:, group["indices"]] * group["correction"][None]
            )
            biases.append(np.mean(candidate_error, axis=1))
            maes.append(np.mean(np.abs(candidate_error), axis=1))
            trends.append(np.abs(candidate_error @ group["slope_weights"]))
            original_biases.append(group["original_bias"])
            original_maes.append(group["original_mae"])
            original_trends.append(group["original_trend"])
        biases_array = np.column_stack(biases)
        maes_array = np.column_stack(maes)
        trends_array = np.column_stack(trends)
        original_bias = float(np.mean(original_biases))
        original_median_mae = float(np.median(original_maes))
        original_median_trend = float(np.median(original_trends))
        aggregate_bias = np.mean(biases_array, axis=1)
        median_mae = np.median(maes_array, axis=1)
        improving_fraction = np.mean(
            maes_array < np.asarray(original_maes)[None], axis=1
        )
        median_trend = np.median(trends_array, axis=1)
        label = f"{PERIODS[period_index][0]}_{PERIODS[period_index][1]}"
        period_metrics[f"bias_improvement_{label}"] = (
            abs(original_bias) - np.abs(aggregate_bias)
        ) / abs(original_bias)
        period_metrics[f"mae_improvement_{label}"] = (
            original_median_mae - median_mae
        ) / original_median_mae
        period_metrics[f"improving_fraction_{label}"] = improving_fraction
        period_metrics[f"trend_error_change_{label}"] = (
            median_trend - original_median_trend
        ) / original_median_trend
        violations.extend(
            [
                np.abs(aggregate_bias) / abs(original_bias) - 1.0,
                median_mae / original_median_mae - 1.0,
                (0.5 - improving_fraction) / 0.5,
                median_trend / (1.10 * original_median_trend) - 1.0,
            ]
        )
    transitioned_hma = factors * hma[None]
    annual_max = np.max(np.abs(np.diff(transitioned_hma, axis=1)), axis=1)
    mean_departure = np.abs(
        np.sum((factors[:, :20] - 1.0) * hma[None, :20], axis=1) / 50.0
    )
    period_metrics["hma_annual_change_max_abs_degC"] = annual_max
    period_metrics["hma_1850_1899_mean_departure_from_v2_degC"] = mean_departure
    violations.extend(
        [
            annual_max / hma_internal_max - 1.0,
            mean_departure / MEAN_LIMIT - 1.0,
        ]
    )
    return period_metrics, np.column_stack(violations)


def spatial_gram(
    correction_fields: np.ndarray, mask: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    normalized = weights / np.sum(weights)
    flattened = correction_fields[:20].reshape(20, -1)
    weighted = normalized.reshape(-1)
    return (flattened * weighted[None]) @ flattened.T


def spatial_rms_from_gram(factors: np.ndarray, gram: np.ndarray) -> np.ndarray:
    departure = factors[:, :20] - 1.0
    squared = np.einsum("bi,ij,bj->b", departure, gram, departure) / 2500.0
    return np.sqrt(np.maximum(squared, 0.0))


def station_detail(best: np.ndarray, groups: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for group in groups:
        candidate_error = group["error"] + best[group["indices"]] * group["correction"]
        rows.append(
            {
                "period": f"{PERIODS[group['period_index']][0]}-{PERIODS[group['period_index']][1]}",
                "station_id": group["station_id"],
                "n_years": len(group["years"]),
                "first_year": int(group["years"][0]),
                "last_year": int(group["years"][-1]),
                "original_bias_degC": group["original_bias"],
                "transition_bias_degC": float(np.mean(candidate_error)),
                "original_mae_degC": group["original_mae"],
                "transition_mae_degC": float(np.mean(np.abs(candidate_error))),
                "original_abs_trend_error_degC_per_century": group["original_trend"],
                "transition_abs_trend_error_degC_per_century": abs(
                    float(group["slope_weights"] @ candidate_error)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "INDEPENDENT_VALIDATION_CONTRACT.md"
    contract.write_text(
        "# Independent validation contract\n\n"
        "Recompute the selected 1880-1920 monotone path directly from the frozen "
        "station-year table and materialized allowed-year Candidate v2 field. "
        "Do not import the search or polytope evaluator. Verify every temporal "
        "and spatial gate and test deterministic monotone perturbation neighborhoods. "
        "Do not read or generate 1921-1939 values and do not promote or modify Candidate v2.\n",
        encoding="utf-8",
    )
    sources = {
        "best_factor": BEST,
        "polytope_claim": CLAIM,
        "station": STATION,
        "full_candidate": FULL,
        "boundary": BOUNDARY,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: file_state(path) for name, path in sources.items()}

    claim = json.loads(CLAIM.read_text(encoding="utf-8"))
    factor_table = pd.read_csv(BEST)
    if not np.array_equal(factor_table["year"].to_numpy(dtype=int), YEARS):
        raise RuntimeError("factor-year-contract-failed")
    best = factor_table["best_factor"].to_numpy(dtype=float)
    if abs(best[0] - 1.0) > 1e-12 or abs(best[-1]) > 1e-12:
        raise RuntimeError("factor-endpoint-contract-failed")
    if np.any(np.diff(best) > 1e-12):
        raise RuntimeError("factor-monotonicity-failed")

    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = station_groups(stations)

    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("locked-year-in-candidate")
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        mask = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        selected = np.isin(years, YEARS)
        correction_fields = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][selected], dtype=np.float64
        )
    if correction_fields.shape[0] != len(YEARS):
        raise RuntimeError("transition-field-year-count-failed")
    hma = weighted_mean(correction_fields, mask, lat)
    hma_internal_max = float(
        json.loads(BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )
    gram = spatial_gram(correction_fields, mask, lat)

    metrics, temporal_violations = evaluate_temporal(
        best[None], groups, hma, hma_internal_max
    )
    spatial_rms = float(spatial_rms_from_gram(best[None], gram)[0])
    all_violations = np.append(temporal_violations[0], spatial_rms / RMS_LIMIT - 1.0)
    passes = bool(np.max(all_violations) <= 1e-10)

    key_map = {
        "bias_improvement_1880_1900": "bias_improvement_1880_1900",
        "bias_improvement_1901_1920": "bias_improvement_1901_1920",
        "mae_improvement_1880_1900": "mae_improvement_1880_1900",
        "mae_improvement_1901_1920": "mae_improvement_1901_1920",
        "improving_fraction_1880_1900": "improving_fraction_1880_1900",
        "improving_fraction_1901_1920": "improving_fraction_1901_1920",
        "trend_error_change_1880_1900": "trend_error_change_1880_1900",
        "trend_error_change_1901_1920": "trend_error_change_1901_1920",
        "hma_annual_change_max_abs_degC": "hma_annual_change_max_abs_degC",
        "hma_1850_1899_mean_departure_from_v2_degC": "hma_1850_1899_mean_departure_from_v2_degC",
    }
    recomputed = {name: float(metrics[source][0]) for name, source in key_map.items()}
    recomputed["hma_1850_1899_spatial_rms_departure_degC"] = spatial_rms
    closure = {
        name: abs(value - float(claim["best"][name]))
        for name, value in recomputed.items()
    }
    max_claim_closure = max(closure.values())

    detail = station_detail(best, groups)
    detail.to_csv(OUTPUT / "station_gate_detail.csv", index=False)

    rng = np.random.default_rng(RNG_SEED)
    neighborhood_rows = []
    total_neighbor_pass = 0
    for epsilon in NEIGHBOR_EPS:
        random_drops = rng.dirichlet(np.ones(len(YEARS) - 1), size=NEIGHBORS_PER_EPS)
        random_factors = np.column_stack(
            [np.ones(NEIGHBORS_PER_EPS), 1.0 - np.cumsum(random_drops, axis=1)]
        )
        neighbors = (1.0 - epsilon) * best[None] + epsilon * random_factors
        neighbor_metrics, neighbor_temporal = evaluate_temporal(
            neighbors, groups, hma, hma_internal_max
        )
        neighbor_rms = spatial_rms_from_gram(neighbors, gram)
        max_violation = np.maximum(
            np.max(neighbor_temporal, axis=1), neighbor_rms / RMS_LIMIT - 1.0
        )
        pass_mask = max_violation <= 1e-10
        total_neighbor_pass += int(np.sum(pass_mask))
        neighborhood_rows.append(
            {
                "epsilon": epsilon,
                "sample_count": NEIGHBORS_PER_EPS,
                "pass_count": int(np.sum(pass_mask)),
                "pass_fraction": float(np.mean(pass_mask)),
                "median_max_abs_factor_difference": float(
                    np.median(np.max(np.abs(neighbors - best[None]), axis=1))
                ),
                "maximum_max_abs_factor_difference": float(
                    np.max(np.max(np.abs(neighbors - best[None]), axis=1))
                ),
                "median_max_normalized_violation": float(np.median(np.maximum(max_violation, 0.0))),
                "maximum_hma_spatial_rms_degC": float(np.max(neighbor_rms)),
            }
        )
    neighborhood = pd.DataFrame(neighborhood_rows)
    neighborhood.to_csv(OUTPUT / "monotone_neighborhood_stability.csv", index=False)

    tapered_hma = best * hma
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(YEARS, best, color="#174A7E", linewidth=2.2)
    axes[0].set_ylabel("Correction factor")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[1].plot(YEARS, hma, color="#C44E52", linewidth=1.8, label="Frozen Candidate v2")
    axes[1].plot(YEARS, tapered_hma, color="#2A9D8F", linewidth=2.2, label="Validated transition")
    axes[1].set_ylabel("HMA correction (degC)")
    axes[1].set_xlabel("Year")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Candidate v2 allowed-year monotone transition")
    fig.tight_layout()
    fig.savefig(OUTPUT / "candidate_v2_monotone_transition_validation.png", dpi=180)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    status = "pass" if passes and max_claim_closure <= 1e-10 and unchanged else "fail"
    summary = {
        "schema": "hma-temperature-candidate-monotone-transition-independent-validation-v1",
        "status": status,
        "candidate_id": claim["candidate_id"],
        "validated_path_label": claim["best"]["label"],
        "best_path_recomputed_metrics": recomputed,
        "maximum_claim_recomputation_difference": max_claim_closure,
        "all_full_gates_pass": passes,
        "maximum_normalized_gate_violation": float(max(np.max(all_violations), 0.0)),
        "station_group_count": len(groups),
        "neighborhood": neighborhood_rows,
        "neighbor_sample_count": len(NEIGHBOR_EPS) * NEIGHBORS_PER_EPS,
        "neighbor_pass_count": total_neighbor_pass,
        "interpretation": (
            "The selected monotone path independently closes every frozen temporal and spatial gate."
            if status == "pass"
            else "The selected path did not independently close every declared gate."
        ),
        "candidate_v2_status": "formally-frozen-candidate-not-production-unchanged",
        "transition_status": "independently-validated-non-promotion-preflight" if status == "pass" else "rejected",
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
        "# Candidate v2 monotone transition independent validation\n\n"
        f"Status: `{status}`.\n\n"
        f"- Full frozen gates pass: `{passes}`.\n"
        f"- Maximum claim recomputation difference: `{max_claim_closure:.3e}`.\n"
        f"- HMA 1850-1899 mean departure: `{recomputed['hma_1850_1899_mean_departure_from_v2_degC']:.6f} degC`.\n"
        f"- HMA 1850-1899 spatial RMS departure: `{spatial_rms:.6f} degC`.\n"
        f"- Maximum annual HMA correction change: `{recomputed['hma_annual_change_max_abs_degC']:.6f} degC`.\n"
        f"- Monotone neighborhood passes: `{total_neighbor_pass}/{len(NEIGHBOR_EPS) * NEIGHBORS_PER_EPS}`.\n"
        "- Candidate v2 remains frozen and is not promoted to production.\n"
        "- No sealed-period value was read or generated.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError("independent-validation-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
