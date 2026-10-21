"""Held-out validation for the optimized double-logistic transition family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import differential_evolution
from scipy.special import expit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import independently_validate_temperature_candidate_monotone_transition_v1 as independent


DRIVER = Path(__file__).resolve()
PROJECT = independent.PROJECT
LOOP = independent.LOOP
ADJUDICATION = LOOP / "temperature_candidate_low_dimensional_transition_families_v4_adjudication/summary.json"
V2 = LOOP / "temperature_candidate_low_dimensional_transition_families_v2_optimization"
FULL_FACTOR = V2 / "optimized_factor_double_logistic.csv"
STATION = independent.STATION
FULL = independent.FULL
BOUNDARY = independent.BOUNDARY
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1/summary.json"
STATE = independent.STATE
FACTS = independent.FACTS
EXECUTION = independent.EXECUTION
OUTPUT = LOOP / "temperature_candidate_double_logistic_holdout_validation_v1"

YEARS = independent.YEARS
PERIODS = independent.PERIODS
SEEDS = (20260903, 20260904)
RMS_LIMIT = 0.05
CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
BOUNDS = [
    (1880.0, 1918.0),
    (0.0, 1.0),
    (np.log(0.10), np.log(15.0)),
    (np.log(0.10), np.log(15.0)),
    (0.001, 0.999),
]


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


def candidate_fingerprint(controller: dict[str, object]) -> tuple[str, str, str]:
    branch = controller["research_branches"]["hma_temperature_candidate_v2"]
    return (
        str(branch["candidate_id"]),
        str(branch["status"]),
        str(branch["evidence"]["sha256"]),
    )


def build_factor(parameters: np.ndarray) -> np.ndarray:
    center_1 = float(parameters[0])
    center_2 = center_1 + (1920.0 - center_1) * float(parameters[1])
    scale_1, scale_2 = np.exp(parameters[2:4])
    weight = float(parameters[4])
    first = expit(-(YEARS - center_1) / scale_1)
    second = expit(-(YEARS - center_2) / scale_2)
    raw = weight * first + (1.0 - weight) * second
    factor = (raw - raw[-1]) / (raw[0] - raw[-1])
    factor[0] = 1.0
    factor[-1] = 0.0
    return np.clip(factor, 0.0, 1.0)


def decoded(parameters: np.ndarray) -> dict[str, float]:
    center_1 = float(parameters[0])
    return {
        "center_1": center_1,
        "center_2": center_1 + (1920.0 - center_1) * float(parameters[1]),
        "scale_1": float(np.exp(parameters[2])),
        "scale_2": float(np.exp(parameters[3])),
        "weight_1": float(parameters[4]),
    }


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def gram_matrix(correction: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights / np.sum(weights)
    fields = np.nan_to_num(correction[:20].reshape(20, -1), nan=0.0)
    return (fields * weights.reshape(-1)[None]) @ fields.T


def period_values(
    factor: np.ndarray, groups: list[dict[str, object]], period_index: int
) -> tuple[dict[str, float], list[dict[str, float]]]:
    current = [group for group in groups if group["period_index"] == period_index]
    if not current:
        raise RuntimeError(f"empty-period:{period_index}")
    rows = []
    for group in current:
        error = group["error"] + factor[group["indices"]] * group["correction"]
        rows.append(
            {
                "station_id": group["station_id"],
                "period_index": period_index,
                "original_bias": group["original_bias"],
                "candidate_bias": float(np.mean(error)),
                "original_mae": group["original_mae"],
                "candidate_mae": float(np.mean(np.abs(error))),
                "original_trend": group["original_trend"],
                "candidate_trend": abs(float(group["slope_weights"] @ error)),
            }
        )
    return aggregate_rows(rows), rows


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    original_bias = float(np.mean([row["original_bias"] for row in rows]))
    candidate_bias = float(np.mean([row["candidate_bias"] for row in rows]))
    original_median_mae = float(np.median([row["original_mae"] for row in rows]))
    candidate_median_mae = float(np.median([row["candidate_mae"] for row in rows]))
    original_median_trend = float(np.median([row["original_trend"] for row in rows]))
    candidate_median_trend = float(np.median([row["candidate_trend"] for row in rows]))
    improving_fraction = float(np.mean([
        row["candidate_mae"] < row["original_mae"] for row in rows
    ]))
    raw = np.asarray(
        [
            abs(candidate_bias) / abs(original_bias) - 1.0,
            candidate_median_mae / original_median_mae - 1.0,
            (0.5 - improving_fraction) / 0.5,
            candidate_median_trend / (1.10 * original_median_trend) - 1.0,
        ],
        dtype=float,
    )
    return {
        "station_group_count": len(rows),
        "bias_improvement": (abs(original_bias) - abs(candidate_bias)) / abs(original_bias),
        "mae_improvement": (original_median_mae - candidate_median_mae) / original_median_mae,
        "improving_fraction": improving_fraction,
        "trend_error_change": (candidate_median_trend - original_median_trend) / original_median_trend,
        "maximum_signed_normalized_gate_value": float(np.max(raw)),
        "passes_period_gates": bool(np.max(raw) <= 1e-10),
        "signed_normalized_gate_values": raw.tolist(),
    }


def fit_values(
    factor: np.ndarray,
    train_groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    gram: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    raw_parts = []
    period_records = {}
    for period_index in sorted({int(group["period_index"]) for group in train_groups}):
        record, _ = period_values(factor, train_groups, period_index)
        raw_parts.extend(record["signed_normalized_gate_values"])
        period_records[str(period_index)] = record
    transitioned_hma = factor * hma
    annual_max = float(np.max(np.abs(np.diff(transitioned_hma))))
    mean_departure = abs(float(np.sum((factor[:20] - 1.0) * hma[:20]) / 50.0))
    departure = factor[:20] - 1.0
    spatial_rms = float(np.sqrt(max(departure @ gram @ departure, 0.0) / 2500.0))
    raw_parts.extend(
        [
            annual_max / hma_internal_max - 1.0,
            mean_departure / 0.05 - 1.0,
            spatial_rms / RMS_LIMIT - 1.0,
        ]
    )
    raw = np.asarray(raw_parts, dtype=float)
    return (
        float(np.max(raw)),
        float(np.sum(np.maximum(raw, 0.0))),
        {
            "periods": period_records,
            "hma_annual_change_max_abs_degC": annual_max,
            "hma_1850_1899_mean_departure_degC": mean_departure,
            "hma_1850_1899_spatial_rms_degC": spatial_rms,
            "maximum_signed_normalized_gate_value": float(np.max(raw)),
            "passes_all_training_gates": bool(np.max(raw) <= 1e-10),
        },
    )


def optimize_fold(
    label: str,
    train_groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    gram: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    runs = []
    for seed in SEEDS:
        def objective(parameters: np.ndarray) -> float:
            maximum, positive_sum, _ = fit_values(
                build_factor(parameters), train_groups, hma, hma_internal_max, gram
            )
            return maximum + 0.002 * positive_sum

        result = differential_evolution(
            objective,
            BOUNDS,
            seed=seed,
            strategy="best1bin",
            maxiter=450,
            popsize=18,
            tol=1e-9,
            atol=1e-11,
            polish=True,
            updating="immediate",
            workers=1,
        )
        factor = build_factor(result.x)
        maximum, positive_sum, train_record = fit_values(
            factor, train_groups, hma, hma_internal_max, gram
        )
        runs.append(
            {
                "label": label,
                "seed": seed,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "maximum_signed_normalized_gate_value": maximum,
                "sum_positive_normalized_gate_violation": positive_sum,
                "passes_all_training_gates": bool(maximum <= 1e-10),
                "parameters": decoded(result.x),
                "raw_parameters": result.x.tolist(),
                "factor": factor.tolist(),
                "training_metrics": train_record,
            }
        )
    best = min(
        runs,
        key=lambda row: (
            row["maximum_signed_normalized_gate_value"],
            row["sum_positive_normalized_gate_violation"],
        ),
    )
    return np.asarray(best["factor"], dtype=float), best, runs


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "HOLDOUT_VALIDATION_CONTRACT.md"
    contract.write_text(
        "# Double-logistic held-out validation\n\n"
        "Refit the fixed five-parameter double-logistic family without each station "
        "identity in turn and without each time period in turn. Parameters never see "
        "held-out temperature errors. Every training fit must pass its available "
        "station and frozen HMA gates. Aggregated leave-one-station-out predictions "
        "must pass the original four station gates in both periods. Both temporal "
        "directions must pass the held-out period gates. Failure rejects the transition "
        "as a generalizable Candidate v3 addendum. No sealed values or production writes.\n",
        encoding="utf-8",
    )
    sources = {
        "adjudication": ADJUDICATION,
        "full_factor": FULL_FACTOR,
        "station": STATION,
        "full_candidate": FULL,
        "boundary": BOUNDARY,
        "freeze": FREEZE,
        "execution_contract": EXECUTION,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: file_state(path) for name, path in sources.items()}
    controller_before = json.loads(STATE.read_text(encoding="utf-8"))
    fingerprint_before = candidate_fingerprint(controller_before)
    if fingerprint_before != (
        CANDIDATE_ID,
        "formally-frozen-candidate-not-production",
        sha256(FREEZE),
    ):
        raise RuntimeError("candidate-controller-fingerprint-failed")
    adjudication = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    if adjudication["status"] != "pass" or adjudication["preferred_for_holdout"] != "double_logistic":
        raise RuntimeError("holdout-precondition-failed")
    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = independent.station_groups(stations)
    station_ids = sorted({str(group["station_id"]) for group in groups})
    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("locked-year-in-candidate")
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        mask = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][np.isin(years, YEARS)],
            dtype=np.float64,
        )
    hma = weighted_mean(correction, mask, lat)
    hma_internal_max = float(
        json.loads(BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )
    gram = gram_matrix(correction, mask, lat)

    fold_summaries = []
    all_run_rows = []
    heldout_rows_by_period = {0: [], 1: []}
    factor_columns = {"year": YEARS}
    for station_id in station_ids:
        train = [group for group in groups if group["station_id"] != station_id]
        heldout = [group for group in groups if group["station_id"] == station_id]
        factor, best, runs = optimize_fold(
            f"leave_station_{station_id}", train, hma, hma_internal_max, gram
        )
        factor_columns[f"leave_station_{station_id}"] = factor
        for row in runs:
            all_run_rows.append({key: value for key, value in row.items() if key not in {"factor", "training_metrics", "raw_parameters", "parameters"}})
        holdout_records = []
        for group in heldout:
            _, rows = period_values(factor, [group], int(group["period_index"]))
            holdout_records.extend(rows)
            heldout_rows_by_period[int(group["period_index"])].extend(rows)
        fold_summaries.append(
            {
                "fold": f"leave_station_{station_id}",
                "training_station_group_count": len(train),
                "heldout_station_group_count": len(heldout),
                "training_pass": best["passes_all_training_gates"],
                "training_maximum_signed_normalized_gate_value": best["maximum_signed_normalized_gate_value"],
                "parameters": best["parameters"],
                "heldout_groups": holdout_records,
            }
        )
    pd.DataFrame(factor_columns).to_csv(OUTPUT / "leave_one_station_out_factors.csv", index=False)
    loo_aggregate = {
        str(period_index): aggregate_rows(rows)
        for period_index, rows in heldout_rows_by_period.items()
    }

    temporal = []
    for train_period, holdout_period in ((0, 1), (1, 0)):
        train = [group for group in groups if group["period_index"] == train_period]
        holdout = [group for group in groups if group["period_index"] == holdout_period]
        factor, best, runs = optimize_fold(
            f"train_period_{train_period}", train, hma, hma_internal_max, gram
        )
        factor_columns = pd.DataFrame({"year": YEARS, "factor": factor})
        factor_columns.to_csv(
            OUTPUT / f"temporal_holdout_factor_train_{train_period}.csv", index=False
        )
        for row in runs:
            all_run_rows.append({key: value for key, value in row.items() if key not in {"factor", "training_metrics", "raw_parameters", "parameters"}})
        heldout_metrics, heldout_rows = period_values(factor, holdout, holdout_period)
        temporal.append(
            {
                "train_period": f"{PERIODS[train_period][0]}-{PERIODS[train_period][1]}",
                "holdout_period": f"{PERIODS[holdout_period][0]}-{PERIODS[holdout_period][1]}",
                "training_pass": best["passes_all_training_gates"],
                "training_maximum_signed_normalized_gate_value": best["maximum_signed_normalized_gate_value"],
                "parameters": best["parameters"],
                "heldout_metrics": heldout_metrics,
                "heldout_groups": heldout_rows,
            }
        )
    pd.DataFrame(all_run_rows).to_csv(OUTPUT / "optimization_run_summary.csv", index=False)
    (OUTPUT / "leave_one_station_out_detail.json").write_text(
        json.dumps(fold_summaries, indent=2), encoding="utf-8"
    )
    (OUTPUT / "temporal_holdout_detail.json").write_text(
        json.dumps(temporal, indent=2), encoding="utf-8"
    )

    all_loo_train_pass = all(item["training_pass"] for item in fold_summaries)
    loo_holdout_pass = all(item["passes_period_gates"] for item in loo_aggregate.values())
    all_temporal_train_pass = all(item["training_pass"] for item in temporal)
    temporal_holdout_pass = all(item["heldout_metrics"]["passes_period_gates"] for item in temporal)
    generalizes = all_loo_train_pass and loo_holdout_pass and all_temporal_train_pass and temporal_holdout_pass

    full_factor = pd.read_csv(FULL_FACTOR)["factor"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    axis.plot(YEARS, full_factor, color="black", linewidth=2.5, label="Full-sample double logistic")
    fold_factor_table = pd.read_csv(OUTPUT / "leave_one_station_out_factors.csv")
    for column in fold_factor_table.columns[1:]:
        axis.plot(YEARS, fold_factor_table[column], color="#4477AA", alpha=0.35, linewidth=1.1)
    for index, item in enumerate(temporal):
        factor = pd.read_csv(OUTPUT / f"temporal_holdout_factor_train_{index}.csv")["factor"]
        axis.plot(YEARS, factor, linestyle="--", linewidth=2.0, label=f"Train {item['train_period']}")
    axis.set_xlabel("Year")
    axis.set_ylabel("Correction factor")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "double_logistic_holdout_factors.png", dpi=180)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    controller_after = json.loads(STATE.read_text(encoding="utf-8"))
    fingerprint_after = candidate_fingerprint(controller_after)
    unchanged = before == after
    fingerprint_unchanged = fingerprint_before == fingerprint_after
    status = "complete" if unchanged and fingerprint_unchanged else "invalidated-by-source-change"
    classification = (
        "double_logistic_transition_generalizes_under_station_and_temporal_holdouts"
        if generalizes
        else "double_logistic_transition_fails_preregistered_holdout_generalization"
    )
    summary = {
        "schema": "hma-temperature-candidate-double-logistic-holdout-validation-v1",
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "station_identity_count": len(station_ids),
        "station_group_count": len(groups),
        "leave_one_station_out": {
            "all_training_fits_pass": all_loo_train_pass,
            "aggregated_holdout_period_metrics": loo_aggregate,
            "both_holdout_periods_pass": loo_holdout_pass,
        },
        "temporal_holdout": {
            "all_training_fits_pass": all_temporal_train_pass,
            "both_holdout_periods_pass": temporal_holdout_pass,
            "folds": temporal,
        },
        "generalization_pass": generalizes,
        "decision": (
            "Advance the smooth five-parameter double-logistic transition to a non-production Candidate v3 addendum package."
            if generalizes
            else "Reject the fitted transition as non-generalizing; retain frozen Candidate v2 as allowed-year-only and do not use the unrestricted annual path."
        ),
        "candidate_v2_status": "formally-frozen-candidate-not-production-unchanged",
        "candidate_fingerprint_unchanged": fingerprint_unchanged,
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_generated": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "before": before,
                "after": after,
                "unchanged": unchanged,
                "controller_before": controller_before,
                "controller_after": controller_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Double-logistic held-out validation\n\n"
        f"Classification: `{classification}`.\n\n"
        f"- All leave-one-station training fits pass: `{all_loo_train_pass}`.\n"
        f"- Both aggregated station holdout periods pass: `{loo_holdout_pass}`.\n"
        f"- Both temporal training fits pass: `{all_temporal_train_pass}`.\n"
        f"- Both opposite-period holdouts pass: `{temporal_holdout_pass}`.\n"
        f"- Overall generalization pass: `{generalizes}`.\n"
        "- Candidate v2 remains frozen and not in production. No sealed value was read or generated.\n",
        encoding="utf-8",
    )
    if status != "complete":
        raise RuntimeError("holdout-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
