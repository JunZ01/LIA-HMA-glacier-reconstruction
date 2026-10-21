"""Globally optimize the preregistered low-dimensional transition families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import differential_evolution
from scipy.special import betainc, expit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import search_temperature_candidate_low_dimensional_transition_families_v1 as base


DRIVER = Path(__file__).resolve()
PROJECT = base.PROJECT
LOOP = base.LOOP
SEARCH = base.OUTPUT
SEARCH_SUMMARY = SEARCH / "summary.json"
OUTPUT = LOOP / "temperature_candidate_low_dimensional_transition_families_v2_optimization"
FACTS = base.FACTS
EXECUTION = base.EXECUTION
STATE = base.STATE
YEARS = base.YEARS
SEEDS = (20260831, 20260901, 20260902)
RMS_LIMIT = 0.05


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


def normalized(values: np.ndarray) -> np.ndarray:
    result = (values - values[-1]) / (values[0] - values[-1])
    result[0] = 1.0
    result[-1] = 0.0
    return np.clip(result, 0.0, 1.0)


def build_information(stations: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    gamma = float(np.exp(parameters[0]))
    counts = (
        stations.assign(valid=np.isfinite(stations["qcf_station_degC"]))
        .groupby("year")["valid"]
        .sum()
        .reindex(YEARS, fill_value=0)
        .to_numpy(dtype=float)
    )
    cumulative = np.cumsum(counts)
    information = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0])
    result = 1.0 - information**gamma
    result[0] = 1.0
    result[-1] = 0.0
    return result


def build_beta(_stations: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    alpha, beta = np.exp(parameters)
    result = 1.0 - betainc(alpha, beta, base.X)
    result[0] = 1.0
    result[-1] = 0.0
    return result


def build_double_logistic(_stations: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    center_1 = float(parameters[0])
    center_2 = center_1 + (1920.0 - center_1) * float(parameters[1])
    scale_1, scale_2 = np.exp(parameters[2:4])
    weight = float(parameters[4])
    first = expit(-(YEARS - center_1) / scale_1)
    second = expit(-(YEARS - center_2) / scale_2)
    return normalized(weight * first + (1.0 - weight) * second)


def build_piecewise(_stations: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    knots = 1880.0 + 40.0 * np.sort(parameters[:4])
    level = float(parameters[4])
    t1, t2, t3, t4 = knots
    values = np.ones(len(YEARS), dtype=np.float64)
    first = (YEARS > t1) & (YEARS < t2)
    values[first] = 1.0 - (1.0 - level) * (YEARS[first] - t1) / max(t2 - t1, 1e-12)
    values[(YEARS >= t2) & (YEARS <= t3)] = level
    second = (YEARS > t3) & (YEARS < t4)
    values[second] = level * (t4 - YEARS[second]) / max(t4 - t3, 1e-12)
    values[YEARS >= t4] = 0.0
    values[0] = 1.0
    values[-1] = 0.0
    return values


FAMILY_DEFINITIONS = {
    "instrument_information_power": {
        "parameter_count": 1,
        "bounds": [(np.log(0.05), np.log(20.0))],
        "builder": build_information,
    },
    "beta_cdf": {
        "parameter_count": 2,
        "bounds": [(np.log(0.05), np.log(60.0)), (np.log(0.05), np.log(60.0))],
        "builder": build_beta,
    },
    "double_logistic": {
        "parameter_count": 5,
        "bounds": [
            (1880.0, 1918.0),
            (0.0, 1.0),
            (np.log(0.10), np.log(15.0)),
            (np.log(0.10), np.log(15.0)),
            (0.001, 0.999),
        ],
        "builder": build_double_logistic,
    },
    "two_stage_piecewise_linear": {
        "parameter_count": 5,
        "bounds": [(0.0, 1.0)] * 4 + [(0.001, 0.999)],
        "builder": build_piecewise,
    },
}


def evaluate(
    factors: np.ndarray,
    groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    gram: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    result = base.gates.evaluate_factors(factors[None], groups, hma, hma_internal_max)
    rms = float(base.spatial_rms(factors[None], gram)[0])
    raw = np.append(result["violations"][0], rms / RMS_LIMIT - 1.0)
    max_raw = float(np.max(raw))
    positive_sum = float(np.sum(np.maximum(raw, 0.0)))
    record = base.gates.row_record("optimized", factors[None], result, 0)
    record.update(
        {
            "hma_1850_1899_spatial_rms_departure_degC": rms,
            "maximum_signed_normalized_gate_value": max_raw,
            "sum_positive_normalized_gate_violation": positive_sum,
            "passes_all_full_gates": bool(max_raw <= 1e-10),
            "signed_normalized_gate_values": raw.tolist(),
        }
    )
    return max_raw, positive_sum, record


def decoded_parameters(name: str, parameters: np.ndarray) -> dict[str, float]:
    if name == "instrument_information_power":
        return {"gamma": float(np.exp(parameters[0]))}
    if name == "beta_cdf":
        return {"alpha": float(np.exp(parameters[0])), "beta": float(np.exp(parameters[1]))}
    if name == "double_logistic":
        center_1 = float(parameters[0])
        return {
            "center_1": center_1,
            "center_2": center_1 + (1920.0 - center_1) * float(parameters[1]),
            "scale_1": float(np.exp(parameters[2])),
            "scale_2": float(np.exp(parameters[3])),
            "weight_1": float(parameters[4]),
        }
    knots = 1880.0 + 40.0 * np.sort(parameters[:4])
    return {
        "transition_1_start": float(knots[0]),
        "transition_1_end": float(knots[1]),
        "transition_2_start": float(knots[2]),
        "transition_2_end": float(knots[3]),
        "plateau_factor": float(parameters[4]),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "GLOBAL_OPTIMIZATION_CONTRACT.md"
    contract.write_text(
        "# Low-dimensional family global optimization\n\n"
        "Optimize only the four V1-preregistered families with three fixed seeds "
        "per family. Minimize the largest signed normalized frozen-gate value, "
        "with a small tie-breaker on summed positive violations. No family, gate, "
        "year, station, or spatial rule changes. No sealed value is read or "
        "generated, and no candidate is promoted.\n",
        encoding="utf-8",
    )
    sources = {
        "search_summary": SEARCH_SUMMARY,
        "full_candidate": base.FULL,
        "station": base.STATION,
        "boundary": base.BOUNDARY,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: file_state(path) for name, path in sources.items()}
    search = json.loads(SEARCH_SUMMARY.read_text(encoding="utf-8"))
    stations = pd.read_csv(base.STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = base.gates.build_station_groups(stations)
    with Dataset(base.FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("locked-year-in-candidate")
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        mask = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][np.isin(years, YEARS)],
            dtype=np.float64,
        )
    hma = base.gates.weighted_mean(correction, mask, lat)
    hma_internal_max = float(
        json.loads(base.BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )
    gram = base.spatial_gram(correction, mask, lat)

    run_rows = []
    best_by_family = {}
    for name, definition in FAMILY_DEFINITIONS.items():
        builder = definition["builder"]

        def objective(parameters: np.ndarray) -> float:
            factors = builder(stations, parameters)
            max_raw, positive_sum, _ = evaluate(
                factors, groups, hma, hma_internal_max, gram
            )
            return max_raw + 0.002 * positive_sum

        family_runs = []
        for seed in SEEDS:
            result = differential_evolution(
                objective,
                definition["bounds"],
                seed=seed,
                strategy="best1bin",
                maxiter=600,
                popsize=20,
                tol=1e-10,
                atol=1e-12,
                polish=True,
                updating="immediate",
                workers=1,
            )
            factors = builder(stations, result.x)
            max_raw, positive_sum, metrics = evaluate(
                factors, groups, hma, hma_internal_max, gram
            )
            decoded = decoded_parameters(name, result.x)
            row = {
                "family": name,
                "parameter_count": definition["parameter_count"],
                "seed": seed,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "objective": float(result.fun),
                "maximum_signed_normalized_gate_value": max_raw,
                "sum_positive_normalized_gate_violation": positive_sum,
                "passes_all_full_gates": bool(max_raw <= 1e-10),
                **decoded,
            }
            run_rows.append(row)
            family_runs.append((max_raw, positive_sum, result.x.copy(), factors, metrics, row))
        family_best = min(family_runs, key=lambda item: (item[0], item[1]))
        best_by_family[name] = family_best
        pd.DataFrame({"year": YEARS, "factor": family_best[3]}).to_csv(
            OUTPUT / f"optimized_factor_{name}.csv", index=False
        )
        (OUTPUT / f"optimized_metrics_{name}.json").write_text(
            json.dumps(
                {
                    "parameters": decoded_parameters(name, family_best[2]),
                    "metrics": family_best[4],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    pd.DataFrame(run_rows).to_csv(OUTPUT / "global_optimization_runs.csv", index=False)

    family_summaries = []
    for name, item in best_by_family.items():
        family_summaries.append(
            {
                "family": name,
                "parameter_count": FAMILY_DEFINITIONS[name]["parameter_count"],
                "passes_all_full_gates": bool(item[0] <= 1e-10),
                "maximum_signed_normalized_gate_value": item[0],
                "sum_positive_normalized_gate_violation": item[1],
                "parameters": decoded_parameters(name, item[2]),
                "metrics": item[4],
            }
        )
    passing = [item for item in family_summaries if item["passes_all_full_gates"]]
    preferred = min(
        passing,
        key=lambda item: (
            item["parameter_count"],
            item["maximum_signed_normalized_gate_value"],
        ),
    ) if passing else None

    unrestricted = pd.read_csv(base.UNRESTRICTED_BEST)["best_factor"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    axis.plot(YEARS, unrestricted, color="black", linewidth=2.4, label="Unrestricted validated path")
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    for color, item in zip(colors, family_summaries):
        name = item["family"]
        label = f"{name} ({item['parameter_count']}p; max={item['maximum_signed_normalized_gate_value']:.3f})"
        axis.plot(YEARS, best_by_family[name][3], color=color, linewidth=1.8, label=label)
    axis.set_xlabel("Year")
    axis.set_ylabel("Correction factor")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimized_low_dimensional_families.png", dpi=180)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    classification = (
        "optimized_low_dimensional_family_passes_requires_holdout_validation"
        if preferred is not None
        else "optimized_preregistered_low_dimensional_families_remain_infeasible"
    )
    summary = {
        "schema": "hma-temperature-candidate-low-dimensional-transition-families-v2-optimization",
        "status": "complete",
        "candidate_id": search["candidate_id"],
        "classification": classification,
        "families": family_summaries,
        "preferred_by_complexity": preferred,
        "decision": (
            "Advance only the lowest-dimensional passing family to held-out validation."
            if preferred is not None
            else "The unrestricted annual path has no supported low-dimensional replacement in the preregistered families and is not adopted scientifically."
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
        "# Optimized low-dimensional transition families\n\n"
        f"Classification: `{classification}`.\n\n"
        + "\n".join(
            f"- `{item['family']}` ({item['parameter_count']} parameters): "
            f"pass=`{item['passes_all_full_gates']}`, maximum signed normalized gate "
            f"value=`{item['maximum_signed_normalized_gate_value']:.6f}`."
            for item in family_summaries
        )
        + "\n\n"
        + (
            f"Preferred by complexity: `{preferred['family']}`; held-out validation is required.\n"
            if preferred is not None
            else "No preregistered low-dimensional family is feasible after global optimization.\n"
        )
        + "Candidate v2 remains frozen and not in production. No sealed value was read or generated.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
