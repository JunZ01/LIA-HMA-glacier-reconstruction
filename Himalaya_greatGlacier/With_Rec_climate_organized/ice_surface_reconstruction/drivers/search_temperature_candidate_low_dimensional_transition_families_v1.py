"""Search preregistered low-dimensional monotone transition families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.special import betainc, expit
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import search_temperature_candidate_monotone_allowed_transition_v1 as gates


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
FULL = gates.FULL
STATION = gates.STATION
BOUNDARY = gates.BOUNDARY
UNRESTRICTED = LOOP / "temperature_candidate_monotone_lp_feasible_polytopes_v3"
UNRESTRICTED_BEST = UNRESTRICTED / "best_full_gate_factor.csv"
UNRESTRICTED_SUMMARY = UNRESTRICTED / "summary.json"
INDEPENDENT = LOOP / "temperature_candidate_monotone_transition_independent_validation_v2/summary.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_low_dimensional_transition_families_v1"

YEARS = np.arange(1880, 1921)
X = (YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0])
RMS_LIMIT = 0.05
SEED = 20260831
FAMILIES = {
    "instrument_information_power": {"parameter_count": 1, "sample_count": 20000},
    "beta_cdf": {"parameter_count": 2, "sample_count": 100000},
    "double_logistic": {"parameter_count": 5, "sample_count": 200000},
    "two_stage_piecewise_linear": {"parameter_count": 5, "sample_count": 200000},
}
TOP_KEEP = 100


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


def normalize_endpoints(values: np.ndarray) -> np.ndarray:
    denominator = values[:, :1] - values[:, -1:]
    result = (values - values[:, -1:]) / denominator
    result[:, 0] = 1.0
    result[:, -1] = 0.0
    return np.clip(result, 0.0, 1.0)


def instrument_information_family(stations: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    counts = (
        stations.assign(valid=np.isfinite(stations["qcf_station_degC"]))
        .groupby("year")["valid"]
        .sum()
        .reindex(YEARS, fill_value=0)
        .to_numpy(dtype=float)
    )
    cumulative = np.cumsum(counts)
    if cumulative[-1] <= cumulative[0]:
        raise RuntimeError("instrument-information-empty")
    information = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0])
    gamma = np.exp(np.linspace(np.log(0.10), np.log(10.0), FAMILIES["instrument_information_power"]["sample_count"]))
    factors = 1.0 - information[None] ** gamma[:, None]
    factors[:, 0] = 1.0
    factors[:, -1] = 0.0
    return factors, pd.DataFrame({"gamma": gamma})


def beta_family() -> tuple[np.ndarray, pd.DataFrame]:
    sampler = qmc.LatinHypercube(d=2, seed=SEED + 1)
    sample = sampler.random(FAMILIES["beta_cdf"]["sample_count"])
    a = np.exp(np.log(0.12) + sample[:, 0] * (np.log(15.0) - np.log(0.12)))
    b = np.exp(np.log(0.12) + sample[:, 1] * (np.log(15.0) - np.log(0.12)))
    factors = 1.0 - betainc(a[:, None], b[:, None], X[None])
    factors[:, 0] = 1.0
    factors[:, -1] = 0.0
    return factors, pd.DataFrame({"alpha": a, "beta": b})


def double_logistic_family() -> tuple[np.ndarray, pd.DataFrame]:
    sampler = qmc.LatinHypercube(d=5, seed=SEED + 2)
    sample = sampler.random(FAMILIES["double_logistic"]["sample_count"])
    center_1 = 1881.0 + 29.0 * sample[:, 0]
    center_2 = center_1 + (1919.0 - center_1) * sample[:, 1]
    scale_1 = np.exp(np.log(0.20) + sample[:, 2] * (np.log(10.0) - np.log(0.20)))
    scale_2 = np.exp(np.log(0.20) + sample[:, 3] * (np.log(10.0) - np.log(0.20)))
    weight = 0.02 + 0.96 * sample[:, 4]
    first = expit(-(YEARS[None] - center_1[:, None]) / scale_1[:, None])
    second = expit(-(YEARS[None] - center_2[:, None]) / scale_2[:, None])
    factors = normalize_endpoints(weight[:, None] * first + (1.0 - weight[:, None]) * second)
    return factors, pd.DataFrame(
        {
            "center_1": center_1,
            "center_2": center_2,
            "scale_1": scale_1,
            "scale_2": scale_2,
            "weight_1": weight,
        }
    )


def piecewise_family() -> tuple[np.ndarray, pd.DataFrame]:
    sampler = qmc.LatinHypercube(d=5, seed=SEED + 3)
    sample = sampler.random(FAMILIES["two_stage_piecewise_linear"]["sample_count"])
    knots = 1880.0 + 40.0 * np.sort(sample[:, :4], axis=1)
    plateau = 0.01 + 0.98 * sample[:, 4]
    factors = np.empty((len(sample), len(YEARS)), dtype=np.float64)
    for index, (current_knots, level) in enumerate(zip(knots, plateau)):
        t1, t2, t3, t4 = current_knots
        values = np.ones(len(YEARS), dtype=np.float64)
        first = (YEARS > t1) & (YEARS < t2)
        values[first] = 1.0 - (1.0 - level) * (YEARS[first] - t1) / max(t2 - t1, 1e-9)
        values[(YEARS >= t2) & (YEARS <= t3)] = level
        second = (YEARS > t3) & (YEARS < t4)
        values[second] = level * (t4 - YEARS[second]) / max(t4 - t3, 1e-9)
        values[YEARS >= t4] = 0.0
        values[0] = 1.0
        values[-1] = 0.0
        factors[index] = values
    return factors, pd.DataFrame(
        {
            "transition_1_start": knots[:, 0],
            "transition_1_end": knots[:, 1],
            "transition_2_start": knots[:, 2],
            "transition_2_end": knots[:, 3],
            "plateau_factor": plateau,
        }
    )


def spatial_gram(correction: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights / np.sum(weights)
    fields = np.nan_to_num(correction[:20].reshape(20, -1), nan=0.0)
    return (fields * weights.reshape(-1)[None]) @ fields.T


def spatial_rms(factors: np.ndarray, gram: np.ndarray) -> np.ndarray:
    departure = factors[:, :20] - 1.0
    return np.sqrt(
        np.maximum(np.einsum("bi,ij,bj->b", departure, gram, departure), 0.0)
        / 2500.0
    )


def search_family(
    name: str,
    factors: np.ndarray,
    parameters: pd.DataFrame,
    groups: list[dict[str, object]],
    hma: np.ndarray,
    hma_internal_max: float,
    gram: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    top: list[tuple[float, float, int, dict[str, object]]] = []
    pass_count = 0
    for left in range(0, len(factors), 5000):
        current = factors[left : left + 5000]
        result = gates.evaluate_factors(current, groups, hma, hma_internal_max)
        rms = spatial_rms(current, gram)
        rms_violation = np.maximum(rms / RMS_LIMIT - 1.0, 0.0)
        score = np.maximum(result["score"], rms_violation)
        sum_violation = result["sum_violation"] + rms_violation
        pass_count += int(np.sum(score <= 1e-10))
        local_order = np.lexsort((sum_violation, score))[:TOP_KEEP]
        for local in local_order:
            global_index = left + int(local)
            record = gates.row_record(
                f"{name}_{global_index}", current, result, int(local)
            )
            record.update(parameters.iloc[global_index].to_dict())
            record.update(
                {
                    "family": name,
                    "parameter_count": FAMILIES[name]["parameter_count"],
                    "sample_index": global_index,
                    "hma_1850_1899_spatial_rms_departure_degC": float(rms[local]),
                    "full_max_normalized_violation": float(score[local]),
                    "full_sum_positive_normalized_violation": float(sum_violation[local]),
                    "passes_all_full_gates": bool(score[local] <= 1e-10),
                }
            )
            top.append((float(score[local]), float(sum_violation[local]), global_index, record))
        top = sorted(top, key=lambda item: (item[0], item[1]))[:TOP_KEEP]
    records = pd.DataFrame([item[3] for item in top])
    best_index = int(top[0][2])
    summary = {
        "family": name,
        "parameter_count": FAMILIES[name]["parameter_count"],
        "sample_count": int(len(factors)),
        "full_gate_pass_count": pass_count,
        "full_gate_pass_fraction": pass_count / len(factors),
        "best_full_max_normalized_violation": float(top[0][0]),
        "best_full_sum_positive_normalized_violation": float(top[0][1]),
        "best_sample_index": best_index,
        "best_parameters": parameters.iloc[best_index].to_dict(),
        "best_metrics": top[0][3],
    }
    return summary, records, factors[best_index].copy()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "LOW_DIMENSIONAL_TRANSITION_CONTRACT.md"
    contract.write_text(
        "# Low-dimensional monotone transition families\n\n"
        "Search four preregistered families with fixed endpoints f(1880)=1 and "
        "f(1920)=0 and non-increasing factors. Rank scientific preference by "
        "parameter count before numerical score: one-parameter cumulative direct-"
        "station information weight, two-parameter beta CDF, five-parameter double "
        "logistic, and five-parameter two-stage piecewise linear. Passing requires "
        "all frozen station, HMA temporal, mean, and spatial RMS gates. This is an "
        "interpretability audit only. No sealed value is read or generated and no "
        "candidate is promoted.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "station": STATION,
        "boundary": BOUNDARY,
        "unrestricted_best": UNRESTRICTED_BEST,
        "unrestricted_summary": UNRESTRICTED_SUMMARY,
        "independent_validation": INDEPENDENT,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: file_state(path) for name, path in sources.items()}
    independent = json.loads(INDEPENDENT.read_text(encoding="utf-8"))
    if independent["status"] != "pass":
        raise RuntimeError("unrestricted-independent-validation-not-pass")
    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = gates.build_station_groups(stations)
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
    hma = gates.weighted_mean(correction, mask, lat)
    hma_internal_max = float(
        json.loads(BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )
    gram = spatial_gram(correction, mask, lat)

    builders = {
        "instrument_information_power": lambda: instrument_information_family(stations),
        "beta_cdf": beta_family,
        "double_logistic": double_logistic_family,
        "two_stage_piecewise_linear": piecewise_family,
    }
    summaries = []
    best_factors = {}
    for name, builder in builders.items():
        factors, parameters = builder()
        if np.any(np.diff(factors, axis=1) > 1e-10):
            raise RuntimeError(f"nonmonotone-family:{name}")
        family_summary, top, best = search_family(
            name, factors, parameters, groups, hma, hma_internal_max, gram
        )
        summaries.append(family_summary)
        best_factors[name] = best
        top.to_csv(OUTPUT / f"top_{name}.csv", index=False)
        pd.DataFrame({"year": YEARS, "factor": best}).to_csv(
            OUTPUT / f"best_factor_{name}.csv", index=False
        )

    family_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in item.items()
                if key not in {"best_parameters", "best_metrics"}
            }
            for item in summaries
        ]
    ).sort_values(["parameter_count", "best_full_max_normalized_violation"])
    family_table.to_csv(OUTPUT / "family_comparison.csv", index=False)
    passing = [item for item in summaries if item["full_gate_pass_count"] > 0]
    preferred = min(
        passing,
        key=lambda item: (
            item["parameter_count"],
            item["best_full_max_normalized_violation"],
            item["best_full_sum_positive_normalized_violation"],
        ),
    ) if passing else None

    unrestricted = pd.read_csv(UNRESTRICTED_BEST)["best_factor"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    axis.plot(YEARS, unrestricted, color="black", linewidth=2.4, label="Unrestricted validated path")
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    for color, item in zip(colors, summaries):
        name = item["family"]
        label = f"{name} ({item['parameter_count']}p; pass={item['full_gate_pass_count']})"
        axis.plot(YEARS, best_factors[name], color=color, linewidth=1.8, label=label)
    axis.set_xlabel("Year")
    axis.set_ylabel("Correction factor")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "low_dimensional_transition_family_comparison.png", dpi=180)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    classification = (
        "low_dimensional_full_gate_family_found_requires_holdout_validation"
        if preferred is not None
        else "no_preregistered_low_dimensional_family_passed_full_gates"
    )
    summary = {
        "schema": "hma-temperature-candidate-low-dimensional-transition-families-v1",
        "status": "complete",
        "candidate_id": independent["candidate_id"],
        "classification": classification,
        "families": summaries,
        "preferred_by_complexity": preferred,
        "decision": (
            "Advance only the lowest-dimensional passing family to station and temporal holdout validation."
            if preferred is not None
            else "Do not use the unrestricted annual path as a scientific transition; retain Candidate v2 as allowed-year-only."
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
        "# Low-dimensional transition family search\n\n"
        f"Classification: `{classification}`.\n\n"
        + "\n".join(
            f"- `{item['family']}` ({item['parameter_count']} parameters): "
            f"{item['full_gate_pass_count']}/{item['sample_count']} pass; best violation "
            f"{item['best_full_max_normalized_violation']:.6f}."
            for item in summaries
        )
        + "\n\n"
        + (
            f"Preferred by complexity: `{preferred['family']}`. It requires holdout validation before any scientific use.\n"
            if preferred is not None
            else "No low-dimensional family passed; the unrestricted path is not scientifically adopted.\n"
        )
        + "Candidate v2 remains formally frozen and not in production. No sealed value was read or generated.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
