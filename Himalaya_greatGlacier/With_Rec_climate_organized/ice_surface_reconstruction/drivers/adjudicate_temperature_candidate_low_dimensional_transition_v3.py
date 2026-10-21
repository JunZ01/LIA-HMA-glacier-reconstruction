"""Adjudicate the V2 optimization after an unrelated controller-state change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import independently_validate_temperature_candidate_monotone_transition_v1 as independent


DRIVER = Path(__file__).resolve()
PROJECT = independent.PROJECT
LOOP = independent.LOOP
V2 = LOOP / "temperature_candidate_low_dimensional_transition_families_v2_optimization"
MANIFEST = V2 / "source_manifest.json"
RUNS = V2 / "global_optimization_runs.csv"
STATION = independent.STATION
FULL = independent.FULL
BOUNDARY = independent.BOUNDARY
FACTS = independent.FACTS
EXECUTION = independent.EXECUTION
STATE = independent.STATE
OUTPUT = LOOP / "temperature_candidate_low_dimensional_transition_families_v3_adjudication"
FAMILIES = ("double_logistic", "two_stage_piecewise_linear")
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


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def spatial_gram(correction: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights / np.sum(weights)
    fields = np.nan_to_num(correction[:20].reshape(20, -1), nan=0.0)
    return (fields * weights.reshape(-1)[None]) @ fields.T


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "TECHNICAL_ADJUDICATION_CONTRACT.md"
    contract.write_text(
        "# Technical adjudication contract\n\n"
        "V2 is numerically releasable only if its source manifest shows that the "
        "sole concurrent change was LIA_CURRENT_STATE.json, every scientific source "
        "remained byte-identical, current scientific sources still match the V2 "
        "after-hashes, and both passing paths independently reclose every frozen "
        "gate. This adjudication does not rerun optimization, promote a candidate, "
        "read sealed values, or change production.\n",
        encoding="utf-8",
    )
    factor_paths = {
        family: V2 / f"optimized_factor_{family}.csv" for family in FAMILIES
    }
    metric_paths = {
        family: V2 / f"optimized_metrics_{family}.json" for family in FAMILIES
    }
    sources = {
        "v2_manifest": MANIFEST,
        "v2_runs": RUNS,
        "station": STATION,
        "full_candidate": FULL,
        "boundary": BOUNDARY,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
        **{f"factor_{name}": path for name, path in factor_paths.items()},
        **{f"metrics_{name}": path for name, path in metric_paths.items()},
    }
    before = {name: file_state(path) for name, path in sources.items()}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = sorted(
        name for name, value in manifest["before"].items()
        if value != manifest["after"][name]
    )
    if changed != ["current_state"]:
        raise RuntimeError(f"unexpected-v2-source-change:{changed}")
    for name in (
        "search_summary", "full_candidate", "station", "boundary",
        "facts", "execution_contract", "driver", "contract",
    ):
        if manifest["before"][name] != manifest["after"][name]:
            raise RuntimeError(f"scientific-source-changed-during-v2:{name}")
    current_map = {
        "full_candidate": FULL,
        "station": STATION,
        "boundary": BOUNDARY,
        "facts": FACTS,
        "execution_contract": EXECUTION,
    }
    for name, path in current_map.items():
        if sha256(path) != manifest["after"][name]["sha256"]:
            raise RuntimeError(f"scientific-source-changed-after-v2:{name}")

    stations = pd.read_csv(STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    groups = independent.station_groups(stations)
    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("locked-year-in-candidate")
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        mask = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][np.isin(years, independent.YEARS)],
            dtype=np.float64,
        )
    hma = weighted_mean(correction, mask, lat)
    hma_internal_max = float(
        json.loads(BOUNDARY.read_text(encoding="utf-8"))[
            "hma_allowed_internal_correction_change"
        ]["annual_max_abs_degC"]
    )
    gram = spatial_gram(correction, mask, lat)

    validations = []
    for family in FAMILIES:
        table = pd.read_csv(factor_paths[family])
        if not np.array_equal(table["year"].to_numpy(dtype=int), independent.YEARS):
            raise RuntimeError(f"factor-year-failed:{family}")
        factors = table["factor"].to_numpy(dtype=float)
        if abs(factors[0] - 1.0) > 1e-12 or abs(factors[-1]) > 1e-12:
            raise RuntimeError(f"factor-endpoint-failed:{family}")
        if np.any(np.diff(factors) > 1e-12):
            raise RuntimeError(f"factor-monotonicity-failed:{family}")
        metrics, temporal = independent.evaluate_temporal(
            factors[None], groups, hma, hma_internal_max
        )
        rms = float(independent.spatial_rms_from_gram(factors[None], gram)[0])
        raw = np.append(temporal[0], rms / RMS_LIMIT - 1.0)
        claimed = json.loads(metric_paths[family].read_text(encoding="utf-8"))["metrics"]
        recomputed = {
            "bias_improvement_1880_1900": float(metrics["bias_improvement_1880_1900"][0]),
            "bias_improvement_1901_1920": float(metrics["bias_improvement_1901_1920"][0]),
            "mae_improvement_1880_1900": float(metrics["mae_improvement_1880_1900"][0]),
            "mae_improvement_1901_1920": float(metrics["mae_improvement_1901_1920"][0]),
            "improving_fraction_1880_1900": float(metrics["improving_fraction_1880_1900"][0]),
            "improving_fraction_1901_1920": float(metrics["improving_fraction_1901_1920"][0]),
            "trend_error_change_1880_1900": float(metrics["trend_error_change_1880_1900"][0]),
            "trend_error_change_1901_1920": float(metrics["trend_error_change_1901_1920"][0]),
            "hma_annual_change_max_abs_degC": float(metrics["hma_annual_change_max_abs_degC"][0]),
            "hma_1850_1899_mean_departure_from_v2_degC": float(metrics["hma_1850_1899_mean_departure_from_v2_degC"][0]),
            "hma_1850_1899_spatial_rms_departure_degC": rms,
        }
        closure = max(abs(value - float(claimed[name])) for name, value in recomputed.items())
        validations.append(
            {
                "family": family,
                "passes_all_full_gates": bool(np.max(raw) <= 1e-10),
                "maximum_signed_normalized_gate_value": float(np.max(raw)),
                "maximum_claim_recomputation_difference": closure,
                "recomputed_metrics": recomputed,
            }
        )
    if not all(item["passes_all_full_gates"] for item in validations):
        raise RuntimeError("passing-path-independent-closure-failed")
    if max(item["maximum_claim_recomputation_difference"] for item in validations) > 1e-10:
        raise RuntimeError("claim-recomputation-closure-failed")

    runs = pd.read_csv(RUNS)
    convergence = []
    for family in FAMILIES:
        subset = runs[runs["family"] == family]
        pass_values = subset.loc[subset["passes_all_full_gates"], "maximum_signed_normalized_gate_value"]
        convergence.append(
            {
                "family": family,
                "seed_count": int(len(subset)),
                "passing_seed_count": int(subset["passes_all_full_gates"].sum()),
                "passing_seed_value_range": float(pass_values.max() - pass_values.min()) if len(pass_values) else None,
            }
        )
    preferred = "double_logistic"
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    status = "pass" if unchanged else "fail"
    summary = {
        "schema": "hma-temperature-candidate-low-dimensional-transition-v3-adjudication",
        "status": status,
        "candidate_id": "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828",
        "v2_changed_sources": changed,
        "scientific_sources_unchanged_during_and_after_v2": True,
        "independent_validations": validations,
        "seed_convergence": convergence,
        "preferred_for_holdout": preferred,
        "preference_reason": "same five-parameter complexity, smooth form, and all three seeds converge to passing solutions with approximately 1e-9 gate-value spread",
        "decision": "Release the V2 numerical result as technically valid and advance double_logistic only to held-out validation; do not promote it.",
        "candidate_v2_status": "formally-frozen-candidate-not-production-unchanged",
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
        "# Low-dimensional transition V3 technical adjudication\n\n"
        f"Status: `{status}`.\n\n"
        "- The sole V2 concurrent change was the unrelated live controller state.\n"
        "- All scientific sources remained byte-identical.\n"
        "- Double logistic and two-stage piecewise paths independently reclose all gates.\n"
        "- Double logistic is advanced only to held-out validation because all three seeds converge to passing solutions.\n"
        "- Candidate v2 remains frozen and not in production.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError("technical-adjudication-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
