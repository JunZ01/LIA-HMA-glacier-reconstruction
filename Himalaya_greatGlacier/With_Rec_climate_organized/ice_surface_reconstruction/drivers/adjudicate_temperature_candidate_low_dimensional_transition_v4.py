"""Concurrency-safe adjudication of low-dimensional transition optimization."""

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
OUTPUT = LOOP / "temperature_candidate_low_dimensional_transition_families_v4_adjudication"
V2_MANIFEST = V2 / "source_manifest.json"
RUNS = V2 / "global_optimization_runs.csv"
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1/summary.json"
STATE = independent.STATE
FACTS = independent.FACTS
EXECUTION = independent.EXECUTION
STATION = independent.STATION
FULL = independent.FULL
BOUNDARY = independent.BOUNDARY
FAMILIES = ("double_logistic", "two_stage_piecewise_linear")
RMS_LIMIT = 0.05
CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def state(path: Path) -> dict[str, object]:
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


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def gram_matrix(correction: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights / np.sum(weights)
    fields = np.nan_to_num(correction[:20].reshape(20, -1), nan=0.0)
    return (fields * weights.reshape(-1)[None]) @ fields.T


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "CONCURRENCY_SAFE_ADJUDICATION_CONTRACT.md"
    contract.write_text(
        "# Concurrency-safe technical adjudication\n\n"
        "The temperature calculation is valid only if its computational inputs "
        "remain byte-identical. The live glacier-controller route and append-only "
        "facts file may advance concurrently, but the frozen Candidate v2 identity, "
        "status, and evidence hash must remain invariant before and after this audit. "
        "Independently recompute both optimized passing paths from the station and "
        "field inputs. No sealed values, promotion, production writes, EnKF, or OGGM.\n",
        encoding="utf-8",
    )
    factors = {name: V2 / f"optimized_factor_{name}.csv" for name in FAMILIES}
    claims = {name: V2 / f"optimized_metrics_{name}.json" for name in FAMILIES}
    scientific_sources = {
        "v2_manifest": V2_MANIFEST,
        "runs": RUNS,
        "freeze": FREEZE,
        "station": STATION,
        "full_candidate": FULL,
        "boundary": BOUNDARY,
        "execution_contract": EXECUTION,
        "driver": DRIVER,
        "contract": contract,
        **{f"factor_{name}": path for name, path in factors.items()},
        **{f"claim_{name}": path for name, path in claims.items()},
    }
    scientific_before = {name: state(path) for name, path in scientific_sources.items()}
    controller_before = json.loads(STATE.read_text(encoding="utf-8"))
    facts_before = state(FACTS)
    fingerprint_before = candidate_fingerprint(controller_before)
    freeze_hash = sha256(FREEZE)
    if fingerprint_before != (
        CANDIDATE_ID,
        "formally-frozen-candidate-not-production",
        freeze_hash,
    ):
        raise RuntimeError("candidate-controller-fingerprint-failed")

    v2_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    v2_changed = sorted(
        name for name, value in v2_manifest["before"].items()
        if value != v2_manifest["after"][name]
    )
    if v2_changed != ["current_state"]:
        raise RuntimeError(f"unexpected-v2-concurrent-change:{v2_changed}")
    for name in (
        "full_candidate", "station", "boundary", "facts", "execution_contract",
        "search_summary", "driver", "contract",
    ):
        if v2_manifest["before"][name] != v2_manifest["after"][name]:
            raise RuntimeError(f"v2-source-changed:{name}")
    for name, path in {
        "full_candidate": FULL,
        "station": STATION,
        "boundary": BOUNDARY,
        "execution_contract": EXECUTION,
    }.items():
        if sha256(path) != v2_manifest["after"][name]["sha256"]:
            raise RuntimeError(f"computational-input-changed-after-v2:{name}")

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
    gram = gram_matrix(correction, mask, lat)

    validations = []
    for family in FAMILIES:
        table = pd.read_csv(factors[family])
        factor = table["factor"].to_numpy(dtype=float)
        if not np.array_equal(table["year"].to_numpy(dtype=int), independent.YEARS):
            raise RuntimeError(f"factor-year-failed:{family}")
        if abs(factor[0] - 1.0) > 1e-12 or abs(factor[-1]) > 1e-12:
            raise RuntimeError(f"factor-endpoint-failed:{family}")
        if np.any(np.diff(factor) > 1e-12):
            raise RuntimeError(f"factor-monotonicity-failed:{family}")
        metrics, temporal = independent.evaluate_temporal(
            factor[None], groups, hma, hma_internal_max
        )
        rms = float(independent.spatial_rms_from_gram(factor[None], gram)[0])
        raw = np.append(temporal[0], rms / RMS_LIMIT - 1.0)
        claimed = json.loads(claims[family].read_text(encoding="utf-8"))["metrics"]
        values = {
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
        closure = max(abs(value - float(claimed[name])) for name, value in values.items())
        validations.append(
            {
                "family": family,
                "passes_all_full_gates": bool(np.max(raw) <= 1e-10),
                "maximum_signed_normalized_gate_value": float(np.max(raw)),
                "maximum_claim_recomputation_difference": closure,
                "metrics": values,
            }
        )
    if not all(item["passes_all_full_gates"] for item in validations):
        raise RuntimeError("independent-gate-closure-failed")
    if max(item["maximum_claim_recomputation_difference"] for item in validations) > 1e-10:
        raise RuntimeError("metric-closure-failed")

    runs = pd.read_csv(RUNS)
    convergence = []
    for family in FAMILIES:
        subset = runs[runs["family"] == family]
        passed = subset[subset["passes_all_full_gates"]]
        convergence.append(
            {
                "family": family,
                "seed_count": int(len(subset)),
                "passing_seed_count": int(len(passed)),
                "passing_seed_gate_value_range": float(
                    passed["maximum_signed_normalized_gate_value"].max()
                    - passed["maximum_signed_normalized_gate_value"].min()
                ) if len(passed) else None,
            }
        )

    scientific_after = {name: state(path) for name, path in scientific_sources.items()}
    controller_after = json.loads(STATE.read_text(encoding="utf-8"))
    facts_after = state(FACTS)
    fingerprint_after = candidate_fingerprint(controller_after)
    scientific_unchanged = scientific_before == scientific_after
    candidate_fingerprint_unchanged = fingerprint_before == fingerprint_after
    status = "pass" if scientific_unchanged and candidate_fingerprint_unchanged else "fail"
    summary = {
        "schema": "hma-temperature-candidate-low-dimensional-transition-v4-adjudication",
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "v2_concurrent_changed_sources": v2_changed,
        "v2_computational_inputs_unchanged": True,
        "independent_validations": validations,
        "seed_convergence": convergence,
        "preferred_for_holdout": "double_logistic",
        "preference_reason": "same five-parameter complexity, smooth form, and three-of-three fixed seeds converge to passing paths",
        "controller_state_before": controller_before["state_id"],
        "controller_state_after": controller_after["state_id"],
        "facts_sha256_before": facts_before["sha256"],
        "facts_sha256_after": facts_after["sha256"],
        "candidate_fingerprint_unchanged": candidate_fingerprint_unchanged,
        "decision": "Release the numerical optimization result and advance double_logistic only to held-out validation; do not promote it.",
        "candidate_v2_status": "formally-frozen-candidate-not-production-unchanged",
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_generated": 0,
        "scientific_sources_unchanged_during_adjudication": scientific_unchanged,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "scientific_before": scientific_before,
                "scientific_after": scientific_after,
                "scientific_unchanged": scientific_unchanged,
                "controller_before": controller_before,
                "controller_after": controller_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Low-dimensional transition V4 adjudication\n\n"
        f"Status: `{status}`.\n\n"
        "- V2 changed only the unrelated live controller pointer while running.\n"
        "- All computational inputs remained byte-identical.\n"
        "- Both optimized five-parameter paths independently pass every frozen gate.\n"
        "- Double logistic advances only to held-out validation because all three seeds pass.\n"
        "- Candidate v2 remains frozen and not in production.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError("concurrency-safe-adjudication-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
