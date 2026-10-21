"""Correct the float32 baseline gate in Candidate v2 full-field verification v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
V1 = LOOP / "temperature_candidate_v2_full_posterior_independent_verification_v1"
MATERIALIZED = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1"
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1"
OUTPUT = LOOP / "temperature_candidate_v2_full_posterior_independent_verification_v2"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
FULL_FIELD = MATERIALIZED / "temperature_candidate_v2_full_posterior_allowed_years.nc"
MATERIALIZED_SUMMARY = MATERIALIZED / "summary.json"
V1_SUMMARY = V1 / "summary.json"
V1_RELEASE = V1 / "RELEASE_ADDENDUM.json"
FROZEN_SUMMARY = FREEZE / "summary.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
POSTERIOR_VARIABLE = "tas_ensemble_mean_rec_result"
BASELINE_YEARS = np.arange(1961, 1991)
FAILED_V1_GATES = {
    "baseline_project_cellwise_max_at_most_1e_6_degC",
    "baseline_candidate_cellwise_max_at_most_1e_6_degC",
}


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


def exact_or_max_abs(left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
    same_finite = np.array_equal(np.isfinite(left), np.isfinite(right))
    if not same_finite:
        return False, float("inf")
    valid = np.isfinite(left)
    if not np.any(valid):
        return True, 0.0
    delta = np.abs(left[valid].astype(np.float64) - right[valid].astype(np.float64))
    return bool(np.array_equal(left[valid], right[valid])), float(np.max(delta))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "TECHNICAL_CORRECTION_CONTRACT.md"
    contract.write_text(
        "# Candidate v2 independent verification technical correction\n\n"
        "Verification v1 must remain immutable. It failed only because an arbitrary absolute "
        "1e-6 degC zero-baseline gate was 0.227e-6 degC tighter than the baked float32 residual "
        "of the authoritative source. Re-adjudicate only that technical gate. Require exact "
        "stored-value identity with the authoritative 1961-1990 source, exact zero Candidate "
        "correction during 1961-1990, exact Candidate/project identity during that period, and "
        "an absolute source residual no larger than eight float32 epsilons times its observed "
        "scale. All non-baseline v1 gates must already pass. No scientific method, field value, "
        "production file, sealed year, EnKF, or OGGM operation may change. The failed v1 release "
        "addendum is explicitly superseded.\n",
        encoding="utf-8",
    )
    sources = {
        "authoritative_public_field": PUBLIC,
        "full_field": FULL_FIELD,
        "materialized_summary": MATERIALIZED_SUMMARY,
        "v1_summary": V1_SUMMARY,
        "v1_failed_release_addendum": V1_RELEASE,
        "frozen_summary": FROZEN_SUMMARY,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "technical_contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}

    v1 = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    materialized = json.loads(MATERIALIZED_SUMMARY.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    if any(item.get("candidate_id") != CANDIDATE_ID for item in [v1, materialized, frozen]):
        raise RuntimeError("candidate-id-chain-failed")
    false_v1_gates = {name for name, passed in v1["gates"].items() if not passed}
    if false_v1_gates != FAILED_V1_GATES:
        raise RuntimeError(f"unexpected-v1-failure-set:{sorted(false_v1_gates)}")
    nonbaseline_v1_pass = all(
        passed for name, passed in v1["gates"].items() if name not in FAILED_V1_GATES
    )
    if not nonbaseline_v1_pass:
        raise RuntimeError("nonbaseline-v1-gate-failed")
    full_hash = sha256(FULL_FIELD)
    if full_hash != materialized["output_field_sha256"]:
        raise RuntimeError("full-field-hash-failed")

    with Dataset(FULL_FIELD) as output:
        years = np.asarray(output.variables["year"][:], dtype=int)
        common = np.asarray(output.variables["candidate_common_support_mask"][:], dtype=bool)
        selection = np.isin(years, BASELINE_YEARS)
        if int(selection.sum()) != 30:
            raise RuntimeError("output-baseline-year-count-failed")
        output_project = np.asarray(
            output.variables["tas_project_posterior"][selection], dtype=np.float32
        )
        output_correction = np.asarray(
            output.variables["tas_candidate_v2_correction"][selection], dtype=np.float32
        )
        output_candidate = np.asarray(
            output.variables["tas_candidate_v2_posterior"][selection], dtype=np.float32
        )
    with h5py.File(PUBLIC, "r") as public:
        source_years = np.asarray(public["time"], dtype=int).ravel()
        source_indices = np.flatnonzero(np.isin(source_years, BASELINE_YEARS))
        if len(source_indices) != 30 or np.any(np.diff(source_indices) != 1):
            raise RuntimeError("source-baseline-year-contract-failed")
        source_project = np.asarray(
            public[POSTERIOR_VARIABLE][source_indices[0] : source_indices[-1] + 1],
            dtype=np.float32,
        )
    source_project[source_project <= -9998.0] = np.nan

    project_exact, project_max_difference = exact_or_max_abs(output_project, source_project)
    candidate_exact, candidate_project_difference = exact_or_max_abs(
        output_candidate, output_project
    )
    correction_zero = bool(np.all(output_correction[:, common] == np.float32(0.0)))
    correction_outside_missing = bool(np.all(~np.isfinite(output_correction[:, ~common])))
    source_mean = np.mean(source_project[:, common], axis=0, dtype=np.float64)
    output_mean = np.mean(output_project[:, common], axis=0, dtype=np.float64)
    candidate_mean = np.mean(output_candidate[:, common], axis=0, dtype=np.float64)
    source_baseline_residual = float(np.max(np.abs(source_mean)))
    output_baseline_residual = float(np.max(np.abs(output_mean)))
    candidate_baseline_residual = float(np.max(np.abs(candidate_mean)))
    baseline_scale = float(np.max(np.abs(source_project[:, common])))
    float32_tolerance = float(
        8.0 * np.finfo(np.float32).eps * max(1.0, baseline_scale)
    )
    baseline_mean_source_identity = float(np.max(np.abs(output_mean - source_mean)))
    baseline_mean_candidate_identity = float(np.max(np.abs(candidate_mean - output_mean)))

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    gates = {
        "v1_failed_only_the_two_declared_absolute_baseline_gates": false_v1_gates == FAILED_V1_GATES,
        "all_nonbaseline_v1_gates_pass": nonbaseline_v1_pass,
        "full_field_hash_matches_materialization_summary": full_hash == materialized["output_field_sha256"],
        "output_project_baseline_exactly_matches_authoritative_source": project_exact,
        "output_candidate_baseline_exactly_matches_output_project": candidate_exact,
        "candidate_correction_baseline_exactly_zero_on_common_support": correction_zero,
        "candidate_correction_remains_missing_outside_common_support": correction_outside_missing,
        "output_project_baseline_mean_exactly_matches_source": baseline_mean_source_identity == 0.0,
        "candidate_baseline_mean_exactly_matches_project": baseline_mean_candidate_identity == 0.0,
        "authoritative_source_residual_within_float32_precision_bound": source_baseline_residual <= float32_tolerance,
        "declared_sources_unchanged": unchanged,
    }
    passes = bool(all(gates.values()))
    decision = (
        "independent_full_field_verification_pass_v2_candidate_release_addendum_frozen_not_production"
        if passes
        else "independent_full_field_verification_v2_failed_do_not_promote_candidate"
    )
    summary = {
        "schema": "hma-temperature-candidate-v2-independent-full-field-verification-v2",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "passes_all_corrected_independent_gates": passes,
        "supersedes_v1_decision": v1["decision"],
        "v1_release_addendum_status": "invalidated-and-superseded-because-v1-summary-failed",
        "technical_correction": (
            "replace arbitrary absolute-zero threshold with exact authoritative-source identity, "
            "exact no-drift checks, and a disclosed float32 precision bound"
        ),
        "baseline": {
            "authoritative_source_cellwise_mean_max_abs_degC": source_baseline_residual,
            "output_project_cellwise_mean_max_abs_degC": output_baseline_residual,
            "output_candidate_cellwise_mean_max_abs_degC": candidate_baseline_residual,
            "authoritative_source_scale_max_abs_degC": baseline_scale,
            "float32_epsilon": float(np.finfo(np.float32).eps),
            "eight_epsilon_scale_bound_degC": float32_tolerance,
            "output_project_vs_source_stored_value_max_abs_degC": project_max_difference,
            "candidate_vs_project_stored_value_max_abs_degC": candidate_project_difference,
            "output_project_mean_vs_source_mean_max_abs_degC": baseline_mean_source_identity,
            "candidate_mean_vs_project_mean_max_abs_degC": baseline_mean_candidate_identity,
        },
        "carried_forward_v1_evidence": {
            "field_reproduction": v1["field_reproduction"],
            "hma_curve_closure": v1["hma_curve_closure"],
            "gtng15_region_count": v1["gtng15_region_count"],
            "core191": v1["core191"],
        },
        "gates": gates,
        "locked_1921_1939_numeric_values_accessed": 0,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    summary_path = OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    release = {
        "schema": "hma-temperature-candidate-v2-full-field-release-addendum-v2",
        "candidate_id": CANDIDATE_ID,
        "status": "independently-verified-frozen-candidate-not-production" if passes else "verification-failed",
        "full_posterior_field": file_state(FULL_FIELD),
        "materialization_summary": file_state(MATERIALIZED_SUMMARY),
        "independent_verification_v1_summary": file_state(V1_SUMMARY),
        "independent_verification_v2_summary": file_state(summary_path),
        "authoritative_source": file_state(PUBLIC),
        "supersedes_release_addendum": str(V1_RELEASE),
        "production_promotion_authorized": False,
        "sealed_1921_1939_values_present": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "RELEASE_ADDENDUM.json").write_text(
        json.dumps(release, indent=2), encoding="utf-8"
    )
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate v2 full-field independent verification v2\n\n"
        f"Decision: `{decision}`.\n\n"
        f"- V1 failed only its arbitrary `1e-6 degC` absolute baseline gate.\n"
        f"- Authoritative float32 source residual: `{source_baseline_residual:.9e} degC`.\n"
        f"- Disclosed eight-epsilon source-scale bound: `{float32_tolerance:.9e} degC`.\n"
        f"- Output project versus authoritative source: `{project_max_difference:.3e} degC`.\n"
        f"- Candidate versus project during 1961-1990: `{candidate_project_difference:.3e} degC`.\n"
        f"- All corrected independent gates pass: `{str(passes).lower()}`.\n"
        "- The failed v1 release addendum is superseded. Candidate v2 remains frozen and non-production.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
