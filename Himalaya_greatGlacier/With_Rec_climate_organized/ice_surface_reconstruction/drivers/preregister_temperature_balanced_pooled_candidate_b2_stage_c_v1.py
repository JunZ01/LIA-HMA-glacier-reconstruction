from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PRIMARY = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v3"
ROBUST = LOOP / "temperature_balanced_pooled_stage_b_robustness_v1"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "primary_summary": PRIMARY / "summary.json",
        "robustness_summary": ROBUST / "summary.json",
        "driver": Path(__file__).resolve(),
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in sources.items()}

    primary = json.loads(sources["primary_summary"].read_text(encoding="utf-8"))
    robust = json.loads(sources["robustness_summary"].read_text(encoding="utf-8"))
    if primary["stage_b_pass"] is not False:
        raise RuntimeError("primary-stage-b-route-must-remain-rejected")
    if robust["balanced_pooled_secondary_passes_all_original_numeric_gates"] is not True:
        raise RuntimeError("balanced-pooled-secondary-did-not-pass")
    if not all(robust["numeric_gates"].values()):
        raise RuntimeError("one-or-more-stage-b-gates-failed")
    if robust["sealed_1921_1939_values_used"]:
        raise RuntimeError("sealed-years-used-in-stage-b")
    if robust["prohibited_source_accessed"]:
        raise RuntimeError("prohibited-source-used-in-stage-b")

    contract = """# Candidate B2 Stage C preregistration

## Scientific status

Candidate B2 is a newly preregistered real-data candidate. It is not a
retroactive promotion of the rejected Stage B primary route and is not a
production reconstruction.

## Frozen method

- Models: CESM, ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L.
- Each model contributes exactly 349 paired temperature/precipitation residual
  members; the pooled residual library therefore has 1396 members.
- The evolving prior mean is the unweighted arithmetic mean of the four model
  evolving means. No model weight or amplitude is fitted to observations.
- A 31-year centered low-pass defines each evolving mean. The two allowed time
  segments are filtered independently so the filter cannot cross 1921-1939.
- State temperature and every proxy operator use the same paired member index.
- Archived proxy6 and proxy9 bivariate PSM definitions and archived diagonal
  RegEM residual variances are unchanged.
- proxy6 and proxy9 are assimilated separately. Their posterior temperature
  fields are averaged with fixed weights 0.5 and 0.5.
- No posterior averaging by climate model is permitted.

## Time and data locks

- Allowed reconstruction segments: 1500-1920 and 1940-2000.
- Years 1921-1939 remain sealed and may not be requested, read, filtered,
  scored, plotted, or used to choose any method.
- 20CRv3 is prohibited and may not be accessed.
- All temperature outputs used for instrumental comparison are expressed as
  anomalies relative to each source's own 1961-1990 climatology.

## Stage C gates

Stage C is observational validation, not automatic production promotion.
Before any production decision it must report, separately by proxy network and
joint fixed average:

1. reconstruction closure and finite-value checks for every allowed year;
2. 1940-2000 official GHCN-M station skill using annual/decadal correlation,
   RMSE, MAE, bias, trend error, and sign agreement;
3. spatial and seasonal residual diagnostics for East, Central, and West HMA;
4. 1850-1920 comparisons against the frozen independent-record evidence
   matrix, without treating CRU and Berkeley as independent proxy families;
5. comparison with the archived CESM production reconstruction and a no-proxy
   balanced-pooled background comparator;
6. source hashes, exact year requests, and confirmation that sealed/prohibited
   data were not accessed.

No gate may be changed after real proxy observations are read. Failure does not
license a tuned shift, time-segment splice, fitted model weight, or access to
the sealed interval.

## Authorization boundary

This preregistration and input preparation do not authorize Stage C EnKF.
Running Candidate B2 with real proxy observations requires a separate explicit
user authorization. Production files and OGGM remain untouched.
"""
    (OUT / "CANDIDATE_B2_STAGE_C_CONTRACT.md").write_text(contract, encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    source_unchanged = before == after
    if not source_unchanged:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": "balanced_pooled_candidate_b2",
        "stage_b_primary_status": "rejected",
        "stage_b_secondary_numeric_gates_passed": True,
        "models": ["CESM", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"],
        "members_per_model": 349,
        "pooled_members": 1396,
        "model_weights": [0.25, 0.25, 0.25, 0.25],
        "proxy_network_weights": {"proxy6": 0.5, "proxy9": 0.5},
        "allowed_periods": [[1500, 1920], [1940, 2000]],
        "sealed_period": [1921, 1939],
        "stage_c_enkf_authorized": False,
        "production_modified": False,
        "oggm_run": False,
        "real_proxy_observations_read": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "source_hashes_unchanged": source_unchanged,
        "source_hashes": before,
        "next_action": "prepare_and_validate_candidate_b2_inputs_without_real_data_enkf",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "# Candidate B2 preregistration\n\n"
        "Candidate B2 is frozen before any real-data EnKF. The rejected separate-"
        "assimilation Stage B route remains rejected. The new candidate uses an "
        "equal four-model pooled covariance and equal four-model evolving mean.\n\n"
        "No production file, real proxy observation, OGGM run, sealed year, or "
        "prohibited source was used. Stage C remains unauthorized pending input "
        "preflight and separate user approval.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
