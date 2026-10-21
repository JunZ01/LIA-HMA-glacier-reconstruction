from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import evaluate_temperature_candidate_b2_stage_c_v3 as evaluator
import run_temperature_candidate_b2_stage_c_real_proxy_v3 as stage_c
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_authorization_guards_v2"
MISSING_RECEIPT = LOOP / "DOES_NOT_EXIST_CANDIDATE_B2_AUTH.json"
REAL_TEST_OUTPUT = LOOP / "DO_NOT_CREATE_UNAUTHORIZED_CANDIDATE_B2_STAGE_C_V2"
EVALUATION_TEST_OUTPUT = LOOP / "DO_NOT_CREATE_SYNTHETIC_AS_REAL_EVALUATION_V2"
STATIC_PACKAGE = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/"
    "candidate_b2_stage_c_static_input_v2.npz"
)


def expect_permission_error(action) -> str:
    try:
        action()
    except PermissionError as error:
        return str(error)
    raise AssertionError("permission error was not raised")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if MISSING_RECEIPT.exists() or REAL_TEST_OUTPUT.exists() or EVALUATION_TEST_OUTPUT.exists():
        raise RuntimeError("guard-fixture-path-preexists")
    driver_error = expect_permission_error(
        lambda: stage_c.main(MISSING_RECEIPT, REAL_TEST_OUTPUT)
    )
    evaluator_error = expect_permission_error(
        lambda: evaluator.main(STATIC_PACKAGE, EVALUATION_TEST_OUTPUT)
    )
    gates = {
        "real_driver_v3_rejects_missing_receipt": "authorization receipt is absent" in driver_error,
        "real_driver_created_no_output": not REAL_TEST_OUTPUT.exists(),
        "evaluator_v3_rejects_non_real_input": "authorization chain is absent" in evaluator_error,
        "evaluator_created_no_output": not EVALUATION_TEST_OUTPUT.exists(),
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
    }
    passed = all(
        value
        for key, value in gates.items()
        if key not in {"real_proxy_observations_read", "stage_c_enkf_run"}
    ) and not gates["real_proxy_observations_read"] and not gates["stage_c_enkf_run"]
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "authorization_guards_pass": bool(passed),
        "gates": gates,
        "driver_rejection": driver_error,
        "evaluator_rejection": evaluator_error,
        "real_proxy_observations_read": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "driver": {"path": str(Path(stage_c.__file__).resolve()), "sha256": core.sha256(Path(stage_c.__file__).resolve())},
        "evaluator": {"path": str(Path(evaluator.__file__).resolve()), "sha256": core.sha256(Path(evaluator.__file__).resolve())},
        "test": {"path": str(DRIVER), "sha256": core.sha256(DRIVER)},
        "static_package": {"path": str(STATIC_PACKAGE), "sha256": core.sha256(STATIC_PACKAGE)},
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("authorization-guard-test-v2-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
