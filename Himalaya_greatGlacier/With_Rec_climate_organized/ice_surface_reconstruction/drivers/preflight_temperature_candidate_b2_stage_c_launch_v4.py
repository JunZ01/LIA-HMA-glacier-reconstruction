from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import evaluate_temperature_candidate_b2_stage_c_v1 as evaluator_v1
import evaluate_temperature_candidate_b2_stage_c_v2 as evaluator_v2
import run_temperature_candidate_b2_stage_c_real_proxy_v1 as driver_v1
import run_temperature_candidate_b2_stage_c_real_proxy_v2 as driver_v2
import run_temperature_candidate_b2_stage_c_real_proxy_v3 as driver_v3
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
STATIC_DIR = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2"
STATIC_PACKAGE = STATIC_DIR / "candidate_b2_stage_c_static_input_v2.npz"
SYNTHETIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_synthetic_preflight_v1/summary.json"
EVALUATOR_PREFLIGHT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluator_preflight_v1/summary.json"
GUARDS = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_authorization_guards_v1/summary.json"
PREVIOUS = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v3"
BASE_CONTRACT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1/"
    "CANDIDATE_B2_STAGE_C_CONTRACT.md"
)
CENTERING_CONTRACT = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers"
    / "temperature_candidate_b2_stage_c_observation_centering_v1_contract.md"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v4"


def canonical_hash(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def checked_summary(path: Path, key: str) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get(key) is not True:
        raise RuntimeError(f"preflight-not-passed:{path}:{key}")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    static = checked_summary(STATIC_DIR / "summary.json", "static_package_pass")
    synthetic = checked_summary(SYNTHETIC, "synthetic_preflight_pass")
    evaluator = checked_summary(EVALUATOR_PREFLIGHT, "evaluator_preflight_pass")
    guards = checked_summary(GUARDS, "authorization_guards_pass")
    previous = checked_summary(PREVIOUS / "summary.json", "preauthorization_launch_ready")
    if previous.get("finite_real_proxy_active_masks_known") is not False:
        raise RuntimeError("real-proxy-mask-boundary-not-preserved")
    if core.sha256(STATIC_PACKAGE) != core.EXPECTED_STATIC_PACKAGE_SHA256:
        raise RuntimeError("static-package-hash")

    implementation = {
        "core_v1": Path(core.__file__).resolve(),
        "real_driver_v1": Path(driver_v1.__file__).resolve(),
        "real_driver_v2": Path(driver_v2.__file__).resolve(),
        "real_driver_v3": Path(driver_v3.__file__).resolve(),
        "evaluator_v1": Path(evaluator_v1.__file__).resolve(),
        "evaluator_v2": Path(evaluator_v2.__file__).resolve(),
    }
    implementation_hashes = {
        name: core.sha256(path) for name, path in implementation.items()
    }
    frozen = {
        "candidate": "balanced_pooled_candidate_b2",
        "allowed_periods": [[1500, 1920], [1940, 2000]],
        "sealed_period": [1921, 1939],
        "models": ["CESM", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"],
        "pooled_members": 1396,
        "model_weights": [0.25, 0.25, 0.25, 0.25],
        "network_weights": {"proxy6": 0.5, "proxy9": 0.5},
        "observation_centering": (
            "raw proxy value minus archived CESM HXb mean over 1500-1848"
        ),
        "static_package_sha256": core.sha256(STATIC_PACKAGE),
        "base_contract_sha256": core.sha256(BASE_CONTRACT),
        "centering_contract_sha256": core.sha256(CENTERING_CONTRACT),
        "implementation_hashes": implementation_hashes,
        "active_real_driver": "run_temperature_candidate_b2_stage_c_real_proxy_v3.py",
        "active_evaluator": "evaluate_temperature_candidate_b2_stage_c_v2.py",
        "authorization_text_required": driver_v1.AUTHORIZATION_TEXT,
        "finite_real_proxy_masks_known_before_authorization": False,
        "production_write_allowed": False,
        "oggm_allowed": False,
        "prohibited_source": "20CRv3",
    }
    fingerprint = canonical_hash(frozen)
    old_manifest = json.loads((PREVIOUS / "launch_manifest.json").read_text(encoding="utf-8"))
    old_fingerprint = old_manifest["launch_fingerprint_sha256"]
    if fingerprint == old_fingerprint:
        raise RuntimeError("driver-upgrade-did-not-change-fingerprint")

    gates = {
        "static_package_pass": True,
        "synthetic_end_to_end_pass": True,
        "evaluator_metric_interface_pass": True,
        "authorization_guards_pass": True,
        "round277_full_value_closure_pass": bool(
            previous["round277_value_closure"]["all_value_closure_pass"]
        ),
        "real_proxy_active_masks_remain_unread": True,
        "all_implementation_dependencies_hashed": len(implementation_hashes) == 6,
        "real_proxy_value_cells_read": 0,
        "stage_c_real_proxy_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
    }
    launch_ready = (
        all(
            gates[name]
            for name in (
                "static_package_pass",
                "synthetic_end_to_end_pass",
                "evaluator_metric_interface_pass",
                "authorization_guards_pass",
                "round277_full_value_closure_pass",
                "real_proxy_active_masks_remain_unread",
                "all_implementation_dependencies_hashed",
            )
        )
        and gates["real_proxy_value_cells_read"] == 0
        and gates["stage_c_real_proxy_enkf_run"] is False
        and gates["production_modified"] is False
        and gates["oggm_run"] is False
    )
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "preauthorization_launch_ready": bool(launch_ready),
        "launch_fingerprint_sha256": fingerprint,
        "supersedes_launch_fingerprint_sha256": old_fingerprint,
        "gates": gates,
        "synthetic_direct_update_closure_degC": synthetic[
            "maximum_direct_member_mean_closure_degC"
        ],
        "round277_full_value_closure_max_error_degC": max(
            item["maximum_abs_error_degC"]
            for item in previous["round277_value_closure"]["numeric_errors"].values()
        ),
        "finite_real_proxy_active_masks_known": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "authorization_missing": True,
        "next_action": "await explicit user authorization, then create receipt bound to v4 fingerprint and v3 driver",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "launch_manifest.json").write_text(
        json.dumps(
            {
                "launch_fingerprint_sha256": fingerprint,
                "supersedes_launch_fingerprint_sha256": old_fingerprint,
                "frozen_configuration": frozen,
                "implementation_paths": {
                    name: str(path) for name, path in implementation.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_schema = {
        "schema": "candidate-b2-stage-c-authorization-v1",
        "authorized": True,
        "authorization_text": driver_v1.AUTHORIZATION_TEXT,
        "authorized_at": "TO_BE_FILLED_ONLY_AFTER_EXPLICIT_USER_AUTHORIZATION",
        "static_package_sha256": core.EXPECTED_STATIC_PACKAGE_SHA256,
        "driver_sha256": implementation_hashes["real_driver_v3"],
        "launch_fingerprint_sha256": fingerprint,
    }
    (OUT / "authorization_receipt_schema.json").write_text(
        json.dumps(receipt_schema, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_PREAUTHORIZATION_REPORT.md").write_text(
        "# Candidate B2 Stage C final preauthorization report v4\n\n"
        f"Launch preparation pass: `{launch_ready}`.\n\n"
        f"Active fingerprint: `{fingerprint}`.\n\n"
        f"Superseded fingerprint: `{old_fingerprint}`.\n\n"
        "The active chain is real driver v3 plus evaluator v2. Every wrapper and "
        "implementation dependency is hashed. The driver rejects missing "
        "authorization; the evaluator rejects static, synthetic, or otherwise "
        "unauthorized inputs. Finite real-proxy masks remain deliberately unknown.\n\n"
        "No real proxy value has been read and no real-data EnKF has run.\n",
        encoding="utf-8",
    )
    manifest_sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "static_summary": STATIC_DIR / "summary.json",
        "synthetic_summary": SYNTHETIC,
        "evaluator_preflight": EVALUATOR_PREFLIGHT,
        "authorization_guards": GUARDS,
        "previous_preflight": PREVIOUS / "summary.json",
        "base_contract": BASE_CONTRACT,
        "centering_contract": CENTERING_CONTRACT,
        "driver": DRIVER,
        **implementation,
    }
    source_manifest = {
        name: {"path": str(path), "sha256": core.sha256(path)}
        for name, path in manifest_sources.items()
    }
    source_manifest.update(
        {
            "real_proxy_observations_read": False,
            "stage_c_enkf_run": False,
            "sealed_period_values_used": False,
            "prohibited_source_accessed": False,
        }
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not launch_ready:
        raise RuntimeError("launch-preflight-v4-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
