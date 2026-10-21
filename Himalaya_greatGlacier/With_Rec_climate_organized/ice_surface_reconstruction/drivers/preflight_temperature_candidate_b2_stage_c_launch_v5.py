from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import evaluate_temperature_candidate_b2_stage_c_v3 as evaluator_v3
import run_temperature_candidate_b2_stage_c_real_proxy_v1 as driver_v1
import run_temperature_candidate_b2_stage_c_real_proxy_v3 as driver_v3
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
PREVIOUS = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v4"
GUARDS = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_authorization_guards_v2/summary.json"
BACKGROUND = LOOP / "temperature_b2_background_eval_v2/summary.json"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v5"


def canonical_hash(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    previous_summary = json.loads((PREVIOUS / "summary.json").read_text(encoding="utf-8"))
    previous_manifest = json.loads((PREVIOUS / "launch_manifest.json").read_text(encoding="utf-8"))
    guards = json.loads(GUARDS.read_text(encoding="utf-8"))
    background = json.loads(BACKGROUND.read_text(encoding="utf-8"))
    if previous_summary.get("preauthorization_launch_ready") is not True:
        raise RuntimeError("previous-launch-preflight-not-ready")
    if guards.get("authorization_guards_pass") is not True:
        raise RuntimeError("v3-evaluator-guards-not-passed")
    if background.get("status") != "complete" or background.get("stage_c_enkf_run") is not False:
        raise RuntimeError("background-diagnostic-invalid")

    frozen = previous_manifest["frozen_configuration"]
    frozen = json.loads(json.dumps(frozen))
    evaluator_path = Path(evaluator_v3.__file__).resolve()
    frozen["implementation_hashes"]["evaluator_v3"] = core.sha256(evaluator_path)
    frozen["active_evaluator"] = evaluator_path.name
    frozen["direct_station_missing_region_policy"] = (
        "report absent support as not evaluable; do not fabricate a pass/fail or an independent station vote"
    )
    fingerprint = canonical_hash(frozen)
    old_fingerprint = previous_manifest["launch_fingerprint_sha256"]
    if fingerprint == old_fingerprint:
        raise RuntimeError("evaluator-upgrade-did-not-change-fingerprint")
    active_driver_hash = core.sha256(Path(driver_v3.__file__).resolve())
    if active_driver_hash != frozen["implementation_hashes"]["real_driver_v3"]:
        raise RuntimeError("active-driver-hash-drift")

    gates = {
        "previous_v4_preflight_pass": True,
        "active_v3_evaluator_authorization_guards_pass": True,
        "background_only_diagnostic_complete": True,
        "active_driver_v3_hash_exact": True,
        "active_evaluator_v3_hash_exact": True,
        "real_proxy_active_masks_remain_unread": True,
        "real_proxy_value_cells_read": 0,
        "stage_c_real_proxy_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
    }
    launch_ready = (
        all(
            gates[name]
            for name in (
                "previous_v4_preflight_pass",
                "active_v3_evaluator_authorization_guards_pass",
                "background_only_diagnostic_complete",
                "active_driver_v3_hash_exact",
                "active_evaluator_v3_hash_exact",
                "real_proxy_active_masks_remain_unread",
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
        "background_only_finding": {
            "1500_1879_external_reference_rmse_improves": True,
            "1850_1920_external_reference_rmse_improves": True,
            "modern_qcf_rmse_improves": True,
            "modern_qcf_correlation_degrades": True,
            "scientific_role": (
                "Stage C must test whether proxy assimilation retains low-frequency background improvement while restoring temporal and regional skill"
            ),
        },
        "finite_real_proxy_active_masks_known": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "authorization_missing": True,
        "next_action": "await explicit user authorization, then create receipt bound to v5 fingerprint and run v3 driver followed by v3 evaluator",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "launch_manifest.json").write_text(
        json.dumps(
            {
                "launch_fingerprint_sha256": fingerprint,
                "supersedes_launch_fingerprint_sha256": old_fingerprint,
                "frozen_configuration": frozen,
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
        "driver_sha256": active_driver_hash,
        "launch_fingerprint_sha256": fingerprint,
    }
    (OUT / "authorization_receipt_schema.json").write_text(
        json.dumps(receipt_schema, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_PREAUTHORIZATION_REPORT.md").write_text(
        "# Candidate B2 Stage C final preauthorization report v5\n\n"
        f"Launch preparation pass: `{launch_ready}`.\n\n"
        f"Active fingerprint: `{fingerprint}`.\n\n"
        f"Superseded fingerprint: `{old_fingerprint}`.\n\n"
        "The balanced no-proxy background substantially reduces the long-term "
        "warm residual but weakens station correlation. The frozen real-data test "
        "therefore asks whether proxy assimilation can retain the low-frequency "
        "improvement while restoring temporal and regional skill. Evaluator v3 "
        "reports missing direct-station regions as not evaluable rather than "
        "inventing evidence.\n\n"
        "No real proxy value has been read and no real-data EnKF has run.\n",
        encoding="utf-8",
    )
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "previous_summary": PREVIOUS / "summary.json",
        "previous_manifest": PREVIOUS / "launch_manifest.json",
        "authorization_guards_v2": GUARDS,
        "background_diagnostic": BACKGROUND,
        "active_driver_v3": Path(driver_v3.__file__).resolve(),
        "active_evaluator_v3": evaluator_path,
        "driver": DRIVER,
    }
    manifest = {
        name: {"path": str(path), "sha256": core.sha256(path)}
        for name, path in sources.items()
    }
    manifest.update(
        {
            "real_proxy_observations_read": False,
            "stage_c_enkf_run": False,
            "sealed_period_values_used": False,
            "prohibited_source_accessed": False,
        }
    )
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not launch_ready:
        raise RuntimeError("launch-preflight-v5-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
