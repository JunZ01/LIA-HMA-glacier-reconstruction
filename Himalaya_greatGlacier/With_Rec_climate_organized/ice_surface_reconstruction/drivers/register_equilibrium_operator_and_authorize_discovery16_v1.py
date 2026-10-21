"""Register the common equilibrium operator and discovery16 staged steady batch.

This controller transaction starts no model. It freezes the operator, cohort,
wave order, gates, and failure policy, and authorizes batch-wide preparation.
Dynamic launch remains gated on immutable preparation and short qualification
evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE182 = BASE / "lia_191_current_route_v10_182/summary.json"
AUDIT_DIR = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v4"
AUDIT = AUDIT_DIR / "summary.json"
ATTRIBUTION = AUDIT_DIR / "static_equilibrium_attribution_93.csv"
COHORTS = BASE / "lia_191_population_extension_cohorts_v1/population_extension_cohort_assignment.csv"
OUT = BASE / "lia_191_discovery16_equilibrium_centered_staged_steady_authorization_v1"
ROUTE183 = BASE / "lia_191_current_route_v10_183"

OPERATOR = "fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1"
EXPECTED_DISCOVERY = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 4588, 5770, 2559]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest().upper()


def main() -> int:
    if OUT.exists() or ROUTE183.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE183}")

    state = load(STATE)
    if state.get("state_id") != "2026-08-31-v10.182-three-terminal-scientific-fail-population-design":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(ROUTE182) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route182-hash-mismatch")
    if sha256(FACTS) != state["scientific_contract"]["sha256"].upper():
        raise RuntimeError("facts-hash-mismatch")
    for path in [AUDIT, ATTRIBUTION, COHORTS]:
        if not path.exists():
            raise FileNotFoundError(path)

    audit = load(AUDIT)
    if audit.get("status") != "read-only-static-equilibrium-audit-pass":
        raise RuntimeError("equilibrium-audit-not-passed")
    if audit.get("operator", {}).get("name") != OPERATOR:
        raise RuntimeError("operator-name-mismatch")
    if int(audit.get("population", -1)) != 93:
        raise RuntimeError("audit-population-mismatch")
    required_checks = [
        "population_count_93", "input_inventory_complete", "registered_input_rows_unambiguous",
        "reference_package_formula", "reference_initial_tendency_sign", "equilibrium_roots",
        "no_dynamics_run", "endpoint_targets_not_used_by_operator", "production_temperature_unchanged",
        "initial_thickness_unchanged", "old_outputs_unchanged", "launch_remains_unauthorized",
    ]
    if not all(audit["checks"].get(key) is True for key in required_checks):
        raise RuntimeError("audit-gate-failure")

    cohort_rows = read_csv(COHORTS)
    discovery_rows = sorted(
        (row for row in cohort_rows if row["cohort"] == "discovery_global_mechanism"),
        key=lambda row: int(row["cohort_order"]),
    )
    heldout_ids = {int(row["system_id"]) for row in cohort_rows if row["cohort"] == "heldout_independent_confirmation"}
    discovery_ids = [int(row["system_id"]) for row in discovery_rows]
    if discovery_ids != EXPECTED_DISCOVERY:
        raise RuntimeError(f"discovery16-membership-drift:{discovery_ids}")
    if set(discovery_ids) & heldout_ids:
        raise RuntimeError("discovery-heldout-overlap")

    attribution_rows = {int(row["system_id"]): row for row in read_csv(ATTRIBUTION)}
    if len(attribution_rows) != 93:
        raise RuntimeError("attribution-population-mismatch")

    manifest_rows: list[dict[str, Any]] = []
    for cohort in discovery_rows:
        sid = int(cohort["system_id"])
        row = attribution_rows[sid]
        if row["cohort"] != "discovery_global_mechanism":
            raise RuntimeError(f"cohort-attribution-mismatch:{sid}")
        residual = float(row["equilibrium_residual_m_ice_yr"])
        if abs(residual) > 1e-10:
            raise RuntimeError(f"equilibrium-residual-gate:{sid}:{residual}")
        stage2_bias = float(row["stage2_temp_bias_c"])
        correction = float(row["equilibrium_delta_t_c_from_stage2_base"])
        member = {
            "system_id": sid,
            "rgi_id": row["rgi_id"],
            "cohort_order": int(row["cohort_order"]),
            "wave": (int(row["cohort_order"]) + 1) // 2,
            "wave_slot": "A" if int(row["cohort_order"]) % 2 else "B",
            "size_class": row["size_class"],
            "retreat_class": row["retreat_class"],
            "stage2_melt_f": float(row["stage2_melt_f"]),
            "stage2_prcp_fac": float(row["stage2_prcp_fac"]),
            "stage2_temp_bias_c": stage2_bias,
            "equilibrium_delta_t_c_from_stage2_base": correction,
            "effective_centered_temp_bias_c": stage2_bias + correction,
            "equilibrium_residual_m_ice_yr": residual,
            "equilibrium_root_final_width_c": float(row["equilibrium_root_final_width_c"]),
            "reference_parent_mb_m_ice_yr": float(row["static_parent_mb_reference_m_ice_yr"]),
            "equilibrium_parent_ela_m": float(row["equilibrium_parent_ela_m"]),
            "equilibrium_accumulation_area_fraction": float(row["equilibrium_accumulation_area_fraction"]),
            "active_cell_count": int(row["active_cell_count"]),
            "active_area_m2": float(row["active_area_m2"]),
            "initial_volume_m3": float(row["initial_volume_m3"]),
            "stable_selection_key": cohort["stable_selection_key"],
        }
        member["registered_operator_fingerprint"] = fingerprint({
            "operator": OPERATOR,
            "audit_sha256": sha256(AUDIT),
            "system_id": sid,
            "rgi_id": member["rgi_id"],
            "stage2_melt_f": member["stage2_melt_f"],
            "stage2_prcp_fac": member["stage2_prcp_fac"],
            "stage2_temp_bias_c": member["stage2_temp_bias_c"],
            "equilibrium_delta_t_c_from_stage2_base": correction,
            "parent_climate_years": [1441, 1470],
            "precipitation_anomaly_mode": "off",
        })
        manifest_rows.append(member)

    waves = []
    for index in range(0, len(manifest_rows), 2):
        pair = manifest_rows[index:index + 2]
        waves.append({
            "wave": index // 2 + 1,
            "system_a": pair[0]["system_id"],
            "system_b": pair[1]["system_id"],
            "maximum_concurrent_gpu_processes": 2,
            "qualification_release": "pending-stage-a-preparation-pass",
            "steady50_release": "pending-stage-b-qualification-pass",
        })

    stage_contract = {
        "stage_a": {
            "name": "batch-wide-static-materialization-and-package-validation",
            "members": 16,
            "authorized_now": True,
            "dynamics": False,
            "all_member_requirements": [
                "immutable source-grid and padded-adapter hashes",
                "exact source-to-adapter thickness conservation and exterior-zero mask",
                "360-month 1441-1470 centered MonthlyTI package with formula parity",
                "unique output and checkpoint roots",
                "per-wave runtime and GPU-memory estimate",
            ],
        },
        "stage_b": {
            "name": "batch-wide-short-centered-qualification",
            "members": 16,
            "authorized_contractually": True,
            "launch_release": "only-after-stage-a-16-of-16-pass-and-new-route",
            "phases": ["0.1-year zero-MB", "1-year equilibrium-centered MonthlyTI"],
            "wave_count": 8,
            "maximum_concurrent_gpu_processes": 2,
        },
        "stage_c": {
            "name": "equilibrium-centered-parent-climate-steady50",
            "members": 16,
            "authorized_contractually": True,
            "launch_release": "only-after-stage-b-formal-adjudication-and-new-route",
            "years": 50,
            "wave_count": 8,
            "maximum_concurrent_gpu_processes": 2,
            "checkpoint_cadence_years": 1,
        },
    }
    gates = {
        "finite_nonnegative": True,
        "outside_domain_maximum_thickness_m": 0.0,
        "mass_closure_relative_tolerance": 1e-10,
        "minimum_timestep_seconds": 0.001,
        "maximum_timestep_seconds": 86400.0,
        "steady_volume_relative_change_maximum": 0.02,
        "registered_length_relative_error_maximum": 0.10,
        "minimum_free_domain_m": 5000.0,
        "annual_atomic_checkpoint_resume_required": True,
        "initial_thickness_modification_allowed": False,
    }
    summary = {
        "schema": "lia-191-discovery16-equilibrium-centered-staged-steady-authorization-v1",
        "status": "operator-registered-discovery16-staged-steady-contract-authorized-stage-a-released",
        "recorded_at": now(),
        "authorization_basis": "explicit user approval: register common equilibrium operator and authorize discovery16 staged steady batch",
        "operator": audit["operator"],
        "operator_registration": {
            "registered": True,
            "registration_scope": "population93 extension research mechanism",
            "discovery_mechanism": OPERATOR,
            "endpoint_targets_used": False,
            "glacier_specific_manual_tuning": False,
            "production_source_climate_files_modified": False,
            "centered_forcing_packages_to_be_materialized": True,
        },
        "discovery16": {
            "count": 16,
            "system_ids": discovery_ids,
            "fixed_membership": True,
            "replacement_allowed": False,
            "heldout16_sealed": True,
            "heldout_system_ids_disclosed_to_execution": False,
        },
        "staged_batch": stage_contract,
        "resource_policy": {
            "fixed_wave_order": True,
            "waves": 8,
            "maximum_concurrent_gpu_processes": 2,
            "four_way_launch_authorized": False,
            "basis": "systems7/79 measured two-way throughput benchmark; no passed four-way expansion gate",
            "failure_isolation": "a failed process affects only its own output; the other process is not stopped",
        },
        "gates": gates,
        "failure_policy": {
            "technical_failure": "one exact-contract versioned retry only after cause identification",
            "scientific_failure": "retain in discovery16 denominator; no per-glacier correction, rescue, or gate change",
            "batch_learning": "analyze pass/fail covariates only after the frozen cohort stage closes",
        },
        "prohibitions": [
            "no per-glacier equilibrium search or manual correction",
            "no discovery16 member replacement",
            "no heldout16 use before discovery adjudication",
            "no historical chain for discovery16",
            "no system2005 warmer launch",
            "no initial-thickness, physics, solver, or gate change",
            "no accepted-count change from steady results alone",
        ],
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queued": 16,
        "stage_a_preparation_authorized": True,
        "dynamic_launch_authorized_now": False,
        "accepted_count_effect": 0,
        "model_run_performed": False,
        "initial_thickness_modified": False,
        "old_output_modified": False,
        "next_legal_action": "prepare and hash all 16 equilibrium-centered planform2d adapters and 1441-1470 MonthlyTI packages as one batch; launch no dynamics until 16-of-16 preparation passes",
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 486. Common equilibrium operator and discovery16 staged steady contract registered", """## 486. Common equilibrium operator and discovery16 staged steady contract registered (2026-09-01)

- The read-only population93 audit passes and registers `fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1` as one new research mechanism. It uses the fixed 1441-1470 parent climate, precipitation anomalies off, common melt/snow/lapse/density constants, bracket `[-10, 10] C`, and 64 bisection iterations to set area-integrated initial net surface mass balance to zero on each frozen initial mask. Endpoint length targets and glacier-specific manual tuning are not used.
- Discovery16 membership is frozen to systems `988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 4588, 5770, 2559`. No member replacement is allowed and heldout16 remains sealed.
- The staged batch is authorized as one cohort: all 16 first receive immutable adapter/package preparation; all 16 then receive the same short zero-MB plus centered-climate qualification; qualifying execution proceeds to the frozen 50-year steady test in eight fixed two-process GPU waves. Dynamic stages require new route snapshots after their preceding all-batch gate.
- The inherited steady gates remain unchanged: 2% volume change, 10% registered-length error, 5 km free domain, exact exterior-zero ice, finite/nonnegative state, 1e-10 relative mass closure, and annual atomic checkpoints. Scientific failures remain in the denominator and receive no individual correction or rescue. A technical failure permits only one exact-contract retry after cause identification.
- Route v10.183 releases batch-wide static preparation only. It starts no model, changes no initial thickness or source climate file, authorizes no historical chain or system-2005 warmer run, and changes no accepted count.""")
    decisions_after = append_section(decisions_before, "## 2026-09-01 - Route v10.183: register common equilibrium operator and authorize discovery16 staged steady batch", """## 2026-09-01 - Route v10.183: register common equilibrium operator and authorize discovery16 staged steady batch

- Register the population93 zero-net-MB equilibrium-centering operator as one disclosed new research mechanism.
- Freeze discovery16 membership and eight two-process GPU waves; do not substitute heldout or expansion members.
- Authorize the full staged batch contract and release only the 16-of-16 static adapter/package preparation step. Short qualification and 50-year steady launches require their own validated route snapshots after upstream batch gates pass.
- Retain every scientific failure in the discovery denominator. Prohibit per-glacier tuning, individual rescue, historical launch, initial-thickness change, and accepted-count change.""")

    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-183",
        "status": "equilibrium-operator-registered-discovery16-staged-steady-authorized-stage-a-preparation",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 16,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_182",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": summary_hash,
        "registered_operator": OPERATOR,
        "discovery16_batch_contract_authorized": True,
        "stage_a_preparation_authorized": True,
        "dynamic_launch_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    state183 = dict(state)
    state183["state_id"] = "2026-09-01-v10.183-equilibrium-operator-discovery16-staged-steady-preparation"
    state183["updated_at"] = now()
    state183["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    state183["authoritative_route"] = {
        "version": "v10.183",
        "path": str((ROUTE183 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.182",
    }
    state183["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 16}
    state183["current_focus"] = {
        "system_id": "discovery16",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "queued-staged-steady-preparation",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "historical_run_active": False,
    }
    branches = dict(state183.get("research_branches", {}))
    branches["population93_equilibrium_centered_planform2d_v1"] = {
        "status": "operator-registered-discovery16-staged-steady-authorized-stage-a-preparation",
        "systems": discovery_ids,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "batch_contract_authorized": True,
        "dynamic_launch_authorized": False,
        "heldout16_sealed": True,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    }
    state183["research_branches"] = branches
    state183["next_action"] = {
        "type": "prepare-discovery16-equilibrium-centered-planform2d-batch-v1",
        "launch_authorized": False,
        "description": summary["next_legal_action"],
    }
    state_bytes = json_bytes(state183)

    stage = Path(tempfile.mkdtemp(prefix=".discovery183_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        with (evidence_stage / "discovery16_member_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        with (evidence_stage / "staged_gpu_waves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(waves[0]))
            writer.writeheader()
            writer.writerows(waves)
        source_manifest = {
            "created_at": now(),
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "sources": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in [ROUTE182, AUDIT, ATTRIBUTION, COHORTS]
            ],
        }
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes(source_manifest))
        (route_stage / "summary.json").write_bytes(route_bytes)

        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE183))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE183, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
