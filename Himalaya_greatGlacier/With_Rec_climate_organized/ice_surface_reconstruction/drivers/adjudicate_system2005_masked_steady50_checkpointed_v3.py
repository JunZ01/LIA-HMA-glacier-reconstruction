"""Adjudicate the completed masked system-2005 steady run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE142 = BASE / "lia_191_current_route_v10_142/summary.json"
RUN = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3/summary.json"
OLD = BASE / "lia_191_torch_planform2d_system2005_steady50_monthly_ti_formal_v2/summary.json"
LAUNCH = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3_launch_v1/summary.json"
LOGS = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3_logs_v1"
OUT = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3_adjudication_v1"
ROUTE143 = BASE / "lia_191_current_route_v10_143"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    value = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + value.rstrip() + newline


def main() -> None:
    for path in (OUT, ROUTE143):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    run = json.loads(RUN.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.142-system2005-masked-steady50-checkpointed-active":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE142) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route142-hash-mismatch")
    if run["status"] != "masked-steady50-fail-stop" or run["completed_years"] != 50.0:
        raise RuntimeError("run-not-terminal-50y")
    expected_checks = dict(run["checks"])
    if expected_checks.pop("steady_volume_change_pass") is not False or not all(expected_checks.values()):
        raise RuntimeError("failure-not-volume-only")
    if (LOGS / "stderr.log").stat().st_size != 0:
        raise RuntimeError("stderr-not-empty")
    latest = json.loads(
        (RUN.parent / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    latest_manifest = Path(latest["manifest"])
    manifest = json.loads(latest_manifest.read_text(encoding="utf-8"))
    if manifest["year"] != 50 or sha256(latest_manifest.parent / manifest["state_file"]) != manifest["state_sha256"]:
        raise RuntimeError("terminal-checkpoint-invalid")

    old_change = float(old["final_volume_relative_change"])
    new_change = float(run["final_volume_relative_change"])
    source_reduction = 1.0 - float(run["cumulative_climate_source_m3"]) / float(
        old["cumulative_climate_source_volume_change_m3"]
    )
    adjudication = {
        "schema": "lia-191-system2005-masked-steady50-checkpointed-v3-adjudication-v1",
        "status": "numerical-domain-pass-scientific-steady-volume-fail",
        "recorded_at": now(),
        "system_id": 2005,
        "run_summary": str(RUN.relative_to(ROOT)),
        "run_summary_sha256": sha256(RUN),
        "launch_summary_sha256": sha256(LAUNCH),
        "pid": launch["pid"],
        "terminal_checkpoint_manifest": str(latest_manifest.relative_to(ROOT)),
        "terminal_checkpoint_state_sha256": manifest["state_sha256"],
        "results": {
            "completed_years": run["completed_years"],
            "runtime_seconds": run["runtime_seconds_this_process"],
            "volume_change_fraction": new_change,
            "volume_gate_fraction": 0.02,
            "length_error_fraction": run["final_root_path_metrics"]["registered_length_relative_error"],
            "free_domain_m": run["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
            "cumulative_source_m3": run["cumulative_climate_source_m3"],
            "mass_budget_residual_m3": run["cumulative_mass_budget_residual_m3"],
            "outside_domain_maximum_thickness_m": run["outside_domain_maximum_thickness_m"],
            "unmasked_volume_change_fraction": old_change,
            "absolute_volume_drift_reduction_fraction_points": old_change - new_change,
            "relative_volume_drift_reduction": 1.0 - new_change / old_change,
            "relative_positive_source_reduction": source_reduction,
        },
        "scientific_decision": {
            "source_mask_mechanism_necessary": True,
            "source_mask_mechanism_sufficient_for_steady_gate": False,
            "unmasked_exterior_area_feedback_removed": True,
            "remaining_failure_is_internal_domain_climate_surface_feedback": True,
            "historical_run_authorized": False,
            "accepted_count_effect": 0,
            "next_action": "read-only annual source, surface-elevation, active-support, and hypsometry decomposition",
        },
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
    }
    adjudication_bytes = payload(adjudication)
    adjudication_hash = hashlib.sha256(adjudication_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-143",
        "status": "system2005-masked-steady50-volume-fail-internal-feedback-audit-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_142",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": adjudication_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": "read-only decomposition of the masked run's annual climate source by surface elevation, support class, and hypsometry; no model launch",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 447. System-2005 masked steady50 scientific volume fail", """## 447. System-2005 masked steady50 numerical/domain pass and scientific volume fail (2026-08-29)

- The sole registered v3 process completed all 50 years in `7983.61 s` with 50 valid annual atomic checkpoints. Length error is `5.7943%`, free domain is `5.54125 km`, cumulative mass residual is `-7.15e-7 m3`, all states are finite/nonnegative, and exterior-domain thickness remains exactly zero.
- The frozen `2%` steady-volume gate fails: final volume change is `+14.6045%`, driven by `+1.179925e9 m3` cumulative climate source. The source-native mask reduces the old unmasked `+68.7603%` drift by `54.1558` percentage points (`78.76%`) and removes the diagnosed exterior-area feedback, but it is not sufficient for steady acceptance.
- Route v10.143 closes this masked-steady attempt as a scientific volume failure with no accepted-count effect. The next action is a read-only annual decomposition of internal-domain climate source, surface elevation, active support, and hypsometry. No historical run, initial-thickness modification, or production-temperature change is authorized.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.143: system 2005 masked steady50 volume fail", """## 2026-08-29 - Route v10.143: system 2005 masked steady50 volume fail

- Classified the completed run as numerical/domain pass and scientific steady-volume fail. The mask removes most of the old runaway but leaves `+14.6045%` internal-domain growth.
- Cleared active and queue counts. Authorized only a read-only internal source/elevation/hypsometry decomposition before any new mechanism is proposed.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.143-system2005-masked-steady50-volume-fail"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.143",
        "path": str((ROUTE143 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.142",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-steady50-volume-fail-internal-feedback-audit-required",
        "completed_years": 50.0,
        "volume_change_fraction": new_change,
        "root_length_error_fraction": run["final_root_path_metrics"]["registered_length_relative_error"],
        "free_domain_m": run["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "audit-system2005-masked-internal-source-elevation-feedback-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }
    stage = Path(tempfile.mkdtemp(prefix=".system2005_masked_steady50_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(adjudication_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE143.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE143 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE143, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.143", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
