"""Adjudicate exact system-2005 reproduction and advance to historical readiness audit."""

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
ROUTE151 = BASE / "lia_191_current_route_v10_151/summary.json"
FORMAL = BASE / "lia_191_system2005_masked_temperature_formal50_v1/summary.json"
REPRO = BASE / "lia_191_system2005_masked_temperature_reproduction_v1/summary.json"
LOG = BASE / "lia_191_system2005_masked_temperature_reproduction_v1_logs_v1/stderr.log"
OUT = BASE / "lia_191_system2005_masked_temperature_reproduction_adjudication_v1"
ROUTE152 = BASE / "lia_191_current_route_v10_152"


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
    for path in (OUT, ROUTE152):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    repro = json.loads(REPRO.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.151-system2005-masked-temperature-reproduction-active":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE151) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route151-hash-mismatch")
    if formal["status"] != "masked-temperature-formal50-pass" or not all(formal["checks"].values()):
        raise RuntimeError("formal-not-pass")
    if repro["status"] != "masked-temperature-reproduction-pass" or not all(repro["checks"].values()):
        raise RuntimeError("reproduction-not-pass")
    if LOG.read_bytes():
        raise RuntimeError("reproduction-stderr-not-empty")
    scalar_fields = (
        "final_volume_m3",
        "final_volume_relative_change",
        "cumulative_climate_source_m3",
        "cumulative_mass_budget_residual_m3",
    )
    if any(formal[field] != repro[field] for field in scalar_fields):
        raise RuntimeError("formal-reproduction-scalar-mismatch")
    if formal["final_root_path_metrics"] != repro["final_root_path_metrics"]:
        raise RuntimeError("formal-reproduction-length-mismatch")

    evidence = {
        "schema": "lia-191-system2005-masked-temperature-reproduction-adjudication-v1",
        "status": "steady-formal-and-independent-reproduction-pass",
        "recorded_at": now(),
        "system_id": 2005,
        "formal": str(FORMAL.relative_to(ROOT)),
        "formal_sha256": sha256(FORMAL),
        "reproduction": str(REPRO.relative_to(ROOT)),
        "reproduction_sha256": sha256(REPRO),
        "exact_reproduction": {
            "all_50_checkpoint_thickness_arrays": repro["checks"]["all_50_checkpoint_thickness_arrays_exact"],
            "final_thickness_array": repro["checks"]["final_thickness_array_exact"],
            "final_domain_mask_array": repro["checks"]["final_domain_mask_array_exact"],
            "final_thickness_array_sha256": repro["final_thickness_array_sha256"],
            "final_domain_mask_array_sha256": repro["final_domain_mask_array_sha256"],
        },
        "steady_result": {
            "temperature_offset_k": repro["temperature_offset_k"],
            "volume_change_fraction": repro["final_volume_relative_change"],
            "root_length_m": repro["final_root_path_metrics"]["observed_active_control_volume_length_m"],
            "root_length_error_fraction": repro["final_root_path_metrics"]["registered_length_relative_error"],
            "free_domain_m": repro["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
            "mass_budget_residual_m3": repro["cumulative_mass_budget_residual_m3"],
        },
        "scientific_decision": {
            "steady_stage_reproduced": True,
            "historical_stage_passed": False,
            "system_accepted": False,
            "next_action": "read-only audit historical forcing, endpoint operators, runtime/checkpoint contract, and model-family promotion requirements before any historical run",
        },
        "accepted_count_effect": 0,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "historical_run": False,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-152",
        "status": "system2005-masked-temperature-steady-reproduced-historical-readiness-audit-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_151",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": "read-only system2005 masked-temperature historical-readiness audit; no model launch, no production temperature change, and no accepted-count change",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 456. System-2005 masked temperature steady reproduction pass", f"""## 456. System-2005 masked temperature steady reproduction pass (2026-08-29)

- The sole registered independent reproduction completed 50 years in `{repro['runtime_seconds_this_process']:.2f} s` and passed every formal numerical, domain, volume, length, free-domain, source-hash, and runtime gate. Its final volume change is `{repro['final_volume_relative_change']:+.4%}`, root-length error `{repro['final_root_path_metrics']['registered_length_relative_error']:.4%}`, and free domain `{repro['final_free_domain_metrics']['minimum_active_cell_center_to_grid_edge_m']/1000:.3f} km`.
- All 50 annual `thickness_m` arrays, final thickness array, and final domain-mask array are bitwise identical to the formal run under the corrected array-content hash contract. Final thickness-array SHA-256 is `{repro['final_thickness_array_sha256']}`; reproduction stderr is empty.
- This completes the masked 2D steady-stage formal and independent reproduction requirements for system 2005. It does not establish a 1500–2000 historical pass, promote the research model family, modify production temperature, or change accepted counts. Route v10.152 authorizes only a read-only historical-readiness audit.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.152: system 2005 steady reproduction passed", """## 2026-08-29 - Route v10.152: system 2005 steady reproduction passed

- Adjudicated exact formal/reproduction agreement for all annual and terminal scientific arrays.
- Cleared active and queued states. Accepted counts remain 58/133 because historical-stage and model-family promotion requirements are not yet satisfied.
- Authorized only a read-only historical-readiness audit before any new launch.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.152-system2005-masked-temperature-steady-reproduced"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.152",
        "path": str((ROUTE152 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.151",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "steady-reproduced-historical-readiness-audit",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-steady-formal-reproduction-pass-historical-readiness-audit",
        "completed_years": 50.0,
        "volume_change_fraction": repro["final_volume_relative_change"],
        "root_length_error_fraction": repro["final_root_path_metrics"]["registered_length_relative_error"],
        "free_domain_m": repro["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "audit-system2005-masked-temperature-historical-readiness-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_temperature_reproduction_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE152.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE152 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE152, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.152", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
