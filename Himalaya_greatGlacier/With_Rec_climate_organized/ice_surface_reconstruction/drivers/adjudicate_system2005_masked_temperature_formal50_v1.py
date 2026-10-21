"""Adjudicate system-2005 formal50 and queue one independent reproduction."""

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
ROUTE148 = BASE / "lia_191_current_route_v10_148/summary.json"
RUN_ROOT = BASE / "lia_191_system2005_masked_temperature_formal50_v1"
RUN = RUN_ROOT / "summary.json"
LOG = BASE / "lia_191_system2005_masked_temperature_formal50_v1_logs_v1/stderr.log"
BASELINE = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3/summary.json"
OUT = BASE / "lia_191_system2005_masked_temperature_formal50_adjudication_v1"
ROUTE149 = BASE / "lia_191_current_route_v10_149"


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
    for path in (OUT, ROUTE149):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    result = json.loads(RUN.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.148-system2005-masked-temperature-formal50-active":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE148) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route148-hash-mismatch")
    if result["status"] != "masked-temperature-formal50-pass" or not all(result["checks"].values()):
        raise RuntimeError("formal50-not-pass")
    if LOG.read_bytes():
        raise RuntimeError("formal50-stderr-not-empty")

    checkpoint_hashes = {}
    for year in range(1, 51):
        directory = RUN_ROOT / f"checkpoints/year_{year:03d}"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state_path = directory / manifest["state_file"]
        if sha256(state_path) != manifest["state_sha256"].upper():
            raise RuntimeError(f"checkpoint-hash-mismatch:{year}")
        checkpoint_hashes[str(year)] = sha256(state_path)

    volume_improvement = float(baseline["final_volume_relative_change"]) - float(result["final_volume_relative_change"])
    reproduction = {
        "system_id": 2005,
        "run_role": "independent-reproduction",
        "duration_years": 50,
        "delta_t_k": float(result["temperature_offset_k"]),
        "initial_state": "unchanged source-native year-zero 2D thickness",
        "domain": "lia_source_mask",
        "solver_and_forcing": "bitwise-identical registered formal scientific contract in a separate output root",
        "annual_atomic_checkpoints": True,
        "hard_gates": {
            "all_formal_scientific_gates": True,
            "final_state_sha256_exact": result["final_state_sha256"],
            "all_50_checkpoint_state_sha256_exact": checkpoint_hashes,
            "stderr_empty": True,
            "source_and_initial_state_hashes_unchanged": True,
        },
        "terminal_rule": "pass completes steady-stage independent reproduction; failure blocks acceptance and requires deterministic-discrepancy adjudication",
    }
    evidence = {
        "schema": "lia-191-system2005-masked-temperature-formal50-adjudication-v1",
        "status": "formal50-pass-reproduction-queued",
        "recorded_at": now(),
        "system_id": 2005,
        "result": str(RUN.relative_to(ROOT)),
        "result_sha256": sha256(RUN),
        "checkpoint_state_sha256": checkpoint_hashes,
        "formal": {
            "temperature_offset_k": result["temperature_offset_k"],
            "runtime_seconds": result["runtime_seconds_this_process"],
            "volume_change_fraction": result["final_volume_relative_change"],
            "baseline_volume_change_fraction": baseline["final_volume_relative_change"],
            "volume_drift_reduction_percentage_points": 100.0 * volume_improvement,
            "root_length_m": result["final_root_path_metrics"]["observed_active_control_volume_length_m"],
            "root_length_error_fraction": result["final_root_path_metrics"]["registered_length_relative_error"],
            "free_domain_m": result["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
            "cumulative_source_m3": result["cumulative_climate_source_m3"],
            "mass_budget_residual_m3": result["cumulative_mass_budget_residual_m3"],
            "final_state_sha256": result["final_state_sha256"],
        },
        "queued_reproduction": reproduction,
        "accepted_count_effect": 0,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "historical_run": False,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-149",
        "status": "system2005-masked-temperature-reproduction-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_148",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "queued_run": reproduction,
        "next_legal_action": "launch exactly one independent 50-year reproduction of the passed system-2005 masked temperature formal contract; no duplicate beyond this registered reproduction and no historical run",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 453. System-2005 masked temperature formal50 pass", f"""## 453. System-2005 masked temperature formal50 pass (2026-08-29)

- The sole registered `+0.3342811518826305 K` masked 2D formal run completed 50 years in `{result['runtime_seconds_this_process']:.2f} s`. Final volume change is `{result['final_volume_relative_change']:+.4%}` versus baseline `{baseline['final_volume_relative_change']:+.4%}`, a reduction of `{100.0*volume_improvement:.3f}` percentage points. Root length is `{result['final_root_path_metrics']['observed_active_control_volume_length_m']/1000:.3f} km` with `{result['final_root_path_metrics']['registered_length_relative_error']:.4%}` error; free domain is `{result['final_free_domain_metrics']['minimum_active_cell_center_to_grid_edge_m']/1000:.3f} km`.
- All 50 annual checkpoint hashes pass. Exterior thickness is exactly zero, cumulative mass residual is `{result['cumulative_mass_budget_residual_m3']:.3e} m3`, all numerical/source/runtime gates pass, stderr is empty, and final-state SHA-256 is `{result['final_state_sha256']}`.
- Route v10.149 queues exactly one independent fresh-year-zero reproduction in a separate output root and preregisters exact final and annual state hashes. No accepted-count change occurs before reproduction adjudication; no production-temperature or historical action is authorized.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.149: system 2005 masked temperature reproduction queued", """## 2026-08-29 - Route v10.149: system 2005 masked temperature reproduction queued

- Adjudicated the 50-year formal masked temperature run as a complete scientific pass under all frozen gates.
- Queued one independent reproduction with exact final-state and all annual checkpoint hashes. Accepted counts remain unchanged pending reproduction.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.149-system2005-masked-temperature-reproduction-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.149",
        "path": str((ROUTE149 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.148",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 1}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "queued-masked-temperature-reproduction",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-formal50-pass-reproduction-queued",
        "completed_years": 50.0,
        "volume_change_fraction": result["final_volume_relative_change"],
        "root_length_error_fraction": result["final_root_path_metrics"]["registered_length_relative_error"],
        "free_domain_m": result["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "launch-system2005-masked-temperature-reproduction-v1",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_temperature_formal_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE149.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE149 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE149, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.149", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
