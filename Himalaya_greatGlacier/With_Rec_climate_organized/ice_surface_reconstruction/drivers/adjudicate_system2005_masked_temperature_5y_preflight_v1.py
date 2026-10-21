"""Adjudicate the completed system-2005 temperature preflight and queue formal50."""

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
ROUTE146 = BASE / "lia_191_current_route_v10_146/summary.json"
RUN_ROOT = BASE / "lia_191_system2005_masked_temperature_5y_preflight_v1"
RUN = RUN_ROOT / "summary.json"
LOG = BASE / "lia_191_system2005_masked_temperature_5y_preflight_v1_logs_v1/stderr.log"
BASELINE = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3/summary.json"
OUT = BASE / "lia_191_system2005_masked_temperature_5y_preflight_adjudication_v1"
ROUTE147 = BASE / "lia_191_current_route_v10_147"


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
    for path in (OUT, ROUTE147):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    result = json.loads(RUN.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.146-system2005-masked-temperature-5y-preflight-active":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE146) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route146-hash-mismatch")
    if result["status"] != "masked-temperature-5y-preflight-pass" or not all(result["checks"].values()):
        raise RuntimeError("preflight-not-pass")
    if LOG.read_bytes():
        raise RuntimeError("preflight-stderr-not-empty")
    checkpoint_hashes = {}
    for year in range(1, 6):
        directory = RUN_ROOT / f"checkpoints/year_{year:03d}"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state_path = directory / manifest["state_file"]
        if sha256(state_path) != manifest["state_sha256"].upper():
            raise RuntimeError(f"checkpoint-hash-mismatch:{year}")
        checkpoint_hashes[str(year)] = {
            "manifest_sha256": sha256(manifest_path),
            "state_sha256": sha256(state_path),
        }

    baseline5 = baseline["checkpoint_metrics"]["5"]
    delta_source = float(result["cumulative_climate_source_m3"]) - float(baseline5["cumulative_source_m3"])
    formal = {
        "system_id": 2005,
        "duration_years": 50,
        "delta_t_k": float(result["temperature_offset_k"]),
        "initial_state": "unchanged source-native 2D thickness; restart from year 0, not the preflight endpoint",
        "domain": "lia_source_mask",
        "forcing": "same 600-month exported MonthlyTI sequence with the registered in-memory offset",
        "solver": "same masked source-half/diffusion/source-half operator and exact frozen constants",
        "annual_atomic_checkpoints": True,
        "hard_gates": {
            "completed_years": 50,
            "absolute_volume_change_fraction_max": 0.02,
            "registered_length_relative_error_max": 0.10,
            "free_domain_m_min": 5000.0,
            "outside_domain_thickness_m_exact": 0.0,
            "cumulative_budget_relative_max": 1.0e-10,
            "finite_nonnegative": True,
            "source_and_initial_state_hashes_unchanged": True,
        },
        "terminal_rule": "pass makes an independent reproduction eligible; failure closes the masked 2D scalar-temperature mechanism",
    }
    evidence = {
        "schema": "lia-191-system2005-masked-temperature-5y-preflight-adjudication-v1",
        "status": "preflight-pass-formal50-queued",
        "recorded_at": now(),
        "system_id": 2005,
        "result": str(RUN.relative_to(ROOT)),
        "result_sha256": sha256(RUN),
        "checkpoint_hashes": checkpoint_hashes,
        "preflight": {
            "temperature_offset_k": result["temperature_offset_k"],
            "runtime_seconds": result["runtime_seconds_this_process"],
            "volume_change_fraction": result["final_volume_relative_change"],
            "root_length_error_fraction": result["final_root_path_metrics"]["registered_length_relative_error"],
            "free_domain_m": result["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
            "cumulative_source_m3": result["cumulative_climate_source_m3"],
            "baseline_5y_source_m3": baseline5["cumulative_source_m3"],
            "source_change_vs_baseline_m3": delta_source,
            "mass_budget_residual_m3": result["cumulative_mass_budget_residual_m3"],
        },
        "queued_formal": formal,
        "accepted_count_effect": 0,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "historical_run": False,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-147",
        "status": "system2005-masked-temperature-formal50-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_146",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "queued_run": formal,
        "next_legal_action": "launch exactly one registered checkpointed 50-year masked 2D temperature formal run for system 2005; no duplicate or historical run",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 451. System-2005 masked temperature 5y preflight pass", f"""## 451. System-2005 masked temperature 5y preflight pass (2026-08-29)

- The sole registered `+0.3342811518826305 K` masked 2D preflight completed 5 years in `{result['runtime_seconds_this_process']:.2f} s`. Volume changed `{result['final_volume_relative_change']:+.4%}`, root-length error was `{result['final_root_path_metrics']['registered_length_relative_error']:.4%}`, free domain was `{result['final_free_domain_metrics']['minimum_active_cell_center_to_grid_edge_m']/1000:.3f} km`, exterior thickness remained exactly zero, and cumulative mass residual was `{result['cumulative_mass_budget_residual_m3']:.3e} m3`.
- Its 5-year climate source was `{result['cumulative_climate_source_m3']/1e6:.3f}e6 m3`, `{delta_source/1e6:.3f}e6 m3` below the baseline run, while the baseline and offset runs had the same `25.904 km` registered root-path length at year 5. All five atomic checkpoint hashes, numerical gates, runtime, and source hashes pass; stderr is empty.
- Route v10.147 queues exactly one fresh-from-year-zero 50-year formal run at the same offset and frozen gates. This is still a research steady mechanism, not production-temperature modification, historical authorization, reproduction, or acceptance.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.147: system 2005 masked temperature formal50 queued", """## 2026-08-29 - Route v10.147: system 2005 masked temperature formal50 queued

- Adjudicated the unique 5-year preflight as a complete scientific pass under all preregistered gates.
- Queued one 50-year formal run from the unchanged year-zero state. Pass permits reproduction only; failure closes this scalar 2D temperature mechanism.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.147-system2005-masked-temperature-formal50-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.147",
        "path": str((ROUTE147 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.146",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 1}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "queued-masked-temperature-formal50",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-5y-preflight-pass-formal50-queued",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "launch-system2005-masked-temperature-formal50-v1",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_temperature_preflight_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE147.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE147 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE147, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.147", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
