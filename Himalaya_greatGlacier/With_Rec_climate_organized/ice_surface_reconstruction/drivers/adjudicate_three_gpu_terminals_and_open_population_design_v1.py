"""Adjudicate the three v10.181 GPU terminals and open read-only population design.

This is a controller transaction.  It starts no model and authorizes no launch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
CONTRACT = ROOT / "LIA_EXECUTION_CONTRACT.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE181 = BASE / "lia_191_current_route_v10_181/summary.json"
COHORTS = BASE / "lia_191_population_extension_cohorts_v1/summary.json"
S2005 = ROOT / "w2005mask2dhistfull_e0p834281_v1/historical_summary_formal.json"
S7 = ROOT / "w7generic2dsteady50_parent1441_1470_v1/summary.json"
S79 = ROOT / "w79generic2dsteady50_parent1441_1470_v1/summary.json"
STDERRS = {
    2005: ROOT / "w2005mask2dhistfull_e0p834281_v1_resume80v1_logs/stderr.log",
    7: ROOT / "w7generic2dsteady50_parent1441_1470_v1_logs/stderr.log",
    79: ROOT / "w79generic2dsteady50_parent1441_1470_v1_logs/stderr.log",
}
OUT = BASE / "lia_191_system2005_system7_79_terminal_adjudication_v1"
ROUTE182 = BASE / "lia_191_current_route_v10_182"


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


def main() -> int:
    if OUT.exists() or ROUTE182.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}:{ROUTE182}")
    state = load(STATE)
    if state.get("state_id") != "2026-08-29-v10.181-system2005-resume-system7-79-steady-active":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(ROUTE181) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route181-hash-mismatch")
    if sha256(FACTS) != state["scientific_contract"]["sha256"].upper():
        raise RuntimeError("facts-hash-mismatch")
    for path in [S2005, S7, S79, COHORTS, *STDERRS.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    if any(path.stat().st_size != 0 for path in STDERRS.values()):
        raise RuntimeError("nonempty-stderr")

    s2005, s7, s79 = load(S2005), load(S7), load(S79)
    for summary, system_id in [(s2005, 2005), (s7, 7), (s79, 79)]:
        if int(summary["system_id"]) != system_id:
            raise RuntimeError(f"system-id-mismatch:{system_id}")
        if not summary["checks"]["finite_nonnegative"] or not summary["checks"]["mass_budget_closure"]:
            raise RuntimeError(f"unexpected-technical-failure:{system_id}")
        if summary["outside_domain_maximum_thickness_m"] != 0.0:
            raise RuntimeError(f"outside-domain-thickness:{system_id}")

    h = s2005["historical_scores_diagnostic_until_full_run"]
    system2005_total_pass = h["lia_total_error_fraction"] <= 0.10 and h["rgi_total_error_fraction"] <= 0.10
    system2005_separation_pass = (
        h["lia_separation_error_fraction"] <= 0.20 and h["rgi_separation_error_fraction"] <= 0.20
    )
    if s2005["actual_calendar_year"] != 2000 or not system2005_total_pass or system2005_separation_pass:
        raise RuntimeError("unexpected-system2005-terminal")
    if s7["checks"]["steady_volume_change_pass"] or s79["checks"]["steady_volume_change_pass"]:
        raise RuntimeError("unexpected-steady-pass")
    if not s7["checks"]["registered_length_error_pass"] or not s79["checks"]["registered_length_error_pass"]:
        raise RuntimeError("unexpected-length-failure")

    rows = []
    for summary, role in [(s2005, "full-historical-colder"), (s7, "generic-steady50"), (s79, "generic-steady50")]:
        sid = int(summary["system_id"])
        root = summary["final_root_path_metrics"]
        row = {
            "system_id": sid,
            "run_role": role,
            "technical_class": "numerically-complete",
            "scientific_class": "scientific-failure",
            "completed_years": summary["completed_years"],
            "accepted_steps": summary["accepted_steps"],
            "rejected_steps": summary["rejected_steps"],
            "minimum_timestep_seconds": summary["counters"]["minimum_accepted_dt_seconds"],
            "volume_change_fraction": summary["final_volume_relative_change"],
            "registered_length_error_fraction": root["registered_length_relative_error"],
            "free_domain_m": summary["final_free_domain_metrics"]["minimum_active_cell_center_to_grid_edge_m"],
            "outside_domain_maximum_thickness_m": summary["outside_domain_maximum_thickness_m"],
            "stderr_empty": True,
            "formal_reason": "steady-volume-gate-fail" if sid in {7, 79} else "historical-separation-gates-fail",
        }
        if sid == 2005:
            row.update({
                "lia_total_error_fraction": h["lia_total_error_fraction"],
                "lia_separation_error_fraction": h["lia_separation_error_fraction"],
                "rgi_total_error_fraction": h["rgi_total_error_fraction"],
                "rgi_separation_error_fraction": h["rgi_separation_error_fraction"],
            })
        rows.append(row)

    conclusion = (
        "The generic masked planform2d solver is numerically conservative and domain-clean, but its registered "
        "parent-climate steady operator is not population-ready: the two independent systems drift +40.62% and "
        "-6.55% in volume. System 2005 completes the full historical interval but fails both frozen separation "
        "gates. This closes the current 2D steady/historical mechanism and requires one population-level "
        "climate-state equilibrium mechanism before any cohort launch."
    )
    summary = {
        "schema": "lia-191-three-gpu-terminal-adjudication-v1",
        "status": "three-numerically-complete-scientific-failures-current-2d-mechanism-closed",
        "recorded_at": now(),
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queued": 0,
        "system_results": {str(row["system_id"]): row for row in rows},
        "system2005_warmer_launched": False,
        "system2005_warmer_retained_in_queue": False,
        "generic_2d_class_promoted": False,
        "population_cohort_launch_authorized": False,
        "scientific_conclusion": conclusion,
        "next_legal_action": (
            "read-only population-93 static climate/state equilibrium attribution using one deterministic "
            "operator; define no glacier-specific manual parameter rescue and launch no discovery member"
        ),
        "model_run_performed": False,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "old_output_modified": False,
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 485. Three GPU terminals close the current generic 2D mechanism", """## 485. Three GPU terminals close the current generic 2D mechanism (2026-08-31)

- Systems `7`, `79`, and `2005` all completed their registered targets with empty stderr, finite/nonnegative states, exact exterior-zero thickness, mass closure, annual checkpoints, and saved final states. They are scientific failures, not technical failures.
- System `7` completes 50 steady years with registered-length error `5.411%` but volume change `+40.622%`; system `79` completes 50 years with length error `9.007%` but volume change `-6.545%`. Both violate the frozen `2%` steady-volume gate while retaining more than `5.5 km` free domain.
- System `2005` completes the full `1470-2000` colder historical chain. Its LIA/RGI total errors are `4.640%` and `3.337%`, but separation errors are `45.041%` and `29.051%`, both above the frozen `20%` gates. The unauthorized warmer branch was never launched and is removed from the queue.
- These results close promotion of the current generic masked planform2d steady/historical mechanism. The next action is a read-only population-level climate/state equilibrium attribution over the 93-system extension pool. No discovery-cohort member, per-glacier rescue, parameter change, thickness change, or accepted-count action is authorized.""")
    decisions_after = append_section(decisions_before, "## 2026-08-31 - Route v10.182: three terminals close current 2D mechanism", """## 2026-08-31 - Route v10.182: three terminals close current 2D mechanism

- Adjudicate systems 7, 79, and 2005 as numerically complete scientific failures under their frozen gates.
- Clear all three stale active processes and the never-authorized system-2005 warmer queue; accepted counts remain unchanged.
- Do not launch the 16-system discovery cohort with the failed mechanism. Begin only the read-only population-93 static climate/state equilibrium attribution needed to define one global mechanism.""")

    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-182",
        "status": "three-terminal-scientific-fail-population-global-mechanism-design",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_181",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": summary_hash,
        "generic_2d_class_promoted": False,
        "population_cohort_launch_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    state182 = dict(state)
    state182["state_id"] = "2026-08-31-v10.182-three-terminal-scientific-fail-population-design"
    state182["updated_at"] = now()
    state182["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state182["authoritative_route"] = {
        "version": "v10.182", "path": str((ROUTE182 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash, "supersedes": "v10.181",
    }
    state182["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state182["current_focus"] = {
        "system_id": "population93",
        "mainline_family": "planform2d-global-climate-state-equilibrium-design",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "historical_run_active": False,
    }
    branches = dict(state182.get("research_branches", {}))
    branches["system2005_pytorch_planform2d"] = {
        "status": "colder-full-historical-scientific-fail-warmer-not-launched",
        "promoted_to_mainline": False, "accepted_count_effect": 0,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "next_legal_action": "closed under current historical endpoint mechanism; no warmer launch",
    }
    branches["generic_gpu_pair_adapter_v1"] = {
        "status": "systems7-79-steady50-independent-pair-scientific-fail",
        "systems": [7, 79], "promoted_to_mainline": False, "accepted_count_effect": 0,
        "launch_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    }
    state182["research_branches"] = branches
    state182["next_action"] = {
        "type": "audit-population93-static-climate-state-equilibrium-v1",
        "launch_authorized": False,
        "description": summary["next_legal_action"],
    }
    state_bytes = json_bytes(state182)

    stage = Path(tempfile.mkdtemp(prefix=".terminal182_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        fields = list(rows[0])
        with (evidence_stage / "task_terminals.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        manifest = {
            "created_at": now(), "driver": str(DRIVER.relative_to(ROOT)), "driver_sha256": sha256(DRIVER),
            "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in [ROUTE181, COHORTS, S2005, S7, S79, *STDERRS.values()]],
        }
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes(manifest))
        (route_stage / "summary.json").write_bytes(route_bytes)
        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE182))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE182, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
