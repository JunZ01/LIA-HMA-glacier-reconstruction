"""Adjudicate the system-2005 one-year masked-climate preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE139 = BASE / "lia_191_current_route_v10_139/summary.json"
RUN = BASE / "lia_191_system2005_masked_climate_1y_preflight_v1/summary.json"
OUT = BASE / "lia_191_system2005_masked_climate_1y_preflight_adjudication_v1"
ROUTE140 = BASE / "lia_191_current_route_v10_140"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    value = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + value.rstrip() + newline


def main() -> None:
    for path in (OUT, ROUTE140):
        if path.exists():
            raise FileExistsError(path)
    state = load(STATE)
    run = load(RUN)
    if state["state_id"] != "2026-08-29-v10.139-system2005-masked-climate-1y-preflight-queued":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE139) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route139-hash-mismatch")
    if run["status"] != "masked-climate-1y-preflight-pass" or not all(run["checks"].values()):
        raise RuntimeError("preflight-not-pass")

    adjudication = {
        "schema": "lia-191-system2005-masked-climate-1y-preflight-adjudication-v1",
        "status": "masked-climate-1y-preflight-formal-pass",
        "recorded_at": now(),
        "system_id": 2005,
        "run_summary": str(RUN.relative_to(ROOT)),
        "run_summary_sha256": sha256(RUN),
        "results": {
            "completed_years": run["completed_years"],
            "wall_seconds": run["wall_seconds"],
            "accepted_steps": run["accepted_steps"],
            "rejected_steps": run["rejected_steps"],
            "initial_volume_m3": run["initial_volume_m3"],
            "final_volume_m3": run["final_volume_m3"],
            "cumulative_climate_source_m3": run["cumulative_climate_source_m3"],
            "fixed_initial_geometry_source_m3": run["fixed_initial_geometry_source_m3"],
            "source_ratio_actual_to_fixed": run["source_ratio_actual_to_fixed"],
            "cumulative_mass_budget_residual_m3": run["cumulative_mass_budget_residual_m3"],
            "outside_domain_maximum_thickness_m": run["outside_domain_maximum_thickness_m"],
            "initial_active_area_m2": run["initial_active_area_m2"],
            "final_active_area_m2": run["final_active_area_m2"],
        },
        "runtime_projection": {
            "observed_seconds_per_year": run["wall_seconds"],
            "naive_50y_wall_seconds": run["wall_seconds"] * 50.0,
            "naive_50y_wall_minutes": run["wall_seconds"] * 50.0 / 60.0,
            "checkpoint_resume_required_before_launch": True,
        },
        "scientific_decision": {
            "masked_climate_mechanism_eligible_for_50y_steady_test": True,
            "runaway_positive_source_removed_over_preflight": True,
            "formal_50y_run_currently_authorized": False,
            "next_required_action": "build and verify an annual-checkpoint/resume 50-year masked steady driver",
            "accepted_count_effect": 0,
        },
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "historical_run": False,
    }
    adjudication_bytes = payload(adjudication)
    adjudication_hash = hashlib.sha256(adjudication_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-140",
        "status": "system2005-masked-climate-1y-preflight-pass-checkpointed-steady-design-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_139",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": adjudication_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": "build and verify an annual-checkpoint/resume 50-year source-mask MonthlyTI steady driver; do not launch before a new route",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 444. System-2005 masked climate one-year preflight pass", """## 444. System-2005 masked climate one-year preflight pass (2026-08-29)

- The exact E-drive PyTorch `2.11.0+cu128` and CUDA `12.8` preflight completed one year in `98.13 s` with 851 accepted and 293 rejected adaptive steps. All registered runtime, input-hash, finite-state, nonnegativity, mass-closure, and source-direction gates pass.
- Exterior-domain thickness remains exactly zero. The actual climate source is `-3.30614e6 m3`, consistent in direction with the frozen-initial source `-5.39481e6 m3`; cumulative mass-budget residual is only `-2.6786e-5 m3`. This removes the prior unmasked branch's false positive area-source feedback over the scientific preflight.
- Route v10.140 authorizes no long run. A 50-year masked steady test is scientifically eligible only after annual checkpoint/resume support is built and verified. The one-year runtime projects naively to about `81.77 min` for 50 years. Initial thickness, production temperature, accepted counts, and historical forcing remain unchanged.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.140: system 2005 masked climate one-year preflight passed", """## 2026-08-29 - Route v10.140: system 2005 masked climate one-year preflight passed

- Accepted the exact CUDA 12.8 one-year masked-climate preflight as a formal mechanism pass: negative climate source, exact zero exterior ice, finite nonnegative state, and mass-budget closure all pass.
- Cleared the queue. The next action is checkpoint/resume construction and verification for a 50-year masked steady run; no historical or long run is yet authorized.""")

    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.140-system2005-masked-climate-1y-preflight-pass"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.140",
        "path": str((ROUTE140 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.139",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-preflight",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-climate-1y-preflight-pass-checkpointed-steady-design-required",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "design-system2005-masked-steady50-checkpoint-resume-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_climate1y_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(adjudication_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE140.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE140 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE140, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.140", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
