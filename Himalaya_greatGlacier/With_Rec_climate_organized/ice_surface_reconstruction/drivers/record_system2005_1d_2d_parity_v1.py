"""Record system-2005 1D/2D parity evidence and advance to mask audit."""

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
CONTRACT = ROOT / "LIA_EXECUTION_CONTRACT.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE135 = BASE / "lia_191_current_route_v10_135/summary.json"
EVIDENCE = BASE / "lia_191_system2005_1d_2d_climate_area_budget_parity_v1/summary.json"
ROUTE136 = BASE / "lia_191_current_route_v10_136"


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
    nl = b"\r\n" if b"\r\n" in content else b"\n"
    value = text.replace("\r\n", "\n").replace("\n", nl.decode()).encode("utf-8")
    return content.rstrip() + nl + nl + value.rstrip() + nl


def main() -> None:
    if ROUTE136.exists():
        raise FileExistsError(ROUTE136)
    state = load(STATE)
    result = load(EVIDENCE)
    if state["state_id"] != "2026-08-29-v10.135-remaining133-current-mechanisms-exhausted":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE135) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route135-hash-mismatch")
    if result["status"] != "read-only-parity-audit-complete":
        raise RuntimeError("parity-audit-not-complete")
    feedback = result["two_d_mass_budget"]["feedback_fraction_of_actual_source"]
    if feedback <= 1.0:
        raise RuntimeError("area-feedback-not-decisive")

    route = {
        "schema": "lia-191-current-route-v10-136",
        "status": "system2005-2d-area-feedback-identified-lia-domain-mask-audit-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_135",
        "result": str(EVIDENCE.relative_to(ROOT)),
        "result_sha256": sha256(EVIDENCE),
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "read-only audit of the existing lia_source_mask as a support/no-flux domain for system 2005",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 440. System-2005 1D/2D climate-area parity audit", """## 440. System-2005 1D/2D climate-area parity audit (2026-08-29)

- The exported MonthlyTI formula matches OGGM exactly and the 2D 50-year volume change closes to the climate source within `0.000155 m3`; neither a formula-sign error nor numerical mass leakage explains the `+68.76%` growth.
- On the frozen initial 2D geometry, the same 50-year climate source is `-0.12139e9 m3`. The actual source is `+5.55526e9 m3`; the excess `+5.67665e9 m3` is created by geometry/support feedback.
- The mass-balance-active 2D area expands from `89.7485` to `170.7449 km2`, while the 1D area changes by only `-1.15%`. Newly activated 2D cells cover `84.7926 km2` and have a fixed-geometry mean balance of `+1.566 m ice yr-1`.
- The current 2D branch therefore fails because diffusion expands support beyond the mapped LIA source domain and activates strong high-elevation accumulation. Existing `geology.nc` already contains `lia_source_mask`; route v10.136 authorizes only a read-only physical-domain audit before any new solver run.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.136: system 2005 2D area feedback identified", """## 2026-08-29 - Route v10.136: system 2005 2D area feedback identified

- Closed climate-formula and numerical-mass-leak hypotheses.
- Identified diffusion-driven support expansion and newly activated positive-accumulation area as the cause of the 2D steady failure.
- No model launch or calibration is authorized. Next perform a read-only audit of the existing mapped LIA source mask as a physical no-flux/support domain.""")
    state136 = dict(state)
    state136["state_id"] = "2026-08-29-v10.136-system2005-2d-area-feedback-identified"
    state136["updated_at"] = now()
    state136["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    state136["authoritative_route"] = {
        "version": "v10.136",
        "path": str((ROUTE136 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.135",
    }
    state136["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "new-mechanism-readonly-design",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
    }
    state136["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "area-support-positive-feedback-identified-not-promoted",
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
        "next_legal_action": "read-only audit of existing lia_source_mask as support/no-flux domain",
    })
    state136["next_action"] = {
        "type": "audit-system2005-lia-source-mask-domain-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_parity_record_", dir=ROOT))
    try:
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(state136))
        ROUTE136.mkdir(parents=False)
        os.replace(stage / "route/summary.json", ROUTE136 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(ROUTE136, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.136", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
