"""Record the source-mask audit and authorize one zero-MB fixture."""

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
ROUTE136 = BASE / "lia_191_current_route_v10_136/summary.json"
EVIDENCE = BASE / "lia_191_system2005_lia_source_mask_domain_audit_v2/summary.json"
ROUTE137 = BASE / "lia_191_current_route_v10_137"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_zero_mb_fixture_v1.json"
MODEL = DRIVER.parent / "run_system2005_masked_zero_mb_fixture_v1.py"
CORE = DRIVER.parent / "torch_planform_2d_masked_frozen_diffusivity_core_v1.py"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
OUTPUT = BASE / "lia_191_system2005_masked_zero_mb_fixture_v1"


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
    for path in (ROUTE137, CONFIG, OUTPUT):
        if path.exists():
            raise FileExistsError(path)
    state = load(STATE)
    evidence = load(EVIDENCE)
    if state["state_id"] != "2026-08-29-v10.136-system2005-2d-area-feedback-identified":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE136) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route136-hash-mismatch")
    if evidence["status"] != "source-native-lia-domain-candidate-pass-readonly":
        raise RuntimeError("source-mask-domain-not-pass")

    route = {
        "schema": "lia-191-current-route-v10-137",
        "status": "system2005-source-mask-zero-mb-fixture-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_136",
        "result": str(EVIDENCE.relative_to(ROOT)),
        "result_sha256": sha256(EVIDENCE),
        "registered_action": "zero-MB masked-diffusion single-step and 0.1-year fixture only",
        "model_driver": str(MODEL.relative_to(ROOT)),
        "model_driver_sha256": sha256(MODEL),
        "masked_core": str(CORE.relative_to(ROOT)),
        "masked_core_sha256": sha256(CORE),
        "config": str(CONFIG.relative_to(ROOT)),
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "climate_run_authorized": False,
        "historical_run_authorized": False,
        "production_temperature_modified": False,
        "next_legal_action": "launch exactly one registered masked zero-MB fixture and consume authorization",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    config = {
        "schema": "lia-191-system2005-masked-zero-mb-fixture-config-v1",
        "project_root": str(ROOT),
        "system_id": 2005,
        "geology": str(GEOLOGY.relative_to(ROOT)),
        "geology_sha256": sha256(GEOLOGY),
        "domain_variable": "lia_source_mask",
        "glen_a": 4.355770498109968e-23,
        "mass_balance": "zero",
        "target_years": 0.1,
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "expected_route": str((ROUTE137 / "summary.json").relative_to(ROOT)),
        "expected_route_sha256": route_hash,
        "initial_thickness_modification_allowed": False,
        "climate_run_authorized": False,
        "historical_run_authorized": False,
    }
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 441. System-2005 source-native LIA domain candidate", """## 441. System-2005 source-native LIA domain candidate (2026-08-29)

- Corrected mask audit v2 supersedes only v1's premature flowline-catchment candidate. The selected domain is the source-native `lia_source_mask` already carried in `geology.nc`; it is grid-aligned, requires no reprojection or fitted buffer, contains every initial active cell, and includes `6.3871 km2` of initially empty in-domain area.
- The runaway 50-year state places `79.7279 km2` of active ice outside this domain. That exterior supplies `123.464e6 m3 yr-1`, `79.97%` of the final fixed-geometry positive climate source.
- Route v10.137 authorizes exactly one zero-MB masked-diffusion fixture: an adaptive single-step check followed by 0.1 year. It must preserve mass, nonnegativity, exact zero exterior thickness, source hashes, and unchanged initial thickness. No climate or historical run is authorized.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.137: system 2005 masked zero-MB fixture queued", """## 2026-08-29 - Route v10.137: system 2005 masked zero-MB fixture queued

- Frozen the source-native `lia_source_mask` as the non-tuned research domain candidate.
- Authorized exactly one zero-MB single-step plus 0.1-year conservation fixture using face-masked conductances. Post-step clipping is not permitted.
- Initial thickness, climate, production files, targets, and accepted count remain unchanged.""")
    state137 = dict(state)
    state137["state_id"] = "2026-08-29-v10.137-system2005-masked-zero-mb-fixture-queued"
    state137["updated_at"] = now()
    state137["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state137["authoritative_route"] = {
        "version": "v10.137", "path": str((ROUTE137 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash, "supersedes": "v10.136",
    }
    state137["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 1}
    state137["current_focus"] = {
        "system_id": "2005", "mainline_family": "source-mask-planform2d-research-fixture",
        "state": "queued", "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
    }
    state137["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "source-native-domain-zero-mb-fixture-queued",
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
        "next_legal_action": route["next_legal_action"],
    })
    state137["next_action"] = {
        "type": "launch-system2005-masked-zero-mb-fixture-v1",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_mask_authorize_", dir=ROOT))
    try:
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "config").write_bytes(payload(config))
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(state137))
        ROUTE137.mkdir(parents=False)
        os.replace(stage / "route/summary.json", ROUTE137 / "summary.json")
        os.replace(stage / "config", CONFIG)
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(ROUTE137, ignore_errors=True)
        CONFIG.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run([os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True)
    print(json.dumps({"status": "queued", "route": "v10.137", "config": str(CONFIG), "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
