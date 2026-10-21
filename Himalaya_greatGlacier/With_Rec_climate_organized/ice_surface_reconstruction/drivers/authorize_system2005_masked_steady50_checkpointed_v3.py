"""Authorize one checkpointed 50-year source-mask steady run for system 2005."""

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
ROUTE140 = BASE / "lia_191_current_route_v10_140/summary.json"
EVIDENCE = BASE / "lia_191_system2005_masked_steady50_checkpoint_contract_v3/summary.json"
ROUTE141 = BASE / "lia_191_current_route_v10_141"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_steady50_checkpointed_v3.json"
DRIVER = HERE.parent / "run_system2005_masked_steady50_checkpointed_v3.py"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
STATIC = BASE / "lia_191_igm_planform2d_system2005_static_v1/static_summary.json"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
OBSERVATION = BASE / "lia_191_igm_planform2d_system2005_static_v1/observation_path_static.csv"
OUTPUT = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3"


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
    for path in (ROUTE141, CONFIG, OUTPUT):
        if path.exists():
            raise FileExistsError(path)
    state = load(STATE)
    evidence = load(EVIDENCE)
    if state["state_id"] != "2026-08-29-v10.140-system2005-masked-climate-1y-preflight-pass":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE140) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route140-hash-mismatch")
    if evidence["status"] != "checkpoint-contract-pass" or not all(evidence["checks"].values()):
        raise RuntimeError("checkpoint-contract-not-pass")
    if evidence["tested_driver_sha256"] != sha256(DRIVER):
        raise RuntimeError("tested-driver-hash-mismatch")

    route = {
        "schema": "lia-191-current-route-v10-141",
        "status": "system2005-masked-steady50-checkpointed-v3-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_140",
        "result": str(EVIDENCE.relative_to(ROOT)),
        "result_sha256": sha256(EVIDENCE),
        "registered_action": "one checkpointed 50-year source-mask MonthlyTI steady run only",
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "config": str(CONFIG.relative_to(ROOT)),
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory plus latest pointer",
        "runtime_estimate_minutes": {"naive": 81.77, "planning_upper": 120.0},
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "historical_run_authorized": False,
        "production_temperature_modified": False,
        "next_legal_action": "launch exactly one registered checkpointed masked 50-year steady process using the E-drive CUDA 12.8 runtime",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    config = {
        "schema": "lia-191-system2005-masked-steady50-checkpointed-config-v1",
        "project_root": str(ROOT),
        "system_id": 2005,
        "target_years": 50,
        "geology": str(GEOLOGY.relative_to(ROOT)),
        "geology_sha256": sha256(GEOLOGY),
        "static_summary": str(STATIC.relative_to(ROOT)),
        "static_summary_sha256": sha256(STATIC),
        "climate_package": str(CLIMATE.relative_to(ROOT)),
        "climate_package_sha256": sha256(CLIMATE),
        "observation_path": str(OBSERVATION.relative_to(ROOT)),
        "observation_path_sha256": sha256(OBSERVATION),
        "domain_variable": "lia_source_mask",
        "glen_a": 4.355770498109968e-23,
        "driver_sha256": sha256(DRIVER),
        "expected_route": str((ROUTE141 / "summary.json").relative_to(ROOT)),
        "expected_route_sha256": route_hash,
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "runtime_site_packages": ".runtime/torch-cu128-v1/site-packages",
        "initial_thickness_modification_allowed": False,
        "historical_run_authorized": False,
        "production_temperature_modified": False,
    }

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 445. System-2005 checkpointed masked steady50 queued", """## 445. System-2005 checkpointed masked steady50 queued (2026-08-29)

- The final prelaunch driver is v3. V1 was never launched and is superseded because its checkpoint directory publication was not atomic; v2 repaired atomic publication; v3 additionally injects the exact frozen constants `rho=900 kg m-3`, `g=9.80665 m s-2`, and `n=3` into every split step.
- The v3 checkpoint contract passes on the real `894 x 880` initial-thickness array: state, elapsed time, proposed timestep, counters, and metrics restore exactly; contract-hash and state-hash tampering are rejected; no temporary checkpoint remains.
- Route v10.141 queues exactly one 50-year source-mask MonthlyTI steady run. It has annual atomic checkpoints and explicit resume, an `81.77 min` naive estimate with a `120 min` planning bound, unchanged initial thickness and inputs, and no historical, production-temperature, or accepted-count authority.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.141: system 2005 checkpointed masked steady50 queued", """## 2026-08-29 - Route v10.141: system 2005 checkpointed masked steady50 queued

- Checkpoint integrity and exact frozen-constant injection pass before launch.
- Authorized exactly one v3 masked 50-year steady process with annual atomic checkpoints. Historical forcing, production files, initial thickness, and accepted counts remain unchanged.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.141-system2005-masked-steady50-checkpointed-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.141",
        "path": str((ROUTE141 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.140",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 1}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "queued",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-steady50-checkpointed-v3-queued",
        "evidence": {"path": str(EVIDENCE.relative_to(ROOT)), "sha256": sha256(EVIDENCE)},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "launch-system2005-masked-steady50-checkpointed-v3",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_masked_steady50_authorize_", dir=ROOT))
    try:
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "config").write_bytes(payload(config))
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        ROUTE141.mkdir(parents=False)
        os.replace(stage / "route/summary.json", ROUTE141 / "summary.json")
        os.replace(stage / "config", CONFIG)
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(ROUTE141, ignore_errors=True)
        CONFIG.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "queued", "route": "v10.141", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
