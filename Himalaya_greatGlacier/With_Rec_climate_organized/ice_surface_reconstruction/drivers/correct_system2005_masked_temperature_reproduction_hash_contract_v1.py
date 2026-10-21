"""Correct reproduction gates from NPZ container hashes to array-content hashes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE149 = BASE / "lia_191_current_route_v10_149/summary.json"
FORMAL_ROOT = BASE / "lia_191_system2005_masked_temperature_formal50_v1"
FORMAL = FORMAL_ROOT / "summary.json"
OUT = BASE / "lia_191_system2005_masked_temperature_reproduction_hash_contract_correction_v1"
ROUTE150 = BASE / "lia_191_current_route_v10_150"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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
    for path in (OUT, ROUTE150):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route149 = json.loads(ROUTE149.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.149-system2005-masked-temperature-reproduction-queued":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE149) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route149-hash-mismatch")
    if formal["status"] != "masked-temperature-formal50-pass":
        raise RuntimeError("formal-not-pass")

    annual_hashes = {}
    for year in range(1, 51):
        with np.load(FORMAL_ROOT / f"checkpoints/year_{year:03d}/state.npz", allow_pickle=False) as archive:
            annual_hashes[str(year)] = array_sha256(np.asarray(archive["thickness_m"], dtype=np.float64))
    with np.load(FORMAL_ROOT / "final_state_50y.npz", allow_pickle=False) as archive:
        final_thickness_hash = array_sha256(np.asarray(archive["thickness_m"], dtype=np.float64))
        final_domain_hash = array_sha256(np.asarray(archive["domain_mask"], dtype=bool))

    reproduction = dict(route149["queued_run"])
    reproduction["hard_gates"] = {
        "all_formal_scientific_gates": True,
        "final_thickness_array_sha256_exact": final_thickness_hash,
        "final_domain_mask_array_sha256_exact": final_domain_hash,
        "all_50_checkpoint_thickness_array_sha256_exact": annual_hashes,
        "stderr_empty": True,
        "source_and_initial_state_hashes_unchanged": True,
    }
    evidence = {
        "schema": "lia-191-system2005-masked-temperature-reproduction-hash-contract-correction-v1",
        "status": "prelaunch-container-hash-contract-corrected",
        "recorded_at": now(),
        "system_id": 2005,
        "supersedes_only": "route-v10.149 NPZ-container exact-hash fields",
        "reason": "NPZ container hashes include metadata and timestamps; scientific reproducibility is the exact dtype, shape, and C-order bytes of registered arrays.",
        "formal_result": str(FORMAL.relative_to(ROOT)),
        "formal_result_sha256": sha256(FORMAL),
        "final_thickness_array_sha256": final_thickness_hash,
        "final_domain_mask_array_sha256": final_domain_hash,
        "annual_thickness_array_sha256": annual_hashes,
        "corrected_queued_reproduction": reproduction,
        "scientific_parameters_changed": False,
        "simulation_launched": False,
        "accepted_count_effect": 0,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-150",
        "status": "system2005-masked-temperature-reproduction-array-hash-contract-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_149",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "queued_run": reproduction,
        "next_legal_action": "launch exactly one independent 50-year reproduction using exact thickness-array content hashes; no duplicate beyond this registered reproduction and no historical run",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 454. System-2005 reproduction array-hash contract correction", """## 454. System-2005 reproduction array-hash contract correction (2026-08-29)

- Before reproduction launch, route v10.149's exact NPZ-file hash requirement was found to include ZIP metadata and timestamps, which are not scientific state and can differ when arrays are bitwise identical. No reproduction had started and no result was observed.
- Route v10.150 replaces only those container hashes with exact hashes over array dtype, shape, and C-order bytes for all 50 annual `thickness_m` arrays plus final thickness and domain-mask arrays. Temperature, forcing, geometry, physics, solver, thresholds, duration, initial state, and all scientific gates are unchanged.
- Exactly one independent reproduction remains queued. Counts remain 58 accepted and 133 remaining until terminal reproduction adjudication.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.150: reproduction array-hash contract corrected", """## 2026-08-29 - Route v10.150: reproduction array-hash contract corrected

- Corrected a prelaunch reproducibility-gate representation error from NPZ container hashes to exact scientific array-content hashes.
- No run had launched and no scientific parameter changed. The sole reproduction remains queued.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.150-system2005-masked-temperature-reproduction-array-hash-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.150",
        "path": str((ROUTE150 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.149",
    }
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "queued-masked-temperature-reproduction-array-hash-contract",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-reproduction-array-hash-contract-queued",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "launch-system2005-masked-temperature-reproduction-v1",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_reproduction_hash_correct_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE150.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE150 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE150, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.150", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
