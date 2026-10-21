"""Record the dominant internal support-exchange mechanism and advance read-only."""

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
ROUTE143 = BASE / "lia_191_current_route_v10_143/summary.json"
DECOMP = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v3/summary.json"
EXCHANGE = BASE / "lia_191_system2005_masked_support_exchange_v1/summary.json"
PLOT = BASE / "lia_191_system2005_masked_internal_source_feedback_plot_v1/summary.json"
OUT = BASE / "lia_191_system2005_masked_internal_feedback_adjudication_v1"
ROUTE144 = BASE / "lia_191_current_route_v10_144"


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
    for path in (OUT, ROUTE144):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    decomp = json.loads(DECOMP.read_text(encoding="utf-8"))
    exchange = json.loads(EXCHANGE.read_text(encoding="utf-8"))
    plot = json.loads(PLOT.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.143-system2005-masked-steady50-volume-fail":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE143) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route143-hash-mismatch")
    if decomp["status"] != "internal-source-decomposition-pass" or not all(decomp["checks"].values()):
        raise RuntimeError("decomposition-not-pass")
    if exchange["status"] != "support-exchange-pass" or not all(exchange["checks"].values()):
        raise RuntimeError("exchange-not-pass")
    if plot["status"] != "plot-complete":
        raise RuntimeError("plot-not-complete")

    evidence = {
        "schema": "lia-191-system2005-masked-internal-feedback-adjudication-v1",
        "status": "internal-support-exchange-dominant-mechanism-confirmed",
        "recorded_at": now(),
        "system_id": 2005,
        "parent_decomposition": str(DECOMP.relative_to(ROOT)),
        "parent_decomposition_sha256": sha256(DECOMP),
        "support_exchange": str(EXCHANGE.relative_to(ROOT)),
        "support_exchange_sha256": sha256(EXCHANGE),
        "plot": plot["figure"],
        "actual_50y_source_m3": decomp["actual_50y_source_m3"],
        "fraction_of_actual_source": decomp["fraction_of_actual_source"],
        "support_exchange_split": {
            "added_high_support_m3": exchange["added_support_effect_m3"],
            "removed_low_support_m3": exchange["removed_support_effect_m3"],
            "added_fraction_of_support": exchange["added_fraction_of_support_effect"],
            "removed_fraction_of_support": exchange["removed_fraction_of_support_effect"],
            "added_fraction_of_actual": exchange["added_fraction_of_actual_source"],
            "removed_fraction_of_actual": exchange["removed_fraction_of_actual_source"],
        },
        "final_year_exchange": exchange["final_year_exchange"],
        "scientific_decision": {
            "external_domain_expansion_is_remaining_cause": False,
            "numerical_mass_leak_is_remaining_cause": False,
            "surface_elevation_only_is_remaining_cause": False,
            "internal_support_exchange_is_dominant": True,
            "new_model_run_authorized": False,
            "next_action": "read-only audit of prior system2005 temperature/static-bias mechanisms before defining any nonduplicate mechanism",
            "accepted_count_effect": 0,
        },
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "historical_run": False,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-144",
        "status": "system2005-internal-support-exchange-confirmed-prior-mechanism-audit-next",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_143",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": "read-only audit all prior system2005 temperature-bias, static-balance, and 2D source-support evidence for duplication and viable brackets; no model launch",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 448. System-2005 internal support exchange mechanism", """## 448. System-2005 masked internal support-exchange mechanism (2026-08-29)

- Exact read-only replay closes the 50-year `+1.179925e9 m3` source. The initial geometry under the same climate is negative (`-121.390e6 m3`), while active-support exchange contributes `+1.005490e9 m3` (`85.22%` of actual source), surface-elevation change on initial support contributes `+133.611e6 m3`, support-surface interaction `+85.371e6 m3`, and within-year geometry `+76.843e6 m3`.
- Support exchange splits into `+351.859e6 m3` from newly active high support and `+653.630e6 m3` from removing low-elevation negative-balance support. Thus `65.01%` of the dominant support effect is loss of former ablation area and `34.99%` is added accumulation area. In year 50 the added cells average `6316 m` and `+2.302 m ice yr-1`; removed cells average `5561 m` and `-1.873 m ice yr-1` on the initial surface.
- The remaining masked failure is therefore not exterior expansion, numerical leakage, or surface uplift alone. Route v10.144 authorizes only a read-only audit of prior system-2005 temperature/static-balance mechanisms before any new run, so closed or duplicate mechanisms are not reopened.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.144: system 2005 internal support exchange confirmed", """## 2026-08-29 - Route v10.144: system 2005 internal support exchange confirmed

- Confirmed that support exchange explains 85.22% of the masked run's positive source; lost low-elevation ablation support is the larger component.
- Authorized no simulation. The next action is an exhaustive read-only prior-mechanism audit before any scalar temperature or other balance mechanism is proposed.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.144-system2005-internal-support-exchange-confirmed"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.144",
        "path": str((ROUTE144 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.143",
    }
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "hold-new-mechanism",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "internal-support-exchange-confirmed-prior-mechanism-audit-required",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "audit-system2005-prior-temperature-static-balance-mechanisms-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }
    stage = Path(tempfile.mkdtemp(prefix=".system2005_internal_feedback_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE144.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE144 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE144, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.144", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
