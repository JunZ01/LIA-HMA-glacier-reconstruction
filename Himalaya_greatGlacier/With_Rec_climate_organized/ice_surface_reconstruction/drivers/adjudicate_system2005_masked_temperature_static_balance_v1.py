"""Register the system-2005 masked static temperature bracket and queue one preflight."""

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
ROUTE144 = BASE / "lia_191_current_route_v10_144/summary.json"
STATIC = BASE / "lia_191_system2005_masked_temperature_static_balance_v1/summary.json"
OUT = BASE / "lia_191_system2005_masked_temperature_static_balance_adjudication_v1"
ROUTE145 = BASE / "lia_191_current_route_v10_145"


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
    for path in (OUT, ROUTE145):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    result = json.loads(STATIC.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.144-system2005-internal-support-exchange-confirmed":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE144) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route144-hash-mismatch")
    if result["status"] != "static-temperature-bracket-pass" or not all(result["checks"].values()):
        raise RuntimeError("static-temperature-audit-not-pass")

    zero_delta = float(result["static_temperature_boundaries"]["zero_source_delta_t_k"])
    positive = float(result["static_temperature_boundaries"]["positive_2pct_delta_t_k"])
    negative = float(result["static_temperature_boundaries"]["negative_2pct_delta_t_k"])
    preflight = {
        "system_id": 2005,
        "duration_years": 5,
        "delta_t_k": zero_delta,
        "initial_state": "unchanged source-native 2D thickness",
        "domain": "lia_source_mask",
        "forcing": "same 600-month exported MonthlyTI sequence with an in-memory uniform temperature offset",
        "solver": "same masked source-half/diffusion/source-half operator and frozen physical constants",
        "annual_atomic_checkpoints": True,
        "hard_gates": {
            "completed_years": 5,
            "absolute_volume_change_fraction_max": 0.02,
            "registered_length_relative_error_max": 0.10,
            "free_domain_m_min": 5000.0,
            "outside_domain_thickness_m_exact": 0.0,
            "cumulative_budget_relative_max": 1.0e-10,
            "finite_nonnegative": True,
            "source_and_initial_state_hashes_unchanged": True,
            "cumulative_source_less_than_baseline_5y_m3": -12739049.93954622,
        },
        "terminal_rule": "pass makes one checkpointed 50-year formal run eligible; failure closes this scalar 2D temperature mechanism",
    }
    evidence = {
        "schema": "lia-191-system2005-masked-temperature-static-balance-adjudication-v1",
        "status": "static-temperature-mechanism-bracketed-preflight-queued",
        "recorded_at": now(),
        "system_id": 2005,
        "static_evidence": str(STATIC.relative_to(ROOT)),
        "static_evidence_sha256": sha256(STATIC),
        "baseline_volume_change_fraction": result["baseline"]["volume_change_fraction"],
        "baseline_root_length_error_fraction": result["baseline"]["root_length_error_fraction"],
        "static_2pct_interval_delta_t_k": [positive, negative],
        "static_zero_source_delta_t_k": zero_delta,
        "scientific_decision": {
            "one_dimensional_temperature_mechanism_reopened": False,
            "old_thickness_lift_candidate_reopened": False,
            "current_masked_2d_temperature_mechanism_is_nonduplicate": True,
            "production_temperature_modified": False,
            "full_historical_authorized": False,
            "accepted_count_effect": 0,
        },
        "queued_preflight": preflight,
    }
    evidence_bytes = payload(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-145",
        "status": "system2005-masked-temperature-5y-preflight-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 1,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_144",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": evidence_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "queued_run": preflight,
        "next_legal_action": "launch exactly one registered 5-year masked 2D temperature-offset preflight for system 2005; no duplicate or historical run",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 449. System-2005 masked temperature static-balance bracket", f"""## 449. System-2005 masked temperature static-balance bracket (2026-08-29)

- A read-only MonthlyTI replay on all 51 saved masked 2D states closes the baseline annual-start source exactly and gives a strictly monotonic temperature response. Retaining the measured baseline within-year residual, the saved-path zero-source offset is `{zero_delta:+.6f} K`; the interpolated static 2% volume-source interval is `{positive:+.6f}` to `{negative:+.6f} K`.
- This does not reopen the closed 1D natural-state temperature family, and the old v4 pass remains ineligible because it inherited a `+2.805264 m` prohibited thickness lift. The current 2D masked replay is nonduplicate because its support geometry and length margin differ from that 1D family.
- Static replay does not predict changed support or length. Route v10.145 therefore queues exactly one 5-year masked 2D preflight at `{zero_delta:+.15f} K`, with unchanged initial thickness and production files. It must pass the frozen 2% volume, 10% length, 5 km free-domain, exact exterior-zero, mass-closure, and source-hash gates before any 50-year run becomes eligible.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.145: system 2005 masked temperature preflight queued", f"""## 2026-08-29 - Route v10.145: system 2005 masked temperature preflight queued

- Closed the prior-mechanism audit without reopening the ineligible 1D/thickness-lift routes.
- Registered the nonduplicate masked 2D static zero-source point `{zero_delta:+.15f} K` and queued exactly one 5-year dynamics preflight under fixed gates. No production temperature, historical forcing, initial thickness, or accepted count changes are authorized.""")

    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.145-system2005-masked-temperature-5y-preflight-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.145",
        "path": str((ROUTE145 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.144",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 1}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "queued-masked-temperature-5y-preflight",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-static-bracketed-5y-preflight-queued",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": evidence_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "launch-system2005-masked-temperature-5y-preflight-v1",
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_temperature_static_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        OUT.mkdir(parents=False)
        ROUTE145.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE145 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ROUTE145, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    validation = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "complete", "route": "v10.145", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
