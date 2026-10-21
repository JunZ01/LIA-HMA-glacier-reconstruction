"""Authorize and freeze the discovery15 Stage B short qualification batch."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE186 = BASE / "lia_191_current_route_v10_186/summary.json"
STAGE_A_ROOT = BASE / "lia_191_discovery15_equilibrium_centered_stage_a_v2"
STAGE_A_SUMMARY = STAGE_A_ROOT / "summary.json"
STAGE_A_MEMBERS = STAGE_A_ROOT / "member_preparation_results.csv"
OUT = BASE / "lia_191_discovery15_stage_b_short_qualification_authorization_v1"
ROUTE187 = BASE / "lia_191_current_route_v10_187"
RESULT_ROOT = BASE / "lia_191_discovery15_stage_b_short_qualification_v1"
QUALIFICATION_DRIVER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_generic_planform2d_gpu_qualification_preflight_v1.py"
LAUNCHER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/launch_discovery15_stage_b_short_qualification_batch_v1.py"
SYSTEMS = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]
WAVES = [[988, 2086], [3185, 801], [4959, 2534], [2570, 705], [2727, 4491], [3016, 3660], [2177], [5770, 2559]]


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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    if OUT.exists() or ROUTE187.exists() or RESULT_ROOT.exists():
        raise FileExistsError("v10.187-versioned-output-exists")
    state = load(STATE)
    if state.get("state_id") != "2026-09-01-v10.186-discovery16-stage-a-terminal-stage-b-decision-required":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    if sha256(ROUTE186) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route186-hash-mismatch")
    stage_a = load(STAGE_A_SUMMARY)
    member_rows = rows(STAGE_A_MEMBERS)
    if stage_a.get("status") != "stage-a-15-of-15-pass":
        raise RuntimeError("stage-a-not-terminal-pass")
    if [int(row["system_id"]) for row in member_rows] != SYSTEMS:
        raise RuntimeError("stage-a-membership-drift")
    if any(row["stage_a_status"] != "pass" for row in member_rows):
        raise RuntimeError("stage-a-member-not-pass")
    if not QUALIFICATION_DRIVER.is_file() or not LAUNCHER.is_file():
        raise FileNotFoundError("stage-b-driver-or-launcher-missing")

    recorded = now()
    qualification_driver_hash = sha256(QUALIFICATION_DRIVER)
    launcher_hash = sha256(LAUNCHER)
    output_roots = [relative(RESULT_ROOT / "members" / f"system_{sid}") for sid in SYSTEMS]
    summary = {
        "schema": "lia-191-discovery15-stage-b-short-qualification-authorization-v1",
        "status": "stage-b-fixed15-short-qualification-authorized",
        "recorded_at": recorded,
        "discovery16_denominator": 16,
        "stage_a_scientific_failure_systems": [4588],
        "authorized_systems": SYSTEMS,
        "authorized_system_count": 15,
        "fixed_gpu_waves": WAVES,
        "maximum_concurrent_gpu_processes": 2,
        "qualification_phases": {
            "zero_mb_years": 0.1,
            "climate_response_years": 1.0,
            "restart_each_phase_from_frozen_source_thickness": True,
            "midpoint_checkpoint_bitwise_restore_each_phase": True,
        },
        "authorized_run_role": "scientific-qualification-preflight",
        "authorized_driver": relative(QUALIFICATION_DRIVER),
        "authorized_driver_sha256": qualification_driver_hash,
        "authorized_launcher": relative(LAUNCHER),
        "authorized_launcher_sha256": launcher_hash,
        "authorized_output_roots": output_roots,
        "stage_a_summary": {"path": relative(STAGE_A_SUMMARY), "sha256": sha256(STAGE_A_SUMMARY)},
        "stage_a_members": {"path": relative(STAGE_A_MEMBERS), "sha256": sha256(STAGE_A_MEMBERS)},
        "member_substitution_allowed": False,
        "parameter_adjustment_allowed": False,
        "steady_authorized": False,
        "historical_authorized": False,
        "accepted_count_effect": 0,
        "next_legal_action": "launch exactly the fixed fifteen Stage B short qualifications in the eight frozen waves; adjudicate all members before any steady decision",
    }
    summary_bytes = json_bytes(summary)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-187",
        "status": "discovery15-stage-b-short-qualification-authorized-self-hash-locked",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 15,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_186",
        "result": relative(OUT / "summary.json"),
        "result_sha256": summary_hash,
        "authorized_systems": SYSTEMS,
        "authorized_run_role": "scientific-qualification-preflight",
        "authorized_driver": relative(QUALIFICATION_DRIVER),
        "authorized_driver_sha256": qualification_driver_hash,
        "driver_self_hash_required": True,
        "authorized_launcher": relative(LAUNCHER),
        "authorized_launcher_sha256": launcher_hash,
        "authorized_output_roots": output_roots,
        "fixed_gpu_waves": WAVES,
        "concurrency": 2,
        "failure_isolation": "stop only the affected member and continue the fixed batch without retry or replacement",
        "qualification_phases": summary["qualification_phases"],
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "physics_modified": False,
        "gates_modified": False,
        "zero_mb_authorized": True,
        "short_climate_response_authorized": True,
        "steady_authorized": False,
        "historical_authorized": False,
        "next_legal_action": summary["next_legal_action"],
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    config_payloads: list[tuple[int, bytes, str]] = []
    row_by_system = {int(row["system_id"]): row for row in member_rows}
    for system_id in SYSTEMS:
        row = row_by_system[system_id]
        static_summary = ROOT / row["static_summary"]
        climate_summary = ROOT / row["climate_summary"]
        climate_config = climate_summary.parent / "config_snapshot.json"
        static = load(static_summary)
        climate = load(climate_summary)
        if static.get("status") != "static-pass" or climate.get("status") != "historical-climate-bridge-pass":
            raise RuntimeError(f"upstream-status-mismatch:{system_id}")
        stage2 = static["stage2_frozen"]
        frozen_keys = ("melt_f", "prcp_fac", "glen_a", "fs", "temp_melt_c", "base_temp_bias_c", "delta_t_c")
        if any(stage2[key] != climate["stage2"][key] for key in frozen_keys):
            raise RuntimeError(f"stage2-drift:{system_id}")
        config = {
            "schema": "lia-191-generic-planform2d-gpu-qualification-preflight-config-v1",
            "project_root": str(ROOT).replace("\\", "/"),
            "system_id": system_id,
            "run_role": "scientific-qualification-preflight",
            "geology": row["geology"].replace("\\", "/"),
            "geology_sha256": row["geology_sha256"].upper(),
            "source_thickness_sha256": row["source_thickness_sha256"].upper(),
            "geology_initial_thickness_sha256": row["geology_initial_thickness_sha256"].upper(),
            "climate_package": row["climate_package"].replace("\\", "/"),
            "climate_package_sha256": row["climate_package_sha256"].upper(),
            "static_summary": row["static_summary"].replace("\\", "/"),
            "static_summary_sha256": row["static_summary_sha256"].upper(),
            "upstream_evidence": row["climate_summary"].replace("\\", "/"),
            "upstream_evidence_sha256": row["climate_summary_sha256"].upper(),
            "upstream_config_snapshot": relative(climate_config),
            "upstream_config_snapshot_sha256": sha256(climate_config),
            "upstream_status": "historical-climate-bridge-pass",
            "source_shape": list(static["source_shape"]),
            "adapter_shape": list(static["padded_shape"]),
            "stage2_frozen": stage2,
            "launch_route": relative(ROUTE187 / "summary.json"),
            "launch_route_sha256": route_hash,
            "driver_sha256": qualification_driver_hash,
            "output_root": relative(RESULT_ROOT / "members" / f"system_{system_id}"),
            "launch_authorized": True,
            "initial_thickness_modification_allowed": False,
            "physics_modified": False,
            "gates_modified": False,
            "steady_authorized": False,
            "historical_authorized": False,
        }
        payload = json_bytes(config)
        config_payloads.append((system_id, payload, hashlib.sha256(payload).hexdigest().upper()))

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    facts_after = append_section(facts_before, "## 489. Discovery15 Stage B short qualification authorized", """## 489. Discovery15 Stage B short qualification authorized (2026-09-01)

- The user explicitly authorizes route v10.187 for the fixed fifteen Stage A-passing discovery members. System 4588 remains a scientific failure in the frozen discovery16 denominator and is neither retried nor replaced.
- Every authorized member runs exactly two independent phases from its immutable source-derived thickness: a 0.1-year zero-MB masked/no-flux qualification and a one-year equilibrium-centered 1441-1470 MonthlyTI response qualification. Midpoint checkpoints must restore bitwise in both phases.
- The fifteen members retain their common registered operator, Stage-2 parameters, geometry, climate package, numerical gates, and eight frozen GPU waves with at most two concurrent processes. No member-specific tuning, substitution, rescue, or exact-contract retry is authorized by this route.
- Route v10.187 authorizes only the short Stage B qualification. It authorizes no 50-year steady run, historical chain, thickness modification, gate change, promotion, or accepted-count change.""")
    decisions_after = append_section(decisions_before, "## 2026-09-01 - Route v10.187: authorize fixed discovery15 Stage B short qualification", """## 2026-09-01 - Route v10.187: authorize fixed discovery15 Stage B short qualification

- Launch exactly the fifteen Stage A-passing members in the eight frozen waves, with at most two concurrent GPU processes.
- Run only 0.1 year zero-MB plus one year equilibrium-centered MonthlyTI, restarting each phase from the frozen source thickness.
- Continue unaffected members after an isolated failure; do not retry, tune, replace, or rescue any member under this route.
- Do not launch 50-year steady or historical dynamics and do not change the accepted count.""")

    state187 = dict(state)
    state187["state_id"] = "2026-09-01-v10.187-discovery15-stage-b-short-qualification-authorized"
    state187["updated_at"] = recorded
    state187["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state187["authoritative_route"] = {"version": "v10.187", "path": relative(ROUTE187 / "summary.json"), "sha256": route_hash, "supersedes": "v10.186"}
    state187["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 15}
    state187["current_focus"] = {
        "system_id": "discovery15-stage-b",
        "mainline_family": "population93-equilibrium-centered-planform2d-v1",
        "state": "launch-authorized-short-qualification",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "historical_run_active": False,
        "evidence": {"path": relative(OUT / "summary.json"), "sha256": summary_hash},
    }
    branches = dict(state187["research_branches"])
    branch = dict(branches["population93_equilibrium_centered_planform2d_v1"])
    branch.update({
        "status": "stage-b-fixed15-short-qualification-authorized",
        "dynamic_launch_authorized": True,
        "stage_b_authorized_systems": SYSTEMS,
        "evidence": {"path": relative(OUT / "summary.json"), "sha256": summary_hash},
        "next_legal_action": summary["next_legal_action"],
    })
    branches["population93_equilibrium_centered_planform2d_v1"] = branch
    state187["research_branches"] = branches
    state187["next_action"] = {"type": "launch-discovery15-stage-b-short-qualification", "launch_authorized": True, "description": summary["next_legal_action"]}
    state_bytes = json_bytes(state187)

    stage = Path(tempfile.mkdtemp(prefix=".discovery187_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        configs_stage = evidence_stage / "configs"
        configs_stage.mkdir()
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        manifest = []
        for system_id, payload, payload_hash in config_payloads:
            name = f"system_{system_id}.json"
            (configs_stage / name).write_bytes(payload)
            manifest.append({"system_id": system_id, "config": f"configs/{name}", "sha256": payload_hash})
        (evidence_stage / "config_manifest.json").write_bytes(json_bytes({"schema": "lia-191-discovery15-stage-b-config-manifest-v1", "systems": SYSTEMS, "waves": WAVES, "configs": manifest}))
        (evidence_stage / "source_manifest.json").write_bytes(json_bytes({
            "created_at": recorded,
            "transaction_driver": relative(DRIVER),
            "transaction_driver_sha256": sha256(DRIVER),
            "qualification_driver": relative(QUALIFICATION_DRIVER),
            "qualification_driver_sha256": qualification_driver_hash,
            "launcher": relative(LAUNCHER),
            "launcher_sha256": launcher_hash,
            "sources": [{"path": relative(path), "sha256": sha256(path)} for path in [ROUTE186, STAGE_A_SUMMARY, STAGE_A_MEMBERS]],
        }))
        (route_stage / "summary.json").write_bytes(route_bytes)
        backups = {path: path.read_bytes() for path in [FACTS, DECISIONS, STATE]}
        shutil.move(str(evidence_stage), str(OUT))
        shutil.move(str(route_stage), str(ROUTE187))
        try:
            FACTS.write_bytes(facts_after)
            DECISIONS.write_bytes(decisions_after)
            STATE.write_bytes(state_bytes)
        except Exception:
            for path, content in backups.items():
                path.write_bytes(content)
            shutil.rmtree(OUT, ignore_errors=True)
            shutil.rmtree(ROUTE187, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
