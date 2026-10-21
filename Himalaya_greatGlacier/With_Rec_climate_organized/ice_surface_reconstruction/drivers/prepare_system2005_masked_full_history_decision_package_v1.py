"""Prepare and authorize the sequential System-2005 masked full-history pair."""

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
ROUTE160 = BASE / "lia_191_current_route_v10_160/summary.json"
ROUTE161 = BASE / "lia_191_current_route_v10_161"
TERMINAL = BASE / "lia_191_system2005_masked_historical_response_preflight_terminal_v1/summary.json"
PREPARATION = BASE / "lia_191_system2005_masked_historical_preparation_v1/summary.json"
OUTPUT = BASE / "lia_191_system2005_masked_full_history_decision_package_v1"
DRIVER = HERE.parent / "run_system2005_masked_historical_branch_v3.py"
V2_DRIVER = HERE.parent / "run_system2005_masked_historical_branch_v2.py"
V1_DRIVER = HERE.parent / "run_system2005_masked_historical_branch_v1.py"
TEST = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/tests/test_system2005_masked_full_history_contract_v1.py"
CONFIG_DIR = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
RUNTIME = ROOT / ".runtime/torch-cu128-v1/site-packages"
MAXIMUM_ACCEPTED_STEPS = 600_000


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode())
    return content.rstrip() + newline + newline + normalized.encode("utf-8").rstrip() + newline


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    configs = {
        "colder": CONFIG_DIR / "system2005_masked_historical_full_endp0p834281_v1_prepared.json",
        "warmer": CONFIG_DIR / "system2005_masked_historical_full_endp1p334281_v1_prepared.json",
    }
    for path in (OUTPUT, ROUTE161, *configs.values()):
        if path.exists():
            raise FileExistsError(path)
    state = load(STATE)
    terminal = load(TERMINAL)
    preparation = load(PREPARATION)
    if state["state_id"] != "2026-08-29-v10.160-system2005-historical-response-preflight-pass":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE160) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route160-hash-mismatch")
    if terminal["status"] != "historical-response-preflight-pass":
        raise RuntimeError("paired-preflight-not-pass")
    if preparation["status"] != "historical-preparation-pass":
        raise RuntimeError("historical-preparation-not-pass")
    full_roots = {
        "colder": ROOT / "w2005mask2dhistfull_e0p834281_v1",
        "warmer": ROOT / "w2005mask2dhistfull_e1p334281_v1",
    }
    if any(path.exists() for path in full_roots.values()):
        raise RuntimeError("full-output-root-already-exists")
    subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )

    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(RUNTIME), "TEMP": str(ROOT / ".runtime"), "TMP": str(ROOT / ".runtime")})
    tests = subprocess.run(
        [str(PYTHON), "-m", "unittest", "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.tests.test_system2005_masked_full_history_contract_v1", "-v"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if "Ran 3 tests" not in tests.stderr or "OK" not in tests.stderr:
        raise RuntimeError("full-target-tests-not-confirmed")

    v1_text = V1_DRIVER.read_text(encoding="utf-8")
    v2_text = V2_DRIVER.read_text(encoding="utf-8")
    v3_text = DRIVER.read_text(encoding="utf-8")
    code_checks = {
        "target_years_derived_from_calendar": "target_years = target_end_year - start_year" in v1_text,
        "engine_target_years_dynamic": "engine.TARGET_YEARS = target_years" in v1_text,
        "visible_months_dynamic": "int(engine.TARGET_YEARS) * 12" in v2_text,
        "checkpoint_range_dynamic": "range(1, target_years + 1)" in v3_text,
        "resume_manifest_supported": "resume_manifest" in v1_text,
    }
    if not all(code_checks.values()):
        raise RuntimeError(f"target-neutral-driver-audit-failed:{code_checks}")

    branches = terminal["branches"]
    seconds_per_year = max(float(branches[name]["runtime_seconds"]) for name in branches) / 10.0
    projected_seconds = seconds_per_year * 530.0
    projected_steps = max(float(branches[name]["accepted_steps"]) for name in branches) / 10.0 * 530.0
    if MAXIMUM_ACCEPTED_STEPS < projected_steps * 1.4:
        raise RuntimeError("accepted-step-headroom-below-40-percent")
    checkpoint_bytes = max((ROOT / f"w2005mask2dhistpre_{suffix}/checkpoints/year_1480/state.npz").stat().st_size for suffix in ("e0p834281_v1", "e1p334281_v1"))
    history_bytes_per_year = max(next((ROOT / f"w2005mask2dhistpre_{suffix}").glob("step_history_part_*.csv")).stat().st_size for suffix in ("e0p834281_v1", "e1p334281_v1")) / 10.0
    projected_bytes = checkpoint_bytes * 530 + history_bytes_per_year * 530 + 10_000_000
    disk = shutil.disk_usage(ROOT.anchor)
    if disk.free < projected_bytes * 4:
        raise RuntimeError("insufficient-disk-headroom")

    templates = {
        "colder": CONFIG_DIR / "system2005_masked_historical_preflight_endp0p834281_v1.json",
        "warmer": CONFIG_DIR / "system2005_masked_historical_preflight_endp1p334281_v1.json",
    }
    endpoints = {"colder": 0.8342811518826305, "warmer": 1.3342811518826305}
    prepared = {}
    for branch in ("colder", "warmer"):
        config = load(templates[branch])
        config.update(
            {
                "run_role": f"full-historical-endpoint-{branch}",
                "target_end_year": 2000,
                "target_years": 530,
                "temperature_offset_2000_k": endpoints[branch],
                "maximum_accepted_steps": MAXIMUM_ACCEPTED_STEPS,
                "driver": str(DRIVER.relative_to(ROOT)).replace("\\", "/"),
                "driver_sha256": sha256(DRIVER),
                "expected_route": str(ROUTE160.relative_to(ROOT)).replace("\\", "/"),
                "expected_route_sha256": sha256(ROUTE160),
                "launch_route": str(ROUTE160.relative_to(ROOT)).replace("\\", "/"),
                "launch_route_sha256": sha256(ROUTE160),
                "launch_authorized": False,
                "output_root": str(full_roots[branch].relative_to(ROOT)),
                "preparation_evidence": str((OUTPUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            }
        )
        configs[branch].write_bytes(encode(config))
        prepared[branch] = {
            "path": str(configs[branch].relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(configs[branch]),
            "temperature_offset_2000_k": endpoints[branch],
            "output_root": str(full_roots[branch].relative_to(ROOT)),
        }

    checks = {
        "paired_preflight_pass": True,
        "checkpoint_fixture_exact": all(preparation["checks"].values()),
        "target_530_checkpoint_tests_pass": True,
        "visible_6360_month_test_pass": True,
        "target_neutral_driver_audit_pass": all(code_checks.values()),
        "accepted_step_headroom_at_least_40_percent": MAXIMUM_ACCEPTED_STEPS >= projected_steps * 1.4,
        "disk_headroom_at_least_four_times_one_branch": disk.free >= projected_bytes * 4,
        "fresh_output_roots": True,
        "sequential_gpu_policy": True,
    }
    decision = {
        "schema": "lia-191-system2005-masked-full-history-decision-package-v1",
        "generated_at": now(),
        "status": "full-history-pair-prepared-and-authorized",
        "system_id": 2005,
        "calendar": [1470, 2000],
        "target_years": 530,
        "visible_months": 6360,
        "package_months": 6372,
        "execution_policy": "sequential one-process-at-a-time; colder first",
        "prepared_configs": prepared,
        "driver": {"path": str(DRIVER.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DRIVER)},
        "full_target_tests": {"path": str(TEST.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TEST), "tests_run": 3, "status": "pass"},
        "code_checks": code_checks,
        "runtime_budget": {"seconds_per_simulated_year": seconds_per_year, "projected_hours_per_branch": projected_seconds / 3600.0, "projected_hours_pair_sequential": 2 * projected_seconds / 3600.0},
        "step_budget": {"projected_accepted_steps_per_branch": projected_steps, "configured_maximum_accepted_steps": MAXIMUM_ACCEPTED_STEPS, "headroom_fraction": MAXIMUM_ACCEPTED_STEPS / projected_steps - 1.0},
        "storage_budget": {"projected_bytes_per_branch": projected_bytes, "projected_bytes_pair": 2 * projected_bytes, "e_drive_free_bytes": disk.free},
        "checks": checks,
        "scientific_reason": "the short pair proves temperature-to-volume response but cannot resolve the 0.009434 K early separation with a 75 m length operator; the full calendar develops the preregistered 0.5 K endpoint separation",
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "accepted_count_effect": 0,
    }
    if not all(checks.values()):
        raise RuntimeError("decision-package-check-failed")

    stage = Path(tempfile.mkdtemp(prefix=".system2005_full_history_decision_", dir=ROOT))
    try:
        evidence_stage = stage / "evidence"
        route_stage = stage / "route"
        evidence_stage.mkdir()
        route_stage.mkdir()
        summary_bytes = encode(decision)
        (evidence_stage / "summary.json").write_bytes(summary_bytes)
        (evidence_stage / "FINAL_RESULT.md").write_text(
            f"# System 2005 masked full-history decision package\n\nThe sequential 1470-2000 pair is prepared. Measured projection: {projected_seconds / 3600:.2f} h per branch and {2 * projected_seconds / 3600:.2f} h for the pair. Accepted-step allowance: {MAXIMUM_ACCEPTED_STEPS:,} versus {projected_steps:,.0f} projected. Pair storage: {2 * projected_bytes / 1e9:.2f} GB; E: free: {disk.free / 1e12:.2f} TB.\n\nDirect 530-checkpoint and 6360-month tests pass. Initial thickness, production temperature, endpoints, physics, climate package, and targets are unchanged.\n",
            encoding="utf-8",
        )
        result_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
        route = {
            "schema": "lia-191-current-route-v10-161",
            "status": "system2005-masked-full-history-pair-queued-colder-launch-authorized",
            "population": 191,
            "accepted": 58,
            "remaining": 133,
            "active": 0,
            "queue": 2,
            "launch_authorized": True,
            "full_historical_authorized": True,
            "supersedes": "lia_191_current_route_v10_160",
            "result": str((OUTPUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": result_hash,
            "driver": str(DRIVER.relative_to(ROOT)).replace("\\", "/"),
            "driver_sha256": sha256(DRIVER),
            "prepared_configs": prepared,
            "launch_order": ["colder", "warmer"],
            "accepted_count_effect": 0,
            "initial_thickness_modification_allowed": False,
            "production_temperature_modified": False,
            "next_legal_action": "materialize a v10.161-bound colder full-history config and launch it exactly once; retain warmer queued and do not run both GPU branches concurrently",
        }
        route_bytes = encode(route)
        route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
        facts_before = FACTS.read_bytes()
        decisions_before = DECISIONS.read_bytes()
        state_before = STATE.read_bytes()
        facts_after = append(
            facts_before,
            "## 465. System-2005 masked full-history pair prepared",
            f"""## 465. System-2005 masked full-history pair prepared (2026-08-29)

- Direct contract tests pass for exactly 530 annual checkpoints and a 6,360-month visible 1470-2000 window. Static audit confirms target-derived years, dynamic duration, month visibility, checkpoint range, and resume support in driver v3.
- Measured runtime projects `{projected_seconds / 3600.0:.2f} h` per branch and `{2 * projected_seconds / 3600.0:.2f} h` sequentially. Each config allows `{MAXIMUM_ACCEPTED_STEPS:,}` accepted steps versus `{projected_steps:,.0f}` projected, with `{MAXIMUM_ACCEPTED_STEPS / projected_steps - 1.0:.1%}` headroom. Projected pair storage is `{2 * projected_bytes / 1e9:.2f} GB`; E: has `{disk.free / 1e12:.2f} TB` free.
- The immutable year-2000 endpoints remain `+0.8342811518826305 K` and `+1.3342811518826305 K`, with common `+0.3342811518826305 K` in 1470 and unchanged initial thickness, physics, climate package, and LIA/RGI targets. Route v10.161 queues both branches sequentially and authorizes only colder first.
- No production-temperature change, simultaneous GPU pair, model-family promotion, acceptance, or count change occurs.""",
        )
        decisions_after = append(
            decisions_before,
            "## 2026-08-29 - Route v10.161: full historical pair prepared and queued",
            """## 2026-08-29 - Route v10.161: full historical pair prepared and queued

- The 530-year checkpoint/month-window tests, target-neutral driver audit, resume fixture, step, runtime, and storage budgets pass.
- Queue both preregistered endpoints sequentially. Authorize exactly one colder launch first; keep warmer queued and prohibit concurrent GPU execution. Counts remain 58/133.""",
        )
        next_state = dict(state)
        next_state["state_id"] = "2026-08-29-v10.161-system2005-masked-full-history-pair-queued"
        next_state["updated_at"] = now()
        next_state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
        next_state["authoritative_route"] = {"version": "v10.161", "path": str((ROUTE161 / "summary.json").relative_to(ROOT)).replace("\\", "/"), "sha256": route_hash, "supersedes": "v10.160"}
        next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 2}
        next_state["current_focus"] = {"system_id": "2005", "mainline_family": "source-mask-planform2d-research-steady", "state": "full-historical-pair-queued-colder-authorized", "initial_thickness_modification_allowed": False, "full_historical_authorized": True, "evidence": {"path": route["result"], "sha256": result_hash}}
        next_state["research_branches"]["system2005_pytorch_planform2d"].update({"status": "masked-full-historical-pair-queued-colder-authorized", "evidence": {"path": route["result"], "sha256": result_hash}, "next_legal_action": route["next_legal_action"]})
        next_state["next_action"] = {"type": "launch-system2005-masked-full-history-colder-v1", "launch_authorized": True, "description": route["next_legal_action"]}
        (route_stage / "summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(encode(next_state))
        OUTPUT.mkdir(parents=False)
        ROUTE161.mkdir(parents=False)
        for item in evidence_stage.iterdir():
            os.replace(item, OUTPUT / item.name)
        os.replace(route_stage / "summary.json", ROUTE161 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        shutil.rmtree(ROUTE161, ignore_errors=True)
        for path in configs.values():
            path.unlink(missing_ok=True)
        if "facts_before" in locals():
            FACTS.write_bytes(facts_before)
            DECISIONS.write_bytes(decisions_before)
            STATE.write_bytes(state_before)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    validation_after = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": decision["status"], "route": "v10.161", "runtime_budget": decision["runtime_budget"], "step_budget": decision["step_budget"], "storage_budget": decision["storage_budget"], "validator": json.loads(validation_after.stdout)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
