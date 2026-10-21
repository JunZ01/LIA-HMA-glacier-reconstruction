"""Rescreen the remaining inventory and queue system 3910 historical endpoints."""

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

import psutil


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
ROUTE130 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_130/summary.json"
RESCREEN_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_remaining133_latest_rescreen_after_system902_v1"
PREP_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_first_historical_bracket_preparation_v1"
ROUTE131_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_131"
CONFIG_DIR = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
DRIVER_MODEL = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2.py"
DRIVER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2"
)
STEADY_ROOT = ROOT / "w445steady3910dtp05mindt30v1"
STEADY_SUMMARY = STEADY_ROOT / "steady_50y_summary.json"
STEADY_CONFIG = STEADY_ROOT / "config_snapshot.json"
STEADY_STATE = STEADY_ROOT / "lia_equilibrium_state_50y.pkl.gz"
INVERSION_SUMMARY = ROOT / "w342inv3910v1/inversion_summary_formal.json"
ZERO_SUMMARY = ROOT / "w348zero3910mindt30v1/zero_mb_summary_formal.json"
CLIMATE_PIXELS = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
RECONSTRUCTION = Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc")
OLD_ROOTS = [ROOT / "w489hist3910rampendp10mindt30v1", ROOT / "w490hist3910rampendp15mindt30v1"]
NEW_ROOTS = [ROOT / "w933hist3910rampendp1000v1", ROOT / "w933hist3910rampendp1500v1"]
LOG_ROOTS = [ROOT / "w933hist3910rampendp1000v1_logs", ROOT / "w933hist3910rampendp1500v1_logs"]
CONFIG_PATHS = [
    CONFIG_DIR / "historical_conservative_3910_endp1000_batch42_checkpoint_v1.json",
    CONFIG_DIR / "historical_conservative_3910_endp1500_batch42_checkpoint_v1.json",
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
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


def config(endpoint: float, output_root: Path) -> dict[str, Any]:
    return {
        "experiment": "synthetic-lia-system-conservative-junction-v1",
        "stage": "historical-forward-ramp-screening",
        "system_id": 3910,
        "main_rgi_id": "RGI60-14.17267",
        "synthetic_rgi_id": "RGI60-14.93910",
        "inversion_root": {"anchor": "project", "sub": "w342inv3910v1"},
        "steady_root": {"anchor": "project", "sub": "w445steady3910dtp05mindt30v1"},
        "output_root": {"anchor": "project", "sub": output_root.name},
        "climate_pixels": {
            "anchor": "validation",
            "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv",
        },
        "climate_id": "ensemble_rec_ref",
        "reconstruction_climate": str(RECONSTRUCTION).replace("\\", "/"),
        "start_year": 1470,
        "end_year": 2000,
        "forward_delta_t_end_c": endpoint,
        "minimum_timestep_seconds": 30.0,
        "minimum_free_downstream_m": 5000.0,
        "lia_total_error_max": 0.1,
        "lia_separation_error_max": 0.2,
        "rgi_total_error_max": 0.1,
        "rgi_separation_error_max": 0.2,
        "checkpoint_interval_years": 1,
        "resume_allowed": True,
        "expected_route_version": "v10.131",
        "full_historical_authorized": True,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "lia_registered_length_m": 14387.90925181418,
        "rgi_registered_length_m": 11145.152806089716,
        "stage2": {
            "melt_f": 2.460524971368185,
            "prcp_fac": 4.15551389902024,
            "glen_a": 3.384156689254701e-24,
            "fs": 0.0,
            "temp_melt_c": -1.0,
            "base_temp_bias_c": 0.0,
            "delta_t_c": 0.5,
            "prcp_anomaly_mode": "off",
        },
    }


def main() -> None:
    state = load_json(STATE)
    if state.get("state_id") != "2026-08-29-v10.130-system902-forward-endpoint-family-closed":
        raise RuntimeError(f"unexpected-state-id:{state.get('state_id')}")
    if sha256(ROUTE130) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route130-hash-mismatch")
    for path in [RESCREEN_DIR, PREP_DIR, ROUTE131_DIR, *NEW_ROOTS, *LOG_ROOTS, *CONFIG_PATHS]:
        if path.exists():
            raise FileExistsError(path)

    inversion = load_json(INVERSION_SUMMARY)
    zero = load_json(ZERO_SUMMARY)
    steady = load_json(STEADY_SUMMARY)
    if inversion.get("status") != "inversion-pass":
        raise RuntimeError("system3910-inversion-not-pass")
    if zero.get("status") != "zero-mb-pass":
        raise RuntimeError("system3910-zero-mb-not-pass")
    if steady.get("status") != "steady-50y-pass":
        raise RuntimeError("system3910-steady-not-pass")
    if abs(float(steady["volume_change_fraction_50y"])) > 0.02:
        raise RuntimeError("system3910-steady-volume-gate")
    if float(steady["final"]["remaining_free_downstream_m"]) < 5000.0:
        raise RuntimeError("system3910-steady-domain-gate")
    if not (14387.90925181418 > 11145.152806089716):
        raise RuntimeError("system3910-endpoint-order-invalid")
    if not STEADY_STATE.is_file():
        raise RuntimeError("system3910-equilibrium-state-missing")

    related = []
    for item in psutil.process_iter(["pid", "cmdline"]):
        joined = " ".join(item.info.get("cmdline") or [])
        if DRIVER_MODULE in joined and "3910" in joined:
            related.append({"pid": item.info["pid"], "command": joined})
    if related:
        raise RuntimeError(f"related-system3910-process-exists:{related}")

    old_status = []
    for path in OLD_ROOTS:
        old_status.append({
            "root": path.name,
            "exists": path.exists(),
            "historical_summary_exists": (path / "historical_summary.json").exists(),
            "formal_summary_exists": (path / "historical_summary_formal.json").exists(),
            "annual_curve_exists": (path / "historical_1470_2000.csv").exists(),
            "reused": False,
        })

    configs = [config(1.0, NEW_ROOTS[0]), config(1.5, NEW_ROOTS[1])]
    config_payloads = [json_bytes(item) for item in configs]
    config_hashes = [hashlib.sha256(item).hexdigest().upper() for item in config_payloads]

    rescreen_rows = [
        {"system_id": 7, "classification": "closed-target-operator-incompatibility", "reason": "RGI length exceeds LIA length under historical-maximum operator"},
        {"system_id": 1574, "classification": "closed-target-operator-incompatibility", "reason": "RGI length exceeds LIA length under historical-maximum operator"},
        {"system_id": 3915, "classification": "closed-input-geometry", "reason": "LIA feature is not exactly one valid Polygon"},
        {"system_id": 902, "classification": "closed-scalar-forward-family", "reason": "v10.130 joint LIA-modern gate conflict"},
        {"system_id": 3910, "classification": "selected-unexhausted-historical-route", "reason": "formal inversion, zero-MB, and 50y steady pass; normal endpoint order; no complete historical output"},
    ]
    rescreen_summary = {
        "schema": "lia-191-remaining133-latest-rescreen-after-system902-v1",
        "status": "read-only-rescreen-complete-system3910-selected",
        "recorded_at": now(),
        "accepted": 58,
        "remaining": 133,
        "selected_system_id": 3910,
        "selection_basis": (
            "system 3910 is the only surviving normal-order system with formal inversion, "
            "formal zero-MB, an exact-50-year steady pass, and no complete historical endpoint"
        ),
        "reopened_closed_mechanism": False,
        "model_process_started": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
    }
    prep = {
        "schema": "lia-191-system3910-first-historical-bracket-preparation-v1",
        "status": "prepared-not-launched",
        "recorded_at": now(),
        "system_id": 3910,
        "main_rgi_id": "RGI60-14.17267",
        "synthetic_rgi_id": "RGI60-14.93910",
        "equilibrium_delta_t_c": 0.5,
        "registered_forward_endpoints_c": [1.0, 1.5],
        "selection_basis": rescreen_summary["selection_basis"],
        "upstream_evidence": {
            "inversion_summary": {"path": str(INVERSION_SUMMARY.relative_to(ROOT)), "sha256": sha256(INVERSION_SUMMARY), "status": inversion["status"]},
            "zero_mb_summary": {"path": str(ZERO_SUMMARY.relative_to(ROOT)), "sha256": sha256(ZERO_SUMMARY), "status": zero["status"]},
            "steady_summary": {"path": str(STEADY_SUMMARY.relative_to(ROOT)), "sha256": sha256(STEADY_SUMMARY), "status": steady["status"]},
            "steady_config_sha256": sha256(STEADY_CONFIG),
            "equilibrium_state_sha256": sha256(STEADY_STATE),
        },
        "steady_metrics": {
            "volume_change_fraction_50y": steady["volume_change_fraction_50y"],
            "final_length_m": steady["final"]["length_m"],
            "lia_length_relative_error": steady["final"]["length_relative_error_lia"],
            "free_downstream_m": steady["final"]["remaining_free_downstream_m"],
        },
        "registered_targets_m": {"lia": 14387.90925181418, "rgi": 11145.152806089716},
        "configs": [
            {"endpoint_c": endpoint, "path": str(path.relative_to(ROOT)), "sha256": digest, "output_root": output.name, "log_root": log.name}
            for endpoint, path, digest, output, log in zip([1.0, 1.5], CONFIG_PATHS, config_hashes, NEW_ROOTS, LOG_ROOTS)
        ],
        "driver_module": DRIVER_MODULE,
        "driver_sha256": sha256(DRIVER_MODEL),
        "climate_pixels_sha256": sha256(CLIMATE_PIXELS),
        "reconstruction_climate_sha256": sha256(RECONSTRUCTION),
        "minimum_timestep_seconds": 30.0,
        "checkpoint_interval_years": 1,
        "old_incomplete_roots": old_status,
        "old_incomplete_roots_reused": False,
        "new_output_and_log_roots_exist": False,
        "related_model_process_count": 0,
        "physics_changed": False,
        "scoring_changed": False,
        "initial_thickness_modification_allowed": False,
        "independent_reproduction_required_for_acceptance": True,
        "accepted_count_effect": 0,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "record route v10.131, validate, and launch exactly one checkpointed process per endpoint",
    }
    prep_bytes = json_bytes(prep)
    prep_hash = hashlib.sha256(prep_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-131",
        "status": "system3910-first-historical-bracket-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 2,
        "launch_authorized": True,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_130",
        "system_id": 3910,
        "registered_forward_endpoints_c": [1.0, 1.5],
        "configs": prep["configs"],
        "preparation_evidence": str((PREP_DIR / "summary.json").relative_to(ROOT)),
        "preparation_evidence_sha256": prep_hash,
        "minimum_timestep_seconds": 30.0,
        "checkpoint_interval_years": 1,
        "duplicate_launch_authorized": False,
        "third_endpoint_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "launch exactly one checkpointed system-3910 process per registered endpoint after validator PASS",
    }
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()

    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append_section(facts_before, "## 435. System-3910 first historical bracket queued", """## 435. System-3910 first historical bracket queued (2026-08-29)

- The incremental latest-evidence rescreen of the remaining 133 systems selected `3910` as the sole unexhausted normal-order historical route. Systems `7/1574` remain closed by the LIA-maximum versus longer-RGI target incompatibility, `3915` remains closed at geometry, and `902` is closed by Section 434.
- System `3910` passes formal inversion, 30-second conservative zero-MB, and the exactly-50-year steady gate at `delta_T=+0.50 K`: volume change `+0.61736%`, final length `15.200 km`, LIA error `5.64426%`, and free domain `5.750 km`.
- Exactly two immutable checkpointed historical endpoints are registered: `+1.00 K` and `+1.50 K`. Old `w489/w490` partial roots contain no annual curve or historical summary and are not reused.
- Route v10.131 authorizes one process per endpoint after validator PASS. Counts remain 58 accepted and 133 remaining, active 0 and queued 2 before launch. Physics, scoring, geometry, initial thickness, and production temperature are unchanged.""")
    decisions_after = append_section(decisions_before, "## 2026-08-29 - Route v10.131: system 3910 first historical bracket queued", """## 2026-08-29 - Route v10.131: system 3910 first historical bracket queued

- Completed the read-only remaining-133 rescreen and selected system `3910` as the only unexhausted normal-order system with all upstream gates passed and no complete historical endpoint.
- Registered new immutable checkpointed `+1.00/+1.50 K` endpoints; the old incomplete roots are preserved but not reused.
- Counts remain 58 accepted and 133 remaining. Exactly two launches are authorized only after validator PASS.""")
    state131 = dict(state)
    state131["state_id"] = "2026-08-29-v10.131-system3910-first-historical-bracket-queued"
    state131["updated_at"] = now()
    state131["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state131["authoritative_route"] = {
        "version": "v10.131",
        "path": str((ROUTE131_DIR / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.130",
    }
    state131["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 2}
    state131["current_focus"] = {
        "system_id": "3910",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "first-historical-bracket-queued",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": True,
        "evidence": {"path": str((PREP_DIR / "summary.json").relative_to(ROOT)), "sha256": prep_hash},
    }
    state131["next_action"] = {
        "type": "launch-system3910-first-historical-bracket-v1",
        "launch_authorized": True,
        "description": "Launch exactly one checkpointed process for each registered +1.00/+1.50 K endpoint; then consume launch authorization.",
    }
    state_bytes = json_bytes(state131)

    stage = Path(tempfile.mkdtemp(prefix=".system3910_prepare_", dir=ROOT))
    try:
        (stage / "rescreen").mkdir()
        (stage / "prep").mkdir()
        (stage / "route").mkdir()
        (stage / "rescreen/summary.json").write_bytes(json_bytes(rescreen_summary))
        with (stage / "rescreen/rescreen_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["system_id", "classification", "reason"])
            writer.writeheader()
            writer.writerows(rescreen_rows)
        (stage / "prep/summary.json").write_bytes(prep_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        for index, payload in enumerate(config_payloads):
            (stage / f"config{index}").write_bytes(payload)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(state_bytes)
        RESCREEN_DIR.mkdir(parents=False)
        PREP_DIR.mkdir(parents=False)
        ROUTE131_DIR.mkdir(parents=False)
        os.replace(stage / "rescreen/summary.json", RESCREEN_DIR / "summary.json")
        os.replace(stage / "rescreen/rescreen_candidates.csv", RESCREEN_DIR / "rescreen_candidates.csv")
        os.replace(stage / "prep/summary.json", PREP_DIR / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE131_DIR / "summary.json")
        for index, path in enumerate(CONFIG_PATHS):
            os.replace(stage / f"config{index}", path)
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(RESCREEN_DIR, ignore_errors=True)
        shutil.rmtree(PREP_DIR, ignore_errors=True)
        shutil.rmtree(ROUTE131_DIR, ignore_errors=True)
        for path in CONFIG_PATHS:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps({
        "status": "prepared-not-launched",
        "selected_system": 3910,
        "registered_endpoints_c": [1.0, 1.5],
        "route": "v10.131",
        "route_sha256": route_hash,
        "preparation_sha256": prep_hash,
        "queued": 2,
        "accepted": 58,
        "remaining": 133,
        "old_incomplete_roots_reused": False,
        "production_temperature_modified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
