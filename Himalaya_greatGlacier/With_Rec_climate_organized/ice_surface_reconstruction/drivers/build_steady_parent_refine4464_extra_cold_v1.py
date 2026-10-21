"""Prepare the frozen 4464 extra-cold steady config; never run OGGM."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_MANIFEST = "w704steady_parent_zero_pass9_manifest_v1.json"
PARENT_MANIFEST_SHA256 = "3f63c938bc3b0047c3f964946553d25e148188fe3a3ef65a313a14e64afe03c7"
COLD_MANIFEST = "w704coldsteady_parent_m0p5_7_manifest_v1.json"
COLD_MANIFEST_SHA256 = "172b7fb712eab8730101df663c1b5714b54658f25e32fc84ee335f3d0469b9d6"
MANIFEST_NAME = "w704steady_parent_refine4464_extra_cold_manifest_v1.json"
CONFIG_REL = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1/"
    "steady_parent_refine4464_extra_cold_v1.json"
)
OUTPUT_NAME = "w704steady_parent_refine4464_extra_cold_v1"
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")
RUNNER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.synthetic_lia_system_steady_50y_v1"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strip_allowed(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    del frozen["output_root"]
    del frozen["stage2"]["delta_t_c"]
    return frozen


def _evidence(root: Path, output_name: str, system_id: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_path = root / output_name / "config_snapshot.json"
    summary_path = root / output_name / "steady_50y_summary.json"
    _require(snapshot_path.is_file() and summary_path.is_file(), f"system-{system_id}-run-evidence-missing")
    snapshot, summary = _read(snapshot_path), _read(summary_path)
    _require(int(snapshot.get("system_id", -1)) == system_id, f"system-{system_id}-snapshot-id-conflict")
    record = {
        "output_root": output_name,
        "config_snapshot": {"path": str(snapshot_path.relative_to(root)), "sha256": _sha(snapshot_path)},
        "steady_50y_summary": {"path": str(summary_path.relative_to(root)), "sha256": _sha(summary_path)},
        "delta_t_c": float(snapshot["stage2"]["delta_t_c"]),
        "status": summary.get("status"),
        "final_year": int(summary["final"]["year"]),
        "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
        "review_flags": list(summary.get("review_flags", [])),
        "runtime_error": summary.get("runtime_error"),
    }
    return snapshot, summary, record


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    parent_manifest_path = root / PARENT_MANIFEST
    cold_manifest_path = root / COLD_MANIFEST
    manifest_path = root / MANIFEST_NAME
    config_path = root / CONFIG_REL
    output_root = root / OUTPUT_NAME
    _require(_sha(parent_manifest_path) == PARENT_MANIFEST_SHA256, "parent-manifest-conflict")
    _require(_sha(cold_manifest_path) == COLD_MANIFEST_SHA256, "cold-manifest-conflict")
    _require(PYTHON.is_file(), "python-missing")
    collisions = [str(p) for p in (manifest_path, config_path, output_root) if p.exists()]
    _require(not collisions, "refusing-to-overwrite:" + repr(collisions))

    parent_manifest, cold_manifest = _read(parent_manifest_path), _read(cold_manifest_path)
    parent_by_id = {int(c["system_id"]): c for c in parent_manifest["cases"]}
    cold_by_id = {int(c["system_id"]): c for c in cold_manifest["cases"]}
    parent_record, cold_record = parent_by_id[4464], cold_by_id[4464]
    for label, item in parent_record["parent_evidence"].items():
        path = root / str(item["path"])
        _require(path.is_file() and _sha(path) == str(item["sha256"]), f"4464-{label}-conflict")

    parent_config_path = root / str(parent_record["config"])
    cold_config_path = root / str(cold_record["config"])
    parent_config, cold_config = _read(parent_config_path), _read(cold_config_path)
    parent_snapshot, parent_summary, parent_run = _evidence(root, str(parent_record["output_root"]), 4464)
    cold_snapshot, cold_summary, cold_run = _evidence(root, str(cold_record["output_root"]), 4464)
    _require(parent_config == parent_snapshot and cold_config == cold_snapshot, "4464-config-run-conflict")
    _require(_strip_allowed(parent_config) == _strip_allowed(cold_config), "4464-parent-cold-contract-conflict")
    _require(float(parent_config["stage2"]["delta_t_c"]) == 0.25, "4464-parent-delta-conflict")
    _require(float(cold_config["stage2"]["delta_t_c"]) == -0.25, "4464-cold-delta-conflict")
    for label, summary in (("parent", parent_summary), ("cold", cold_summary)):
        _require(int(summary["final"]["year"]) == 50, f"4464-{label}-incomplete")
        _require(float(summary["volume_change_fraction_50y"]) < 0.0, f"4464-{label}-not-negative")

    # Record and enforce the no-retry exclusion for 5127.
    excluded_record = cold_by_id[5127]
    excluded_snapshot, excluded_summary, excluded_run = _evidence(root, str(excluded_record["output_root"]), 5127)
    excluded_flags = set(excluded_summary.get("review_flags", []))
    _require(float(excluded_snapshot["stage2"]["delta_t_c"]) == 0.25, "5127-cold-delta-conflict")
    _require("cfl-error" in excluded_flags and "50y-incomplete" in excluded_flags, "5127-exclusion-conflict")
    _require(int(excluded_summary["final"]["year"]) < 50, "5127-exclusion-unexpected-completion")

    payload = copy.deepcopy(parent_config)
    payload["stage2"]["delta_t_c"] = -0.75
    payload["output_root"] = {"anchor": "project", "sub": OUTPUT_NAME}
    _require(_strip_allowed(payload) == _strip_allowed(parent_config), "4464-frozen-field-conflict")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read(config_path) == payload, "4464-config-roundtrip-conflict")
    _require(not output_root.exists(), "output-created-during-preparation")

    command = (
        "& '" + str(PYTHON) + "' -m " + RUNNER_MODULE
        + " --config '" + str(config_path) + "'"
    )
    manifest = {
        "status": "config-built-not-run",
        "stage": "system-4464-extra-cold-minus-0.75K-v1",
        "count": 1,
        "parent_manifest": {"path": PARENT_MANIFEST, "sha256": PARENT_MANIFEST_SHA256},
        "cold_manifest": {"path": COLD_MANIFEST, "sha256": COLD_MANIFEST_SHA256},
        "allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "minimum_timestep_changed": False,
        "oggm_started_by_builder": False,
        "static_validation": "pass",
        "case": {
            "system_id": 4464,
            "config": CONFIG_REL,
            "output_root": OUTPUT_NAME,
            "target_delta_t_c": -0.75,
            "frozen_parent_evidence": parent_record["parent_evidence"],
            "parent_temperature_config": {"path": str(parent_config_path.relative_to(root)), "sha256": _sha(parent_config_path)},
            "parent_run": parent_run,
            "cold_run": cold_run,
            "run_command_powershell": command,
        },
        "excluded": [{
            "system_id": 5127,
            "reason": "cold-cfl-error-50y-incomplete-no-midpoint-no-retry",
            "minimum_timestep_changed": False,
            "evidence": excluded_run,
        }],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path), "static_validation": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
