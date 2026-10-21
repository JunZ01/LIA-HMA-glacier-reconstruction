"""Build the next five frozen steady-temperature refinement configs.

The builder performs only provenance checks and exclusive JSON creation.  It
does not import OGGM or execute a glacier model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_MANIFEST = "w704steady_parent_zero_pass9_manifest_v1.json"
PARENT_MANIFEST_SHA256 = (
    "3f63c938bc3b0047c3f964946553d25e148188fe3a3ef65a313a14e64afe03c7"
)
COLD_MANIFEST = "w704coldsteady_parent_m0p5_7_manifest_v1.json"
COLD_MANIFEST_SHA256 = (
    "172b7fb712eab8730101df663c1b5714b54658f25e32fc84ee335f3d0469b9d6"
)
MANIFEST_NAME = "w704steady_parent_refine5_manifest_v1.json"
CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
RUNNER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.synthetic_lia_system_steady_50y_v1"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")

# system_id: (target delta-T, decision, required completed cold evidence)
TARGETS = {
    1645: (-0.25, "extra-cold-after-negative-cold", True),
    1880: (0.25, "midpoint-of-negative-parent-positive-cold", True),
    3654: (-0.375, "extra-cold-after-negative-cold", True),
    4607: (0.0, "cold-from-negative-parent", False),
    5536: (0.5, "cold-from-negative-parent", False),
}
EXCLUDED_SYSTEM_ID = 3523


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _strip_allowed(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    del frozen["output_root"]
    del frozen["stage2"]["delta_t_c"]
    return frozen


def _validated_run_evidence(
    root: Path,
    system_id: int,
    output_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    output_root = root / output_name
    snapshot_path = output_root / "config_snapshot.json"
    summary_path = output_root / "steady_50y_summary.json"
    _require(snapshot_path.is_file(), f"system-{system_id}-snapshot-missing:{output_name}")
    _require(summary_path.is_file(), f"system-{system_id}-summary-missing:{output_name}")
    snapshot = _read_json(snapshot_path)
    summary = _read_json(summary_path)
    _require(
        int(snapshot.get("system_id", -1)) == system_id,
        f"system-{system_id}-snapshot-id-conflict:{output_name}",
    )
    evidence = {
        "output_root": output_name,
        "config_snapshot": {
            "path": str(snapshot_path.relative_to(root)),
            "sha256": _sha256(snapshot_path),
        },
        "steady_50y_summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": _sha256(summary_path),
        },
        "status": summary.get("status"),
        "delta_t_c": float(snapshot["stage2"]["delta_t_c"]),
        "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
        "final_year": int(summary["final"]["year"]),
        "review_flags": list(summary.get("review_flags", [])),
        "runtime_error": summary.get("runtime_error"),
    }
    return snapshot, summary, evidence


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    parent_manifest_path = root / PARENT_MANIFEST
    cold_manifest_path = root / COLD_MANIFEST
    manifest_path = root / MANIFEST_NAME
    _require(_sha256(parent_manifest_path) == PARENT_MANIFEST_SHA256, "parent-manifest-conflict")
    _require(_sha256(cold_manifest_path) == COLD_MANIFEST_SHA256, "cold-manifest-conflict")
    _require(PYTHON.is_file(), f"python-missing:{PYTHON}")
    parent_manifest = _read_json(parent_manifest_path)
    cold_manifest = _read_json(cold_manifest_path)
    _require(parent_manifest.get("static_validation") == "pass", "parent-manifest-not-valid")
    _require(cold_manifest.get("static_validation") == "pass", "cold-manifest-not-valid")
    parent_by_id = {int(case["system_id"]): case for case in parent_manifest["cases"]}
    cold_by_id = {int(case["system_id"]): case for case in cold_manifest["cases"]}

    destinations = [manifest_path]
    output_roots: list[Path] = []
    for system_id in TARGETS:
        destinations.append(config_root / f"steady_parent_refine5_{system_id}_v1.json")
        output_roots.append(root / f"w704steady_parent_refine5_{system_id}_v1")
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite-existing-paths:" + repr(collisions))

    prepared: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for system_id, (target_delta, decision, needs_cold) in TARGETS.items():
        _require(system_id in parent_by_id, f"system-{system_id}-missing-parent-record")
        parent_record = parent_by_id[system_id]
        for label, item in parent_record["parent_evidence"].items():
            path = root / str(item["path"])
            _require(path.is_file(), f"system-{system_id}-{label}-missing")
            _require(_sha256(path) == str(item["sha256"]), f"system-{system_id}-{label}-conflict")

        parent_config_path = root / str(parent_record["config"])
        _require(parent_config_path.is_file(), f"system-{system_id}-parent-config-missing")
        parent_config = _read_json(parent_config_path)
        parent_snapshot, parent_summary, parent_run = _validated_run_evidence(
            root, system_id, str(parent_record["output_root"])
        )
        _require(parent_config == parent_snapshot, f"system-{system_id}-parent-config-run-conflict")
        _require(parent_summary.get("status") == "steady-50y-fail", f"system-{system_id}-parent-status-conflict")
        _require(int(parent_summary["final"]["year"]) == 50, f"system-{system_id}-parent-incomplete")
        _require(float(parent_summary["volume_change_fraction_50y"]) < 0.0, f"system-{system_id}-parent-not-negative")

        cold_run = None
        if needs_cold:
            _require(system_id in cold_by_id, f"system-{system_id}-cold-record-missing")
            cold_record = cold_by_id[system_id]
            cold_config_path = root / str(cold_record["config"])
            _require(cold_config_path.is_file(), f"system-{system_id}-cold-config-missing")
            cold_config = _read_json(cold_config_path)
            cold_snapshot, cold_summary, cold_run = _validated_run_evidence(
                root, system_id, str(cold_record["output_root"])
            )
            _require(cold_config == cold_snapshot, f"system-{system_id}-cold-config-run-conflict")
            _require(_strip_allowed(cold_config) == _strip_allowed(parent_config), f"system-{system_id}-cold-contract-conflict")
            _require(int(cold_summary["final"]["year"]) == 50, f"system-{system_id}-cold-incomplete")
            if system_id == 1880:
                _require(float(cold_summary["volume_change_fraction_50y"]) > 0.0, "system-1880-cold-not-positive")
            else:
                _require(float(cold_summary["volume_change_fraction_50y"]) < 0.0, f"system-{system_id}-cold-not-negative")

        output_name = f"w704steady_parent_refine5_{system_id}_v1"
        config_path = config_root / f"steady_parent_refine5_{system_id}_v1.json"
        payload = copy.deepcopy(parent_config)
        payload["stage2"]["delta_t_c"] = float(target_delta)
        payload["output_root"] = {"anchor": "project", "sub": output_name}
        _require(_strip_allowed(payload) == _strip_allowed(parent_config), f"system-{system_id}-frozen-field-conflict")
        _require(float(payload["stage2"]["delta_t_c"]) == target_delta, f"system-{system_id}-target-delta-conflict")

        command = (
            f"& {_ps_quote(str(PYTHON))} -m {RUNNER_MODULE} "
            f"--config {_ps_quote(str(config_path))}"
        )
        record = {
            "system_id": system_id,
            "decision": decision,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "parent_delta_t_c": float(parent_config["stage2"]["delta_t_c"]),
            "target_delta_t_c": float(target_delta),
            "frozen_parent_evidence": parent_record["parent_evidence"],
            "parent_temperature_config": {
                "path": str(parent_config_path.relative_to(root)),
                "sha256": _sha256(parent_config_path),
            },
            "parent_run": parent_run,
            "cold_run": cold_run,
            "run_command_powershell": command,
        }
        prepared.append((config_path, payload, record))

    # Hard exclusion: 3523 cold run is incomplete because of CFL and gets no config.
    _require(EXCLUDED_SYSTEM_ID in cold_by_id, "excluded-3523-cold-record-missing")
    excluded_record = cold_by_id[EXCLUDED_SYSTEM_ID]
    _, excluded_summary, excluded_evidence = _validated_run_evidence(
        root, EXCLUDED_SYSTEM_ID, str(excluded_record["output_root"])
    )
    excluded_flags = set(excluded_summary.get("review_flags", []))
    _require("cfl-error" in excluded_flags and "50y-incomplete" in excluded_flags, "excluded-3523-cfl-evidence-conflict")
    _require(int(excluded_summary["final"]["year"]) < 50, "excluded-3523-unexpected-completion")

    config_root.mkdir(parents=True, exist_ok=True)
    for config_path, payload, _ in prepared:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    for config_path, payload, _ in prepared:
        _require(_read_json(config_path) == payload, f"roundtrip-conflict:{config_path}")
    _require(all(not path.exists() for path in output_roots), "output-created-during-preparation")

    manifest = {
        "status": "configs-built-not-run",
        "stage": "parent-temperature-refinement-five-v1",
        "count": len(prepared),
        "parent_manifest": {"path": PARENT_MANIFEST, "sha256": PARENT_MANIFEST_SHA256},
        "cold_manifest": {"path": COLD_MANIFEST, "sha256": COLD_MANIFEST_SHA256},
        "runner_module": RUNNER_MODULE,
        "python": str(PYTHON),
        "allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "frozen_contract": parent_manifest["contract"],
        "oggm_started_by_builder": False,
        "static_validation": "pass",
        "excluded": [{"system_id": EXCLUDED_SYSTEM_ID, "reason": "cold-cfl-error-50y-incomplete", "evidence": excluded_evidence}],
        "cases": [record for _, _, record in prepared],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read_json(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "count": manifest["count"], "manifest": str(manifest_path), "static_validation": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
