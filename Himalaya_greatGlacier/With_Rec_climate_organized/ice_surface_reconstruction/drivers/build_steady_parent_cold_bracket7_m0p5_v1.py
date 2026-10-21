"""Build seven parent-minus-0.5 K steady-state bracket configs.

Preparation only: this module validates frozen parent evidence and writes new
JSON files with exclusive-create semantics.  It neither imports nor runs OGGM.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_IDS = (1645, 1880, 3523, 3654, 3909, 4464, 5127)
PARENT_MANIFEST = "w704steady_parent_zero_pass9_manifest_v1.json"
PARENT_MANIFEST_SHA256 = (
    "3f63c938bc3b0047c3f964946553d25e148188fe3a3ef65a313a14e64afe03c7"
)
MANIFEST_NAME = "w704coldsteady_parent_m0p5_7_manifest_v1.json"
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
DELTA_T_OFFSET_C = -0.5


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _without_allowed_changes(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    del frozen["output_root"]
    del frozen["stage2"]["delta_t_c"]
    return frozen


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    parent_manifest_path = root / PARENT_MANIFEST
    manifest_path = root / MANIFEST_NAME

    _require(parent_manifest_path.is_file(), "parent-manifest-missing")
    _require(
        _sha256(parent_manifest_path) == PARENT_MANIFEST_SHA256,
        "parent-manifest-sha256-conflict",
    )
    _require(PYTHON.is_file(), f"python-missing:{PYTHON}")
    parent_manifest = _read_json(parent_manifest_path)
    _require(parent_manifest.get("static_validation") == "pass", "parent-not-validated")
    parent_by_id = {
        int(case["system_id"]): case for case in parent_manifest["cases"]
    }
    _require(
        all(system_id in parent_by_id for system_id in SYSTEM_IDS),
        "parent-manifest-system-set-conflict",
    )

    destinations = [manifest_path]
    output_roots: list[Path] = []
    for system_id in SYSTEM_IDS:
        destinations.append(
            config_root / f"steady_parent_cold_bracket_m0p5_{system_id}_v1.json"
        )
        output_roots.append(
            root / f"w704coldsteady_parent_m0p5_{system_id}_v1"
        )
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite-existing-paths:" + repr(collisions))

    prepared: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for system_id in SYSTEM_IDS:
        parent_record = parent_by_id[system_id]
        parent_config_path = root / str(parent_record["config"])
        parent_output_root = root / str(parent_record["output_root"])
        parent_snapshot_path = parent_output_root / "config_snapshot.json"
        parent_summary_path = parent_output_root / "steady_50y_summary.json"
        for path in (parent_config_path, parent_snapshot_path, parent_summary_path):
            _require(path.is_file(), f"system-{system_id}-missing-parent:{path}")

        # Recheck every provenance fingerprint frozen in the parent manifest.
        for label, evidence in parent_record["parent_evidence"].items():
            evidence_path = root / str(evidence["path"])
            _require(evidence_path.is_file(), f"system-{system_id}-{label}-missing")
            _require(
                _sha256(evidence_path) == str(evidence["sha256"]),
                f"system-{system_id}-{label}-sha256-conflict",
            )

        parent_config = _read_json(parent_config_path)
        parent_snapshot = _read_json(parent_snapshot_path)
        parent_summary = _read_json(parent_summary_path)
        _require(
            parent_config == parent_snapshot,
            f"system-{system_id}-parent-run-config-conflict",
        )
        _require(
            int(parent_config.get("system_id", -1)) == system_id,
            f"system-{system_id}-parent-config-id-conflict",
        )
        _require(
            str(parent_config["output_root"]["sub"]) == str(parent_record["output_root"]),
            f"system-{system_id}-parent-output-root-conflict",
        )
        _require(
            parent_summary.get("status") == "steady-50y-fail",
            f"system-{system_id}-unexpected-parent-status",
        )
        parent_dv50 = float(parent_summary["volume_change_fraction_50y"])
        _require(parent_dv50 < 0.0, f"system-{system_id}-parent-dv50-not-negative")
        _require(
            parent_config["stage2"] == parent_record["stage2"],
            f"system-{system_id}-parent-stage2-conflict",
        )

        parent_delta_t = float(parent_config["stage2"]["delta_t_c"])
        cold_delta_t = parent_delta_t + DELTA_T_OFFSET_C
        output_name = f"w704coldsteady_parent_m0p5_{system_id}_v1"
        config_path = (
            config_root / f"steady_parent_cold_bracket_m0p5_{system_id}_v1.json"
        )
        payload = copy.deepcopy(parent_config)
        payload["stage2"]["delta_t_c"] = cold_delta_t
        payload["output_root"] = {"anchor": "project", "sub": output_name}

        # The output root and delta-T are the only permitted changes.
        _require(
            _without_allowed_changes(payload) == _without_allowed_changes(parent_config),
            f"system-{system_id}-non-delta-physics-or-contract-change",
        )
        _require(
            float(payload["stage2"]["delta_t_c"]) == parent_delta_t - 0.5,
            f"system-{system_id}-delta-t-offset-conflict",
        )

        command = (
            f"& {_ps_quote(str(PYTHON))} -m {RUNNER_MODULE} "
            f"--config {_ps_quote(str(config_path))}"
        )
        record = {
            "system_id": system_id,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "parent_delta_t_c": parent_delta_t,
            "cold_delta_t_c": cold_delta_t,
            "parent_volume_change_fraction_50y": parent_dv50,
            "frozen_parent_evidence": parent_record["parent_evidence"],
            "parent_temperature_config": {
                "path": str(parent_config_path.relative_to(root)),
                "sha256": _sha256(parent_config_path),
            },
            "parent_run_config_snapshot": {
                "path": str(parent_snapshot_path.relative_to(root)),
                "sha256": _sha256(parent_snapshot_path),
            },
            "parent_steady_50y_summary": {
                "path": str(parent_summary_path.relative_to(root)),
                "sha256": _sha256(parent_summary_path),
            },
            "run_command_powershell": command,
        }
        prepared.append((config_path, payload, record))

    config_root.mkdir(parents=True, exist_ok=True)
    for config_path, payload, _ in prepared:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    for config_path, payload, _ in prepared:
        _require(_read_json(config_path) == payload, f"roundtrip-conflict:{config_path}")
    _require(
        all(not path.exists() for path in output_roots),
        "steady-output-created-during-preparation",
    )

    manifest = {
        "status": "configs-built-not-run",
        "stage": "parent-temperature-cold-bracket-minus-0.5K-v1",
        "count": len(prepared),
        "parent_manifest": {
            "path": PARENT_MANIFEST,
            "sha256": PARENT_MANIFEST_SHA256,
        },
        "runner_module": RUNNER_MODULE,
        "python": str(PYTHON),
        "delta_t_offset_c": DELTA_T_OFFSET_C,
        "frozen_contract": parent_manifest["contract"],
        "allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "oggm_started_by_builder": False,
        "static_validation": "pass",
        "cases": [record for _, _, record in prepared],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read_json(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "count": manifest["count"],
                "manifest": str(manifest_path),
                "static_validation": manifest["static_validation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
