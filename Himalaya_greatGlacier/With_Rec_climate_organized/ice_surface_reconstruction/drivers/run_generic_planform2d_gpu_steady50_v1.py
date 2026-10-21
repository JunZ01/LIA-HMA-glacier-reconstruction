"""Run a frozen-input generic planform-2D 50-year GPU steady stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .generic_planform2d_adapter_contract_v1 import array_sha256
from . import run_system2005_masked_steady50_checkpointed_v1 as engine
from . import run_system2005_masked_steady50_checkpointed_v3 as steady_driver
from .run_generic_planform2d_gpu_qualification_preflight_v1 import (
    STAGE2_KEYS,
    validate_frozen_inputs,
)


SCHEMA = "lia-191-generic-planform2d-gpu-steady50-config-v1"
RESULT_SCHEMA = "lia-191-generic-planform2d-gpu-steady50-result-v1"
RUN_ROLE = "generic-planform2d-gpu-steady50"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prelaunch(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or config.get("run_role") != RUN_ROLE:
        raise RuntimeError("generic-steady-config-contract-mismatch")
    if int(config.get("target_years", 0)) != engine.TARGET_YEARS:
        raise RuntimeError("generic-steady-target-years-mismatch")
    if not bool(config.get("launch_authorized", False)):
        raise RuntimeError("generic-steady-launch-not-authorized")
    if bool(config.get("initial_thickness_modification_allowed", True)):
        raise RuntimeError("initial-thickness-modification-not-frozen")
    if bool(config.get("physics_modified", True)) or bool(config.get("gates_modified", True)):
        raise RuntimeError("physics-or-gates-modified")
    if engine.sha256(Path(__file__)) != str(config["driver_sha256"]).upper():
        raise RuntimeError("driver-hash-mismatch")

    root = Path(config["project_root"])
    path_keys = (
        "geology",
        "static_summary",
        "climate_package",
        "observation_path",
        "upstream_evidence",
        "upstream_config_snapshot",
        "qualification_summary",
        "parent_climate_summary",
        "expected_route",
    )
    paths = {name: root / config[name] for name in path_keys}
    for name, path in paths.items():
        if engine.sha256(path) != str(config[f"{name}_sha256"]).upper():
            raise RuntimeError(f"{name}-hash-mismatch")

    route = _load_json(paths["expected_route"])
    authorized = [int(value) for value in route.get("authorized_systems", [])]
    if not route.get("launch_authorized") or int(config["system_id"]) not in authorized:
        raise RuntimeError("route-does-not-authorize-system")
    if route.get("authorized_run_role") != RUN_ROLE:
        raise RuntimeError("route-run-role-mismatch")

    parent = _load_json(paths["parent_climate_summary"])
    qualification = _load_json(paths["qualification_summary"])
    if parent.get("status") != "historical-climate-bridge-pass":
        raise RuntimeError("parent-climate-summary-not-pass")
    if [int(value) for value in parent.get("calendar", [])] != [1441, 1470]:
        raise RuntimeError("parent-climate-calendar-mismatch")
    if int(parent.get("sequence_month_count", 0)) != 360:
        raise RuntimeError("parent-climate-month-count-mismatch")
    if qualification.get("status") != "qualification-preflight-pass":
        raise RuntimeError("qualification-summary-not-pass")
    if int(qualification.get("system_id", -1)) != int(config["system_id"]):
        raise RuntimeError("qualification-system-mismatch")

    with np.load(paths["climate_package"], allow_pickle=False) as archive:
        years = np.asarray(archive["sequence_years"], dtype=np.int64)
        months = np.asarray(archive["sequence_months"], dtype=np.int64)
        package_checks = {
            "month_count_360": years.size == 360 and months.size == 360,
            "calendar_starts_1441_01": int(years[0]) == 1441 and int(months[0]) == 1,
            "calendar_ends_1470_12": int(years[-1]) == 1470 and int(months[-1]) == 12,
            "finite_climate_arrays": all(
                np.isfinite(np.asarray(archive[key], dtype=np.float64)).all()
                for key in ("temp", "prcp", "grad")
            ),
        }
    if not all(package_checks.values()):
        raise RuntimeError("parent-climate-package-contract-failed")

    with Dataset(paths["geology"], "r") as dataset:
        initial = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
    frozen_checks = validate_frozen_inputs(config, paths, initial)
    frozen_checks["qualification_thickness_hash_matches"] = (
        str(qualification["geology_initial_thickness_sha256"]).upper()
        == str(config["geology_initial_thickness_sha256"]).upper()
        == array_sha256(initial)
    )
    frozen_checks["qualification_source_hash_matches"] = (
        str(qualification["source_thickness_sha256"]).upper()
        == str(config["source_thickness_sha256"]).upper()
    )
    for key in STAGE2_KEYS:
        frozen_checks[f"qualification_stage2_{key}_matches"] = (
            qualification["stage2_frozen"][key] == config["stage2_frozen"][key]
        )
    if not all(frozen_checks.values()):
        failed = [key for key, value in frozen_checks.items() if not value]
        raise RuntimeError("frozen-input-contract-failed:" + ",".join(failed))
    return {
        "paths": paths,
        "parent_package_checks": package_checks,
        "frozen_input_checks": frozen_checks,
    }


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = _load_json(config_path)
    prelaunch = validate_prelaunch(config)
    result = steady_driver.run(config_path, resume_manifest)
    passed = result.get("status") == "masked-steady50-pass"
    result.update(
        {
            "schema": RESULT_SCHEMA,
            "status": "generic-masked-steady50-pass" if passed else "generic-masked-steady50-fail-stop",
            "run_role": RUN_ROLE,
            "stage2_frozen": config["stage2_frozen"],
            "source_thickness_sha256": config["source_thickness_sha256"],
            "geology_initial_thickness_sha256": config["geology_initial_thickness_sha256"],
            "parent_climate_calendar": [1441, 1470],
            "parent_climate_repeated_over_target_years": True,
            "parent_package_checks": prelaunch["parent_package_checks"],
            "frozen_input_checks": prelaunch["frozen_input_checks"],
            "qualification_summary_sha256": config["qualification_summary_sha256"],
            "initial_thickness_modified": False,
            "physics_modified": False,
            "gates_modified": False,
            "historical_run": False,
            "accepted_count_effect": 0,
        }
    )
    output = Path(config["project_root"]) / config["output_root"]
    engine.atomic_bytes(output / "summary.json", engine.json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    result = run(args.config, args.resume_manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result["status"] != "generic-masked-steady50-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
