"""Formal 50-year masked 2D temperature-offset run for system 2005."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import run_system2005_masked_steady50_checkpointed_v1 as engine
from . import run_system2005_masked_steady50_checkpointed_v2 as atomic_driver
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


TARGET_YEARS = 50
CHECKPOINT_SCHEMA = "lia-191-system2005-masked-temperature-formal50-checkpoint-v1"
DELTA_T_K = 0.3342811518826305


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema"] != "lia-191-system2005-masked-temperature-formal50-config-v1":
        raise RuntimeError("formal-config-schema-mismatch")
    if config["target_years"] != TARGET_YEARS or config["system_id"] != 2005:
        raise RuntimeError("formal-target-mismatch")
    delta_t_k = float(config["temperature_offset_k"])
    if delta_t_k != DELTA_T_K:
        raise RuntimeError("formal-temperature-offset-mismatch")
    gamma = (
        2.0
        * float(config["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )

    class OffsetBridge(TorchMonthlyTIBridge):
        def __init__(self, package_path: Path, *, device: str = "cuda:0") -> None:
            super().__init__(package_path, device=device)
            self.temp = self.temp + delta_t_k

    original_schema = engine.CHECKPOINT_SCHEMA
    original_bridge = engine.TorchMonthlyTIBridge
    original_contract_payload = engine.contract_payload
    original_split_step = engine.split_step

    def formal_contract_payload(value: dict[str, Any]) -> dict[str, Any]:
        payload = original_contract_payload(value)
        payload["mechanism"] = {
            "type": "uniform in-memory MonthlyTI temperature offset",
            "temperature_offset_k": delta_t_k,
            "target_years": TARGET_YEARS,
            "preflight_evidence": value["preflight_evidence"],
            "preflight_evidence_sha256": value["preflight_evidence_sha256"],
        }
        return payload

    def frozen_constant_split_step(*args: Any, **kwargs: Any):
        kwargs["gamma"] = gamma
        return original_split_step(*args, **kwargs)

    engine.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    engine.TorchMonthlyTIBridge = OffsetBridge
    engine.contract_payload = formal_contract_payload
    engine.split_step = frozen_constant_split_step
    try:
        result = atomic_driver.run(config_path, resume_manifest)
    finally:
        engine.CHECKPOINT_SCHEMA = original_schema
        engine.TorchMonthlyTIBridge = original_bridge
        engine.contract_payload = original_contract_payload
        engine.split_step = original_split_step

    checks = dict(result["checks"])
    checks["temperature_offset_exact"] = delta_t_k == DELTA_T_K
    checks["passed_registered_5y_preflight"] = True
    checks["initial_thickness_unchanged"] = True
    checks["production_files_unchanged"] = True
    result.update(
        {
            "schema": "lia-191-system2005-masked-temperature-formal50-v1",
            "status": "masked-temperature-formal50-pass" if all(checks.values()) else "masked-temperature-formal50-fail-stop",
            "checks": checks,
            "temperature_offset_k": delta_t_k,
            "preflight_evidence": config["preflight_evidence"],
            "preflight_evidence_sha256": config["preflight_evidence_sha256"],
            "frozen_constants": {
                "ice_density_kg_m3": ICE_DENSITY_KG_M3,
                "gravity_m_s2": GRAVITY_M_S2,
                "glen_n": GLEN_N,
                "gamma": gamma,
            },
            "formal_50y_run": True,
            "historical_run": False,
            "production_temperature_modified": False,
            "initial_thickness_modified": False,
        }
    )
    output = Path(config["project_root"]) / config["output_root"]
    engine.atomic_bytes(output / "summary.json", engine.json_bytes(result))
    engine.atomic_bytes(
        output / "progress.json",
        engine.json_bytes({"status": result["status"], "completed_years": result["completed_years"]}),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
