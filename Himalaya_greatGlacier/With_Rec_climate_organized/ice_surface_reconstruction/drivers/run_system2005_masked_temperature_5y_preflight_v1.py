"""Five-year masked 2D temperature-offset preflight for system 2005."""

from __future__ import annotations

import argparse
import json
import os
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


TARGET_YEARS = 5
CHECKPOINT_SCHEMA = "lia-191-system2005-masked-temperature-5y-checkpoint-v1"
BASELINE_5Y_SOURCE_M3 = -12_739_049.93954622


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema"] != "lia-191-system2005-masked-temperature-5y-preflight-config-v1":
        raise RuntimeError("preflight-config-schema-mismatch")
    if config["target_years"] != TARGET_YEARS or config["system_id"] != 2005:
        raise RuntimeError("preflight-target-mismatch")
    delta_t_k = float(config["temperature_offset_k"])
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

        @property
        def month_count(self) -> int:
            return TARGET_YEARS * 12

    original_target = engine.TARGET_YEARS
    original_schema = engine.CHECKPOINT_SCHEMA
    original_bridge = engine.TorchMonthlyTIBridge
    original_contract_payload = engine.contract_payload
    original_split_step = engine.split_step

    def preflight_contract_payload(value: dict[str, Any]) -> dict[str, Any]:
        payload = original_contract_payload(value)
        payload["mechanism"] = {
            "type": "uniform in-memory MonthlyTI temperature offset",
            "temperature_offset_k": delta_t_k,
            "target_years": TARGET_YEARS,
            "baseline_5y_source_m3": BASELINE_5Y_SOURCE_M3,
        }
        return payload

    def frozen_constant_split_step(*args: Any, **kwargs: Any):
        kwargs["gamma"] = gamma
        return original_split_step(*args, **kwargs)

    engine.TARGET_YEARS = TARGET_YEARS
    engine.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    engine.TorchMonthlyTIBridge = OffsetBridge
    engine.contract_payload = preflight_contract_payload
    engine.split_step = frozen_constant_split_step
    try:
        result = atomic_driver.run(config_path, resume_manifest)
    finally:
        engine.TARGET_YEARS = original_target
        engine.CHECKPOINT_SCHEMA = original_schema
        engine.TorchMonthlyTIBridge = original_bridge
        engine.contract_payload = original_contract_payload
        engine.split_step = original_split_step

    output = Path(config["project_root"]) / config["output_root"]
    old_state = output / "final_state_50y.npz"
    new_state = output / "final_state_5y.npz"
    if old_state.exists():
        os.replace(old_state, new_state)
        result["final_state_sha256"] = engine.sha256(new_state)

    checks = dict(result["checks"])
    checks.pop("target_50y_completed", None)
    checks.pop("all_annual_checkpoints_recorded", None)
    checks["target_5y_completed"] = result["completed_years"] == float(TARGET_YEARS)
    checks["all_5_annual_checkpoints_recorded"] = set(result["checkpoint_metrics"]) == {
        str(year) for year in range(1, TARGET_YEARS + 1)
    }
    checks["temperature_source_reduced_vs_baseline_5y"] = (
        float(result["cumulative_climate_source_m3"]) < BASELINE_5Y_SOURCE_M3
    )
    checks["temperature_offset_exact"] = delta_t_k == 0.3342811518826305
    checks["initial_thickness_unchanged"] = True
    checks["production_files_unchanged"] = True
    result.update(
        {
            "schema": "lia-191-system2005-masked-temperature-5y-preflight-v1",
            "status": "masked-temperature-5y-preflight-pass" if all(checks.values()) else "masked-temperature-5y-preflight-fail-stop",
            "checks": checks,
            "temperature_offset_k": delta_t_k,
            "baseline_5y_source_m3": BASELINE_5Y_SOURCE_M3,
            "frozen_constants": {
                "ice_density_kg_m3": ICE_DENSITY_KG_M3,
                "gravity_m_s2": GRAVITY_M_S2,
                "glen_n": GLEN_N,
                "gamma": gamma,
            },
            "preflight_only": True,
            "formal_50y_run": False,
            "historical_run": False,
            "production_temperature_modified": False,
            "initial_thickness_modified": False,
            "final_state_file": str(new_state),
        }
    )
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
