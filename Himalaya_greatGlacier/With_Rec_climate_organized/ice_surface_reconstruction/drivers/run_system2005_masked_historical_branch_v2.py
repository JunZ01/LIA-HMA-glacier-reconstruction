"""Calendar-window wrapper for the system-2005 masked historical branch v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_system2005_masked_historical_branch_v1 as v1
from . import run_system2005_masked_steady50_checkpointed_v1 as engine
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge


def run(config_path: Path, resume_manifest: Path | None = None):
    original_month_count = TorchMonthlyTIBridge.month_count

    def registered_window_month_count(self: TorchMonthlyTIBridge) -> int:
        return min(int(self.sequence_indices.numel()), int(engine.TARGET_YEARS) * 12)

    TorchMonthlyTIBridge.month_count = property(registered_window_month_count)
    try:
        result = v1.run(config_path, resume_manifest)
    finally:
        TorchMonthlyTIBridge.month_count = original_month_count
    result["schema"] = "lia-191-system2005-masked-historical-branch-v2"
    result["v1_prelaunch_issue"] = (
        "v1 exposed all 6372 package months to the base engine, whose exact target-window "
        "guard requires 12 months per simulated year"
    )
    result["v2_change"] = (
        "expose exactly the leading 12*target_years calendar positions; climate arrays, "
        "sequence order, forcing formula, dynamics, and all gates are unchanged"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["project_root"]) / config["output_root"]
    engine.atomic_bytes(output / "historical_summary_formal.json", engine.json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
