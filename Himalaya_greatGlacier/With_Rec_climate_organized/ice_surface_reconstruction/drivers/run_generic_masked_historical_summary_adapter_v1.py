"""System-neutral adjudication for an already-produced historical summary.

This adapter never starts dynamics. It applies the registered target-year
checkpoint contract and writes a new versioned summary only when explicitly
invoked with an existing input summary and a distinct output path.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .generic_planform2d_adapter_contract_v1 import checkpoint_range_complete


def adjudicate(result: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    adapted = deepcopy(dict(result))
    target_years = int(config["target_years"])
    checkpoint_metrics = adapted.get("checkpoint_metrics", {})
    if not isinstance(checkpoint_metrics, Mapping):
        raise RuntimeError("checkpoint-metrics-not-a-mapping")
    checkpoint_pass = checkpoint_range_complete(checkpoint_metrics.keys(), target_years)
    checks = adapted.setdefault("checks", {})
    numerical = adapted.setdefault("historical_numerical_checks", {})
    if not isinstance(checks, dict) or not isinstance(numerical, dict):
        raise RuntimeError("historical-checks-not-mappings")
    checks["all_annual_checkpoints_recorded"] = checkpoint_pass
    numerical["all_annual_checkpoints_recorded"] = checkpoint_pass
    adapted["schema"] = "lia-191-generic-masked-historical-summary-adapter-v1"
    adapted["system_id"] = int(config["system_id"])
    adapted["target_years"] = target_years
    adapted["status"] = (
        "historical-preflight-branch-pass"
        if all(bool(value) for value in numerical.values())
        else "historical-preflight-branch-technical-fail"
    )
    adapted["adapter_contract"] = {
        "dynamics_run_by_adapter": False,
        "initial_thickness_modified": False,
        "old_summary_modified": False,
        "checkpoint_range_uses_target_years": True,
    }
    return adapted


def run(config_path: Path, input_summary: Path, output_summary: Path) -> dict[str, object]:
    if output_summary.exists():
        raise FileExistsError(output_summary)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = json.loads(input_summary.read_text(encoding="utf-8"))
    result = adjudicate(source, config)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.input_summary, args.output_summary), indent=2))


if __name__ == "__main__":
    main()
