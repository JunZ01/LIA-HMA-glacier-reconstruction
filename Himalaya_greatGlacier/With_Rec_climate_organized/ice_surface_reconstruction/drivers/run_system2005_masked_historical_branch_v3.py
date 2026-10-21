"""Target-window checkpoint adjudication wrapper for historical branch v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_system2005_masked_historical_branch_v2 as v2
from . import run_system2005_masked_steady50_checkpointed_v1 as engine


def run(config_path: Path, resume_manifest: Path | None = None):
    result = v2.run(config_path, resume_manifest)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_years = int(config["target_years"])
    checkpoint_pass = set(result["checkpoint_metrics"]) == {
        str(year) for year in range(1, target_years + 1)
    }
    result["checks"]["all_annual_checkpoints_recorded"] = checkpoint_pass
    result["historical_numerical_checks"]["all_annual_checkpoints_recorded"] = checkpoint_pass
    result["status"] = (
        "historical-preflight-branch-pass"
        if all(result["historical_numerical_checks"].values())
        else "historical-preflight-branch-technical-fail"
    )
    result["schema"] = "lia-191-system2005-masked-historical-branch-v3"
    result["v2_terminal_issue"] = (
        "the reused base summary compared checkpoint keys with a literal 1..50 set"
    )
    result["v3_change"] = (
        "compare checkpoint keys with 1..target_years; dynamics, saved states, forcing, "
        "solver, and every numerical gate are unchanged"
    )
    output = Path(config["project_root"]) / config["output_root"]
    engine.atomic_bytes(output / "historical_summary_formal.json", engine.json_bytes(result))
    engine.atomic_bytes(
        output / "progress.json",
        engine.json_bytes(
            {
                "status": result["status"],
                "actual_calendar_year": result["actual_calendar_year"],
                "completed_years": result["completed_years"],
                "accepted_steps": result["accepted_steps"],
                "rejected_steps": result["rejected_steps"],
            }
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
