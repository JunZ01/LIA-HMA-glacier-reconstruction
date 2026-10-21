"""One-hour adaptive zero-MB extension for system 2005.

This version reuses the validated 600-second step-doubling engine with a
versioned one-hour contract: checkpoints at 600, 1200, 1800, and 3600 seconds,
and a candidate timestep cap of 300 seconds.  The five-percent error gate and
all numerical, mass, boundary, and no-clipping rules remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import torch_planform_2d_system2005_adaptive_relaxation_600s_v1 as engine


TARGET_SECONDS = 3600.0
CHECKPOINT_SECONDS = (600.0, 1200.0, 1800.0, 3600.0)
MAXIMUM_DT_SECONDS = 300.0


def run(
    geology: Path,
    static_summary_path: Path,
    convergence_path: Path,
    relaxation_60s_path: Path,
    relaxation_600s_path: Path,
    output: Path,
) -> dict[str, object]:
    prerequisite = json.loads(relaxation_600s_path.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "adaptive-600s-pass-extension-not-authorized":
        raise RuntimeError("adaptive-600s-prerequisite-not-pass")

    original_target = engine.TARGET_SECONDS
    original_checkpoints = engine.CHECKPOINT_SECONDS
    original_maximum_dt = engine.MAXIMUM_DT_SECONDS
    try:
        engine.TARGET_SECONDS = TARGET_SECONDS
        engine.CHECKPOINT_SECONDS = CHECKPOINT_SECONDS
        engine.MAXIMUM_DT_SECONDS = MAXIMUM_DT_SECONDS
        summary = engine.run(
            geology,
            static_summary_path,
            convergence_path,
            relaxation_60s_path,
            output,
        )
    finally:
        engine.TARGET_SECONDS = original_target
        engine.CHECKPOINT_SECONDS = original_checkpoints
        engine.MAXIMUM_DT_SECONDS = original_maximum_dt

    final_checks = dict(summary["final_checks"])
    target_pass = bool(final_checks.pop("target_600s_completed"))
    final_checks["target_3600s_completed"] = target_pass
    passed = all(bool(value) for value in final_checks.values())
    summary.update(
        {
            "schema": "lia-191-torch-planform2d-system2005-adaptive-relaxation-3600s-v1",
            "status": "adaptive-3600s-pass-extension-not-authorized" if passed else "adaptive-3600s-fail-stop",
            "contract_reuses_validated_600s_engine": True,
            "scientific_operator_changed": False,
            "source_prerequisite_600s_status": prerequisite["status"],
            "final_checks": final_checks,
            "adjudication_required": "review-300s-candidate-accuracy-area-and-throughput-before-extension",
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "progress.json").write_text(
        json.dumps({"status": summary["status"], "physical_time_s": summary["completed_seconds"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--relaxation-60s-summary", type=Path, required=True)
    parser.add_argument("--relaxation-600s-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.geology,
                arguments.static_summary,
                arguments.convergence_summary,
                arguments.relaxation_60s_summary,
                arguments.relaxation_600s_summary,
                arguments.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
