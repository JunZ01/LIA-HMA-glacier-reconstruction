"""Technical-retry wrapper for the system-2005 one-day GPU diagnostic.

Version 1 stopped before any solver call because the launcher supplied the
GPU smoke summary instead of the authoritative planform static summary.  The
scientific operator and all numerical gates are unchanged here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import torch_planform_2d_system2005_adaptive_relaxation_86400s_v1 as engine


def run(
    geology: Path,
    static_summary_path: Path,
    convergence_path: Path,
    relaxation_60s_path: Path,
    relaxation_3600s_path: Path,
    output: Path,
) -> dict[str, object]:
    summary = engine.run(
        geology,
        static_summary_path,
        convergence_path,
        relaxation_60s_path,
        relaxation_3600s_path,
        output,
    )
    summary.update(
        {
            "schema": "lia-191-torch-planform2d-system2005-adaptive-relaxation-86400s-v2",
            "technical_retry_of": (
                "lia-191-torch-planform2d-system2005-adaptive-relaxation-86400s-v1"
            ),
            "technical_retry_reason": "v1-wrong-static-summary-path-before-solver-call",
            "scientific_contract_changed_from_v1": False,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--relaxation-60s-summary", type=Path, required=True)
    parser.add_argument("--relaxation-3600s-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.geology,
                arguments.static_summary,
                arguments.convergence_summary,
                arguments.relaxation_60s_summary,
                arguments.relaxation_3600s_summary,
                arguments.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
