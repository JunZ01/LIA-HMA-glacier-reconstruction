"""Technical retry for the relaxed monotone-transition polytope audit.

The V2 audit correctly builds its linear constraints from the LP station
representation, but it accidentally passes that reduced representation to the
full nonlinear gate evaluator.  This adapter supplies the evaluator with the
complete station representation while leaving the LP systems, candidate
paths, thresholds, random seed, and optimization unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import resolve_temperature_candidate_monotone_lp_feasible_polytopes_v2 as core


DRIVER = Path(__file__).resolve()
OUTPUT = (
    core.LOOP
    / "temperature_candidate_monotone_lp_feasible_polytopes_v3"
)


def main() -> int:
    stations = pd.read_csv(core.STATION)
    if stations["year"].between(1921, 1939).any():
        raise RuntimeError("locked-year-in-station")
    full_groups = core.searchbase.build_station_groups(stations)

    original_evaluate = core.searchbase.evaluate_factors

    def evaluate_with_full_groups(factors, _lp_groups, hma, hma_internal_max):
        return original_evaluate(factors, full_groups, hma, hma_internal_max)

    core.OUTPUT = OUTPUT
    core.DRIVER = DRIVER
    core.searchbase.evaluate_factors = evaluate_with_full_groups
    try:
        result = core.main()
    finally:
        core.searchbase.evaluate_factors = original_evaluate

    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "hma-temperature-candidate-monotone-lp-feasible-polytopes-v3",
            "technical_retry": {
                "supersedes_failed_attempt": str(
                    core.LOOP / "temperature_candidate_monotone_lp_feasible_polytopes_v2"
                ),
                "failure": "LP-only station records were passed to the full nonlinear evaluator",
                "correction": "Full station records are used only by the nonlinear evaluator",
                "scientific_contract_changed": False,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "TECHNICAL_RETRY.md").write_text(
        "# Technical retry\n\n"
        "V2 stopped before candidate evaluation because its reduced LP station "
        "records do not contain the original bias and MAE fields required by "
        "the full nonlinear gate evaluator. V3 changes only that data handoff. "
        "All scientific inputs, thresholds, random seed, LP systems, candidate "
        "families, and optimization settings are unchanged. No sealed value is "
        "read or generated.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
