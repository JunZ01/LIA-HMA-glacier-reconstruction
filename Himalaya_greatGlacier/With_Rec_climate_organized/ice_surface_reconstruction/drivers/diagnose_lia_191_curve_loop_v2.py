"""Corrected Core-191 curve-loop diagnosis.

Version 2 separates non-converged spin-up chains from genuine technical
failures. It reuses the read-only metrics implemented in version 1 and does
not launch OGGM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ice_surface_reconstruction.drivers import diagnose_lia_191_curve_loop as v1


def diagnose_chains(
    scores: pd.DataFrame,
    tolerance: float = v1.TOLERANCE,
    min_separation_m: float = v1.MIN_SEPARATION_M,
) -> pd.DataFrame:
    out = v1.diagnose_chains(scores, tolerance, min_separation_m)
    nontechnical_unstable = ~out["technical_failure_bool"] & ~out["stable_chain"]
    out.loc[nontechnical_unstable, "curve_class"] = "spinup_not_converged"
    out.loc[nontechnical_unstable, "next_action"] = (
        "repair_or_reestimate_stable_initial_state"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=v1.controller_root() / "scores.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v1.controller_root() / "diagnostics/curve_loop_v2",
    )
    parser.add_argument("--tolerance", type=float, default=v1.TOLERANCE)
    parser.add_argument(
        "--min-separation-m", type=float, default=v1.MIN_SEPARATION_M
    )
    args = parser.parse_args()

    chains = diagnose_chains(
        pd.read_csv(args.scores), args.tolerance, args.min_separation_m
    )
    systems = v1.summarize_systems(chains)
    summary = {
        "schema_version": 2,
        "supersedes": "curve_loop_v1",
        "diagnostic_only": True,
        "new_oggm_runs": 0,
        "spinup_length_used_as_lia_constraint": False,
        "lia_constraint": "maximum forward length",
        "modern_constraint": "year-2000 forward length",
        "endpoint_tolerance_fraction_of_lia_rgi_separation": args.tolerance,
        "spinup_stability_limit_fraction": v1.STABILITY_LIMIT,
        "minimum_separation_m": args.min_separation_m,
        "chain_count": len(chains),
        "system_count": len(systems),
        "stable_chain_count": int(chains["stable_chain"].sum()),
        "systems_with_stable_chain": int(systems["stable_chain_count"].gt(0).sum()),
        "technical_failure_count": int(chains["technical_failure_bool"].sum()),
        "chain_class_counts": {
            str(k): int(v) for k, v in chains["curve_class"].value_counts().items()
        },
        "best_system_class_counts": {
            str(k): int(v) for k, v in systems["best_curve_class"].value_counts().items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chains.to_csv(args.output_dir / "curve_loop_chain_diagnostics.csv", index=False)
    systems.to_csv(args.output_dir / "curve_loop_system_best.csv", index=False)
    chains[chains["curve_class"].str.startswith("joint_endpoint")].to_csv(
        args.output_dir / "curve_loop_candidate_chains.csv", index=False
    )
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
