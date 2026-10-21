"""Summarize a completed Core-191 tradeoff pilot without launching OGGM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    stage2 = pd.read_csv(args.stage2_members)
    numeric = [
        "lia_unique_id",
        "delta_t",
        "melt_f",
        "spinup_stability_fraction",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "amplitude_ratio_to_target",
    ]
    for column in numeric:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["stable"] = (
        ~as_bool(scores["technical_failure"])
        & as_bool(scores["spinup_converged"])
        & scores["summary_status"].eq("ok")
        & scores["spinup_stability_fraction"].le(0.02)
    )
    scores["accepted_bool"] = as_bool(scores["accepted"])

    stage2_base = stage2[
        stage2["ensemble_role"].astype(str).eq("mid-turnover")
    ].copy()
    stage2_base["lia_unique_id"] = stage2_base["lia_unique_id"].astype(int)
    stage2_base = stage2_base[["lia_unique_id", "rgi_id", "melt_f"]].rename(
        columns={"melt_f": "melt_f_stage2"}
    )
    if stage2_base.duplicated(["lia_unique_id", "rgi_id"]).any():
        raise ValueError("Stage-2 mid-turnover rows are not unique by LIA system and RGI ID")
    scores = scores.merge(
        stage2_base, on=["lia_unique_id", "rgi_id"], how="left", validate="many_to_one"
    )
    scores["melt_f_stage2"] = pd.to_numeric(scores["melt_f_stage2"], errors="raise")
    scores["melt_f_multiplier"] = scores["melt_f"] / scores["melt_f_stage2"]

    system_rows: list[dict] = []
    level_rows: list[dict] = []
    best_rows: list[pd.Series] = []
    for system_id, group in scores.groupby("lia_unique_id", sort=True):
        stable = group[group["stable"]]
        best = (
            stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
            if not stable.empty
            else None
        )
        if best is not None:
            best_rows.append(best)
        system_rows.append(
            {
                "lia_unique_id": int(system_id),
                "chain_count": int(len(group)),
                "technical_failure_count": int(as_bool(group["technical_failure"]).sum()),
                "stable_chain_count": int(len(stable)),
                "accepted_chain_count": int(group["accepted_bool"].sum()),
                "best_delta_t": np.nan if best is None else best["delta_t"],
                "best_melt_f": np.nan if best is None else best["melt_f"],
                "best_melt_f_multiplier": (
                    np.nan if best is None else best["melt_f_multiplier"]
                ),
                "best_e_lia_sep": np.nan if best is None else best["e_lia_sep"],
                "best_e_rgi_sep": np.nan if best is None else best["e_rgi_sep"],
                "best_objective_sep": np.nan if best is None else best["objective_sep"],
                "best_amplitude_ratio_to_target": (
                    np.nan if best is None else best["amplitude_ratio_to_target"]
                ),
            }
        )
        for multiplier, level in group.groupby("melt_f_multiplier", sort=True):
            level_stable = level[level["stable"]]
            level_best = (
                level_stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
                if not level_stable.empty
                else None
            )
            level_rows.append(
                {
                    "lia_unique_id": int(system_id),
                    "melt_f_multiplier": float(multiplier),
                    "chain_count": int(len(level)),
                    "stable_chain_count": int(len(level_stable)),
                    "accepted_chain_count": int(level["accepted_bool"].sum()),
                    "best_delta_t": np.nan if level_best is None else level_best["delta_t"],
                    "best_e_lia_sep": (
                        np.nan if level_best is None else level_best["e_lia_sep"]
                    ),
                    "best_e_rgi_sep": (
                        np.nan if level_best is None else level_best["e_rgi_sep"]
                    ),
                    "best_objective_sep": (
                        np.nan if level_best is None else level_best["objective_sep"]
                    ),
                    "best_amplitude_ratio_to_target": (
                        np.nan
                        if level_best is None
                        else level_best["amplitude_ratio_to_target"]
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched = scores.sort_values(
        ["lia_unique_id", "melt_f_multiplier", "delta_t"]
    )
    enriched.to_csv(args.output_dir / "tradeoff_scores_enriched.csv", index=False)
    pd.DataFrame(system_rows).to_csv(
        args.output_dir / "tradeoff_system_summary.csv", index=False
    )
    pd.DataFrame(level_rows).to_csv(
        args.output_dir / "tradeoff_level_summary.csv", index=False
    )
    pd.DataFrame(best_rows).to_csv(
        args.output_dir / "tradeoff_best_stable_rows.csv", index=False
    )
    enriched[enriched["accepted_bool"]].to_csv(
        args.output_dir / "tradeoff_accepted_rows.csv", index=False
    )
    print(pd.DataFrame(system_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
