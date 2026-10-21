"""Summarize and plot the first class-based continuous parameter batches."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULT_ROOT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
OUTPUT = (
    RESULT_ROOT
    / "lia_191_controller_v1"
    / "diagnostics"
    / "class_batch_outcome_v1"
)

EXPERIMENT_CLASS = {
    "lia_191_class_C1_glen_pilot_v1": "C1_peak_short_rgi_long",
    "lia_191_class_C2_glen_pilot_v1": "C2_peak_short_rgi_near",
    "lia_191_class_C3_prcp_pilot_v1": "C3_both_short",
    "lia_191_class_C2_glen_promotion_v1": "C2_peak_short_rgi_near",
    "lia_191_class_C2_delta_recenter_v1": "C2_peak_short_rgi_near",
    "lia_191_class_C1_glen_spatial_pilot_v1": "C1_peak_short_rgi_long",
    "lia_191_class_C3_prcp_spatial_pilot_v1": "C3_both_short",
    "lia_191_class_C1_glen_spatial_promotion_v1": "C1_peak_short_rgi_long",
    "lia_191_class_C4_glen_regular_pilot_v1": "C4_peak_near_rgi_short",
    "lia_191_class_C4_glen_spatial_pilot_v1": "C4_peak_near_rgi_short",
    "lia_191_class_C5_glen_spatial_pilot_v1": "C5_peak_near_rgi_long",
}

CLASS_DECISIONS = {
    "C1_peak_short_rgi_long": {
        "family": "glen_a",
        "decision": "closed_after_representative_and_promotion",
        "reason": "two representative improvements triggered promotion, but no representative or promoted chain passed the formal dual gate",
    },
    "C2_peak_short_rgi_near": {
        "family": "glen_a_then_one_delta_recenter",
        "decision": "one_formal_pass_then_closed",
        "reason": "glen_a x0.75 improved three of four representatives; one permitted delta-T recenter produced formal pass system 1546",
    },
    "C3_both_short": {
        "family": "prcp_fac",
        "decision": "closed_no_promotion",
        "reason": "only one representative improved and technical/domain failure rate exceeded 30%",
    },
    "C4_peak_near_rgi_short": {
        "family": "glen_a",
        "decision": "closed_no_promotion",
        "reason": "only one of two geometry-valid representatives improved; third representative failed the existing outlet gate",
    },
    "C5_peak_near_rgi_long": {
        "family": "glen_a",
        "decision": "closed_no_promotion",
        "reason": "both tested directions worsened the sole class member",
    },
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    frames = []
    for experiment, response_class in EXPERIMENT_CLASS.items():
        frame = pd.read_csv(RESULT_ROOT / experiment / "scores.csv")
        frame["experiment"] = experiment
        frame["response_class"] = response_class
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True, sort=False)
    scores["technical_failure_bool"] = _as_bool(scores["technical_failure"])
    scores["spinup_converged_bool"] = _as_bool(scores["spinup_converged"])
    scores["length_observation_valid_bool"] = _as_bool(scores["length_observation_valid"])
    scores["accepted_bool"] = _as_bool(scores["accepted"])
    scores["stable_valid_bool"] = (
        ~scores["technical_failure_bool"]
        & scores["spinup_converged_bool"]
        & scores["length_observation_valid_bool"]
        & pd.to_numeric(scores["spinup_stability_fraction"], errors="coerce").le(0.02)
    )

    rows = []
    for response_class, group in scores.groupby("response_class", sort=True):
        decision = CLASS_DECISIONS[response_class]
        rows.append(
            {
                "response_class": response_class,
                "tested_family": decision["family"],
                "new_chain_count": int(len(group)),
                "technical_failure_count": int(group["technical_failure_bool"].sum()),
                "stable_valid_count": int(group["stable_valid_bool"].sum()),
                "observation_overflow_count": int(
                    _as_bool(group["observation_domain_overflow"]).sum()
                ),
                "formal_pass_count": int(group["accepted_bool"].sum()),
                "formal_pass_systems": "|".join(
                    map(
                        str,
                        sorted(
                            group.loc[group["accepted_bool"], "lia_unique_id"]
                            .astype(int)
                            .unique()
                        ),
                    )
                ),
                "decision": decision["decision"],
                "reason": decision["reason"],
            }
        )
    outcome = pd.DataFrame(rows)

    inventory_summary_path = (
        RESULT_ROOT
        / "lia_191_controller_v1"
        / "diagnostics"
        / "dual_gate_inventory_after_class_batches_v1"
        / "formal_inventory_summary.json"
    )
    inventory = json.loads(inventory_summary_path.read_text(encoding="utf-8"))
    registry = json.loads(
        (
            RESULT_ROOT
            / "lia_191_frozen_curve_registry_v1"
            / "registry_summary.json"
        ).read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": 1,
        "class_count": len(CLASS_DECISIONS),
        "new_chain_count": int(len(scores)),
        "new_technical_failure_count": int(scores["technical_failure_bool"].sum()),
        "new_stable_valid_count": int(scores["stable_valid_bool"].sum()),
        "new_formal_pass_chain_count": int(scores["accepted_bool"].sum()),
        "new_formal_pass_systems": sorted(
            scores.loc[scores["accepted_bool"], "lia_unique_id"].astype(int).unique().tolist()
        ),
        "formal_inventory_chain_count": int(inventory["score_row_count"]),
        "formal_inventory_system_count": int(inventory["systems_with_formal_dual_pass"]),
        "frozen_registry_count": int(registry["candidate_count"]),
        "frozen_registry_systems": registry["systems"],
        "user_review_policy": "review consolidated outcomes after class batches, not glacier by glacier",
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "all_new_chain_scores.csv", index=False)
    outcome.to_csv(OUTPUT / "class_batch_outcome.csv", index=False)
    scores.loc[scores["accepted_bool"]].to_csv(OUTPUT / "new_formal_passes.csv", index=False)
    (OUTPUT / "class_batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    valid = scores.loc[scores["stable_valid_bool"]].copy()
    valid["e_lia_sep"] = pd.to_numeric(valid["e_lia_sep"], errors="coerce")
    valid["e_rgi_sep"] = pd.to_numeric(valid["e_rgi_sep"], errors="coerce")
    valid = valid.dropna(subset=["e_lia_sep", "e_rgi_sep"])
    colors = {
        name: color
        for name, color in zip(
            sorted(CLASS_DECISIONS),
            ["#4472c4", "#70ad47", "#ed7d31", "#a5a5a5", "#7030a0"],
        )
    }
    figure, axis = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)
    axis.axvspan(0, 0.2, color="#d9ead3", alpha=0.45)
    axis.axhspan(0, 0.2, color="#d9ead3", alpha=0.45)
    axis.add_patch(
        plt.Rectangle((0, 0), 0.2, 0.2, fill=False, edgecolor="#2e7d32", linewidth=2)
    )
    for class_name, group in valid.groupby("response_class", sort=True):
        axis.scatter(
            group["e_lia_sep"],
            group["e_rgi_sep"],
            s=45,
            alpha=0.78,
            label=class_name.split("_", 1)[0],
            color=colors[class_name],
        )
    accepted = valid.loc[valid["accepted_bool"]]
    axis.scatter(
        accepted["e_lia_sep"],
        accepted["e_rgi_sep"],
        marker="*",
        s=190,
        facecolor="#ffd966",
        edgecolor="black",
        linewidth=1.0,
        label="formal pass",
        zorder=5,
    )
    axis.set_xlabel("LIA peak error / |LIA-RGI separation|")
    axis.set_ylabel("Year-2000 error / |LIA-RGI separation|")
    axis.set_title("Class-batch formal endpoint diagnostics")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    x99 = float(np.nanpercentile(valid["e_lia_sep"], 99)) if len(valid) else 1.0
    y99 = float(np.nanpercentile(valid["e_rgi_sep"], 99)) if len(valid) else 1.0
    axis.set_xlim(0, max(0.6, x99 * 1.08))
    axis.set_ylim(0, max(0.6, y99 * 1.08))
    figure.savefig(OUTPUT / "class_batch_endpoint_scatter.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nCLASS OUTCOME")
    print(outcome.to_string(index=False))


if __name__ == "__main__":
    main()
