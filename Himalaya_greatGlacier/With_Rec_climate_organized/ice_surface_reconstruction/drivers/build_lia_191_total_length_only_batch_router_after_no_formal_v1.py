"""Rebuild the total-length-only class router after no-formal class batches."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import build_lia_191_total_length_only_batch_router_v1 as _base
from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
DIAGNOSTICS = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
)
INPUT = DIAGNOSTICS / "dual_gate_inventory_after_no_formal_N3_v1"
OUTPUT = DIAGNOSTICS / "total_length_only_batch_router_after_no_formal_v1"


def main() -> None:
    systems = pd.read_csv(INPUT / "system_dual_gate_inventory.csv")
    chains = pd.read_csv(INPUT / "chain_dual_gate_inventory.csv")
    systems["lia_unique_id"] = systems["lia_unique_id"].astype(int)
    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    target_ids = set(
        systems.loc[
            systems["formal_inventory_route"].eq("total_length_only"),
            "lia_unique_id",
        ].astype(int)
    )
    if len(target_ids) != 60:
        raise RuntimeError(f"expected 60 total_length_only systems, found {len(target_ids)}")

    chains["stable_chain"] = _base._as_bool(chains["stable_chain"])
    chains["total_length_pass"] = _base._as_bool(chains["total_length_pass"])
    eligible = chains[
        chains["lia_unique_id"].isin(target_ids)
        & chains["stable_chain"]
        & chains["total_length_pass"]
    ].copy()
    eligible["signed_lia_sep"] = (
        eligible["forward_peak_m"] - eligible["lia_target_m"]
    ) / eligible["lia_rgi_separation_m_formal"]
    eligible["signed_rgi_sep"] = (
        eligible["forward_2000_m"] - eligible["rgi_target_m"]
    ) / eligible["lia_rgi_separation_m_formal"]
    anchors = (
        eligible.sort_values(
            ["lia_unique_id", "objective_sep_formal", "score_priority"],
            ascending=[True, True, False],
        )
        .groupby("lia_unique_id", as_index=False)
        .head(1)
        .copy()
    )
    if set(anchors["lia_unique_id"].astype(int)) != target_ids:
        missing = sorted(target_ids - set(anchors["lia_unique_id"].astype(int)))
        raise RuntimeError(f"missing anchors: {missing}")

    history_rows = []
    for system_id, group in chains[
        chains["lia_unique_id"].isin(target_ids)
    ].groupby("lia_unique_id"):
        history_rows.append(
            {"lia_unique_id": int(system_id), **_base._history_flags(group)}
        )
    anchors = anchors.merge(
        pd.DataFrame(history_rows), on="lia_unique_id", how="left", validate="one_to_one"
    )
    anchors["lia_endpoint_state"] = anchors["signed_lia_sep"].map(
        _base._endpoint_state
    )
    anchors["rgi_endpoint_state"] = anchors["signed_rgi_sep"].map(
        _base._endpoint_state
    )
    anchors["response_class"] = [
        _base._response_class(lia, rgi)
        for lia, rgi in zip(
            anchors["lia_endpoint_state"], anchors["rgi_endpoint_state"]
        )
    ]
    anchors["class_diagnosis"] = anchors["response_class"].map(
        lambda value: _base.CLASS_META[value]["diagnosis"]
    )
    anchors["proposed_family"] = anchors["response_class"].map(
        lambda value: _base.CLASS_META[value]["pilot_family"]
    )
    anchors["class_question"] = anchors["response_class"].map(
        lambda value: _base.CLASS_META[value]["question"]
    )
    anchors["log_target_separation_m"] = np.log10(
        anchors["lia_rgi_separation_m_formal"].clip(lower=1.0)
    )

    frozen = _base._frozen_ids()
    states, reasons = [], []
    for row in anchors.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        if system_id in frozen:
            states.append("excluded")
            reasons.append("already_frozen")
        elif system_id in _base.MANUAL_EXCLUSIONS:
            states.append("excluded")
            reasons.append(_base.MANUAL_EXCLUSIONS[system_id])
        elif bool(row.completed_layer3):
            states.append("closed")
            reasons.append("declared_layer3_already_completed")
        elif row.proposed_family == "glen_a" and bool(row.tested_glen_a):
            states.append("closed")
            reasons.append("proposed_family_already_tested")
        elif row.proposed_family == "prcp_fac" and bool(row.tested_prcp_fac):
            states.append("closed")
            reasons.append("proposed_family_already_tested")
        else:
            states.append("eligible_class_pilot")
            reasons.append("new_class_level_mechanism_question")
    anchors["batch_state"] = states
    anchors["batch_reason"] = reasons

    reps = []
    for _, group in anchors[
        anchors["batch_state"].eq("eligible_class_pilot")
    ].groupby("response_class", sort=True):
        reps.append(_base._select_representatives(group.reset_index(drop=True)))
    representatives = (
        pd.concat(reps, ignore_index=True) if reps else anchors.iloc[0:0].copy()
    )
    anchors["is_representative"] = anchors["lia_unique_id"].isin(
        set(representatives["lia_unique_id"].astype(int))
    )
    summary_rows = []
    for class_name, group in anchors.groupby("response_class", sort=True):
        rep_ids = representatives.loc[
            representatives["response_class"].eq(class_name), "lia_unique_id"
        ].astype(int).tolist()
        summary_rows.append(
            {
                "response_class": class_name,
                "system_count": int(len(group)),
                "eligible_count": int(
                    group["batch_state"].eq("eligible_class_pilot").sum()
                ),
                "closed_or_excluded_count": int(
                    (~group["batch_state"].eq("eligible_class_pilot")).sum()
                ),
                "proposed_family": _base.CLASS_META[class_name]["pilot_family"],
                "representative_ids": "|".join(map(str, rep_ids)),
                "class_question": _base.CLASS_META[class_name]["question"],
            }
        )
    class_summary = pd.DataFrame(summary_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    anchors.sort_values(["response_class", "lia_unique_id"]).to_csv(
        OUTPUT / "system_classification.csv", index=False
    )
    representatives.to_csv(OUTPUT / "representative_pilot_queue.csv", index=False)
    class_summary.to_csv(OUTPUT / "class_summary.csv", index=False)
    summary = {
        "source_inventory": str(INPUT),
        "total_length_only_system_count": int(len(anchors)),
        "eligible_class_pilot_count": int(
            anchors["batch_state"].eq("eligible_class_pilot").sum()
        ),
        "closed_count": int(anchors["batch_state"].eq("closed").sum()),
        "excluded_count": int(anchors["batch_state"].eq("excluded").sum()),
        "class_counts": {
            row.response_class: int(row.system_count)
            for row in class_summary.itertuples(index=False)
        },
        "representatives": {
            row.response_class: [
                int(value)
                for value in str(row.representative_ids).split("|")
                if value
            ]
            for row in class_summary.itertuples(index=False)
        },
        "launched_oggm": False,
    }
    (OUTPUT / "batch_router_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
