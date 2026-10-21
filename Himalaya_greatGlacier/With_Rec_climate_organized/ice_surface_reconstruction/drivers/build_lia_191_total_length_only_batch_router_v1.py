"""Build a class-based batch queue for formal total-length-only systems.

This driver is read-only with respect to OGGM states.  It consumes the latest
formal dual-gate inventory, selects one stable total-length-pass anchor per
system, assigns deterministic endpoint-response classes, records previously
tested parameter families, and chooses representative systems for bounded
class-level mechanism pilots.

No OGGM model is advanced by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
INVENTORY_DIR = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_layer3_2266_v1"
)
OUTPUT_DIR = INVENTORY_DIR.parent / "total_length_only_batch_router_v1"
FROZEN_REGISTRY = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_frozen_curve_registry_v1"
)

FORMAL_TOLERANCE = 0.20
MANUAL_EXCLUSIONS = {
    1044: "manual_provisional_outside_controller_sources",
    1545: "named_configuration_closed",
    3584: "named_configuration_closed",
    2387: "blocked_dynamic_review",
    5508: "blocked_dynamic_review",
}


CLASS_META = {
    "C1_peak_short_rgi_long": {
        "diagnosis": "forward peak is short while year 2000 is long; simulated retreat amplitude is too small",
        "pilot_family": "glen_a",
        "question": "Can a bounded glen_a direction probe increase effective retreat amplitude without losing either endpoint?",
    },
    "C2_peak_short_rgi_near": {
        "diagnosis": "year 2000 is already near RGI while the LIA peak is short",
        "pilot_family": "glen_a",
        "question": "Can a bounded glen_a response-time probe raise the historical peak while retaining the modern endpoint?",
    },
    "C3_both_short": {
        "diagnosis": "both historical peak and year-2000 endpoint are short; the trajectory level is too low",
        "pilot_family": "prcp_fac",
        "question": "Can a bounded fixed-climatology prcp_fac increase the curve level while preserving the simulated amplitude?",
    },
    "C4_peak_near_rgi_short": {
        "diagnosis": "the LIA peak is near target while year 2000 is short; retreat is too strong",
        "pilot_family": "glen_a",
        "question": "Can a bounded glen_a direction probe reduce retreat without losing the LIA peak?",
    },
    "C5_peak_near_rgi_long": {
        "diagnosis": "the LIA peak is near target while year 2000 is long; retreat is too weak",
        "pilot_family": "glen_a",
        "question": "Does the system-2266 glen_a response mechanism generalize to this near-peak/long-modern class?",
    },
    "C6_other": {
        "diagnosis": "endpoint sign pattern is outside the five declared response classes",
        "pilot_family": "manual_diagnosis",
        "question": "Review the signed endpoint response before declaring a parameter family.",
    },
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _endpoint_state(value: float) -> str:
    if value < -FORMAL_TOLERANCE:
        return "short"
    if value > FORMAL_TOLERANCE:
        return "long"
    return "near"


def _response_class(lia_state: str, rgi_state: str) -> str:
    mapping = {
        ("short", "long"): "C1_peak_short_rgi_long",
        ("short", "near"): "C2_peak_short_rgi_near",
        ("short", "short"): "C3_both_short",
        ("near", "short"): "C4_peak_near_rgi_short",
        ("near", "long"): "C5_peak_near_rgi_long",
    }
    return mapping.get((lia_state, rgi_state), "C6_other")


def _frozen_ids() -> set[int]:
    if not FROZEN_REGISTRY.exists():
        return set()
    out: set[int] = set()
    for path in FROZEN_REGISTRY.glob("system_*"):
        try:
            out.add(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return out


def _history_flags(group: pd.DataFrame) -> dict[str, object]:
    layers = set(group["score_layer"].dropna().astype(str))
    sources = set(group["score_source"].dropna().astype(str))
    text = "|".join(sorted(layers | sources)).lower()
    return {
        "tested_delta_t": "delta" in text,
        "tested_melt_f": "melt" in text,
        "tested_glen_a": "glen" in text,
        "tested_temp_melt": "temp_melt" in text,
        "tested_prcp_fac": "prcp" in text,
        "completed_layer3": "layer3_single_parameter" in layers,
        "score_layers": "|".join(sorted(layers)),
        "score_sources": "|".join(sorted(sources)),
    }


def _representative_count(n: int) -> int:
    if n >= 12:
        return 5
    if n >= 6:
        return 4
    if n >= 3:
        return 3
    return n


def _select_representatives(group: pd.DataFrame) -> pd.DataFrame:
    """Select a medoid plus deterministic spread in response/size space."""

    n_select = _representative_count(len(group))
    if n_select == 0:
        return group.iloc[0:0].copy()
    if len(group) <= n_select:
        out = group.copy()
        out["representative_rank"] = np.arange(1, len(out) + 1)
        return out

    feature_names = [
        "signed_lia_sep",
        "signed_rgi_sep",
        "amplitude_ratio_to_target",
        "log_target_separation_m",
    ]
    values = group[feature_names].astype(float).to_numpy()
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = (values - center) / scale
    center_distance = np.sqrt(np.sum(z**2, axis=1))
    selected = [int(np.argmin(center_distance))]
    while len(selected) < n_select:
        remaining = [idx for idx in range(len(group)) if idx not in selected]
        distances = []
        for idx in remaining:
            nearest = min(float(np.linalg.norm(z[idx] - z[j])) for j in selected)
            distances.append((nearest, -int(group.iloc[idx]["lia_unique_id"]), idx))
        selected.append(max(distances)[2])

    out = group.iloc[selected].copy()
    out["representative_rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    systems = pd.read_csv(INVENTORY_DIR / "system_dual_gate_inventory.csv")
    chains = pd.read_csv(INVENTORY_DIR / "chain_dual_gate_inventory.csv")
    systems["lia_unique_id"] = pd.to_numeric(systems["lia_unique_id"], errors="raise").astype(int)
    chains["lia_unique_id"] = pd.to_numeric(chains["lia_unique_id"], errors="raise").astype(int)

    target_ids = set(
        systems.loc[
            systems["formal_inventory_route"].eq("total_length_only"), "lia_unique_id"
        ].tolist()
    )
    if len(target_ids) != 49:
        raise RuntimeError(f"Expected 49 total_length_only systems, found {len(target_ids)}")

    chains["stable_chain"] = _as_bool(chains["stable_chain"])
    chains["total_length_pass"] = _as_bool(chains["total_length_pass"])
    chains["formal_dual_pass"] = _as_bool(chains["formal_dual_pass"])
    eligible_anchors = chains.loc[
        chains["lia_unique_id"].isin(target_ids)
        & chains["stable_chain"]
        & chains["total_length_pass"]
    ].copy()
    if eligible_anchors.empty:
        raise RuntimeError("No stable total-length-pass anchors were found")

    eligible_anchors["signed_lia_sep"] = (
        eligible_anchors["forward_peak_m"] - eligible_anchors["lia_target_m"]
    ) / eligible_anchors["lia_rgi_separation_m_formal"]
    eligible_anchors["signed_rgi_sep"] = (
        eligible_anchors["forward_2000_m"] - eligible_anchors["rgi_target_m"]
    ) / eligible_anchors["lia_rgi_separation_m_formal"]
    anchors = (
        eligible_anchors.sort_values(
            ["lia_unique_id", "objective_sep_formal", "score_priority"],
            ascending=[True, True, False],
        )
        .groupby("lia_unique_id", as_index=False)
        .head(1)
        .copy()
    )
    if set(anchors["lia_unique_id"]) != target_ids:
        missing = sorted(target_ids - set(anchors["lia_unique_id"]))
        raise RuntimeError(f"Missing total-length-pass anchors for systems: {missing}")

    history_rows = []
    for system_id, group in chains.loc[chains["lia_unique_id"].isin(target_ids)].groupby("lia_unique_id"):
        history_rows.append({"lia_unique_id": int(system_id), **_history_flags(group)})
    history = pd.DataFrame(history_rows)

    anchors = anchors.merge(history, on="lia_unique_id", how="left", validate="one_to_one")
    anchors["lia_endpoint_state"] = anchors["signed_lia_sep"].map(_endpoint_state)
    anchors["rgi_endpoint_state"] = anchors["signed_rgi_sep"].map(_endpoint_state)
    anchors["response_class"] = [
        _response_class(lia, rgi)
        for lia, rgi in zip(anchors["lia_endpoint_state"], anchors["rgi_endpoint_state"])
    ]
    anchors["class_diagnosis"] = anchors["response_class"].map(
        lambda value: CLASS_META[value]["diagnosis"]
    )
    anchors["proposed_family"] = anchors["response_class"].map(
        lambda value: CLASS_META[value]["pilot_family"]
    )
    anchors["class_question"] = anchors["response_class"].map(
        lambda value: CLASS_META[value]["question"]
    )
    anchors["log_target_separation_m"] = np.log10(
        pd.to_numeric(anchors["lia_rgi_separation_m_formal"], errors="coerce").clip(lower=1.0)
    )

    frozen = _frozen_ids()
    review_state = []
    review_reason = []
    for row in anchors.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        if system_id in frozen:
            review_state.append("excluded")
            review_reason.append("already_frozen")
        elif system_id in MANUAL_EXCLUSIONS:
            review_state.append("excluded")
            review_reason.append(MANUAL_EXCLUSIONS[system_id])
        elif bool(row.completed_layer3):
            review_state.append("closed")
            review_reason.append("declared_layer3_already_completed")
        elif row.proposed_family == "glen_a" and bool(row.tested_glen_a):
            review_state.append("closed")
            review_reason.append("proposed_family_already_tested")
        elif row.proposed_family == "prcp_fac" and bool(row.tested_prcp_fac):
            review_state.append("closed")
            review_reason.append("proposed_family_already_tested")
        else:
            review_state.append("eligible_class_pilot")
            review_reason.append("new_class_level_mechanism_question")
    anchors["batch_state"] = review_state
    anchors["batch_reason"] = review_reason

    representative_frames = []
    for class_name, group in anchors.loc[
        anchors["batch_state"].eq("eligible_class_pilot")
    ].groupby("response_class", sort=True):
        selected = _select_representatives(group.reset_index(drop=True))
        selected["class_system_count_eligible"] = len(group)
        representative_frames.append(selected)
    representatives = (
        pd.concat(representative_frames, ignore_index=True)
        if representative_frames
        else anchors.iloc[0:0].copy()
    )

    anchors["is_representative"] = anchors["lia_unique_id"].isin(
        set(representatives["lia_unique_id"].astype(int))
    )
    anchors = anchors.sort_values(["response_class", "lia_unique_id"]).reset_index(drop=True)

    summary_rows = []
    for class_name, group in anchors.groupby("response_class", sort=True):
        meta = CLASS_META[class_name]
        rep_ids = representatives.loc[
            representatives["response_class"].eq(class_name), "lia_unique_id"
        ].astype(int).tolist()
        summary_rows.append(
            {
                "response_class": class_name,
                "diagnosis": meta["diagnosis"],
                "system_count": int(len(group)),
                "eligible_count": int(group["batch_state"].eq("eligible_class_pilot").sum()),
                "closed_or_excluded_count": int((~group["batch_state"].eq("eligible_class_pilot")).sum()),
                "proposed_family": meta["pilot_family"],
                "representative_count": len(rep_ids),
                "representative_ids": "|".join(map(str, rep_ids)),
                "class_question": meta["question"],
            }
        )
    class_summary = pd.DataFrame(summary_rows)

    queue_columns = [
        "response_class",
        "representative_rank",
        "lia_unique_id",
        "rgi_id",
        "task_id",
        "output_dir",
        "delta_t",
        "melt_f",
        "prcp_fac_multiplier",
        "glen_a",
        "temp_melt",
        "signed_lia_sep",
        "signed_rgi_sep",
        "amplitude_ratio_to_target",
        "lia_rgi_separation_m_formal",
        "geometry_variant",
        "proposed_family",
        "class_question",
    ]
    representative_queue = representatives.loc[:, queue_columns].sort_values(
        ["response_class", "representative_rank"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(OUTPUT_DIR / "system_classification.csv", index=False)
    class_summary.to_csv(OUTPUT_DIR / "class_summary.csv", index=False)
    representative_queue.to_csv(OUTPUT_DIR / "representative_pilot_queue.csv", index=False)
    summary = {
        "source_inventory": str(INVENTORY_DIR),
        "formal_tolerance": FORMAL_TOLERANCE,
        "total_length_only_system_count": int(len(anchors)),
        "eligible_class_pilot_count": int(anchors["batch_state"].eq("eligible_class_pilot").sum()),
        "closed_count": int(anchors["batch_state"].eq("closed").sum()),
        "excluded_count": int(anchors["batch_state"].eq("excluded").sum()),
        "representative_count": int(len(representative_queue)),
        "class_counts": {
            row.response_class: int(row.system_count) for row in class_summary.itertuples(index=False)
        },
        "representatives": {
            row.response_class: [int(value) for value in str(row.representative_ids).split("|") if value]
            for row in class_summary.itertuples(index=False)
        },
        "ogmm_advanced": False,
    }
    (OUTPUT_DIR / "batch_router_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nCLASS SUMMARY")
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
