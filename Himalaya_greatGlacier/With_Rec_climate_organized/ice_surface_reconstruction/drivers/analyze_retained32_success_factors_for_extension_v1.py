"""Read-only success-factor analysis for extending the retained 32 LIA curves.

This analysis deliberately does not authorize or start model runs.  It separates
route effects from target/geometry effects, then ranks the remaining systems for
future *new-mechanism* investigation.  The ranking is not a success probability.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE = BASE / "lia_191_current_route_v10_33/lia_191_current_route.csv"
ACCEPTED = BASE / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
SHAPES = BASE / "lia_191_accepted_58_curve_shape_screen_v1/accepted_58_curve_shape_metrics.csv"
INVENTORY = BASE / "lia_191_remaining133_mechanism_exhaustion_after_system3910_v1/remaining133_mechanism_inventory.csv"
RESPONSES = BASE / "lia_191_remaining135_response_groups_after_2580_v1/complete_response_groups.csv"
OUT = BASE / "lia_191_retained32_success_factor_extension_v1"


def robust_scale(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    center = float(numeric.median())
    q25, q75 = numeric.quantile([0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return center, scale


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()), float(numeric.quantile(0.25)), float(numeric.quantile(0.75))


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    OUT.mkdir(parents=True)

    route = pd.read_csv(ROUTE)
    route = route.rename(columns={"lia_unique_id": "system_id"})
    route["system_id"] = route["system_id"].astype(int)
    accepted = pd.read_csv(ACCEPTED)
    accepted["system_id"] = accepted["system_id"].astype(int)
    shapes = pd.read_csv(SHAPES)
    shapes["system_id"] = shapes["system_id"].astype(int)
    inventory = pd.read_csv(INVENTORY)
    inventory["system_id"] = inventory["system_id"].astype(int)
    responses = pd.read_csv(RESPONSES)
    responses["system_id"] = responses["system_id"].astype(int)

    accepted_meta = accepted[["system_id", "route"]].rename(columns={"route": "accepted_route"})
    shape_meta = shapes[[
        "system_id", "upward_variation_share", "maximum_rebound_km",
        "smoothed_curve_range_km", "maximum_rebound_range_fraction", "net_change_km",
        "temporary_exclusion_nearly_monotonic_decline",
    ]].copy()
    features = route.merge(accepted_meta, on="system_id", how="left")
    features = features.merge(shape_meta, on="system_id", how="left")
    features = features.merge(
        inventory[["system_id", "current_class", "mechanism_class", "closure_reason"]],
        on="system_id", how="left", suffixes=("", "_inventory"),
    )
    response_columns = [
        "system_id", "steady_volume_change_fraction_50y", "steady_final_free_downstream_m",
        "historical_minimum_timestep_seconds", "e_lia_total_formal", "e_rgi_total_formal",
        "e_lia_separation_formal", "e_rgi_separation_formal", "formal_objective_sum",
        "lia_pass", "rgi_pass", "peak_minus_target_m", "final_minus_target_m", "response_group",
    ]
    features = features.merge(responses[response_columns], on="system_id", how="left")

    features["lia_length_km"] = pd.to_numeric(features["lia_registered_length_m"]) / 1000.0
    features["rgi_length_km"] = pd.to_numeric(features["rgi_registered_length_m"]) / 1000.0
    features["target_retreat_km"] = features["lia_length_km"] - features["rgi_length_km"]
    features["target_retreat_fraction"] = features["target_retreat_km"] / features["lia_length_km"]
    features["target_direction"] = np.where(features["target_retreat_km"] > 0, "retreat", "advance")
    features["rgi_region"] = features["rgi_id"].astype(str).str.extract(r"RGI60-(\d+)\.")[0]
    features["accepted58"] = features["accepted_route"].notna()
    excluded = features["temporary_exclusion_nearly_monotonic_decline"].astype(str).str.lower().eq("true")
    features["shape_group"] = np.where(
        ~features["accepted58"], "remaining133", np.where(excluded, "excluded26", "retained32")
    )
    features["nonfrozen_retained_reference"] = (
        features["shape_group"].eq("retained32") & ~features["accepted_route"].eq("frozen-legacy")
    )

    if len(features) != 191 or features["accepted58"].sum() != 58:
        raise RuntimeError("population-ledger-mismatch")
    if features["shape_group"].eq("retained32").sum() != 32:
        raise RuntimeError("retained32-count-mismatch")
    if features["nonfrozen_retained_reference"].sum() != 10:
        raise RuntimeError("nonfrozen-retained-reference-count-mismatch")

    group_rows: list[dict[str, object]] = []
    for name, group in features.groupby("shape_group", sort=False):
        row: dict[str, object] = {
            "group": name,
            "count": len(group),
            "retreat_target_count": int(group["target_direction"].eq("retreat").sum()),
            "valid_single_polygon_count": int(group["outline_state"].eq("valid-single-polygon").sum()),
        }
        for column in ["lia_length_km", "rgi_length_km", "target_retreat_km", "target_retreat_fraction"]:
            med, q25, q75 = median_iqr(group[column])
            row[f"{column}_median"] = med
            row[f"{column}_q25"] = q25
            row[f"{column}_q75"] = q75
        group_rows.append(row)
    group_summary = pd.DataFrame(group_rows)

    comparison_rows: list[dict[str, object]] = []
    for scope, subset in [
        ("all_accepted_route_confounded", features[features["accepted58"]]),
        ("nonfrozen_only_route_reduced", features[features["accepted58"] & ~features["accepted_route"].eq("frozen-legacy")]),
    ]:
        retained = subset[subset["shape_group"].eq("retained32")]
        excluded_group = subset[subset["shape_group"].eq("excluded26")]
        for column in ["lia_length_km", "rgi_length_km", "target_retreat_km", "target_retreat_fraction"]:
            rmed, rq25, rq75 = median_iqr(retained[column])
            emed, eq25, eq75 = median_iqr(excluded_group[column])
            comparison_rows.append({
                "scope": scope,
                "feature": column,
                "retained_n": len(retained),
                "excluded_n": len(excluded_group),
                "retained_median": rmed,
                "retained_q25": rq25,
                "retained_q75": rq75,
                "excluded_median": emed,
                "excluded_q25": eq25,
                "excluded_q75": eq75,
                "median_difference_retained_minus_excluded": rmed - emed,
            })
    comparison = pd.DataFrame(comparison_rows)

    confounding = (
        features[features["accepted58"]]
        .groupby(["accepted_route", "shape_group"], dropna=False)
        .size().rename("count").reset_index()
    )
    route_totals = confounding.groupby("accepted_route")["count"].transform("sum")
    confounding["within_route_fraction"] = confounding["count"] / route_totals

    # Similarity uses only target-scale variables and a non-frozen retained reference,
    # so the score cannot simply learn the frozen-legacy label.
    features["log_lia_length"] = np.log(features["lia_length_km"].clip(lower=0.001))
    features["log_abs_separation"] = np.log(features["target_retreat_km"].abs().clip(lower=0.001))
    similarity_columns = ["log_lia_length", "target_retreat_fraction", "log_abs_separation"]
    centers_scales = {column: robust_scale(features[column]) for column in similarity_columns}
    z = np.column_stack([
        (features[column].to_numpy(float) - centers_scales[column][0]) / centers_scales[column][1]
        for column in similarity_columns
    ])
    reference_mask = features["nonfrozen_retained_reference"].to_numpy(bool)
    reference = z[reference_mask]
    distances = np.sqrt(((z[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    nearest_distance = distances.min(axis=1)
    features["nonfrozen_retained_similarity"] = np.exp(-0.5 * nearest_distance**2)

    geometry_score = features["outline_state"].map({
        "valid-single-polygon": 1.0,
        "non-polygon:MultiPolygon": 0.35,
        "invalid-polygon": 0.0,
    }).fillna(0.2)
    target_score = features["target_direction"].eq("retreat").astype(float)
    response_score = np.full(len(features), 0.40)
    has_response = features["response_group"].notna().to_numpy()
    objective = pd.to_numeric(features["formal_objective_sum"], errors="coerce").to_numpy()
    response_score[has_response] = np.clip(1.0 / (1.0 + objective[has_response]), 0.0, 1.0)
    lia_pass = features["lia_pass"].astype(str).str.lower().eq("true").to_numpy()
    rgi_pass = features["rgi_pass"].astype(str).str.lower().eq("true").to_numpy()
    response_score = np.maximum(response_score, np.where(lia_pass | rgi_pass, 0.80, 0.0))
    unreachable = features["response_group"].eq("initial-peak-lattice-unreachable").to_numpy()
    response_score[unreachable] = np.minimum(response_score[unreachable], 0.20)

    features["geometry_readiness_score"] = geometry_score
    features["target_compatibility_score"] = target_score
    features["existing_response_evidence_score"] = response_score
    features["new_mechanism_priority_score"] = (
        0.45 * features["nonfrozen_retained_similarity"]
        + 0.25 * features["geometry_readiness_score"]
        + 0.15 * features["target_compatibility_score"]
        + 0.15 * features["existing_response_evidence_score"]
    )

    remaining = features[~features["accepted58"]].copy()
    remaining["comparable_to_current_success_family"] = (
        remaining["target_direction"].eq("retreat")
        & remaining["outline_state"].eq("valid-single-polygon")
    )
    remaining["shortlist_eligible"] = (
        remaining["comparable_to_current_success_family"]
        & ~remaining["response_group"].eq("technical-floor-exhausted-not-retryable")
    )
    remaining = remaining.sort_values(
        ["comparable_to_current_success_family", "new_mechanism_priority_score", "system_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    remaining["new_mechanism_priority_rank"] = np.arange(1, len(remaining) + 1)
    remaining["ranking_interpretation"] = (
        "read-only research priority; current mechanism is closed; not a success probability"
    )

    output_columns = [
        "system_id", "rgi_id", "rgi_region", "shape_group", "accepted_route", "outline_state",
        "lia_length_km", "rgi_length_km", "target_retreat_km", "target_retreat_fraction",
        "target_direction", "synthetic_route_attempted", "current_route", "current_gate",
        "current_class", "mechanism_class", "response_group", "formal_objective_sum",
        "lia_pass", "rgi_pass", "steady_volume_change_fraction_50y",
        "historical_minimum_timestep_seconds", "nonfrozen_retained_similarity",
        "geometry_readiness_score", "target_compatibility_score", "existing_response_evidence_score",
        "new_mechanism_priority_score", "comparable_to_current_success_family",
        "shortlist_eligible", "new_mechanism_priority_rank", "ranking_interpretation",
    ]
    features.to_csv(OUT / "system_feature_table_191.csv", index=False)
    group_summary.to_csv(OUT / "group_summary.csv", index=False)
    comparison.to_csv(OUT / "retained32_vs_excluded26_feature_comparison.csv", index=False)
    confounding.to_csv(OUT / "route_shape_confounding.csv", index=False)
    remaining[output_columns].to_csv(OUT / "remaining133_new_mechanism_priority.csv", index=False)
    shortlist = remaining[remaining["shortlist_eligible"]].head(20)
    shortlist[output_columns].to_csv(OUT / "top20_new_mechanism_shortlist.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    colors = {"retained32": "#167C80", "excluded26": "#E76F51", "remaining133": "#4D6A9F"}
    ordered = ["retained32", "excluded26", "remaining133"]
    data = [features.loc[features["shape_group"].eq(group), "target_retreat_fraction"] * 100 for group in ordered]
    parts = axes[0, 0].violinplot(data, showmedians=True, showextrema=False)
    for body, group in zip(parts["bodies"], ordered):
        body.set_facecolor(colors[group]); body.set_alpha(0.72)
    axes[0, 0].set_xticks(range(1, 4), ["retained 32", "excluded 26", "remaining 133"])
    axes[0, 0].axhline(0, color="#444444", lw=0.8)
    axes[0, 0].set_ylabel("Target retreat fraction (%)")
    axes[0, 0].set_title("Endpoint target geometry")

    data = [features.loc[features["shape_group"].eq(group), "lia_length_km"] for group in ordered]
    axes[0, 1].boxplot(data, labels=["retained 32", "excluded 26", "remaining 133"], showfliers=False)
    axes[0, 1].set_ylabel("LIA target length (km)")
    axes[0, 1].set_title("Glacier target scale")

    pivot = confounding.pivot(index="accepted_route", columns="shape_group", values="count").fillna(0)
    pivot = pivot.reindex(columns=["retained32", "excluded26"], fill_value=0)
    x = np.arange(len(pivot))
    axes[1, 0].bar(x, pivot["retained32"], color=colors["retained32"], label="retained")
    axes[1, 0].bar(x, pivot["excluded26"], bottom=pivot["retained32"], color=colors["excluded26"], label="excluded")
    axes[1, 0].set_xticks(x, [label.replace("-", "\n") for label in pivot.index], fontsize=8)
    axes[1, 0].set_ylabel("Accepted curves")
    axes[1, 0].set_title("Shape label is confounded by route")
    axes[1, 0].legend(frameon=False)

    plot_short = shortlist.sort_values("new_mechanism_priority_score").tail(20)
    axes[1, 1].barh(plot_short["system_id"].astype(str), plot_short["new_mechanism_priority_score"], color="#2A9D8F")
    axes[1, 1].set_xlabel("Research-priority score")
    axes[1, 1].set_title("Top comparable remaining systems")
    axes[1, 1].set_xlim(0, 1)
    fig.suptitle("Why the retained curves succeeded, and where extension is most defensible", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "success_factor_extension_diagnostics.png", dpi=180)
    plt.close(fig)

    retained_nonfrozen = features[features["nonfrozen_retained_reference"]]
    accepted_retreat = int(features.loc[features["accepted58"], "target_direction"].eq("retreat").sum())
    remaining_advance = int(remaining["target_direction"].eq("advance").sum())
    comparable_count = int(remaining["comparable_to_current_success_family"].sum())
    summary = {
        "schema": "lia-191-retained32-success-factor-extension-v1",
        "status": "read-only-success-factor-analysis-complete",
        "population": 191,
        "accepted": 58,
        "retained_shape": 32,
        "excluded_nearly_monotonic": 26,
        "remaining": 133,
        "retained_nonfrozen_reference_count": int(len(retained_nonfrozen)),
        "accepted_retreat_target_count": accepted_retreat,
        "remaining_advance_target_count": remaining_advance,
        "remaining_valid_retreat_comparable_count": comparable_count,
        "route_confounding": {
            "frozen_legacy_retained": 22,
            "frozen_legacy_excluded": 0,
            "nonfrozen_retained": 10,
            "nonfrozen_excluded": 26,
            "conclusion": "curve-shape success cannot be learned from the 32 label without controlling route",
        },
        "principal_findings": [
            "All 58 accepted systems have retreat-direction endpoint targets; 17 remaining systems require advance and are outside the demonstrated success family.",
            "Thirty-one of the retained 32 have valid single-polygon outlines, but valid outlines are also common among failures and are necessary rather than sufficient.",
            "All 22 frozen-legacy curves are retained, so route construction is the strongest observed shape-label confounder.",
            "The defensible extension pool is the 94 remaining systems with valid single polygons and retreat-direction targets, prioritized by similarity to the 10 non-frozen retained curves and existing endpoint evidence.",
            "All 133 current registered mechanisms are closed or unreachable; the ranking is for a genuinely new mechanism, not permission to rerun old parameter families.",
        ],
        "top20_system_ids": [int(value) for value in shortlist["system_id"]],
        "ranking_definition": {
            "nonfrozen_retained_similarity_weight": 0.45,
            "geometry_readiness_weight": 0.25,
            "target_compatibility_weight": 0.15,
            "existing_response_evidence_weight": 0.15,
            "interpretation": "research priority only; not a calibrated success probability",
        },
        "model_run_performed": False,
        "old_output_modified": False,
        "controller_state_modified": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
