"""Save trend-level diagnostics from the completed increment attribution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def trend_per_century(years: pd.Series, values: pd.Series) -> float:
    x = years.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    return float(100.0 * np.polyfit(x - x.mean(), y, 1)[0])


def main() -> int:
    output = project_root() / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    annual = pd.read_csv(output / "annual_proxy_contributions_1901_1920.csv")
    rows = []
    for proxy_index, group in annual.groupby("proxy_index"):
        first = group.iloc[0]
        innovation_trend = trend_per_century(group["year"], group["innovation"])
        contribution_trend = trend_per_century(
            group["year"], group["hma_increment_contribution_mm_day"]
        )
        mean_gain = float(group["hma_gain_weight"].mean())
        rows.append(
            {
                "proxy_index": int(proxy_index),
                "contribution_trend_mm_day_per_century": contribution_trend,
                "innovation_trend_proxy_units_per_century": innovation_trend,
                "mean_hma_gain_weight": mean_gain,
                "gain_times_innovation_trend": mean_gain * innovation_trend,
                "latitude": float(first["latitude"]),
                "longitude": float(first["longitude"]),
                "proxy_type": first["proxy_type"],
                "psm_method": first["psm_method"],
                "psm_correlation": float(first["psm_correlation"]),
                "psm_pvalue": float(first["psm_pvalue"]),
            }
        )
    trends = pd.DataFrame(rows)
    trends["absolute_contribution_trend"] = trends[
        "contribution_trend_mm_day_per_century"
    ].abs()
    site_counts = trends.groupby(["latitude", "longitude"]).size().rename("site_record_count")
    trends = trends.join(site_counts, on=["latitude", "longitude"])
    trends["psm_significance"] = np.where(
        trends["psm_pvalue"] <= 0.05, "p<=0.05", "p>0.05"
    )
    trends["site_class"] = np.where(
        trends["site_record_count"] > 1, "duplicate_site", "singleton"
    )
    trends["longitude_band"] = pd.cut(
        trends["longitude"],
        [0, 80, 90, 100, 200],
        labels=["<80E", "80-90E", "90-100E", ">=100E"],
    )
    trends = trends.sort_values("absolute_contribution_trend", ascending=False)
    trends.to_csv(output / "proxy_trend_contribution_1901_1920.csv", index=False)

    grouping_rows = []
    for grouping in ["psm_significance", "site_class", "longitude_band"]:
        for label, group in trends.groupby(grouping, observed=True):
            grouping_rows.append(
                {
                    "grouping": grouping,
                    "label": str(label),
                    "proxy_count": int(len(group)),
                    "signed_contribution_trend_mm_day_per_century": float(
                        group["contribution_trend_mm_day_per_century"].sum()
                    ),
                    "absolute_contribution_trend_mm_day_per_century": float(
                        group["absolute_contribution_trend"].sum()
                    ),
                }
            )
    grouping_table = pd.DataFrame(grouping_rows)
    grouping_table.to_csv(output / "trend_contribution_groups_1901_1920.csv", index=False)

    total_trend = float(trends["contribution_trend_mm_day_per_century"].sum())
    absolute_total = float(trends["absolute_contribution_trend"].sum())
    top5_fraction = float(trends.head(5)["absolute_contribution_trend"].sum() / absolute_total)
    top10_fraction = float(trends.head(10)["absolute_contribution_trend"].sum() / absolute_total)
    weak = trends["psm_pvalue"] > 0.05
    duplicate = trends["site_record_count"] > 1
    negative_innovation = trends["innovation_trend_proxy_units_per_century"] < 0
    positive_gain = trends["mean_hma_gain_weight"] > 0
    dominant_quadrant = negative_innovation & positive_gain

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        trends["innovation_trend_proxy_units_per_century"],
        trends["mean_hma_gain_weight"],
        c=trends["contribution_trend_mm_day_per_century"],
        cmap="RdBu_r",
        vmin=-float(trends["absolute_contribution_trend"].max()),
        vmax=float(trends["absolute_contribution_trend"].max()),
        s=28,
        alpha=0.8,
    )
    axes[0].axhline(0.0, color="#444444", lw=0.8)
    axes[0].axvline(0.0, color="#444444", lw=0.8)
    axes[0].set_xlabel("Innovation trend (proxy units/century)")
    axes[0].set_ylabel("Mean HMA Kalman gain weight")
    axes[0].set_title("Trend contribution = innovation trend x gain")
    axes[0].grid(alpha=0.2)
    figure.colorbar(
        scatter,
        ax=axes[0],
        label="HMA contribution trend (mm/day/century)",
    )
    ranked = trends.head(20).sort_values(
        "contribution_trend_mm_day_per_century"
    )
    colors = np.where(
        ranked["contribution_trend_mm_day_per_century"] < 0,
        "#2166ac",
        "#b2182b",
    )
    axes[1].barh(
        [f"#{value}" for value in ranked["proxy_index"]],
        ranked["contribution_trend_mm_day_per_century"],
        color=colors,
    )
    axes[1].axvline(0.0, color="#444444", lw=0.8)
    axes[1].set_xlabel("Contribution to 1901-1920 HMA trend (mm/day/century)")
    axes[1].set_title("Largest absolute trend contributions")
    axes[1].grid(axis="x", alpha=0.2)
    figure.savefig(output / "figure_trend_attribution.png", dpi=220)
    plt.close(figure)

    summary = {
        "date": "2026-08-13",
        "years": [1901, 1920],
        "locked_final_period_accessed": False,
        "proxy_trend_sum_mm_day_per_century": total_trend,
        "top5_absolute_trend_fraction": top5_fraction,
        "top10_absolute_trend_fraction": top10_fraction,
        "negative_innovation_proxy_count": int(negative_innovation.sum()),
        "positive_innovation_proxy_count": int((~negative_innovation).sum()),
        "negative_innovation_positive_gain_count": int(dominant_quadrant.sum()),
        "negative_innovation_positive_gain_signed_trend_mm_day_per_century": float(
            trends.loc[
                dominant_quadrant, "contribution_trend_mm_day_per_century"
            ].sum()
        ),
        "absolute_contribution_vs_absolute_innovation_trend_correlation": float(
            trends["absolute_contribution_trend"].corr(
                trends["innovation_trend_proxy_units_per_century"].abs()
            )
        ),
        "absolute_contribution_vs_absolute_gain_correlation": float(
            trends["absolute_contribution_trend"].corr(
                trends["mean_hma_gain_weight"].abs()
            )
        ),
        "weak_psm_signed_trend_fraction": float(
            trends.loc[weak, "contribution_trend_mm_day_per_century"].sum()
            / total_trend
        ),
        "duplicate_site_signed_trend_fraction": float(
            trends.loc[duplicate, "contribution_trend_mm_day_per_century"].sum()
            / total_trend
        ),
        "decision": (
            "Do not target a small proxy blacklist. The excessive decline is a "
            "distributed projection of widespread negative proxy innovations through "
            "the frozen prior-to-HMA covariance. The next controlled experiment must "
            "separate PSM/innovation chronology from covariance mapping."
        ),
    }
    (output / "trend_attribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
