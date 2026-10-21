"""Correct the regional interpretation of the fixed-support longitude audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v2"
YEARS = np.arange(1901, 1921, dtype=int)
REGIONS = ("west_lt80", "central_80_90", "east_90_100", "far_east_ge100")
PRODUCTS = ("cru", "gpcc", "rec_result")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    source_root = result_root / "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v1"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "v1_scores": source_root / "longitude_support_scores.csv",
        "v1_curves": source_root / "longitude_support_hma_curves_1901_1920.csv",
        "v1_summary": source_root / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}
    scores = pd.read_csv(sources["v1_scores"])
    curves = pd.read_csv(sources["v1_curves"])
    v1 = json.loads(sources["v1_summary"].read_text(encoding="utf-8"))
    if v1["locked_1921_1939_accessed"] or not np.array_equal(curves["year"].to_numpy(dtype=int), YEARS):
        raise RuntimeError("v1-period-contract-failed")

    eligible = scores[(scores["support_level"] == "dual_any") & (scores["gridcell_count"] >= 5)].copy()
    eligible["rec_result_is_negative"] = eligible["rec_result_trend_mm_day_per_century"] < 0
    eligible["rec_result_more_negative_than_both_references"] = (
        eligible["rec_result_trend_mm_day_per_century"]
        < eligible[["cru_trend_mm_day_per_century", "gpcc_trend_mm_day_per_century"]].min(axis=1)
    )
    all_more_negative = bool(eligible["rec_result_more_negative_than_both_references"].all())
    negative_region_count = int(eligible["rec_result_is_negative"].sum())
    west_dual_any = int(eligible.loc[eligible["longitude_region"].eq("west_lt80"), "gridcell_count"].iloc[0])
    total_dual_any = int(eligible["gridcell_count"].sum())
    dual_high = scores[scores["support_level"].eq("dual_high")]
    west_dual_high = int(dual_high.loc[dual_high["longitude_region"].eq("west_lt80"), "gridcell_count"].iloc[0])
    total_dual_high = int(dual_high["gridcell_count"].sum())
    classification = (
        "reconstruction_trend_more_negative_than_both_references_in_all_supported_bands"
        if all_more_negative
        else "regional_reference_conflict_not_consistent"
    )

    output.mkdir(parents=True)
    eligible.to_csv(output / "eligible_supported_longitude_bands.csv", index=False)
    scores.to_csv(output / "longitude_support_scores_v1_reproduced.csv", index=False)
    curves.to_csv(output / "longitude_support_curves_v1_reproduced.csv", index=False)

    colors = {"cru": "#6A3D9A", "gpcc": "#777777", "rec_result": "#D55E00"}
    labels = {"cru": "CRU", "gpcc": "GPCC", "rec_result": "rec_result"}
    titles = {"west_lt80": "<80E", "central_80_90": "80-90E", "east_90_100": "90-100E", "far_east_ge100": ">=100E"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.reshape(-1), REGIONS):
        row = scores[(scores["longitude_region"] == region) & (scores["support_level"] == "dual_any")].iloc[0]
        count = int(row["gridcell_count"])
        for product in PRODUCTS:
            all_values = curves[f"{region}_all_joint_{product}_mm_day"]
            axis.plot(YEARS, all_values, color=colors[product], linewidth=1.0, linestyle="--", alpha=0.4)
            if count:
                axis.plot(YEARS, curves[f"{region}_dual_any_{product}_mm_day"], color=colors[product], linewidth=2.0 if product == "rec_result" else 1.6, label=labels[product])
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.set_title(f"{titles[region]}: dual-any n={count}")
        axis.set_ylabel("Precipitation anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=3, title="Solid: dual-any; dashed: all joint")
    fig.savefig(output / "fixed_support_longitude_curves_corrected_interpretation.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "v1_numerical_results_reproduced_without_change": True,
        "correction": "V1 used a misleading classification label. Central and eastern rec_result trends are positive, not negative; they are nevertheless more negative than both reference trends.",
        "eligible_supported_regions": eligible.to_dict(orient="records"),
        "eligible_region_count": int(len(eligible)),
        "regions_with_negative_rec_result_trend": negative_region_count,
        "regions_with_rec_result_more_negative_than_both_references": int(eligible["rec_result_more_negative_than_both_references"].sum()),
        "west_share_of_dual_any_cells": west_dual_any / total_dual_any,
        "west_share_of_dual_high_cells": west_dual_high / total_dual_high,
        "classification": classification,
        "interpretation": "The negative aggregate supported-cell trend is west-dominated. The cross-region result is an amplitude/relative-trend conflict: rec_result is more negative than both references in every supported band, but only the western band has a negative rec_result trend.",
        "interpretation_boundary": "Eastern samples are small and >=100E has no dual support. This does not validate either reference product or authorize a production correction.",
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-v2-summary")
    lines = [
        "# Fixed-support Longitude Audit v2",
        "",
        "V1 numerical values are unchanged; only the classification is corrected.",
        "",
    ]
    for _, row in eligible.iterrows():
        lines.append(
            f"- {row['longitude_region']} n={int(row['gridcell_count'])}: CRU/GPCC/rec_result "
            f"`{row['cru_trend_mm_day_per_century']:+.3f}` / `{row['gpcc_trend_mm_day_per_century']:+.3f}` / "
            f"`{row['rec_result_trend_mm_day_per_century']:+.3f} mm/day/century`."
        )
    lines.extend([
        "",
        f"West contains `{west_dual_any}/{total_dual_any}` dual-any and `{west_dual_high}/{total_dual_high}` dual-high cells.",
        f"Classification: `{classification}`.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
