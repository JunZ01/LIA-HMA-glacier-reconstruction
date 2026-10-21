"""Analyze the frozen central/eastern precipitation operator attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = ROOT / "climate_reconstruction_precipitation_regional_operator_attribution_v1"
MAT = OUTPUT / "regional_operator_decomposition.mat"
METADATA = ROOT / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv"
SOURCE_MAP = ROOT / "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_mapping.csv"
HMA_TRENDS = ROOT / "climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1/operator_decomposition_trends.csv"
MATLAB_DRIVER = PROJECT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/decompose_precipitation_significant_psm_operator_regional_v2.m"
DRIVER = Path(__file__).resolve()

REGIONS = ["hma", "west_lt80", "central_80_90", "east_ge90"]
BRANCHES = ["baseline", "combined_rebuild", "temperature_only", "precipitation_only", "zero_selected_anomaly"]
YEARS = np.arange(1901, 1921)


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(np.polyfit(YEARS[finite], values[finite], 1)[0] * 100.0) if finite.sum() >= 3 else np.nan


def concentration(values: np.ndarray) -> dict[str, float | int]:
    negative = np.sort(np.abs(values[np.isfinite(values) & (values < 0)]))[::-1]
    if not len(negative):
        return {"negative_count": 0, "top5_negative_share": np.nan, "top10_negative_share": np.nan, "effective_negative_count": np.nan}
    weights = negative / negative.sum()
    return {"negative_count": int(len(negative)), "top5_negative_share": float(weights[:5].sum()), "top10_negative_share": float(weights[:10].sum()), "effective_negative_count": float(1.0 / np.sum(weights**2))}


def main() -> int:
    sources = {"regional_mat": MAT, "metadata": METADATA, "source_map": SOURCE_MAP, "hma_trends": HMA_TRENDS, "matlab_driver": MATLAB_DRIVER, "analysis_driver": DRIVER}
    for path in sources.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (OUTPUT / "summary.json").exists(): raise FileExistsError(OUTPUT / "summary.json")
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text("\n".join(["# Regional Precipitation Operator Attribution Contract", "", "- Frozen inputs: prior, 500 members, proxy observations, residual covariance, 65 significant bivariate operators, and all nonselected operators.", "- Targets: full HMA, west <80E, central 80-90E, and east >=90E on the native 2-degree prior grid.", "- Report exact algebraic closure, individual proxy trends, b1-temperature and b2-precipitation pathway effects, proxy-location groups, and b2-sign groups.", "- Call a negative contribution concentrated only when top five records explain at least 50% or the effective negative contributor count is at most 10.", "- Do not tune, delete proxies, alter PSMs, run production EnKF, access 1921-1939, or run OGGM."]) + "\n", encoding="utf-8")
    sources["contract"] = contract
    before = {name: state(path) for name, path in sources.items()}

    with h5py.File(MAT, "r") as handle:
        total = np.asarray(handle["branch_total_increment"], dtype=float).transpose(2, 1, 0)
        selected_total = np.asarray(handle["branch_selected_contribution"], dtype=float).transpose(2, 1, 0)
        nonselected_total = np.asarray(handle["branch_nonselected_contribution"], dtype=float).transpose(2, 1, 0)
        contribution = np.asarray(handle["branch_contribution"], dtype=float).transpose(3, 2, 1, 0)
        closure = np.asarray(handle["branch_closure_error"], dtype=float)
        selected = np.asarray(handle["selected"], dtype=int).ravel() - 1
        combined_closure = float(np.asarray(handle["combined_closure"]).squeeze())
        hma_baseline_closure = float(np.asarray(handle["hma_baseline_closure"]).squeeze())
        innovation_invariance = float(np.asarray(handle["innovation_invariance"]).squeeze())
        region_cell_counts = np.asarray(handle["region_cell_counts"], dtype=int).ravel()
    if total.shape != (20, 5, 4) or contribution.shape != (202, 20, 5, 4): raise RuntimeError("regional-array-shape-failed")
    max_closure = float(np.max(np.abs(closure)))
    if max(combined_closure, hma_baseline_closure, innovation_invariance, max_closure) > 1e-12: raise RuntimeError("regional-closure-failed")

    metadata = pd.read_csv(METADATA)
    source_map = pd.read_csv(SOURCE_MAP)[["proxy_index", "source_family", "data_name", "archive"]]
    metadata = metadata.merge(source_map, on="proxy_index", how="left", validate="one_to_one")
    metadata["source_family"] = metadata.source_family.fillna("unmapped_non_hma_network")
    metadata["data_name"] = metadata.data_name.fillna("unknown")
    metadata["archive"] = metadata.archive.fillna("unknown")
    metadata["selected_significant_bivariate"] = False
    metadata.loc[selected, "selected_significant_bivariate"] = True
    metadata["proxy_location_region"] = np.select([metadata.longitude < 80, (metadata.longitude >= 80) & (metadata.longitude < 90), metadata.longitude >= 90], ["west_lt80", "central_80_90", "east_ge90"], default="unknown")
    metadata["b2_sign"] = np.where(metadata.coefficient_b2 >= 0, "b2_nonnegative", "b2_negative")
    metadata["documented_seasonality"] = "not_stored_in_frozen_proxy_data"

    annual_rows = []
    trend_rows = []
    region_rows = []
    proxy_tables = []
    for rr, region in enumerate(REGIONS):
        branch_trends = {branch: slope(total[:, bb, rr]) for bb, branch in enumerate(BRANCHES)}
        selected_trends = {branch: slope(selected_total[:, bb, rr]) for bb, branch in enumerate(BRANCHES)}
        nonselected_trends = {branch: slope(nonselected_total[:, bb, rr]) for bb, branch in enumerate(BRANCHES)}
        total_effect = branch_trends["baseline"] - branch_trends["zero_selected_anomaly"]
        temperature_effect = branch_trends["temperature_only"] - branch_trends["zero_selected_anomaly"]
        precipitation_effect = branch_trends["precipitation_only"] - branch_trends["zero_selected_anomaly"]
        nonadditivity = total_effect - temperature_effect - precipitation_effect
        for yy, year in enumerate(YEARS):
            for bb, branch in enumerate(BRANCHES):
                annual_rows.append({"year": int(year), "region": region, "branch": branch, "total_increment_mm_day": total[yy, bb, rr], "selected_contribution_mm_day": selected_total[yy, bb, rr], "nonselected_contribution_mm_day": nonselected_total[yy, bb, rr]})
        for branch in BRANCHES:
            trend_rows.append({"region": region, "series": f"increment_{branch}", "trend_mm_day_per_century": branch_trends[branch]})
            trend_rows.append({"region": region, "series": f"selected_contribution_{branch}", "trend_mm_day_per_century": selected_trends[branch]})
            trend_rows.append({"region": region, "series": f"nonselected_contribution_{branch}", "trend_mm_day_per_century": nonselected_trends[branch]})
        for name, value in (("effect_total_selected_operator_anomaly", total_effect), ("effect_temperature_operator_anomaly", temperature_effect), ("effect_precipitation_operator_anomaly", precipitation_effect), ("effect_nonadditivity", nonadditivity)):
            trend_rows.append({"region": region, "series": name, "trend_mm_day_per_century": value})

        local = metadata.copy()
        for bb, branch in enumerate(BRANCHES):
            local[f"trend_{branch}_mm_day_per_century"] = [slope(contribution[index, :, bb, rr]) for index in range(len(metadata))]
        local["effect_total_selected_operator_anomaly"] = local.trend_baseline_mm_day_per_century - local.trend_zero_selected_anomaly_mm_day_per_century
        local["effect_temperature_operator_anomaly"] = local.trend_temperature_only_mm_day_per_century - local.trend_zero_selected_anomaly_mm_day_per_century
        local["effect_precipitation_operator_anomaly"] = local.trend_precipitation_only_mm_day_per_century - local.trend_zero_selected_anomaly_mm_day_per_century
        local["effect_nonadditivity"] = local.effect_total_selected_operator_anomaly - local.effect_temperature_operator_anomaly - local.effect_precipitation_operator_anomaly
        local["target_region"] = region
        local["absolute_baseline_trend"] = local.trend_baseline_mm_day_per_century.abs()
        proxy_tables.append(local)
        selected_values = local.loc[local.selected_significant_bivariate, "trend_baseline_mm_day_per_century"].to_numpy(float)
        concentration_metrics = concentration(selected_values)
        selected_effect_fraction = abs(total_effect) / abs(branch_trends["baseline"]) if branch_trends["baseline"] != 0 else np.nan
        precipitation_dominates = abs(precipitation_effect) > abs(temperature_effect)
        concentrated = bool(concentration_metrics["top5_negative_share"] >= 0.5 or concentration_metrics["effective_negative_count"] <= 10)
        region_rows.append({"region": region, "native_cell_count": int(region_cell_counts[rr]), "baseline_trend_mm_day_per_century": branch_trends["baseline"], "selected_baseline_contribution_trend": selected_trends["baseline"], "nonselected_baseline_contribution_trend": nonselected_trends["baseline"], "selected_operator_total_effect": total_effect, "temperature_operator_effect": temperature_effect, "precipitation_operator_effect": precipitation_effect, "nonadditivity": nonadditivity, "selected_operator_effect_fraction_of_baseline_abs": selected_effect_fraction, "precipitation_pathway_dominates_temperature": precipitation_dominates, **concentration_metrics, "negative_selected_contribution_concentrated": concentrated})

    annual = pd.DataFrame(annual_rows); trends = pd.DataFrame(trend_rows); proxy = pd.concat(proxy_tables, ignore_index=True); regional = pd.DataFrame(region_rows)
    annual.to_csv(OUTPUT / "regional_operator_annual_curves.csv", index=False)
    trends.to_csv(OUTPUT / "regional_operator_trends.csv", index=False)
    proxy.to_csv(OUTPUT / "regional_proxy_trend_attribution.csv", index=False)
    regional.to_csv(OUTPUT / "regional_mechanism_summary.csv", index=False)

    selected_proxy = proxy[proxy.selected_significant_bivariate].copy()
    groups = []
    for (target_region, location_region, b2_sign), group in selected_proxy.groupby(["target_region", "proxy_location_region", "b2_sign"]):
        groups.append({"target_region": target_region, "proxy_location_region": location_region, "b2_sign": b2_sign, "proxy_count": len(group), "baseline_trend_sum": group.trend_baseline_mm_day_per_century.sum(), "total_operator_effect_sum": group.effect_total_selected_operator_anomaly.sum(), "temperature_effect_sum": group.effect_temperature_operator_anomaly.sum(), "precipitation_effect_sum": group.effect_precipitation_operator_anomaly.sum(), "nonadditivity_sum": group.effect_nonadditivity.sum()})
    group_table = pd.DataFrame(groups)
    group_table.to_csv(OUTPUT / "selected_proxy_location_b2_groups.csv", index=False)
    top = proxy[proxy.target_region.isin(["central_80_90", "east_ge90"])].sort_values(["target_region", "trend_baseline_mm_day_per_century"]).groupby("target_region", as_index=False, group_keys=False).head(25)
    top.to_csv(OUTPUT / "central_east_top25_negative_proxy_trends.csv", index=False)

    hma_reference = pd.read_csv(HMA_TRENDS).set_index("series").trend_mm_day_per_century
    hma_checks = {
        "baseline_trend_error": abs(regional.loc[regional.region == "hma", "baseline_trend_mm_day_per_century"].iloc[0] - hma_reference["increment_baseline"]),
        "total_effect_trend_error": abs(regional.loc[regional.region == "hma", "selected_operator_total_effect"].iloc[0] - hma_reference["effect_total_selected_operator_anomaly"]),
        "temperature_effect_trend_error": abs(regional.loc[regional.region == "hma", "temperature_operator_effect"].iloc[0] - hma_reference["effect_temperature_operator_anomaly"]),
        "precipitation_effect_trend_error": abs(regional.loc[regional.region == "hma", "precipitation_operator_effect"].iloc[0] - hma_reference["effect_precipitation_operator_anomaly"]),
    }
    if max(hma_checks.values()) > 1e-12: raise RuntimeError("hma-trend-reference-closure-failed")

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    colors = {"baseline": "#111111", "temperature_only": "#009E73", "precipitation_only": "#D55E00", "zero_selected_anomaly": "#7F7F7F"}
    for col, region in enumerate(["central_80_90", "east_ge90"]):
        ax = axes[0, col]
        for branch in colors:
            data = annual[(annual.region == region) & (annual.branch == branch)]
            ax.plot(data.year, data.total_increment_mm_day, color=colors[branch], marker="o", markersize=3, linewidth=2 if branch == "baseline" else 1.4, label=branch)
        ax.axhline(0, color="#888888", linewidth=0.7); ax.set_title(f"{region}: branch increments"); ax.set_ylabel("Increment (mm/day)")
        ax.legend(frameon=False, fontsize=8)
        effects = regional[regional.region == region].iloc[0]
        labels = ["total", "temperature", "precipitation", "nonadditivity"]
        values = [effects.selected_operator_total_effect, effects.temperature_operator_effect, effects.precipitation_operator_effect, effects.nonadditivity]
        axes[1, col].bar(labels, values, color=["#0072B2", "#009E73", "#D55E00", "#CC79A7"]); axes[1, col].axhline(0, color="#333333", linewidth=0.8); axes[1, col].set_ylabel("Trend effect (mm/day/century)"); axes[1, col].set_title(f"{region}: operator pathways")
        ranked = proxy[proxy.target_region == region].nsmallest(15, "trend_baseline_mm_day_per_century").sort_values("trend_baseline_mm_day_per_century")
        bar_colors = np.where(ranked.selected_significant_bivariate, "#D55E00", "#777777")
        axes[2, col].barh(ranked.proxy_index.astype(str), ranked.trend_baseline_mm_day_per_century, color=bar_colors); axes[2, col].axvline(0, color="#333333", linewidth=0.8); axes[2, col].set_xlabel("Proxy trend contribution (mm/day/century)"); axes[2, col].set_title(f"{region}: 15 most negative records")
    fig.suptitle("1901-1920 central/eastern precipitation trend attribution", fontsize=15)
    fig.savefig(OUTPUT / "central_east_regional_operator_attribution.png", dpi=220); plt.close(fig)

    decisions = {}
    for region in ("central_80_90", "east_ge90"):
        row = regional[regional.region == region].iloc[0]
        if not row.precipitation_pathway_dominates_temperature:
            route = "temperature_or_nonlinear_pathway_review"
        elif row.negative_selected_contribution_concentrated:
            route = "targeted_top_proxy_source_and_leave_group_out_sensitivity"
        else:
            route = "broad_seasonal_psm_structure_rebuild"
        decisions[region] = route
    summary = {"status": "complete", "period": [1901, 1920], "locked_1921_1939_accessed": False, "selected_proxy_count": int(len(selected)), "validation": {"combined_closure": combined_closure, "hma_baseline_closure": hma_baseline_closure, "innovation_invariance": innovation_invariance, "branch_increment_closure": max_closure, "hma_reference_trend_errors": hma_checks}, "regional_metrics": regional.to_dict(orient="records"), "routes": decisions}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Central/Eastern Precipitation Operator Attribution", "", "All regional increments close algebraically and the full-HMA result exactly reproduces the archived decomposition.", ""]
    for region in ("central_80_90", "east_ge90"):
        row = regional[regional.region == region].iloc[0]
        lines.append(f"- {region}: baseline={row.baseline_trend_mm_day_per_century:.3f}, selected total effect={row.selected_operator_total_effect:.3f}, temperature={row.temperature_operator_effect:.3f}, precipitation={row.precipitation_operator_effect:.3f}, nonadditivity={row.nonadditivity:.3f} mm/day/century; top-five negative share={row.top5_negative_share:.3f}, effective negative contributors={row.effective_negative_count:.1f}; route={decisions[region]}.")
    lines.extend(["", "Seasonality is not stored in the frozen proxy operator metadata, so a broad result routes to a seasonal-PSM rebuild rather than post-hoc record deletion."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in sources.items()}; manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "locked_1921_1939_accessed": False}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]: raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
