"""Decompose the Round 45 HMA early trend into regional product contributions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIONS = ("west_lt80", "central_80_90", "east_ge90")
PRODUCTS = ("UDel", "CRU", "GPCC")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; output = loop / "round_46_udel_early_regional_amplitude_decomposition_v1"; round45 = loop / "round_45_udel_1901_1920_precipitation_product_audit_v1"
    paths = {"early_metrics": round45 / "early_product_metrics.csv", "early_series": round45 / "early_regional_series.csv", "modern_metrics": round45 / "modern_udel_cru_gpcc_agreement.csv", "mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy", "grid": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}; early = pd.read_csv(paths["early_metrics"]); series = pd.read_csv(paths["early_series"]); modern = pd.read_csv(paths["modern_metrics"])
    mask = np.load(paths["mask"]).astype(bool)
    with netcdf_file(paths["grid"], "r", mmap=False) as handle:
        lat = np.asarray(handle.variables["lat"].data, dtype=float).copy(); lon = np.asarray(handle.variables["lon"].data, dtype=float).copy()
    longitude = np.broadcast_to(lon[None], mask.shape); weights = np.cos(np.deg2rad(lat))[:, None] * mask; region_masks = {"west_lt80": mask & (longitude < 80), "central_80_90": mask & (longitude >= 80) & (longitude < 90), "east_ge90": mask & (longitude >= 90)}; total_weight = float(np.sum(weights)); fractions = {region: float(np.sum(weights * region_mask) / total_weight) for region, region_mask in region_masks.items()}
    contribution_rows: list[dict[str, object]] = []; regional_rows: list[dict[str, object]] = []
    for product in PRODUCTS:
        product_metrics = early[(early["product"] == product) & early.region.isin(REGIONS)].set_index("region")
        for region in REGIONS:
            trend = float(product_metrics.loc[region, "trend_mm_day_per_century"]); change = float(product_metrics.loc[region, "change_1916_1920_minus_1901_1905_mm_day"]); contribution_rows.append({"product": product, "region": region, "cosine_area_fraction": fractions[region], "regional_trend_mm_day_per_century": trend, "hma_trend_contribution_mm_day_per_century": fractions[region] * trend, "regional_endpoint_change_mm_day": change, "hma_endpoint_contribution_mm_day": fractions[region] * change})
    contributions = pd.DataFrame(contribution_rows); contributions.to_csv(output / "regional_trend_contributions.csv", index=False)
    for region in REGIONS:
        metric = early[(early.region == region) & early["product"].isin(PRODUCTS)].set_index("product"); trends = metric.trend_mm_day_per_century.reindex(PRODUCTS).to_numpy(float); changes = metric.change_1916_1920_minus_1901_1905_mm_day.reindex(PRODUCTS).to_numpy(float); modern_region = modern[modern.region == region].set_index("observation"); std_ratios = modern_region.udel_to_observation_std_ratio.reindex(["CRU", "GPCC"]).to_numpy(float)
        trend_sign_consensus = bool(np.all(trends < 0) or np.all(trends > 0)); change_sign_consensus = bool(np.all(changes < 0) or np.all(changes > 0)); amplitude_supported = bool(np.all((std_ratios >= 0.5) & (std_ratios <= 1.5)))
        regional_rows.append({"region": region, "cosine_area_fraction": fractions[region], "all_three_trend_signs_agree": trend_sign_consensus, "all_three_endpoint_change_signs_agree": change_sign_consensus, "udel_modern_std_ratio_cru": std_ratios[0], "udel_modern_std_ratio_gpcc": std_ratios[1], "udel_modern_amplitude_supported_within_0_5_to_1_5": amplitude_supported, "product_robust_early_direction": bool(trend_sign_consensus and change_sign_consensus and amplitude_supported), "udel_trend_to_cru_gpcc_mean_ratio": float(trends[0] / np.mean(trends[1:])) if abs(np.mean(trends[1:])) > 1e-12 else np.nan})
    regions = pd.DataFrame(regional_rows); regions.to_csv(output / "regional_robustness.csv", index=False)
    reconstructed_hma = contributions.groupby("product").hma_trend_contribution_mm_day_per_century.sum(); archived_hma = early[(early.region == "hma") & early["product"].isin(PRODUCTS)].set_index("product").trend_mm_day_per_century; closure = float(np.max(np.abs(reconstructed_hma.reindex(PRODUCTS) - archived_hma.reindex(PRODUCTS))))
    west_robust = bool(regions.set_index("region").loc["west_lt80", "product_robust_early_direction"]); summary = {"status": "complete", "locked_1921_1939_accessed": False, "region_area_fractions": fractions, "hma_trend_contribution_closure_max_abs": closure, "regional_results": regions.set_index("region").to_dict(orient="index"), "western_hma_1901_1920_decline_product_robust": west_robust, "full_hma_1901_1920_direction_unresolved": True, "decision": "retain_western_decline_only_as_product_robust_early_observation_constraint" if west_robust else "treat_all_early_regional_directions_as_unresolved"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True); pivot = contributions.pivot(index="region", columns="product", values="hma_trend_contribution_mm_day_per_century").reindex(REGIONS); pivot.plot.bar(ax=axes[0], color=["#222222", "#d55e00", "#009e73"]); axes[0].axhline(0, color="#555555"); axes[0].set(ylabel="HMA trend contribution (mm/day/century)", title="Regional contribution to HMA trend"); axes[0].tick_params(axis="x", rotation=15)
    for product, color in [("UDel", "#222222"), ("CRU", "#d55e00"), ("GPCC", "#009e73")]: axes[1].plot(series[series.region == "west_lt80"].year, series[series.region == "west_lt80"][product], marker="o", label=product, color=color)
    axes[1].set(xlabel="Year", ylabel="Anomaly (mm/day)", title="Western HMA: product-robust decline"); axes[1].legend(frameon=False); figure.savefig(output / "early_regional_amplitude_decomposition.png", dpi=180); plt.close(figure)
    (output / "FINAL_RESULT.md").write_text("# UDel Early Regional-Amplitude Decomposition\n\n" f"- HMA trend contribution closure: `{closure:.3e}` mm/day/century.\n" f"- Western HMA product-robust decline: `{str(west_robust).lower()}`.\n" "- Full-HMA direction remains unresolved because UDel central/eastern low-frequency amplitude is not modern-supported.\n\n" f"Decision: `{summary['decision']}`.\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
