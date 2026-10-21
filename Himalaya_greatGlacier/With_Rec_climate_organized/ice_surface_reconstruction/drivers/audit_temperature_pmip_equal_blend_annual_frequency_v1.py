"""Audit whether the fixed PMIP blend is defensible at annual frequency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
BUILD = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1"
CANDIDATE = BUILD / "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
VERIFICATION = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_verification_v2/summary.json"
)
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_frequency_audit_v1"
KEY_REGIONS = ["13-08", "14-03", "15-01", "15-02"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def separate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = pd.Series(values).rolling(31, center=True, min_periods=31).mean().to_numpy(float)
    return low, values - low


def corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Annual-Frequency Audit Contract\n\n"
        "Evaluate the already verified direct annual 50:50 field without fitting. Separate each HMA "
        "and GTN-G annual series into a centered 31-year low-pass and residual high-pass over "
        "1500-1899. Direct annual promotion requires: median regional high-pass correlation change "
        "versus the current background >=-0.05; at least 12/15 regions must not lose more than 0.10 "
        "high-pass correlation; candidate/reference high-pass standard-deviation ratio must be "
        "0.67-1.5 in at least 12/15 regions; HMA candidate/background high-pass correlation >=0.8; "
        "and the 1849-1850 candidate jump must be <=3 standard deviations in HMA and all four key "
        "regions. Low-pass scores are diagnostic because that scale already passed. Failure rejects "
        "only direct annual blending and routes PMIP to a low-frequency correction design. No "
        "1901-1939 value is read.\n",
        encoding="utf-8",
    )
    paths = {
        "candidate": CANDIDATE, "verification": VERIFICATION,
        "assignment": ASSIGNMENT, "support": SUPPORT,
        "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    if not verification["passes_all_gates"]:
        raise RuntimeError("Annual candidate is not independently verified")

    with Dataset(CANDIDATE) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)[hma_mask]
        fields = {
            "reference": np.asarray(dataset.variables["tas_reference_family_balanced"][:], dtype=float)[:, hma_mask],
            "background": np.asarray(dataset.variables["tas_background_six_member"][:], dtype=float)[:, hma_mask],
            "forced_mean": np.asarray(dataset.variables["tas_forced_equal_three_model_mean"][:], dtype=float)[:, hma_mask],
            "candidate": np.asarray(dataset.variables["tas_equal_half_candidate"][:], dtype=float)[:, hma_mask],
        }
    if not np.array_equal(years, np.arange(1500, 1900)):
        raise RuntimeError("Unexpected candidate years")
    lat2, _ = np.meshgrid(lat, lon, indexing="ij")
    point_lat = lat2[hma_mask]
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    names = {"hma": "High Mountain Asia"}
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domain_masks[region.o2region] = grid_mask[hma_mask]
        names[region.o2region] = region.full_name

    series_rows = []
    metric_rows = []
    for domain, raw_mask in domain_masks.items():
        mask = raw_mask & common
        series = {name: domain_mean(values, mask, point_lat) for name, values in fields.items()}
        separated = {name: separate(values) for name, values in series.items()}
        for index, year in enumerate(years):
            series_rows.append({
                "domain": domain, "domain_name": names[domain], "year": int(year),
                **{f"{name}_degC": float(values[index]) for name, values in series.items()},
                **{f"{name}_low31_degC": float(separated[name][0][index]) for name in series},
                **{f"{name}_high31_degC": float(separated[name][1][index]) for name in series},
            })
        reference_low, reference_high = separated["reference"]
        background_low, background_high = separated["background"]
        forced_low, forced_high = separated["forced_mean"]
        candidate_low, candidate_high = separated["candidate"]
        valid = np.isfinite(reference_high)
        background_corr = corr(background_high, reference_high)
        candidate_corr = corr(candidate_high, reference_high)
        candidate_background_corr = corr(candidate_high, background_high)
        reference_std = float(np.std(reference_high[valid], ddof=1))
        background_std = float(np.std(background_high[valid], ddof=1))
        candidate_std = float(np.std(candidate_high[valid], ddof=1))
        low_valid = np.isfinite(reference_low)
        background_low_rmse = float(np.sqrt(np.mean((background_low[low_valid] - reference_low[low_valid]) ** 2)))
        candidate_low_rmse = float(np.sqrt(np.mean((candidate_low[low_valid] - reference_low[low_valid]) ** 2)))
        differences = np.diff(series["candidate"])
        splice_index = int(np.flatnonzero(years[:-1] == 1849)[0])
        splice_jump = float(differences[splice_index])
        typical = np.delete(differences, splice_index)
        splice_sigma = float(abs(splice_jump - np.mean(typical)) / np.std(typical, ddof=1))
        metric_rows.append({
            "domain": domain, "domain_name": names[domain],
            "background_reference_high31_correlation": background_corr,
            "candidate_reference_high31_correlation": candidate_corr,
            "forced_reference_high31_correlation": corr(forced_high, reference_high),
            "candidate_background_high31_correlation": candidate_background_corr,
            "candidate_minus_background_high31_correlation": candidate_corr - background_corr,
            "background_reference_high31_std_ratio": background_std / reference_std,
            "candidate_reference_high31_std_ratio": candidate_std / reference_std,
            "low31_rmse_improvement_vs_background": 1 - candidate_low_rmse / background_low_rmse,
            "candidate_1849_1850_jump_degC": splice_jump,
            "candidate_1849_1850_jump_sigma": splice_sigma,
        })
    metrics = pd.DataFrame(metric_rows)
    series_table = pd.DataFrame(series_rows)
    regional = metrics[metrics.domain.ne("hma")]
    hma = metrics[metrics.domain.eq("hma")].iloc[0]
    key = metrics[metrics.domain.isin(KEY_REGIONS)]
    std_ok = regional.candidate_reference_high31_std_ratio.between(0.67, 1.5)
    gates = {
        "median_regional_highpass_correlation_change_at_least_minus_0p05": bool(
            regional.candidate_minus_background_high31_correlation.median() >= -0.05
        ),
        "at_least_12_regions_lose_no_more_than_0p10_highpass_correlation": bool(
            (regional.candidate_minus_background_high31_correlation >= -0.10).sum() >= 12
        ),
        "highpass_std_ratio_between_0p67_and_1p5_in_at_least_12_regions": bool(std_ok.sum() >= 12),
        "hma_candidate_background_highpass_correlation_at_least_0p8": bool(
            hma.candidate_background_high31_correlation >= 0.8
        ),
        "hma_and_key_region_splice_jumps_at_most_3_sigma": bool(
            hma.candidate_1849_1850_jump_sigma <= 3 and (key.candidate_1849_1850_jump_sigma <= 3).all()
        ),
    }
    pass_all = bool(all(gates.values()))
    metrics.to_csv(OUTPUT / "annual_frequency_domain_metrics.csv", index=False)
    series_table.to_csv(OUTPUT / "annual_frequency_domain_curves.csv", index=False)

    hma_series = series_table[series_table.domain.eq("hma")].set_index("year")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for column, color, label in [
        ("reference_low31_degC", "#111111", "Reference"),
        ("background_low31_degC", "#D55E00", "Current background"),
        ("candidate_low31_degC", "#0072B2", "Direct annual 50:50"),
    ]:
        axes[0].plot(hma_series.index, hma_series[column], color=color, linewidth=1.8, label=label)
    axes[0].set(title="HMA 31-year low-frequency component", ylabel="degC")
    axes[0].legend(frameon=False, ncol=3)
    for column, color, label in [
        ("reference_high31_degC", "#111111", "Reference"),
        ("background_high31_degC", "#D55E00", "Current background"),
        ("candidate_high31_degC", "#0072B2", "Direct annual 50:50"),
    ]:
        axes[1].plot(hma_series.index, hma_series[column], color=color, linewidth=0.9, alpha=0.85, label=label)
    axes[1].set(title="HMA residual annual component", ylabel="degC")
    ordered = regional.sort_values("candidate_minus_background_high31_correlation")
    colors = np.where(ordered.candidate_minus_background_high31_correlation >= 0, "#0072B2", "#D55E00")
    axes[2].bar(ordered.domain, ordered.candidate_minus_background_high31_correlation, color=colors)
    axes[2].axhline(-0.05, color="#333333", linestyle="--", linewidth=1.0)
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set(title="Change in high-frequency correlation with reference", ylabel="Candidate minus background", xlabel="GTN-G region")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    figure = OUTPUT / "temperature_equal_blend_annual_frequency_audit.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    changed = [name for name in paths if before[name] != after[name]]
    numerical_unchanged = not any(name in changed for name in ["candidate", "verification", "assignment", "support"])
    decision = "retain_direct_annual_blend_for_next_validation" if pass_all else "reject_direct_annual_blend_design_low_frequency_only_correction"
    summary = {
        "status": "complete", "locked_1901_1939_values_read": 0,
        "numerical_sources_unchanged": numerical_unchanged,
        "changed_non_numerical_paths": [name for name in changed if name in ["facts"]],
        "median_regional_highpass_correlation_change": float(regional.candidate_minus_background_high31_correlation.median()),
        "regions_losing_more_than_0p10_highpass_correlation": int((regional.candidate_minus_background_high31_correlation < -0.10).sum()),
        "regions_with_accepted_highpass_std_ratio": int(std_ok.sum()),
        "hma_candidate_background_highpass_correlation": float(hma.candidate_background_high31_correlation),
        "hma_candidate_reference_highpass_correlation_change": float(hma.candidate_minus_background_high31_correlation),
        "hma_splice_jump_sigma": float(hma.candidate_1849_1850_jump_sigma),
        "max_key_region_splice_jump_sigma": float(key.candidate_1849_1850_jump_sigma.max()),
        "gates": gates, "passes_all_direct_annual_gates": pass_all,
        "decision": decision,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "changed": changed}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Annual-Frequency Audit\n\n"
        f"- Median regional high-pass correlation change: `{summary['median_regional_highpass_correlation_change']:+.3f}`.\n"
        f"- Regions losing more than 0.10 correlation: `{summary['regions_losing_more_than_0p10_highpass_correlation']}/15`.\n"
        f"- Regions with accepted high-pass variance ratio: `{summary['regions_with_accepted_highpass_std_ratio']}/15`.\n"
        f"- HMA candidate/background high-pass correlation: `{summary['hma_candidate_background_highpass_correlation']:.3f}`.\n"
        f"- HMA 1849-1850 jump: `{summary['hma_splice_jump_sigma']:.2f} sigma`.\n"
        f"- Direct annual gates pass: `{str(pass_all).lower()}`.\n\nDecision: `{decision}`.\n",
        encoding="utf-8",
    )
    if not numerical_unchanged:
        raise RuntimeError("A numerical source changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
