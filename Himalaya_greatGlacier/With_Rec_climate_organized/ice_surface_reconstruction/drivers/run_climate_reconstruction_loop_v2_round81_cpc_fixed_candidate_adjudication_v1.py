"""Adjudicate frozen precipitation candidates against NOAA CPC Unified."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

import acquire_round80_cpc_unified_allowed_subset_v1 as acquisition_helpers
import run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1 as product_helpers

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
ROUND80 = LOOP / "round_80_cpc_unified_allowed_subset_acquisition_v3"
OUTPUT = LOOP / "round_81_cpc_fixed_candidate_adjudication_v1"
PUBLIC_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_input_spatial_audit_v3"
    / "hma_mask_on_netcdf_grid.npy"
)
UDEL = LOOP / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc"
PRECL_DIR = LOOP / "round_57_noaa_precl_allowed_subset_acquisition_v3"
PRECL_CHUNKS = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
PRECL_PATHS = [PRECL_DIR / f"precl_{start}_{end}.nc" for start, end in PRECL_CHUNKS]
DRIVER = Path(__file__).resolve()
BLOCKS = ((1979, 1989), (1990, 2000))


def center_field(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return values - np.nanmean(values, axis=0, keepdims=True)


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def residualize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return values - np.sum(values * weights[None, :], axis=1, keepdims=True)


def regional_metrics(candidate: np.ndarray, target: np.ndarray, years: np.ndarray) -> dict[str, object]:
    rmse = float(np.sqrt(np.mean((candidate - target) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target**2)))
    full_skill = float(1.0 - rmse / zero_rmse)
    block_skills: list[float] = []
    for start, end in BLOCKS:
        selected = (years >= start) & (years <= end)
        block_rmse = float(np.sqrt(np.mean((candidate[selected] - target[selected]) ** 2)))
        block_zero = float(np.sqrt(np.mean(target[selected] ** 2)))
        block_skills.append(float(1.0 - block_rmse / block_zero))
    corr = correlation(candidate, target)
    gates = {
        "full_rmse_improvement_at_least_5pct": full_skill >= 0.05,
        "both_frozen_blocks_improve": bool(np.all(np.asarray(block_skills) > 0)),
        "maximum_block_regression_at_most_5pct": float(np.max(-np.asarray(block_skills))) <= 0.05,
        "annual_correlation_at_least_0_30": corr >= 0.30,
    }
    return {
        "rmse_mm_day": rmse,
        "zero_rmse_mm_day": zero_rmse,
        "full_rmse_improvement": full_skill,
        "annual_correlation": corr,
        "block_skills": block_skills,
        "gates": gates,
        "passes": bool(all(gates.values())),
    }


def spatial_metrics(candidate: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    candidate_residual = residualize(candidate, weights)
    target_residual = residualize(target, weights)
    difference = candidate_residual - target_residual
    mse = float(np.mean(np.sum(difference**2 * weights[None], axis=1)))
    zero_mse = float(np.mean(np.sum(target_residual**2 * weights[None], axis=1)))
    skill = float(1.0 - mse / zero_mse)
    correlations = np.asarray(
        [correlation(candidate_residual[index], target_residual[index]) for index in range(target.shape[0])]
    )
    median_corr = float(np.median(correlations))
    positive_fraction = float(np.mean(correlations > 0))
    gates = {
        "residual_mse_skill_positive": skill > 0,
        "median_pattern_correlation_at_least_0_10": median_corr >= 0.10,
        "at_least_60pct_years_positive": positive_fraction >= 0.60,
    }
    return {
        "residual_mse_skill": skill,
        "median_annual_pattern_correlation": median_corr,
        "positive_year_fraction": positive_fraction,
        "gates": gates,
        "passes": bool(all(gates.values())),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 81 Contract",
                "",
                "## Objective",
                "Use NOAA CPC Unified as a new fixed cross-product target to adjudicate the six existing public precipitation candidates without fitting or changing any candidate.",
                "",
                "## Frozen Data And Preprocessing",
                "- Years: the 21 Round80-v3 eligible years, 1979-1984 and 1986-2000.",
                "- Candidates: HGAO, O, LMR, local_enkf=(HGAO+O)/2, member_equal=(HGAO+O+LMR)/3, and family_balanced=((HGAO+O)/2+LMR)/2.",
                "- Each field is centered independently at each grid cell over the 21 common years. No scaling, regression, or refitting is allowed.",
                "- Report HMA and western-HMA regional curves. The prior provisional decision is western, so only western gates determine advancement.",
                "- Before candidate interpretation, compare CPC regional variability with CRU, GPCC, UDel, and PREC/L on common support.",
                "- CPC is a distinct daily gauge-analysis pipeline, not claimed to have independent station inputs from all four products.",
                "",
                "## Frozen Western Candidate Gates",
                "- Regional: full-period RMSE improvement over zero >=5%, both 1979-1989 and 1990-2000 blocks improve, no block regresses >5%, and annual correlation >=0.30.",
                "- Spatial residual: MSE skill >0, median annual pattern correlation >=0.10, and >=60% of years have positive pattern correlation.",
                "- A candidate advances only if both western regional and western spatial-residual gates pass.",
                "",
                "## Safety",
                "The locked 1921-1939 final-test period is not read or accessed. Original files remain read-only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cpc_path = ROUND80 / "cpc_unified_precip_eligible_1979_2000_public_grid_annual_mm_day.nc"
    round80_summary = json.loads((ROUND80 / "summary.json").read_text(encoding="utf-8"))
    if round80_summary["decision"] != "advance_cpc_to_fixed_candidate_adjudication":
        raise RuntimeError("round80-did-not-advance")
    with h5py.File(cpc_path, "r") as handle:
        years = np.asarray(handle["year"][...], dtype=int)
        latitudes = np.asarray(handle["lat"][...], dtype=float)
        longitudes = np.asarray(handle["lon"][...], dtype=float)
        cpc = np.asarray(handle["precip_annual_mean_mm_day"][...], dtype=float)
    if years.tolist() != round80_summary["eligible_years"] or np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("cpc-year-axis-or-lock-failed")

    mask = np.load(HMA_MASK).astype(bool)
    with h5py.File(PUBLIC_SOURCE, "r") as handle:
        source_years = np.asarray(handle["time"][...], dtype=int).ravel()
        full_indices = np.flatnonzero((source_years >= 1979) & (source_years <= 2000))
        if not np.array_equal(source_years[full_indices], np.arange(1979, 2001)):
            raise RuntimeError("public-year-axis-failed")
        keep = source_years[full_indices] != 1985
        selection = slice(int(full_indices[0]), int(full_indices[-1]) + 1)
        if not np.array_equal(np.asarray(handle["lat"][...], dtype=float), latitudes) or not np.array_equal(
            np.asarray(handle["lon"][...], dtype=float), longitudes
        ):
            raise RuntimeError("public-cpc-coordinate-mismatch")
        observations = {
            name: product_helpers.clean(handle[variable], selection)[keep]
            for name, variable in product_helpers.OBSERVATIONS.items()
        }
        members = {
            name: product_helpers.clean(handle[variable], selection)[keep]
            for name, variable in product_helpers.MEMBERS.items()
        }
    legacy_years = product_helpers.YEARS
    legacy_keep = np.isin(legacy_years, years)
    observations["UDel"] = product_helpers.annualize_udel(UDEL, latitudes, longitudes)[legacy_keep]
    observations["PREC/L"] = product_helpers.annualize_precl(PRECL_PATHS, latitudes, longitudes)[legacy_keep]
    if not np.array_equal(legacy_years[legacy_keep], years):
        raise RuntimeError("legacy-product-year-selection-failed")

    candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean([finite_mean([members["HGAO"], members["O"]]), members["LMR"]]),
    }
    fields = {"CPC": center_field(cpc)}
    fields.update({name: center_field(values) for name, values in observations.items()})
    candidate_fields = {name: center_field(values) for name, values in candidates.items()}

    region_masks = {
        "HMA": mask.copy(),
        "west": mask & (longitudes[None, :] < 80.0),
    }
    compatibility_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    result_map: dict[str, dict[str, object]] = {name: {} for name in candidates}
    common_counts: dict[str, int] = {}

    for region, region_mask in region_masks.items():
        common = region_mask.copy()
        for values in fields.values():
            common &= np.all(np.isfinite(values), axis=0)
        for values in candidate_fields.values():
            common &= np.all(np.isfinite(values), axis=0)
        if common.sum() < 50:
            raise RuntimeError(f"insufficient-common-support:{region}:{int(common.sum())}")
        common_counts[region] = int(common.sum())
        cpc_series = weighted_series(fields["CPC"], common, latitudes)

        for product in product_helpers.OBSERVATIONS | {"UDel": "", "PREC/L": ""}:
            product_series = weighted_series(fields[product], common, latitudes)
            std_ratio = float(np.std(cpc_series, ddof=1) / np.std(product_series, ddof=1))
            corr = correlation(cpc_series, product_series)
            compatibility_rows.append(
                {
                    "region": region,
                    "product": product,
                    "common_cell_count": int(common.sum()),
                    "annual_correlation": corr,
                    "cpc_to_product_std_ratio": std_ratio,
                    "correlation_gate_at_least_0_50": corr >= 0.50,
                    "std_ratio_gate_0_5_to_1_5": 0.5 <= std_ratio <= 1.5,
                    "passes": corr >= 0.50 and 0.5 <= std_ratio <= 1.5,
                }
            )

        weights = np.cos(np.deg2rad(np.broadcast_to(latitudes[:, None], common.shape)[common]))
        weights = weights / weights.sum()
        target_pixels = fields["CPC"][:, common]
        for name, candidate_field in candidate_fields.items():
            candidate_series = weighted_series(candidate_field, common, latitudes)
            regional = regional_metrics(candidate_series, cpc_series, years)
            spatial = spatial_metrics(candidate_field[:, common], target_pixels, weights)
            advances = bool(regional["passes"] and spatial["passes"]) if region == "west" else False
            result_map[name][region] = {
                "regional": regional,
                "spatial_residual": spatial,
                "advances_on_western_cpc": advances,
            }
            candidate_rows.append(
                {
                    "candidate": name,
                    "region": region,
                    "common_cell_count": int(common.sum()),
                    "regional_full_rmse_improvement": regional["full_rmse_improvement"],
                    "regional_annual_correlation": regional["annual_correlation"],
                    "regional_passes": regional["passes"],
                    "spatial_residual_mse_skill": spatial["residual_mse_skill"],
                    "spatial_median_pattern_correlation": spatial["median_annual_pattern_correlation"],
                    "spatial_positive_year_fraction": spatial["positive_year_fraction"],
                    "spatial_passes": spatial["passes"],
                    "advances_on_western_cpc": advances,
                }
            )
            for (start, end), skill in zip(BLOCKS, regional["block_skills"]):
                block_rows.append(
                    {"candidate": name, "region": region, "start_year": start, "end_year": end, "rmse_improvement": skill}
                )
            for index, year in enumerate(years):
                curve_rows.append(
                    {"year": int(year), "region": region, "series": name, "anomaly_mm_day": float(candidate_series[index])}
                )
        for index, year in enumerate(years):
            curve_rows.append(
                {"year": int(year), "region": region, "series": "CPC", "anomaly_mm_day": float(cpc_series[index])}
            )

    compatibility = pd.DataFrame(compatibility_rows)
    scores = pd.DataFrame(candidate_rows)
    blocks = pd.DataFrame(block_rows)
    curves = pd.DataFrame(curve_rows)
    compatibility.to_csv(OUTPUT / "cpc_observation_compatibility.csv", index=False)
    scores.to_csv(OUTPUT / "cpc_candidate_scores.csv", index=False)
    blocks.to_csv(OUTPUT / "cpc_candidate_block_scores.csv", index=False)
    curves.to_csv(OUTPUT / "cpc_candidate_regional_curves.csv", index=False)

    advancing = [name for name, result in result_map.items() if result["west"]["advances_on_western_cpc"]]
    western_compatibility_count = int(compatibility.query("region == 'west'")["passes"].sum())
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "years": years.astype(int).tolist(),
        "excluded_years": [1985],
        "common_cell_counts": common_counts,
        "western_observation_compatibility_paths_passing": western_compatibility_count,
        "candidate_results": result_map,
        "advancing_western_candidates": advancing,
        "decision": "retain_cpc_supported_fixed_candidate" if advancing else "close_existing_fixed_precipitation_candidates_against_cpc",
        "candidate_refit_or_scaling_performed": False,
        "cpc_station_independence_claimed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {
        "CPC": "#111111",
        "HGAO": "#0072B2",
        "O": "#D55E00",
        "LMR": "#009E73",
        "local_enkf": "#CC79A7",
        "member_equal": "#56B4E9",
        "family_balanced": "#E69F00",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, constrained_layout=True)
    for axis, region in zip(axes, ("HMA", "west")):
        selected = curves[curves.region == region]
        for series in colors:
            values = selected[selected.series == series].sort_values("year")
            axis.plot(
                values.year,
                values.anomaly_mm_day,
                color=colors[series],
                linewidth=2.8 if series == "CPC" else 1.25,
                label=series,
            )
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_title(f"{region}: frozen precipitation candidates versus NOAA CPC Unified")
        axis.set_ylabel("Annual anomaly (mm/day)")
        axis.legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Year")
    figure.savefig(OUTPUT / "cpc_candidate_regional_curves.png", dpi=190)
    plt.close(figure)

    result_lines = [
        "# Round 81 CPC Fixed-Candidate Adjudication",
        "",
        f"- Eligible years: `{len(years)}`; excluded: `1985`.",
        f"- Common cells HMA/west: `{common_counts['HMA']}/{common_counts['west']}`.",
        f"- Western CPC compatibility paths passing: `{western_compatibility_count}/4`.",
    ]
    for name in candidates:
        west_result = result_map[name]["west"]
        region = west_result["regional"]
        spatial = west_result["spatial_residual"]
        result_lines.append(
            f"- {name}: west regional skill/correlation `{region['full_rmse_improvement']:.2%}/{region['annual_correlation']:.3f}`, "
            f"spatial skill/median correlation `{spatial['residual_mse_skill']:.2%}/{spatial['median_annual_pattern_correlation']:.3f}`, "
            f"advances `{str(west_result['advances_on_western_cpc']).lower()}`."
        )
    result_lines.extend(
        [
            "",
            f"Advancing candidates: `{advancing}`.",
            f"Decision: `{summary['decision']}`.",
            "CPC is treated as a distinct daily gauge-analysis product, not as station-independent evidence.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    source_paths = [DRIVER, contract, cpc_path, ROUND80 / "summary.json", PUBLIC_SOURCE, HMA_MASK, UDEL, *PRECL_PATHS]
    source_states_before = [acquisition_helpers.file_record(path) for path in source_paths]
    source_states_after = [acquisition_helpers.file_record(path) for path in source_paths]
    unchanged = source_states_before == source_states_after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": source_states_before,
                "sources_after": source_states_after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
