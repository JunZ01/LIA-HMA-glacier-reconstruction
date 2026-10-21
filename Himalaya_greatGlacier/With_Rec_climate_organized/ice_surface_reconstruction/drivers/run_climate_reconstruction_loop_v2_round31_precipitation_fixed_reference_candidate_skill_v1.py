"""Screen fixed public precipitation references against a zero-anomaly baseline."""

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


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
MEMBERS = {
    "HGAO": "pr_rec_ref_HGAO_c05d",
    "LMR": "pr_rec_ref_LMR_c05d",
    "O": "pr_rec_ref_O_c05d",
}
OBSERVATIONS = {
    "CRU": "pr_obs_cru_pre_05deg",
    "GPCC": "pr_obs_gpcp_05deg",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with np.errstate(invalid="ignore"):
        return values - np.nanmean(values[baseline], axis=0)[None]


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def centered_smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half:index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def score(
    candidate: np.ndarray,
    observation: np.ndarray,
    region_mask: np.ndarray,
    latitudes: np.ndarray,
    time_mask: np.ndarray,
) -> dict[str, float | int]:
    candidate_selected = candidate[time_mask]
    observation_selected = observation[time_mask]
    common = (
        region_mask
        & np.all(np.isfinite(candidate_selected), axis=0)
        & np.all(np.isfinite(observation_selected), axis=0)
    )
    if common.sum() == 0:
        raise RuntimeError("empty-common-support")
    candidate_series = weighted_series(candidate_selected, common, latitudes)
    observation_series = weighted_series(observation_selected, common, latitudes)
    candidate_rmse = float(np.sqrt(np.mean((candidate_series - observation_series) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(observation_series**2)))
    candidate_pixel_rmse = np.sqrt(np.mean((candidate_selected[:, common] - observation_selected[:, common]) ** 2, axis=0))
    zero_pixel_rmse = np.sqrt(np.mean(observation_selected[:, common] ** 2, axis=0))
    annual_correlation = correlation(candidate_series, observation_series)
    lowpass_correlation = (
        correlation(centered_smooth(candidate_series), centered_smooth(observation_series))
        if time_mask.sum() == len(YEARS)
        else np.nan
    )
    return {
        "common_cell_count": int(common.sum()),
        "regional_rmse_mm_day": candidate_rmse,
        "zero_regional_rmse_mm_day": zero_rmse,
        "regional_rmse_improvement_over_zero": float(1 - candidate_rmse / zero_rmse),
        "annual_correlation": annual_correlation,
        "lowpass_correlation_15y": lowpass_correlation,
        "median_pixel_rmse_mm_day": float(np.median(candidate_pixel_rmse)),
        "zero_median_pixel_rmse_mm_day": float(np.median(zero_pixel_rmse)),
        "median_pixel_rmse_improvement_over_zero": float(1 - np.median(candidate_pixel_rmse) / np.median(zero_pixel_rmse)),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_31_precipitation_fixed_reference_candidate_skill_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "public_reference_source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round19_scores": loop / "round_19_reference_family_balance_v1/candidate_scores.csv",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}
    contract = """# Round 31 Contract: Fixed Public Precipitation Reference Skill

## Question

Does any fixed public precipitation member or conservative, unfitted combination carry robust modern HMA anomaly skill against both GPCC and CRU?

## Frozen Candidates

- Individual HGAO, O, and LMR members.
- Local EnKF mean: equal HGAO/O mean.
- Member-equal mean: equal HGAO/LMR/O mean.
- Family-balanced mean: equal local-EnKF/LMR mean.
- No fitted weights or candidate-specific tuning is permitted. Each candidate passes or fails independently.

## Validation and Gates

- Use 1940-2000, independent 1961-1990 grid-cell recentering, the frozen HMA/west/central/east masks, GPCC and CRU, and five frozen temporal blocks.
- The baseline predicts zero anomaly after recentering.
- Promotion requires median block/region/product RMSE improvement at least 5%, no pair-block regression worse than 5%, more than 60% improving pair-blocks, whole-HMA improvement in at least four of five blocks for each observation, median full-period annual and 15-year correlations at least 0.30, and nonnegative HMA pixel-RMSE improvement for both observations.
- All gates must pass. Passing permits only a pre-instrumental family-consensus audit, not a production field.

## Exclusions

- No 1921-1939 access, proxy fitting, EnKF, localization, OGGM, or source modification.
"""
    (output / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((all_years >= 1940) & (all_years <= 2000))
        if not np.array_equal(all_years[indices], YEARS):
            raise RuntimeError("year-selection-contract-failed")
        if np.any((all_years[indices] >= 1921) & (all_years[indices] <= 1939)):
            raise RuntimeError("locked-period-accessed")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        members = {name: recenter(clean(handle[variable], selection)) for name, variable in MEMBERS.items()}
        observations = {name: recenter(clean(handle[variable], selection)) for name, variable in OBSERVATIONS.items()}
    candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean([finite_mean([members["HGAO"], members["O"]]), members["LMR"]]),
    }
    longitude_grid = np.broadcast_to(longitudes[None], mask.shape)
    region_masks = {
        name: mask & (longitude_grid >= lower) & (longitude_grid < upper)
        for name, (lower, upper) in REGIONS.items()
    }

    full_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    full_mask = np.ones(len(YEARS), dtype=bool)
    for candidate_name, candidate in candidates.items():
        for observation_name, observation in observations.items():
            for region_name, region_mask in region_masks.items():
                full_rows.append({
                    "candidate": candidate_name,
                    "observation": observation_name,
                    "region": region_name,
                    **score(candidate, observation, region_mask, latitudes, full_mask),
                })
                for fold, (start, end) in enumerate(BLOCKS, start=1):
                    time_mask = (YEARS >= start) & (YEARS <= end)
                    block_rows.append({
                        "candidate": candidate_name,
                        "observation": observation_name,
                        "region": region_name,
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        **score(candidate, observation, region_mask, latitudes, time_mask),
                    })
    full = pd.DataFrame(full_rows)
    blocks = pd.DataFrame(block_rows)
    full.to_csv(output / "fixed_candidate_full_period_scores.csv", index=False)
    blocks.to_csv(output / "fixed_candidate_block_scores.csv", index=False)

    archived = pd.read_csv(paths["round19_scores"])
    archived = archived[(archived.variable == "pr") & archived.candidate.isin(["member_equal", "family_balanced"])]
    closure = full.merge(
        archived,
        on=["observation", "region", "candidate"],
        validate="one_to_one",
        suffixes=("_new", "_round19"),
    )
    if len(closure) != 16:
        raise RuntimeError("round19-closure-row-count-failed")
    round19_rmse_closure = float(np.max(np.abs(closure.regional_rmse_mm_day - closure.regional_rmse)))
    round19_pixel_closure = float(np.max(np.abs(closure.median_pixel_rmse_mm_day - closure.median_pixel_rmse)))
    if max(round19_rmse_closure, round19_pixel_closure) > 1e-12:
        raise RuntimeError("round19-score-closure-failed")

    candidate_summary: dict[str, dict[str, object]] = {}
    for candidate_name in candidates:
        candidate_blocks = blocks[blocks.candidate == candidate_name]
        candidate_full = full[full.candidate == candidate_name]
        improvements = candidate_blocks.regional_rmse_improvement_over_zero.to_numpy(float)
        hma_blocks = candidate_blocks[candidate_blocks.region == "hma"]
        observation_fold_counts = {
            observation: int(np.sum(hma_blocks[hma_blocks.observation == observation].regional_rmse_improvement_over_zero > 0))
            for observation in OBSERVATIONS
        }
        hma_full = candidate_full[candidate_full.region == "hma"]
        pixel_by_observation = {
            row.observation: float(row.median_pixel_rmse_improvement_over_zero)
            for row in hma_full.itertuples()
        }
        gates = {
            "median_block_rmse_improvement_at_least_5pct": float(np.median(improvements)) >= 0.05,
            "maximum_pair_block_regression_at_most_5pct": float(np.max(-improvements)) <= 0.05,
            "improving_pair_block_fraction_over_60pct": float(np.mean(improvements > 0)) > 0.60,
            "both_observations_improve_hma_at_least_4_of_5_blocks": all(count >= 4 for count in observation_fold_counts.values()),
            "median_full_annual_correlation_at_least_0_30": float(candidate_full.annual_correlation.median()) >= 0.30,
            "median_full_lowpass_correlation_at_least_0_30": float(candidate_full.lowpass_correlation_15y.median()) >= 0.30,
            "both_observations_hma_pixel_rmse_nonnegative": all(value >= 0 for value in pixel_by_observation.values()),
        }
        candidate_summary[candidate_name] = {
            "median_block_rmse_improvement_over_zero": float(np.median(improvements)),
            "maximum_pair_block_regression": float(np.max(-improvements)),
            "improving_pair_block_fraction": float(np.mean(improvements > 0)),
            "hma_improving_fold_counts_by_observation": observation_fold_counts,
            "median_full_annual_correlation": float(candidate_full.annual_correlation.median()),
            "median_full_lowpass_correlation_15y": float(candidate_full.lowpass_correlation_15y.median()),
            "hma_pixel_rmse_improvement_by_observation": pixel_by_observation,
            "gates": gates,
            "passes": bool(all(gates.values())),
        }
    passing = [name for name, item in candidate_summary.items() if item["passes"]]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "round19_regional_rmse_max_abs_closure": round19_rmse_closure,
        "round19_pixel_rmse_max_abs_closure": round19_pixel_closure,
        "candidate_count": len(candidates),
        "candidate_results": candidate_summary,
        "passing_candidates": passing,
        "any_fixed_public_precipitation_candidate_passes": bool(passing),
        "decision": "advance_passing_candidates_to_preinstrumental_consensus" if passing else "retain_public_precipitation_references_as_comparison_only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    order = list(candidates)
    values = [
        blocks[blocks.candidate == name].regional_rmse_improvement_over_zero.to_numpy(float) * 100
        for name in order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].boxplot(values, labels=order, showfliers=False)
    axes[0].axhline(0, color="#444444")
    axes[0].axhline(5, color="#9b2226", linestyle="--")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(title="Five-block regional skill", ylabel="RMSE improvement over zero anomaly (%)")
    for name in order:
        selected = full[full.candidate == name]
        axes[1].scatter(selected.annual_correlation, selected.lowpass_correlation_15y, label=name, s=34, alpha=0.75)
    axes[1].axvline(0.30, color="#9b2226", linestyle="--")
    axes[1].axhline(0.30, color="#9b2226", linestyle="--")
    axes[1].set(title="Full-period temporal agreement", xlabel="Annual correlation", ylabel="15-year correlation")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Fixed public precipitation reference candidates", fontsize=14)
    fig.savefig(output / "precipitation_fixed_reference_candidate_skill.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Fixed Public Precipitation Reference Skill",
        "",
        f"- Round 19 RMSE closure: `{round19_rmse_closure:.3e}`; pixel closure: `{round19_pixel_closure:.3e}`.",
    ]
    for name, item in candidate_summary.items():
        lines.append(
            f"- {name}: median block improvement `{item['median_block_rmse_improvement_over_zero']:.2%}`, "
            f"annual correlation `{item['median_full_annual_correlation']:.3f}`, "
            f"15-year correlation `{item['median_full_lowpass_correlation_15y']:.3f}`, passes `{str(item['passes']).lower()}`."
        )
    lines.extend(["", f"Passing candidates: `{passing}`.", "", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
