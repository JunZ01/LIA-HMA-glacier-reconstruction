"""Test a ModE-Sim forced prior-mean correction on frozen modern OOF gates."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1 as base


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_forced_prior_mean_oof_v1"
ROUND115_FIELDS = LOOP / (
    "round_115_temperature_low_rank_oof_v2/temperature_low_rank_oof_fields.npz"
)
MODE_SIM = LOOP / (
    "round_74_mode_sim_allowed_temperature_acquisition_v3/"
    "mode_sim_temp2_allowed_hma_bbox.nc"
)
ROUND113_GATES = LOOP / (
    "round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv"
)
LAMBDA_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    driver_root = Path(__file__).resolve().parent
    common = base.load_common(driver_root)
    paths = {
        "round115_fields": ROUND115_FIELDS,
        "mode_sim": MODE_SIM,
        "round113_gates": ROUND113_GATES,
        "round115_driver": driver_root / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
        "common_scoring_driver": driver_root / "temperature_fold_native_oenkf_increment_cv_v1/run_experiment_v1.py",
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: base.file_state(path) for name, path in paths.items()}

    with np.load(ROUND115_FIELDS) as saved:
        years = np.asarray(saved["years"], dtype=int)
        grid_lat = np.asarray(saved["grid_lat"], dtype=float)
        grid_lon = np.asarray(saved["grid_lon"], dtype=float)
        hma_mask = np.asarray(saved["hma_mask"], dtype=bool)
        observed_support = np.asarray(saved["observed_support"], dtype=bool)
        legacy = np.asarray(saved["equal_six_legacy"], dtype=float)
        observations = {
            name.removeprefix("observation_"): np.asarray(saved[name], dtype=float)
            for name in saved.files
            if name.startswith("observation_")
        }
    if not np.array_equal(years, base.YEARS):
        raise RuntimeError("Round115 year contract failed")
    if legacy.shape != (336, 61) or observed_support.shape != (336,):
        raise RuntimeError("Round115 state contract failed")
    if int(observed_support.sum()) != 99 or len(observations) != 5:
        raise RuntimeError("Round115 observed support contract failed")

    with h5py.File(MODE_SIM, "r") as handle:
        mode_years = np.asarray(handle["time"], dtype=int).ravel()
        mode_indices = np.asarray(
            [int(np.flatnonzero(mode_years == year)[0]) for year in years], dtype=int
        )
        selected_years = mode_years[mode_indices]
        if not np.array_equal(selected_years, years):
            raise RuntimeError("ModE-Sim year coverage failed")
        if np.any((selected_years >= 1921) & (selected_years <= 1939)):
            raise RuntimeError("Locked period access attempted")
        mode_lat = np.asarray(handle["lat"], dtype=float).ravel()
        mode_lon = np.asarray(handle["lon"], dtype=float).ravel()
        mode_cube = np.asarray(
            handle["temp2_anomaly_1961_1990"][mode_indices], dtype=float
        )
    mode_state = common.cube_to_state(
        common.source_grid(mode_cube, mode_lat, mode_lon, grid_lat, grid_lon)
    )
    if not np.all(np.isfinite(mode_state[observed_support])):
        raise RuntimeError("ModE-Sim incomplete on frozen observed support")
    mode_state = base.recenter(mode_state)

    full_weights = common.region_weights(grid_lat, grid_lon, hma_mask)
    weights = {name: values * observed_support for name, values in full_weights.items()}
    if any(np.sum(values > 0) < 3 for values in weights.values()):
        raise RuntimeError("Frozen regional support failed")

    oof = np.full_like(legacy, np.nan)
    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(base.BLOCKS, start=1):
        test = (years >= start) & (years <= end)
        train = ~test
        objectives: dict[float, float] = {}
        for current_lambda in LAMBDA_GRID:
            candidate = legacy + current_lambda * mode_state
            objective = base.regional_rmse_objective(
                candidate[:, train], train, observations, weights, common
            )
            objectives[current_lambda] = objective
            score_rows.append(
                {
                    "outer_fold": fold,
                    "outer_period": f"{start}-{end}",
                    "lambda": current_lambda,
                    "training_median_region_product_rmse_degC": objective,
                }
            )
        selected_lambda = min(
            objectives, key=lambda value: (objectives[value], value)
        )
        oof[:, test] = legacy[:, test] + selected_lambda * mode_state[:, test]
        selection_rows.append(
            {
                "outer_fold": fold,
                "outer_period": f"{start}-{end}",
                "selected_lambda": selected_lambda,
                "training_median_region_product_rmse_degC": objectives[selected_lambda],
                "lambda_zero_training_rmse_degC": objectives[0.0],
                "selected_improvement_over_lambda_zero": (
                    1.0 - objectives[selected_lambda] / objectives[0.0]
                ),
            }
        )
    if not np.all(np.isfinite(oof[observed_support])):
        raise RuntimeError("Forced-prior OOF stitch incomplete")
    candidate = base.recenter(oof)
    branches = {
        "forced_prior_mean_oof": candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(
        branches, observations, weights, common
    )
    candidate_pairs = pairs[pairs.branch.eq("forced_prior_mean_oof")]
    candidate_blocks = blocks[blocks.branch.eq("forced_prior_mean_oof")]
    candidate_spatial = spatial[spatial.branch.eq("forced_prior_mean_oof")]
    relative = candidate_pairs.improvement_vs_equal_six.to_numpy(float)
    relative_paths = candidate_blocks.improvement_vs_equal_six.to_numpy(float)
    absolute = candidate_pairs.skill_vs_zero.to_numpy(float)
    gates = {
        "M_REL_1_median_rmse_improvement_over_equal_six_at_least_0_05": bool(
            np.nanmedian(relative) >= 0.05
        ),
        "M_REL_2_worst_pair_regression_at_most_0_05": bool(
            np.nanmin(relative) >= -0.05 - 1e-12
        ),
        "M_REL_3_fraction_block_region_product_paths_improving_at_least_0_75": bool(
            np.mean(relative_paths > 0) >= 0.75
        ),
        "M_ABS_1_median_rmse_skill_over_zero_at_least_0_05": bool(
            np.nanmedian(absolute) >= 0.05
        ),
        "M_ABS_2_worst_rmse_skill_over_zero_at_least_minus_0_05": bool(
            np.nanmin(absolute) >= -0.05 - 1e-12
        ),
        "M_ABS_3_fraction_positive_regional_skill_at_least_0_75": bool(
            np.mean(absolute > 0) >= 0.75
        ),
        "M_ABS_4_median_annual_regional_correlation_at_least_0_30": bool(
            np.nanmedian(candidate_pairs.annual_correlation) >= 0.30
        ),
        "M_SPAT_1_median_spatial_skill_positive": bool(
            np.nanmedian(candidate_spatial.spatial_residual_skill) > 0
        ),
        "M_SPAT_2_worst_spatial_skill_at_least_minus_0_05": bool(
            np.nanmin(candidate_spatial.spatial_residual_skill) >= -0.05 - 1e-12
        ),
        "M_SPAT_3_fraction_positive_spatial_skill_at_least_0_75": bool(
            np.mean(candidate_spatial.spatial_residual_skill > 0) >= 0.75
        ),
        "M_SPAT_4_median_spatial_pattern_correlation_at_least_0_20": bool(
            np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)
            >= 0.20
        ),
    }
    metrics = {
        "median_rmse_improvement_over_equal_six": float(np.nanmedian(relative)),
        "worst_pair_rmse_improvement_over_equal_six": float(np.nanmin(relative)),
        "fraction_block_region_product_paths_improving": float(
            np.mean(relative_paths > 0)
        ),
        "median_skill_over_zero": float(np.nanmedian(absolute)),
        "worst_skill_over_zero": float(np.nanmin(absolute)),
        "fraction_positive_regional_skill": float(np.mean(absolute > 0)),
        "median_annual_regional_correlation": float(
            np.nanmedian(candidate_pairs.annual_correlation)
        ),
        "median_15year_regional_correlation": float(
            np.nanmedian(candidate_pairs.correlation_15year)
        ),
        "median_spatial_residual_skill": float(
            np.nanmedian(candidate_spatial.spatial_residual_skill)
        ),
        "worst_spatial_residual_skill": float(
            np.nanmin(candidate_spatial.spatial_residual_skill)
        ),
        "fraction_positive_spatial_skill": float(
            np.mean(candidate_spatial.spatial_residual_skill > 0)
        ),
        "median_spatial_pattern_correlation": float(
            np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)
        ),
    }
    selected_lambdas = [float(row["selected_lambda"]) for row in selection_rows]
    passes = bool(all(gates.values()) and any(value > 0 for value in selected_lambdas))

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / "outer_fold_lambda_selection.csv", index=False
    )
    pd.DataFrame(score_rows).to_csv(
        OUTPUT / "training_lambda_scores.csv", index=False
    )
    pairs.to_csv(OUTPUT / "regional_product_scores.csv", index=False)
    blocks.to_csv(OUTPUT / "block_region_product_scores.csv", index=False)
    curves.to_csv(OUTPUT / "regional_curves.csv", index=False)
    spatial.to_csv(OUTPUT / "spatial_product_scores.csv", index=False)
    np.savez_compressed(
        OUTPUT / "forced_prior_mean_oof_fields.npz",
        years=years,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=hma_mask,
        observed_support=observed_support,
        mode_sim=mode_state,
        candidate=candidate,
        equal_six_legacy=legacy,
        **{f"observation_{name}": values for name, values in observations.items()},
    )

    hma = curves[curves.region.eq("hma")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for series, color, width in (
        ("forced_prior_mean_oof", "#0072B2", 2.2),
        ("equal_six_legacy", "#D55E00", 1.8),
    ):
        values = hma[(hma.kind == "reconstruction") & (hma.series == series)]
        axes[0].plot(
            values.year, values.value_degC, color=color, linewidth=width, label=series
        )
    observed = (
        hma[hma.kind.eq("observation")]
        .groupby("year", as_index=False)
        .value_degC.mean()
    )
    axes[0].plot(
        observed.year,
        observed.value_degC,
        color="#111111",
        linewidth=1.8,
        label="five-product mean",
    )
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("Forced-prior-mean blocked out-of-fold test")
    axes[0].legend(frameon=False, ncol=3)
    selections = pd.DataFrame(selection_rows)
    axes[1].bar(
        selections.outer_period,
        selections.selected_lambda,
        color="#009E73",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Selected ModE-Sim lambda")
    axes[1].set_xlabel("Held-out block")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure_path = OUTPUT / "temperature_forced_prior_mean_oof.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "candidate": "equal_six_plus_lambda_times_mode_sim",
        "lambda_grid": list(LAMBDA_GRID),
        "selected_lambdas": selected_lambdas,
        "all_modern_gates_pass": bool(all(gates.values())),
        "nonzero_lambda_selected": bool(any(value > 0 for value in selected_lambdas)),
        "advance_to_lia_candidate": passes,
        "metrics": metrics,
        "gates": gates,
        "locked_1921_1939_accessed": False,
        "lia_reference_opened": False,
        "decision": (
            "advance_forced_prior_mean_to_full_candidate"
            if passes
            else "close_additive_forced_prior_mean_route"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Forced-Prior-Mean Temperature OOF",
        "",
        f"- Selected lambdas: `{selected_lambdas}`.",
        f"- All modern gates pass: `{str(summary['all_modern_gates_pass']).lower()}`.",
        f"- Median RMSE improvement: `{100 * metrics['median_rmse_improvement_over_equal_six']:+.2f}%`.",
        f"- Worst pair improvement: `{100 * metrics['worst_pair_rmse_improvement_over_equal_six']:+.2f}%`.",
        f"- Median spatial residual skill: `{100 * metrics['median_spatial_residual_skill']:+.2f}%`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"before": before, "after": after, "source_hashes_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
