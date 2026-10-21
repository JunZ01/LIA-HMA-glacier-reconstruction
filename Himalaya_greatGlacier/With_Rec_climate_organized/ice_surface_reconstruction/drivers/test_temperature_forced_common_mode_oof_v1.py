"""Test a low-frequency ModE-Sim HMA common-mode correction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1 as base
import test_temperature_forced_low_frequency_oof_v1 as parent_tools


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_forced_common_mode_oof_v1"
PARENT = RESULTS / "climate_reconstruction_temperature_forced_low_frequency_oof_v1"
PARENT_FIELDS = PARENT / "forced_low_frequency_oof_fields.npz"
ROUND113_GATES = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv"
)
LAMBDA_GRID = parent_tools.LAMBDA_GRID


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    driver_root = Path(__file__).resolve().parent
    common = base.load_common(driver_root)
    paths = {
        "parent_fields": PARENT_FIELDS,
        "parent_summary": PARENT / "summary.json",
        "parent_manifest": PARENT / "source_manifest.json",
        "round113_gates": ROUND113_GATES,
        "round115_driver": driver_root / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
        "parent_driver": driver_root / "test_temperature_forced_low_frequency_oof_v1.py",
        "common_scoring_driver": driver_root / "temperature_fold_native_oenkf_increment_cv_v1/run_experiment_v1.py",
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: base.file_state(path) for name, path in paths.items()}

    with np.load(PARENT_FIELDS) as saved:
        years = np.asarray(saved["years"], dtype=int)
        grid_lat = np.asarray(saved["grid_lat"], dtype=float)
        grid_lon = np.asarray(saved["grid_lon"], dtype=float)
        hma_mask = np.asarray(saved["hma_mask"], dtype=bool)
        observed_support = np.asarray(saved["observed_support"], dtype=bool)
        legacy = np.asarray(saved["equal_six_legacy"], dtype=float)
        legacy_low = np.asarray(saved["legacy_low_frequency"], dtype=float)
        mode_low = np.asarray(saved["mode_sim_low_frequency"], dtype=float)
        observations = {
            name.removeprefix("observation_"): np.asarray(saved[name], dtype=float)
            for name in saved.files
            if name.startswith("observation_")
        }
    if not np.array_equal(years, base.YEARS):
        raise RuntimeError("Parent year contract failed")
    if int(observed_support.sum()) != 99 or len(observations) != 5:
        raise RuntimeError("Parent support contract failed")

    full_weights = common.region_weights(grid_lat, grid_lon, hma_mask)
    weights = {name: values * observed_support for name, values in full_weights.items()}
    legacy_common = common.weighted_series(legacy_low, weights["hma"])
    mode_common = common.weighted_series(mode_low, weights["hma"])
    common_residual_series = mode_common - legacy_common
    common_residual = np.broadcast_to(common_residual_series[None, :], legacy.shape).copy()

    oof = np.full_like(legacy, np.nan)
    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(base.BLOCKS, start=1):
        test = (years >= start) & (years <= end)
        train = ~test
        objectives: dict[float, float] = {}
        for current_lambda in LAMBDA_GRID:
            trial = base.recenter(legacy + current_lambda * common_residual)
            objective = base.regional_rmse_objective(
                trial[:, train], train, observations, weights, common
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
        selected_lambda = min(objectives, key=lambda value: (objectives[value], value))
        selected_trial = base.recenter(legacy + selected_lambda * common_residual)
        oof[:, test] = selected_trial[:, test]
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
        raise RuntimeError("Common-mode OOF stitch incomplete")
    candidate = base.recenter(oof)
    branches = {
        "forced_low_frequency_oof": candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(
        branches, observations, weights, common
    )
    metrics, gates = parent_tools.gate_metrics(pairs, blocks, spatial)
    selected_lambdas = [float(row["selected_lambda"]) for row in selection_rows]
    passes = bool(all(gates.values()) and any(value > 0 for value in selected_lambdas))

    pd.DataFrame(selection_rows).to_csv(OUTPUT / "outer_fold_lambda_selection.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUTPUT / "training_lambda_scores.csv", index=False)
    pairs.to_csv(OUTPUT / "regional_product_scores.csv", index=False)
    blocks.to_csv(OUTPUT / "block_region_product_scores.csv", index=False)
    curves.to_csv(OUTPUT / "regional_curves.csv", index=False)
    spatial.to_csv(OUTPUT / "spatial_product_scores.csv", index=False)
    pd.DataFrame(
        {
            "year": years,
            "legacy_hma_lp31_degC": legacy_common,
            "mode_sim_hma_lp31_degC": mode_common,
            "forced_common_residual_degC": common_residual_series,
        }
    ).to_csv(OUTPUT / "forced_common_mode_series.csv", index=False)
    np.savez_compressed(
        OUTPUT / "forced_common_mode_oof_fields.npz",
        years=years,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=hma_mask,
        observed_support=observed_support,
        common_residual=common_residual,
        candidate=candidate,
        equal_six_legacy=legacy,
        **{f"observation_{name}": values for name, values in observations.items()},
    )

    hma = curves[curves.region.eq("hma")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for series, color, width in (
        ("forced_low_frequency_oof", "#0072B2", 2.2),
        ("equal_six_legacy", "#D55E00", 1.8),
    ):
        values = hma[(hma.kind == "reconstruction") & (hma.series == series)]
        axes[0].plot(values.year, values.value_degC, color=color, linewidth=width, label=series)
    observed = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(observed.year, observed.value_degC, color="#111111", linewidth=1.8, label="five-product mean")
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("31-year forced HMA common-mode blocked OOF test")
    axes[0].legend(frameon=False, ncol=3)
    selections = pd.DataFrame(selection_rows)
    axes[1].bar(selections.outer_period, selections.selected_lambda, color="#009E73")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Selected common-mode lambda")
    axes[1].set_xlabel("Held-out block")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure_path = OUTPUT / "temperature_forced_common_mode_oof.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "candidate": "equal_six_plus_lambda_times_hma_lp31_mode_minus_hma_lp31_equal_six",
        "lowpass_width_years": 31,
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
            "advance_forced_common_mode_to_full_candidate"
            if passes
            else "close_forced_common_mode_route"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Forced Common-Mode Temperature OOF\n\n"
        f"- Selected lambdas: `{selected_lambdas}`.\n"
        f"- All modern gates pass: `{str(summary['all_modern_gates_pass']).lower()}`.\n"
        f"- Median RMSE improvement: `{100 * metrics['median_rmse_improvement_over_equal_six']:+.2f}%`.\n"
        f"- Worst pair improvement: `{100 * metrics['worst_pair_rmse_improvement_over_equal_six']:+.2f}%`.\n"
        f"- Median spatial residual skill: `{100 * metrics['median_spatial_residual_skill']:+.2f}%`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
