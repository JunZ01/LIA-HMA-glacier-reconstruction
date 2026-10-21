"""Select HMA reference-anchor strength and timescale by nested family CV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_temperature_hma_reference_anchored_low_frequency_v1 as parent
import build_temperature_hma_reference_anchored_nested_lambda_v1 as nested_parent
import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1 as round115


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PARENT = RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_nested_lambda_v1"
CURVE_SOURCE = RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1"
OUTPUT = RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_nested_timescale_v1"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
WIDTH_GRID = (31, 51, 101)
LAMBDA_GRID = (0.0, 0.25, 0.50, 0.75, 1.00)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lowpass(values: np.ndarray, width: int) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = width // 2
    for segment_left, segment_right in ((0, 921), (940, 1001)):
        for index in range(segment_left, segment_right):
            left = max(segment_left, index - half)
            right = min(segment_right, index + half + 1)
            current = values[left:right]
            if np.any(np.isfinite(current)):
                result[index] = float(np.nanmean(current))
    return result


def make_candidate(
    project: np.ndarray, anchor: np.ndarray, width: int, strength: float
) -> np.ndarray:
    return parent.recenter(project + strength * lowpass(anchor - project, width))


def window_rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = parent.window_values(left) - parent.window_values(right)
    return float(np.sqrt(np.mean(delta * delta)))


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "curve_source": CURVE_SOURCE / "temperature_hma_reference_anchored_candidate_1000_2000.csv",
        "nested_parent_summary": PARENT / "summary.json",
        "nested_parent_manifest": PARENT / "source_manifest.json",
        "public": PUBLIC,
        "hma_mask": HMA_MASK,
        "facts": PROJECT / "LIA_PROJECT_FACTS.md",
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    curves = pd.read_csv(paths["curve_source"])
    if not np.array_equal(curves["year"].to_numpy(int), parent.YEARS):
        raise RuntimeError("Curve year contract failed")
    project_series = curves["project_hma_degC"].to_numpy(float)
    family_series = {
        name: curves[f"reference_{name}_hma_degC"].to_numpy(float)
        for name in parent.FAMILY_ORDER
    }
    lia = (parent.YEARS >= 1500) & (parent.YEARS <= 1899)

    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    pair_outer_errors: dict[tuple[int, float], list[float]] = {
        (width, strength): [] for width in WIDTH_GRID for strength in LAMBDA_GRID
    }
    for outer_held in parent.FAMILY_ORDER:
        training = [name for name in parent.FAMILY_ORDER if name != outer_held]
        objectives: dict[tuple[int, float], float] = {}
        for width in WIDTH_GRID:
            for strength in LAMBDA_GRID:
                errors = []
                for inner_held in training:
                    anchor_names = [name for name in training if name != inner_held]
                    anchor = np.mean(
                        np.stack([family_series[name] for name in anchor_names]), axis=0
                    )
                    trial = make_candidate(project_series, anchor, width, strength)
                    error = window_rmse(trial, family_series[inner_held])
                    errors.append(error)
                    inner_rows.append(
                        {
                            "outer_held_out_family": outer_held,
                            "inner_held_out_family": inner_held,
                            "width_years": width,
                            "lambda": strength,
                            "inner_50year_rmse_degC": error,
                        }
                    )
                objectives[(width, strength)] = float(np.median(errors))
        selected_width, selected_strength = min(
            objectives,
            key=lambda pair: (objectives[pair], pair[1], pair[0]),
        )
        outer_anchor = np.mean(
            np.stack([family_series[name] for name in training]), axis=0
        )
        outer_candidate = make_candidate(
            project_series, outer_anchor, selected_width, selected_strength
        )
        project_annual = parent.rmse(project_series, family_series[outer_held], lia)
        candidate_annual = parent.rmse(outer_candidate, family_series[outer_held], lia)
        project_window = window_rmse(project_series, family_series[outer_held])
        candidate_window = window_rmse(outer_candidate, family_series[outer_held])
        outer_rows.append(
            {
                "outer_held_out_family": outer_held,
                "selected_width_years": selected_width,
                "selected_lambda": selected_strength,
                "inner_selected_median_50year_rmse_degC": objectives[
                    (selected_width, selected_strength)
                ],
                "annual_rmse_improvement": 1.0 - candidate_annual / project_annual,
                "window_rmse_improvement": 1.0 - candidate_window / project_window,
                "candidate_50year_rmse_degC": candidate_window,
            }
        )
        for pair in pair_outer_errors:
            trial = make_candidate(project_series, outer_anchor, pair[0], pair[1])
            pair_outer_errors[pair].append(
                window_rmse(trial, family_series[outer_held])
            )
    inner_table = pd.DataFrame(inner_rows)
    outer_table = pd.DataFrame(outer_rows)
    inner_table.to_csv(OUTPUT / "nested_family_width_lambda_scores.csv", index=False)
    outer_table.to_csv(OUTPUT / "outer_family_validation_metrics.csv", index=False)

    final_rows = []
    for (width, strength), errors in pair_outer_errors.items():
        final_rows.append(
            {
                "width_years": width,
                "lambda": strength,
                "median_outer_family_50year_rmse_degC": float(np.median(errors)),
                "mean_outer_family_50year_rmse_degC": float(np.mean(errors)),
                "worst_outer_family_50year_rmse_degC": float(np.max(errors)),
            }
        )
    final_selection = pd.DataFrame(final_rows)
    selected = final_selection.sort_values(
        ["median_outer_family_50year_rmse_degC", "lambda", "width_years"]
    ).iloc[0]
    final_width = int(selected["width_years"])
    final_strength = float(selected["lambda"])
    final_selection["selected"] = (
        final_selection["width_years"].eq(final_width)
        & final_selection["lambda"].eq(final_strength)
    )
    final_selection.to_csv(OUTPUT / "final_width_lambda_selection.csv", index=False)
    final_anchor = np.mean(
        np.stack([family_series[name] for name in parent.FAMILY_ORDER]), axis=0
    )
    final_candidate = make_candidate(
        project_series, final_anchor, final_width, final_strength
    )

    hma_mask = np.load(HMA_MASK).astype(bool)
    with h5py.File(PUBLIC, "r") as handle:
        latitude = np.asarray(handle["lat"], dtype=float).ravel()
        observations = {
            name: round21.weighted_series(
                nested_parent.read_modern_observation(handle[variable]), hma_mask, latitude
            )
            for name, variable in round115.OBSERVATION_VARIABLES.items()
        }
    modern = (parent.YEARS >= 1940) & (parent.YEARS <= 2000)
    modern_rows = []
    for name, observed in observations.items():
        control_rmse = parent.rmse(project_series, observed, modern)
        candidate_rmse = parent.rmse(final_candidate, observed, modern)
        control_corr = parent.correlation(project_series, observed, modern)
        candidate_corr = parent.correlation(final_candidate, observed, modern)
        modern_rows.append(
            {
                "observation": name,
                "project_rmse_degC": control_rmse,
                "candidate_rmse_degC": candidate_rmse,
                "rmse_improvement": 1.0 - candidate_rmse / control_rmse,
                "project_correlation": control_corr,
                "candidate_correlation": candidate_corr,
                "correlation_change": candidate_corr - control_corr,
            }
        )
    modern_table = pd.DataFrame(modern_rows)
    modern_table.to_csv(OUTPUT / "modern_hma_product_validation.csv", index=False)

    project_windows = parent.window_values(project_series)
    anchor_windows = parent.window_values(final_anchor)
    candidate_windows = parent.window_values(final_candidate)
    pd.DataFrame(
        {
            "start_year": [start for start, _ in parent.WINDOWS],
            "end_year": [end for _, end in parent.WINDOWS],
            "project_degC": project_windows,
            "four_family_anchor_degC": anchor_windows,
            "candidate_degC": candidate_windows,
            "project_minus_anchor_degC": project_windows - anchor_windows,
            "candidate_minus_anchor_degC": candidate_windows - anchor_windows,
        }
    ).to_csv(OUTPUT / "final_candidate_lia_50year_windows.csv", index=False)
    annual = curves.copy()
    annual["selected_width_years"] = final_width
    annual["selected_lambda"] = final_strength
    annual["nested_timescale_candidate_hma_degC"] = final_candidate
    annual.to_csv(OUTPUT / "temperature_hma_nested_timescale_candidate_1000_2000.csv", index=False)

    gates = {
        "all_four_held_out_annual_rmse_improve": bool(
            (outer_table["annual_rmse_improvement"] > 0).all()
        ),
        "all_four_held_out_50year_rmse_improve_at_least_20pct": bool(
            (outer_table["window_rmse_improvement"] >= 0.20).all()
        ),
        "median_held_out_50year_rmse_improvement_at_least_50pct": bool(
            outer_table["window_rmse_improvement"].median() >= 0.50
        ),
        "modern_median_rmse_no_regression": bool(
            modern_table["rmse_improvement"].median() >= 0
        ),
        "modern_worst_product_rmse_regression_at_most_5pct": bool(
            modern_table["rmse_improvement"].min() >= -0.05 - 1e-12
        ),
        "modern_median_correlation_drop_at_most_0p02": bool(
            modern_table["correlation_change"].median() >= -0.02 - 1e-12
        ),
    }
    passes = bool(all(gates.values()))

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    paleo_plot = (parent.YEARS >= 1500) & (parent.YEARS <= 1900)
    axes[0].plot(parent.YEARS[paleo_plot], lowpass(project_series, final_width)[paleo_plot], color="#D55E00", linewidth=2.0, label="Project low-pass")
    axes[0].plot(parent.YEARS[paleo_plot], lowpass(final_anchor, final_width)[paleo_plot], color="#222222", linewidth=2.0, linestyle="--", label="Four-family anchor low-pass")
    axes[0].plot(parent.YEARS[paleo_plot], lowpass(final_candidate, final_width)[paleo_plot], color="#0072B2", linewidth=2.3, label=f"Candidate ({final_width} yr, lambda={final_strength:g})")
    axes[0].set(title="Nested timescale HMA candidate", ylabel="Temperature anomaly (degC)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].bar(outer_table.outer_held_out_family, 100 * outer_table.window_rmse_improvement, color="#009E73")
    axes[1].axhline(20, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set(title="Outer-family 50-year RMSE improvement", ylabel="Improvement (%)")
    modern_years = parent.YEARS[modern]
    axes[2].plot(modern_years, project_series[modern], color="#D55E00", linewidth=1.7, label="Project")
    axes[2].plot(modern_years, final_candidate[modern], color="#0072B2", linewidth=2.1, label="Candidate")
    obs_mean = np.mean(np.stack([values[modern] for values in observations.values()]), axis=0)
    axes[2].plot(modern_years, obs_mean, color="#111111", linewidth=1.7, label="Five-product mean")
    axes[2].set(title="Modern HMA mean validation", xlabel="Year", ylabel="Temperature anomaly (degC)")
    axes[2].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "temperature_hma_reference_anchored_nested_timescale.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "scope": "hma_regional_mean_only",
        "selected_final_width_years": final_width,
        "selected_final_lambda": final_strength,
        "outer_selected_pairs": {
            row.outer_held_out_family: {
                "width_years": int(row.selected_width_years),
                "lambda": float(row.selected_lambda),
            }
            for row in outer_table.itertuples(index=False)
        },
        "median_outer_annual_rmse_improvement": float(
            outer_table["annual_rmse_improvement"].median()
        ),
        "median_outer_50year_rmse_improvement": float(
            outer_table["window_rmse_improvement"].median()
        ),
        "worst_outer_50year_rmse_improvement": float(
            outer_table["window_rmse_improvement"].min()
        ),
        "modern_median_rmse_improvement": float(
            modern_table["rmse_improvement"].median()
        ),
        "modern_worst_rmse_improvement": float(
            modern_table["rmse_improvement"].min()
        ),
        "project_lia_median_minus_anchor_degC": float(
            np.median(project_windows - anchor_windows)
        ),
        "candidate_lia_median_minus_anchor_degC": float(
            np.median(candidate_windows - anchor_windows)
        ),
        "gates": gates,
        "passes_hma_mean_candidate_gates": passes,
        "locked_1921_1939_accessed": False,
        "phase_problem_used_for_tuning": False,
        "spatial_field_release_authorized": False,
        "decision": (
            "accept_nested_timescale_hma_mean_candidate"
            if passes
            else "close_reference_anchored_low_frequency_hma_mean_family"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# HMA Reference-Anchored Nested Timescale\n\n"
        f"- Final pair: `{final_width} years`, `lambda={final_strength}`.\n"
        f"- Passed HMA-mean gates: `{str(passes).lower()}`.\n"
        f"- Median outer 50-year improvement: `{100 * summary['median_outer_50year_rmse_improvement']:+.2f}%`.\n"
        f"- Worst outer 50-year improvement: `{100 * summary['worst_outer_50year_rmse_improvement']:+.2f}%`.\n"
        f"- Modern median RMSE improvement: `{100 * summary['modern_median_rmse_improvement']:+.2f}%`.\n"
        f"- LIA median warm gap: `{summary['project_lia_median_minus_anchor_degC']:+.3f}` to `{summary['candidate_lia_median_minus_anchor_degC']:+.3f} degC`.\n"
        "- Spatial-field release authorized: `false`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: sha256(path) for name, path in paths.items()}
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
