"""Build an HMA-mean low-frequency candidate with family-held-out validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1 as round115


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
CONFIRMATION = RESULTS / "climate_reconstruction_temperature_lia_warm_bias_confirmation_v1/summary.json"

YEARS = np.arange(1000, 2001)
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
FAMILY_ORDER = ["local_enkf", "lmr", "neukom", "phyda"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_allowed_field(dataset: h5py.Dataset) -> np.ndarray:
    result = np.full((len(YEARS), dataset.shape[1], dataset.shape[2]), np.nan)
    result[:921] = round21.clean_slice(dataset, slice(0, 921))
    result[940:] = round21.clean_slice(dataset, slice(940, 1001))
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with np.errstate(invalid="ignore"):
        reference = np.nanmean(result[baseline], axis=0)
    return result - reference[None, :, :]


def lowpass51(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = 25
    for left_segment, right_segment in ((0, 921), (940, 1001)):
        for index in range(left_segment, right_segment):
            left = max(left_segment, index - half)
            right = min(right_segment, index + half + 1)
            current = values[left:right]
            if np.any(np.isfinite(current)):
                result[index] = float(np.nanmean(current))
    return result


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    return values - float(np.nanmean(values[baseline]))


def candidate(project: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return recenter(project + lowpass51(anchor - project))


def rmse(left: np.ndarray, right: np.ndarray, selected: np.ndarray) -> float:
    valid = selected & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) == 0:
        return float("nan")
    delta = left[valid] - right[valid]
    return float(np.sqrt(np.mean(delta * delta)))


def correlation(left: np.ndarray, right: np.ndarray, selected: np.ndarray) -> float:
    valid = selected & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def window_values(values: np.ndarray) -> np.ndarray:
    output = []
    for start, end in WINDOWS:
        selected = (YEARS >= start) & (YEARS <= end)
        if int(selected.sum()) != 50 or not np.all(np.isfinite(values[selected])):
            raise RuntimeError(f"Window coverage failed: {start}-{end}")
        output.append(float(np.mean(values[selected])))
    return np.asarray(output)


def main() -> int:
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "public": PUBLIC,
        "hma_mask": HMA_MASK,
        "warm_bias_confirmation": CONFIRMATION,
        "facts": PROJECT / "LIA_PROJECT_FACTS.md",
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    confirmation = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    if not confirmation.get("systematic_lia_warm_bias_confirmed", False):
        raise RuntimeError("Warm-bias confirmation prerequisite failed")

    hma_mask = np.load(HMA_MASK).astype(bool)
    with h5py.File(PUBLIC, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        latitude = np.asarray(handle["lat"], dtype=float).ravel()
        if not np.array_equal(source_years, YEARS):
            raise RuntimeError("Public year contract failed")
        project_field = read_allowed_field(handle[round115.LEGACY_VARIABLE])
        members = {
            name: round21.read_allowed_member(handle[variable], YEARS)
            for name, variable in round21.MEMBERS.items()
        }
        observation_fields = {
            name: read_allowed_field(handle[variable])
            for name, variable in round115.OBSERVATION_VARIABLES.items()
        }
    if hma_mask.shape != project_field.shape[1:]:
        raise RuntimeError("HMA mask shape failed")
    project_series = round21.weighted_series(project_field, hma_mask, latitude)
    family_fields = {
        "local_enkf": round21.finite_mean([members["HGAO"], members["O"]]),
        "lmr": members["LMR"],
        "neukom": round21.finite_mean(
            [field for name, field in members.items() if name.startswith("Nk_")]
        ),
        "phyda": members["PHYDA"],
    }
    family_series = {
        name: round21.weighted_series(field, hma_mask, latitude)
        for name, field in family_fields.items()
    }
    observation_series = {
        name: round21.weighted_series(field, hma_mask, latitude)
        for name, field in observation_fields.items()
    }

    lia = (YEARS >= 1500) & (YEARS <= 1899)
    loo_rows: list[dict[str, object]] = []
    loo_window_rows: list[dict[str, object]] = []
    loo_candidates: dict[str, np.ndarray] = {}
    for held_out in FAMILY_ORDER:
        anchor = np.mean(
            np.stack([family_series[name] for name in FAMILY_ORDER if name != held_out]),
            axis=0,
        )
        current = candidate(project_series, anchor)
        loo_candidates[held_out] = current
        project_annual_rmse = rmse(project_series, family_series[held_out], lia)
        candidate_annual_rmse = rmse(current, family_series[held_out], lia)
        project_windows = window_values(project_series)
        candidate_windows = window_values(current)
        held_windows = window_values(family_series[held_out])
        project_window_rmse = float(np.sqrt(np.mean((project_windows - held_windows) ** 2)))
        candidate_window_rmse = float(np.sqrt(np.mean((candidate_windows - held_windows) ** 2)))
        loo_rows.append(
            {
                "held_out_family": held_out,
                "project_annual_rmse_degC": project_annual_rmse,
                "candidate_annual_rmse_degC": candidate_annual_rmse,
                "annual_rmse_improvement": 1.0 - candidate_annual_rmse / project_annual_rmse,
                "project_50year_rmse_degC": project_window_rmse,
                "candidate_50year_rmse_degC": candidate_window_rmse,
                "window_rmse_improvement": 1.0 - candidate_window_rmse / project_window_rmse,
                "candidate_annual_correlation": correlation(current, family_series[held_out], lia),
            }
        )
        for (start, end), project_value, candidate_value, held_value in zip(
            WINDOWS, project_windows, candidate_windows, held_windows
        ):
            loo_window_rows.append(
                {
                    "held_out_family": held_out,
                    "start_year": start,
                    "end_year": end,
                    "project_degC": project_value,
                    "candidate_degC": candidate_value,
                    "held_out_reference_degC": held_value,
                    "project_error_degC": project_value - held_value,
                    "candidate_error_degC": candidate_value - held_value,
                }
            )
    loo_table = pd.DataFrame(loo_rows)
    loo_table.to_csv(OUTPUT / "paleo_family_leave_one_out_metrics.csv", index=False)
    pd.DataFrame(loo_window_rows).to_csv(
        OUTPUT / "paleo_family_leave_one_out_50year_windows.csv", index=False
    )

    four_family_anchor = np.mean(
        np.stack([family_series[name] for name in FAMILY_ORDER]), axis=0
    )
    final_candidate = candidate(project_series, four_family_anchor)
    modern = (YEARS >= 1940) & (YEARS <= 2000)
    modern_rows: list[dict[str, object]] = []
    for name, observed in observation_series.items():
        control_rmse = rmse(project_series, observed, modern)
        candidate_rmse = rmse(final_candidate, observed, modern)
        modern_rows.append(
            {
                "observation": name,
                "project_rmse_degC": control_rmse,
                "candidate_rmse_degC": candidate_rmse,
                "rmse_improvement": 1.0 - candidate_rmse / control_rmse,
                "project_correlation": correlation(project_series, observed, modern),
                "candidate_correlation": correlation(final_candidate, observed, modern),
                "correlation_change": (
                    correlation(final_candidate, observed, modern)
                    - correlation(project_series, observed, modern)
                ),
            }
        )
    modern_table = pd.DataFrame(modern_rows)
    modern_table.to_csv(OUTPUT / "modern_hma_product_validation.csv", index=False)

    final_windows = window_values(final_candidate)
    project_windows = window_values(project_series)
    anchor_windows = window_values(four_family_anchor)
    final_bias = final_windows - anchor_windows
    project_bias = project_windows - anchor_windows
    final_window_table = pd.DataFrame(
        {
            "start_year": [start for start, _ in WINDOWS],
            "end_year": [end for _, end in WINDOWS],
            "project_degC": project_windows,
            "four_family_anchor_degC": anchor_windows,
            "candidate_degC": final_windows,
            "project_minus_anchor_degC": project_bias,
            "candidate_minus_anchor_degC": final_bias,
        }
    )
    final_window_table.to_csv(OUTPUT / "final_candidate_lia_50year_windows.csv", index=False)

    annual = pd.DataFrame(
        {
            "year": YEARS,
            "project_hma_degC": project_series,
            "four_family_anchor_hma_degC": four_family_anchor,
            "candidate_hma_degC": final_candidate,
            **{f"reference_{name}_hma_degC": values for name, values in family_series.items()},
        }
    )
    annual.to_csv(OUTPUT / "temperature_hma_reference_anchored_candidate_1000_2000.csv", index=False)

    gates = {
        "all_four_held_out_annual_rmse_improve": bool(
            (loo_table["annual_rmse_improvement"] > 0).all()
        ),
        "all_four_held_out_50year_rmse_improve_at_least_20pct": bool(
            (loo_table["window_rmse_improvement"] >= 0.20).all()
        ),
        "median_held_out_50year_rmse_improvement_at_least_50pct": bool(
            loo_table["window_rmse_improvement"].median() >= 0.50
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
    selected = (YEARS >= 1500) & (YEARS <= 1900)
    axes[0].plot(YEARS[selected], lowpass51(project_series)[selected], color="#D55E00", linewidth=2.0, label="Project LP51")
    axes[0].plot(YEARS[selected], lowpass51(four_family_anchor)[selected], color="#222222", linewidth=2.0, linestyle="--", label="Four-family anchor LP51")
    axes[0].plot(YEARS[selected], lowpass51(final_candidate)[selected], color="#0072B2", linewidth=2.3, label="Candidate LP51")
    axes[0].set(title="HMA low-frequency temperature, 1500-1900", ylabel="Temperature anomaly (degC)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].bar(loo_table.held_out_family, 100 * loo_table.window_rmse_improvement, color="#009E73")
    axes[1].axhline(20, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set(title="Paleo held-out-family 50-year RMSE improvement", ylabel="Improvement (%)")
    modern_years = YEARS[modern]
    axes[2].plot(modern_years, project_series[modern], color="#D55E00", linewidth=1.7, label="Project")
    axes[2].plot(modern_years, final_candidate[modern], color="#0072B2", linewidth=2.1, label="Candidate")
    obs_mean = np.mean(np.stack([values[modern] for values in observation_series.values()]), axis=0)
    axes[2].plot(modern_years, obs_mean, color="#111111", linewidth=1.7, label="Five-product mean")
    axes[2].set(title="Modern HMA mean validation", xlabel="Year", ylabel="Temperature anomaly (degC)")
    axes[2].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = OUTPUT / "temperature_hma_reference_anchored_low_frequency.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "scope": "hma_regional_mean_only",
        "lowpass_width_years": 51,
        "paleo_period": [1500, 1899],
        "modern_period": [1940, 2000],
        "median_held_out_annual_rmse_improvement": float(
            loo_table["annual_rmse_improvement"].median()
        ),
        "median_held_out_50year_rmse_improvement": float(
            loo_table["window_rmse_improvement"].median()
        ),
        "modern_median_rmse_improvement": float(
            modern_table["rmse_improvement"].median()
        ),
        "modern_worst_rmse_improvement": float(
            modern_table["rmse_improvement"].min()
        ),
        "project_lia_median_minus_anchor_degC": float(np.median(project_bias)),
        "candidate_lia_median_minus_anchor_degC": float(np.median(final_bias)),
        "gates": gates,
        "passes_hma_mean_candidate_gates": passes,
        "locked_1921_1939_accessed": False,
        "phase_problem_used_for_tuning": False,
        "spatial_field_release_authorized": False,
        "decision": (
            "accept_hma_mean_diagnostic_candidate_and_test_regional_extension"
            if passes
            else "reject_reference_anchored_hma_mean_candidate"
        ),
        "dependence_caveat": (
            "Reference families are not fully independent. Leave-one-family-out tests transfer across families "
            "but do not create a new independent paleoclimate observation."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# HMA Reference-Anchored Low-Frequency Temperature\n\n"
        f"- Passed HMA-mean gates: `{str(passes).lower()}`.\n"
        f"- Median held-out 50-year RMSE improvement: `{100 * summary['median_held_out_50year_rmse_improvement']:+.2f}%`.\n"
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
