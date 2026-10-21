"""Test HMA LIA temperature robustness across three ModE-Sim historical sets."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = tuple((start, start + 49) for start in range(1450, 1850, 50))
SET_VARIABLES = {
    "set_1420_1_pmip4_analogue_sic": "set_1420_1_temp2_anomaly_1961_1990",
    "set_1420_2_eva_analogue_sic": "set_1420_2_temp2_anomaly_1961_1990",
    "set_1420_3_pmip4_climatology_sic": "set_1420_3_temp2_anomaly_1961_1990",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
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


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[None], axis=1) / np.sum(weights)


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        result[index] = float(np.mean(values[index - half : index + half + 1]))
    return result


def extract_coarse(
    field: np.ndarray, lat: np.ndarray, lon: np.ndarray, support: pd.DataFrame
) -> np.ndarray:
    cells = []
    for row in support.itertuples(index=False):
        i = np.flatnonzero(np.isclose(lat, row.latitude, atol=1e-8))
        j = np.flatnonzero(np.isclose(lon, row.longitude, atol=1e-8))
        if len(i) != 1 or len(j) != 1:
            raise RuntimeError("support-grid-match-failed")
        cells.append(field[:, int(i[0]), int(j[0])])
    return np.stack(cells, axis=1)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_77_mode_sim_historical_set_robustness_v1"
    round76 = loop / "round_76_mode_sim_subset_allowed_temperature_acquisition_v2"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round70 = loop / "round_70_mode_ra_lia_regional_adjudication_v2"
    paths = {
        "three_set_field": round76 / "mode_sim_three_historical_sets_allowed_hma_bbox.nc",
        "round76_summary": round76 / "summary.json",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "round70_windows": round70 / "mode_ra_lia_50year_adjudication.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    round76_summary = json.loads(paths["round76_summary"].read_text(encoding="utf-8"))
    if not round76_summary["passes_acquisition_preflight"]:
        raise RuntimeError("round76-v2-did-not-pass")
    overview = round76_summary["official_experiment_overview"]
    if not (
        overview["1420-1"]["volcanic_forcing_types"] == ["PMIP4"]
        and overview["1420-2"]["volcanic_forcing_types"] == ["EVA"]
        and overview["1420-3"]["volcanic_forcing_types"] == ["PMIP4"]
        and overview["1420-1"]["sic_forcing_types"] == ["analogues"]
        and overview["1420-2"]["sic_forcing_types"] == ["analogues"]
        and overview["1420-3"]["sic_forcing_types"] == ["climatology"]
    ):
        raise RuntimeError("official-set-metadata-conflict")

    with h5py.File(paths["three_set_field"], "r") as handle:
        years = np.asarray(handle["historical_year"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        fields = {
            label: np.asarray(handle[variable], dtype=float)
            for label, variable in SET_VARIABLES.items()
        }
    if not np.array_equal(years, np.arange(1421, 1850)):
        raise RuntimeError("historical-year-axis-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-present")
    if not all(np.all(np.isfinite(field)) for field in fields.values()):
        raise RuntimeError("nonfinite-three-set-field")
    support = pd.read_csv(paths["coarse_support"])
    if len(support) != 140:
        raise RuntimeError("coarse-support-count-failed")
    weights = np.cos(np.deg2rad(support.latitude.to_numpy(float))) * support.fine_hma_cell_count.to_numpy(float)
    coarse = {
        label: extract_coarse(field, lat, lon, support) for label, field in fields.items()
    }
    curves = {label: weighted_series(value, weights) for label, value in coarse.items()}
    annual = pd.DataFrame({"year": years, **{f"{label}_hma_degC": value for label, value in curves.items()}})
    annual.to_csv(output / "three_set_annual_hma_curves.csv", index=False)

    control = pd.read_csv(paths["round70_windows"])
    control = control[control.end_year <= 1849].reset_index(drop=True)
    if list(zip(control.start_year, control.end_year)) != list(WINDOWS):
        raise RuntimeError("round70-window-control-conflict")
    regional_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    set_labels = list(SET_VARIABLES)
    for window_index, (start, end) in enumerate(WINDOWS):
        selected = (years >= start) & (years <= end)
        if int(selected.sum()) != 50:
            raise RuntimeError(f"incomplete-window:{start}:{end}")
        window_fields = {
            label: np.mean(coarse[label][selected], axis=0) for label in set_labels
        }
        set_means = np.asarray(
            [weighted_mean(window_fields[label], weights) for label in set_labels]
        )
        set_center = float(np.mean(set_means))
        set_std = float(np.std(set_means, ddof=1))
        reference_min = float(control.loc[window_index, "reference_four_family_min_degC"])
        reference_max = float(control.loc[window_index, "reference_four_family_max_degC"])
        reference_central = float(control.loc[window_index, "reference_four_family_central_degC"])
        project = float(control.loc[window_index, "project_corrected_mean_degC"])
        expanded_min = min(reference_min, float(set_means.min()))
        expanded_max = max(reference_max, float(set_means.max()))
        model_family_central = float((4 * reference_central + set_center) / 5)
        regional_rows.append(
            {
                "start_year": start,
                "end_year": end,
                **{f"{label}_mean_degC": value for label, value in zip(set_labels, set_means)},
                "three_set_mean_degC": set_center,
                "three_set_std_degC": set_std,
                "three_set_range_degC": float(set_means.max() - set_means.min()),
                "absolute_common_signal_to_set_std_ratio": abs(set_center) / set_std if set_std > 0 else np.inf,
                "all_three_sets_negative": bool(np.all(set_means < 0)),
                "set2_outside_pmip4_set1_set3_interval": bool(
                    set_means[1] < min(set_means[0], set_means[2])
                    or set_means[1] > max(set_means[0], set_means[2])
                ),
                "set2_minus_set1_confounded_eva_member_contrast_degC": set_means[1] - set_means[0],
                "set3_minus_set1_pmip4_boundary_contrast_degC": set_means[2] - set_means[0],
                "project_corrected_mean_degC": project,
                "four_family_reference_central_degC": reference_central,
                "expanded_reference_min_degC": expanded_min,
                "expanded_reference_max_degC": expanded_max,
                "project_above_expanded_reference_max": bool(project > expanded_max),
                "project_minus_model_family_central_degC": project - model_family_central,
            }
        )

        residuals = {}
        residual_rms = {}
        for label in set_labels:
            regional_mean = weighted_mean(window_fields[label], weights)
            residuals[label] = window_fields[label] - regional_mean
            residual_rms[label] = weighted_rms(residuals[label], weights)
        pairwise = {
            f"{left}__{right}": correlation(residuals[left], residuals[right])
            for left, right in combinations(set_labels, 2)
        }
        mean_residual = np.mean(np.stack(list(residuals.values())), axis=0)
        deviations = np.stack([residuals[label] - mean_residual for label in set_labels])
        spatial_rows.append(
            {
                "start_year": start,
                "end_year": end,
                **{f"pair_correlation_{name}": value for name, value in pairwise.items()},
                "minimum_pairwise_residual_correlation": float(min(pairwise.values())),
                "median_pairwise_residual_correlation": float(np.median(list(pairwise.values()))),
                **{f"{label}_residual_rms_degC": residual_rms[label] for label in set_labels},
                "three_set_mean_residual_rms_degC": weighted_rms(mean_residual, weights),
                "cross_set_residual_spread_rms_degC": float(
                    np.sqrt(np.mean([weighted_rms(item, weights) ** 2 for item in deviations]))
                ),
            }
        )
    regional = pd.DataFrame(regional_rows)
    spatial = pd.DataFrame(spatial_rows)
    regional.to_csv(output / "three_set_regional_50year_robustness.csv", index=False)
    spatial.to_csv(output / "three_set_spatial_50year_robustness.csv", index=False)

    negative_count = int(regional.all_three_sets_negative.sum())
    median_signal_ratio = float(regional.absolute_common_signal_to_set_std_ratio.median())
    median_cross_set_range = float(regional.three_set_range_degC.median())
    set2_outside_count = int(regional.set2_outside_pmip4_set1_set3_interval.sum())
    median_min_pair_corr = float(spatial.minimum_pairwise_residual_correlation.median())
    worst_min_pair_corr = float(spatial.minimum_pairwise_residual_correlation.min())
    warm_bias_count = int(regional.project_above_expanded_reference_max.sum())
    gates = {
        "all_three_sets_negative_in_at_least_7_of_8_windows": negative_count >= 7,
        "median_common_signal_to_cross_set_std_ratio_at_least_2": median_signal_ratio >= 2,
        "median_minimum_pairwise_spatial_correlation_at_least_0_8": median_min_pair_corr >= 0.8,
        "worst_minimum_pairwise_spatial_correlation_at_least_0_5": worst_min_pair_corr >= 0.5,
        "project_above_expanded_reference_max_in_at_least_7_of_8_windows": warm_bias_count >= 7,
        "locked_1921_1939_absent": True,
    }
    summary = {
        "status": "complete",
        "years": [1421, 1849],
        "complete_fifty_year_windows": [list(window) for window in WINDOWS],
        "locked_1921_1939_accessed": False,
        "configuration_count": 3,
        "configuration_independence": "sensitivity_configurations_not_independent_evidence_families",
        "one_factor_volcanic_attribution_authorized": False,
        "regional": {
            "all_three_sets_negative_window_count": negative_count,
            "median_absolute_common_signal_to_set_std_ratio": median_signal_ratio,
            "median_cross_set_range_degC": median_cross_set_range,
            "set2_outside_pmip4_set1_set3_interval_count": set2_outside_count,
            "median_abs_set2_minus_set1_confounded_contrast_degC": float(
                np.median(np.abs(regional.set2_minus_set1_confounded_eva_member_contrast_degC))
            ),
            "median_abs_set3_minus_set1_pmip4_boundary_contrast_degC": float(
                np.median(np.abs(regional.set3_minus_set1_pmip4_boundary_contrast_degC))
            ),
        },
        "spatial": {
            "median_minimum_pairwise_residual_correlation": median_min_pair_corr,
            "worst_minimum_pairwise_residual_correlation": worst_min_pair_corr,
            "median_cross_set_residual_spread_rms_degC": float(
                spatial.cross_set_residual_spread_rms_degC.median()
            ),
            "median_three_set_mean_residual_rms_degC": float(
                spatial.three_set_mean_residual_rms_degC.median()
            ),
        },
        "project_warm_bias": {
            "project_above_expanded_reference_max_window_count": warm_bias_count,
            "median_project_minus_model_family_central_degC": float(
                regional.project_minus_model_family_central_degC.median()
            ),
        },
        "gates": gates,
        "passes_historical_set_robustness": bool(all(gates.values())),
        "decision": (
            "mode_sim_lia_cold_signal_is_robust_across_available_forcing_boundary_configurations"
            if all(gates.values())
            else "mode_sim_lia_signal_is_configuration_sensitive"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    labels = ["Set 1: PMIP4, analogue SIC", "Set 2: EVA, analogue SIC", "Set 3: PMIP4, climatology SIC"]
    for color, label, set_label in zip(colors, labels, set_labels):
        axes[0, 0].plot(years, smooth(curves[set_label]), color=color, linewidth=1.7, label=label)
    axes[0, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 0].set(title="HMA annual temperature, 15-year means", ylabel="Anomaly (degC)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    x = np.arange(len(regional))
    window_labels = [f"{row.start_year}-{row.end_year}" for row in regional.itertuples()]
    axes[0, 1].fill_between(x, regional[[f"{label}_mean_degC" for label in set_labels]].min(axis=1), regional[[f"{label}_mean_degC" for label in set_labels]].max(axis=1), color="#BBBBBB", alpha=0.5, label="Three-set range")
    axes[0, 1].plot(x, regional.three_set_mean_degC, color="#111111", marker="o", label="Three-set mean")
    axes[0, 1].plot(x, regional.project_corrected_mean_degC, color="#CC79A7", marker="s", label="Corrected project")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xticks(x, window_labels, rotation=35, ha="right", fontsize=8)
    axes[0, 1].set(title="Regional 50-year sensitivity envelope", ylabel="Anomaly (degC)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].bar(x, regional.absolute_common_signal_to_set_std_ratio, color="#E69F00")
    axes[1, 0].axhline(2, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set_xticks(x, window_labels, rotation=35, ha="right", fontsize=8)
    axes[1, 0].set(title="Common regional signal versus set spread", ylabel="|Mean| / set SD")

    axes[1, 1].plot(x, spatial.minimum_pairwise_residual_correlation, color="#0072B2", marker="o", label="Minimum pair correlation")
    axes[1, 1].plot(x, spatial.median_pairwise_residual_correlation, color="#009E73", marker="s", label="Median pair correlation")
    axes[1, 1].axhline(0.8, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 1].set_xticks(x, window_labels, rotation=35, ha="right", fontsize=8)
    axes[1, 1].set(title="Spatial residual robustness", ylabel="Pattern correlation", ylim=(0, 1.02))
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.savefig(output / "mode_sim_historical_set_robustness.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-Sim Historical-Set Robustness",
                "",
                f"- All three sets are cold in `{negative_count}/8` complete windows.",
                f"- Median common-signal/set-SD ratio: `{median_signal_ratio:.2f}`; median set range `{median_cross_set_range:.3f}` degC.",
                f"- Median/worst minimum pairwise spatial correlation: `{median_min_pair_corr:.3f}` / `{worst_min_pair_corr:.3f}`.",
                f"- Project remains above the expanded reference maximum in `{warm_bias_count}/8` windows.",
                f"- Set 2 lies outside the two PMIP4-set interval in `{set2_outside_count}/8` windows; this contrast remains confounded by SST/SIC member choices.",
                f"- All robustness gates pass: `{str(summary['passes_historical_set_robustness']).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
