"""Test robustness of the Round 19 temperature family-balanced reference."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_round19():
    path = Path(__file__).with_name("run_climate_reconstruction_loop_v2_round19_reference_family_balance_v1.py")
    spec = importlib.util.spec_from_file_location("round19_family_balance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("round19-import-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_aggregates(comparison: pd.DataFrame, candidate: str) -> dict[str, object]:
    values = comparison[comparison.candidate == candidate]
    hma = values[values.region == "hma"]
    median_regional = float(values.regional_rmse_improvement.median())
    maximum_regression = max(0.0, float(np.max(-values.regional_rmse_improvement.to_numpy(float))))
    positive_fraction = float(np.mean(values.regional_rmse_improvement > 0))
    median_pixel = float(hma.pixel_rmse_improvement.median())
    median_lowpass = float(values.lowpass_correlation_change.median())
    gates = {
        "median_regional_rmse_improvement_at_least_2pct": median_regional >= 0.02,
        "maximum_pair_regression_at_most_5pct": maximum_regression <= 0.05,
        "more_than_half_pairs_improve": positive_fraction > 0.5,
        "median_hma_pixel_rmse_does_not_worsen": median_pixel >= 0,
        "median_regional_lowpass_correlation_does_not_decrease": median_lowpass >= 0,
    }
    return {
        "median_regional_rmse_improvement": median_regional,
        "maximum_pair_regression": maximum_regression,
        "positive_pair_fraction": positive_fraction,
        "median_hma_pixel_rmse_improvement": median_pixel,
        "median_regional_lowpass_correlation_change": median_lowpass,
        "round19_gates": gates,
        "passes_round19_gates": bool(all(gates.values())),
    }


def main() -> int:
    base = load_round19()
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_20_temperature_family_balance_robustness_v1"
    netcdf = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    round19_comparison = loop / "round_19_reference_family_balance_v1/family_balance_comparison.csv"
    paths = {
        "netcdf": netcdf,
        "hma_mask": mask_path,
        "round19_comparison": round19_comparison,
        "round19_driver": Path(base.__file__).resolve(),
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: base.state(path) for name, path in paths.items()}

    hma_mask = np.load(mask_path).astype(bool)
    score_rows: list[dict[str, object]] = []
    curve_rows: dict[str, object] = {"year": base.YEARS}
    with h5py.File(netcdf, "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((all_years >= 1940) & (all_years <= 2000))
        if not np.array_equal(all_years[indices], base.YEARS):
            raise RuntimeError("year-selection-failed")
        if np.any((all_years[indices] >= 1921) & (all_years[indices] <= 1939)):
            raise RuntimeError("locked-period-accessed")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        longitude_grid = np.broadcast_to(longitudes[None, :], hma_mask.shape)

        members = {
            name: base.recenter(base.clean(handle[dataset], selection))
            for name, dataset in base.MEMBERS["tas"].items()
        }
        observations = {
            name: base.recenter(base.clean(handle[dataset], selection))
            for name, dataset in base.OBSERVATIONS["tas"].items()
        }
        control = base.finite_mean(list(members.values()))
        local_enkf = base.finite_mean([members["HGAO"], members["O"]])
        neukom = base.finite_mean([members[name] for name in members if name.startswith("Nk_")])
        families = {
            "local_enkf": local_enkf,
            "lmr": members["LMR"],
            "neukom": neukom,
            "phyda": members["PHYDA"],
        }
        candidates = {
            "primary_four_family": base.finite_mean(list(families.values())),
            "five_family_hgao_o_separate": base.finite_mean([members["HGAO"], members["O"], members["LMR"], neukom, members["PHYDA"]]),
        }
        for omitted in families:
            candidates[f"leave_out_{omitted}"] = base.finite_mean([value for name, value in families.items() if name != omitted])

        curve_rows["member_equal"] = base.weighted_series(control, hma_mask, latitudes)
        for candidate_name, candidate in candidates.items():
            curve_rows[candidate_name] = base.weighted_series(candidate, hma_mask, latitudes)
        for observation_name, observation in observations.items():
            curve_rows[f"obs_{observation_name}"] = base.weighted_series(observation, hma_mask, latitudes)
            for region, (lower, upper) in base.REGIONS.items():
                region_mask = hma_mask & (longitude_grid >= lower) & (longitude_grid < upper)
                control_metrics = base.pair_metrics(control, observation, region_mask, latitudes)
                for candidate_name, candidate in candidates.items():
                    metrics = base.pair_metrics(candidate, observation, region_mask, latitudes)
                    score_rows.append({
                        "candidate": candidate_name,
                        "observation": observation_name,
                        "region": region,
                        **metrics,
                        "regional_rmse_improvement": float(1 - metrics["regional_rmse"] / control_metrics["regional_rmse"]),
                        "pixel_rmse_improvement": float(1 - metrics["median_pixel_rmse"] / control_metrics["median_pixel_rmse"]),
                        "annual_correlation_change": float(metrics["regional_correlation"] - control_metrics["regional_correlation"]),
                        "lowpass_correlation_change": float(metrics["regional_lowpass_correlation_15y"] - control_metrics["regional_lowpass_correlation_15y"]),
                        "absolute_trend_error_change": float(abs(metrics["regional_trend_difference_per_century"]) - abs(control_metrics["regional_trend_difference_per_century"])),
                    })

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "robustness_candidate_scores.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "hma_temperature_robustness_curves.csv", index=False)
    aggregates = {candidate: candidate_aggregates(scores, candidate) for candidate in scores.candidate.unique()}

    old = pd.read_csv(round19_comparison)
    old = old[old.variable == "tas"].sort_values(["observation", "region"]).reset_index(drop=True)
    primary = scores[scores.candidate == "primary_four_family"].sort_values(["observation", "region"]).reset_index(drop=True)
    closure_columns = ["regional_rmse_improvement", "pixel_rmse_improvement", "lowpass_correlation_change"]
    closure_errors = {
        column: float(np.max(np.abs(primary[column].to_numpy(float) - old[column].to_numpy(float))))
        for column in closure_columns
    }
    closure_passes = bool(all(value <= 1e-12 for value in closure_errors.values()))
    leave_out = [name for name in aggregates if name.startswith("leave_out_")]
    positive_leave_out_count = int(sum(aggregates[name]["median_regional_rmse_improvement"] > 0 for name in leave_out))
    worst_leave_out_median = float(min(aggregates[name]["median_regional_rmse_improvement"] for name in leave_out))
    robustness_gates = {
        "round19_primary_metric_closure": closure_passes,
        "five_family_passes_round19_gates": bool(aggregates["five_family_hgao_o_separate"]["passes_round19_gates"]),
        "at_least_three_of_four_leave_out_medians_positive": positive_leave_out_count >= 3,
        "no_leave_out_median_regression_over_2pct": worst_leave_out_median >= -0.02,
    }
    passes = bool(all(robustness_gates.values()))
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "primary_round19_metric_closure_max_abs": closure_errors,
        "candidate_aggregates": aggregates,
        "positive_leave_one_family_out_count": positive_leave_out_count,
        "worst_leave_one_family_out_median_improvement": worst_leave_out_median,
        "robustness_gates": robustness_gates,
        "passes_robustness": passes,
        "decision": "permit_versioned_allowed_period_temperature_reference_field" if passes else "retain_family_balance_as_modern_period_sensitivity_only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    order = list(aggregates)
    values = [scores[scores.candidate == name].regional_rmse_improvement.to_numpy(float) * 100 for name in order]
    axis.boxplot(values, labels=[name.replace("_", "\n") for name in order], showfliers=False)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.axhline(2, color="#9b2226", linestyle="--", linewidth=1)
    axis.set_ylabel("Regional RMSE improvement over member equal (%)")
    axis.set_title("Temperature family-balance robustness, 1940-2000")
    fig.savefig(output / "temperature_family_balance_robustness.png", dpi=180)
    plt.close(fig)

    lines = ["# Temperature Family-Balance Robustness", ""]
    for candidate, item in aggregates.items():
        lines.append(f"- {candidate}: median RMSE improvement `{item['median_regional_rmse_improvement']:.3%}`, positive pairs `{item['positive_pair_fraction']:.1%}`, Round 19 gates `{str(item['passes_round19_gates']).lower()}`.")
    lines.extend([
        "",
        f"- Positive leave-one-family-out candidates: `{positive_leave_out_count}/4`.",
        f"- Worst leave-one-family-out median improvement: `{worst_leave_out_median:.3%}`.",
        f"- Robustness passes: `{str(passes).lower()}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: base.state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
