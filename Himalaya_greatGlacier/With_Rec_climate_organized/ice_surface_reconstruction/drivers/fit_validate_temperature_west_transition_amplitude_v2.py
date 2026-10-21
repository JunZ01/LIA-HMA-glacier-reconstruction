"""Fit a low-frequency western transition attenuation and validate it out of sample."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fit_validate_temperature_west_transition_amplitude_v1 as base


INPUT = base.RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2/western_temperature_transition_1812_1900.csv"
OUTPUT = base.RESULTS / "climate_reconstruction_temperature_west_transition_amplitude_oos_v2"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Supersede v1 because the upstream Upper Indus curve had been smoothed twice.\n"
        "- Fit only to Upper Indus over 1861-1900; Kumrat remains untouched holdout evidence.\n"
        "- Keep the project curve unchanged through 1860 and ramp adjustment through 1880.\n"
        "- Smooth the correction basis once over 11 years because this is a low-frequency transition diagnostic.\n"
        "- Do not access 1901-1920 or locked 1921-1939 data.\n",
        encoding="utf-8",
    )
    before = base.sha256(INPUT)
    frame = pd.read_csv(INPUT)
    if (int(frame.year.min()), int(frame.year.max())) != (1812, 1900):
        raise RuntimeError("Input interval is not the frozen safe prefix")
    years = frame.year.to_numpy(int)
    project = frame.project_west_baseline_standardized.to_numpy(float)
    reference = frame.reference_west_baseline_standardized.to_numpy(float)
    train = frame.upper_indus_baseline_standardized.to_numpy(float)
    holdout = frame.kumrat_hindu_kush_baseline_standardized.to_numpy(float)
    anchor_mask = base.selected(years, base.ANCHOR_PERIOD)
    fit_mask = base.selected(years, base.FIT_PERIOD)
    late_mask = base.selected(years, base.LATE_PERIOD)
    anchor = float(np.mean(project[anchor_mask]))
    ramp = np.clip((years - 1860.0) / (base.RAMP_END - 1860.0), 0.0, 1.0)
    raw_basis = ramp * (project - anchor)
    basis = pd.Series(raw_basis).rolling(11, center=True, min_periods=1).mean().to_numpy(float)
    basis[years <= 1860] = 0.0
    attenuation = float(np.clip(
        np.dot(basis[fit_mask], (project - train)[fit_mask]) / np.dot(basis[fit_mask], basis[fit_mask]),
        0.0,
        0.50,
    ))
    candidate = project - attenuation * basis
    if not np.array_equal(candidate[years <= 1860], project[years <= 1860]):
        raise RuntimeError("Pre-1861 closure failed")

    metrics: dict[str, dict[str, float]] = {}
    for role, target in [("fitted_on_upper_indus", train), ("heldout_kumrat", holdout)]:
        values = {
            "project_rmse": base.rmse(project, target, fit_mask),
            "candidate_rmse": base.rmse(candidate, target, fit_mask),
            "reference_rmse": base.rmse(reference, target, fit_mask),
            "project_correlation": base.corr(project, target, fit_mask),
            "candidate_correlation": base.corr(candidate, target, fit_mask),
            "reference_correlation": base.corr(reference, target, fit_mask),
        }
        values["candidate_rmse_improvement_vs_project_fraction"] = 1.0 - values["candidate_rmse"] / values["project_rmse"]
        values["candidate_correlation_change_vs_project"] = values["candidate_correlation"] - values["project_correlation"]
        metrics[role] = values

    late_means = {
        "upper_indus": float(np.mean(train[late_mask])),
        "kumrat_holdout": float(np.mean(holdout[late_mask])),
        "project": float(np.mean(project[late_mask])),
        "candidate": float(np.mean(candidate[late_mask])),
        "reference": float(np.mean(reference[late_mask])),
    }
    correction = candidate - project
    gates = {
        "heldout_rmse_improves": metrics["heldout_kumrat"]["candidate_rmse"] < metrics["heldout_kumrat"]["project_rmse"],
        "heldout_late_mean_error_improves": abs(late_means["candidate"] - late_means["kumrat_holdout"]) < abs(late_means["project"] - late_means["kumrat_holdout"]),
        "heldout_correlation_regression_below_0p05": metrics["heldout_kumrat"]["candidate_correlation_change_vs_project"] > -0.05,
        "positive_warming_preserved": late_means["candidate"] > 0.15,
        "pre1861_exact_closure": bool(np.array_equal(candidate[years <= 1860], project[years <= 1860])),
        "maximum_correction_step_below_0p10_sd": float(np.max(np.abs(np.diff(correction)))) < 0.10,
    }

    out = frame.copy()
    out["west_transition_ramp"] = ramp
    out["smoothed_correction_basis"] = basis
    out["west_diagnostic_candidate_baseline_standardized"] = candidate
    out["candidate_minus_project_baseline_sd"] = correction
    out.to_csv(OUTPUT / "west_transition_amplitude_candidate_1812_1900.csv", index=False)
    pd.DataFrame([{"evidence_role": role, **values} for role, values in metrics.items()]).to_csv(OUTPUT / "fit_holdout_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    for values, color, width, label, style in [
        (train, "#E28C2B", 2.0, "Upper Indus (fit only)", "-"),
        (holdout, "#B92D27", 2.2, "Kumrat Hindu Kush (holdout)", "-"),
        (project, "#777777", 1.8, "Original project west", "-"),
        (candidate, "#276FBF", 2.5, "Attenuated west diagnostic", "-"),
        (reference, "#111111", 1.5, "Reference west", "--"),
    ]:
        ax.plot(years, values, color=color, linewidth=width, label=label, linestyle=style)
    ax.axvline(1860, color="#888888", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    ax.set(xlim=(1812, 1900), xlabel="Year", ylabel="Temperature relative to 1812-1860 (baseline SD)", title="Western transition amplitude: Upper Indus fit, Kumrat holdout")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "west_transition_amplitude_oos_validation.png", dpi=260)
    plt.close(fig)

    summary = {
        "status": "complete",
        "supersedes": "climate_reconstruction_temperature_west_transition_amplitude_oos_v1",
        "fitted_attenuation_fraction": attenuation,
        "final_low_frequency_amplitude_scale": 1.0 - attenuation,
        "metrics": metrics,
        "late_1881_1900_means_baseline_sd": late_means,
        "maximum_absolute_correction_baseline_sd": float(np.max(np.abs(correction))),
        "maximum_adjacent_correction_step_baseline_sd": float(np.max(np.abs(np.diff(correction)))),
        "gates": gates,
        "all_regional_diagnostic_gates_pass": bool(all(gates.values())),
        "candidate_status": "regional_diagnostic_only_not_production",
        "reason_not_production": "Two western records with differing target seasons cannot validate a full HMA annual field.",
        "input_hash_unchanged": before == base.sha256(INPUT),
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"input": str(INPUT), "sha256": before}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
