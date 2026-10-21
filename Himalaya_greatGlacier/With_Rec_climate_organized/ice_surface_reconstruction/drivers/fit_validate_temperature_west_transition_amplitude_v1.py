"""Fit a west-HMA transition-amplitude sensitivity and test it out of sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
INPUT = RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v1/western_temperature_transition_1812_1900.csv"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_transition_amplitude_oos_v1"
FIT_PERIOD = (1861, 1900)
ANCHOR_PERIOD = (1841, 1860)
LATE_PERIOD = (1881, 1900)
RAMP_END = 1880


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    return (years >= period[0]) & (years <= period[1])


def rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))


def corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Fit only to Upper Indus over 1861-1900; Kumrat Hindu Kush is untouched holdout evidence.\n"
        "- Keep the project curve exactly unchanged through 1860.\n"
        "- Ramp the amplitude adjustment linearly from 1861 through 1880.\n"
        "- Fit one bounded attenuation parameter analytically, not by outcome-filtered scanning.\n"
        "- Candidate remains regional diagnostic because proxy seasons and target variables differ.\n"
        "- Do not access 1901-1920 or locked 1921-1939 data.\n",
        encoding="utf-8",
    )
    before = sha256(INPUT)
    frame = pd.read_csv(INPUT)
    if frame.year.min() != 1812 or frame.year.max() != 1900:
        raise RuntimeError("Input is not the frozen safe 1812-1900 interval")
    years = frame.year.to_numpy(int)
    project = frame.project_west_baseline_standardized.to_numpy(float)
    reference = frame.reference_west_baseline_standardized.to_numpy(float)
    train = frame.upper_indus_baseline_standardized.to_numpy(float)
    holdout = frame.kumrat_hindu_kush_baseline_standardized.to_numpy(float)

    anchor_mask = selected(years, ANCHOR_PERIOD)
    fit_mask = selected(years, FIT_PERIOD)
    late_mask = selected(years, LATE_PERIOD)
    anchor = float(np.mean(project[anchor_mask]))
    ramp = np.clip((years - 1860.0) / (RAMP_END - 1860.0), 0.0, 1.0)
    adjustment_basis = ramp * (project - anchor)
    numerator = float(np.dot(adjustment_basis[fit_mask], (project - train)[fit_mask]))
    denominator = float(np.dot(adjustment_basis[fit_mask], adjustment_basis[fit_mask]))
    attenuation = float(np.clip(numerator / denominator, 0.0, 0.50))
    final_amplitude_scale = 1.0 - attenuation
    candidate = project - attenuation * adjustment_basis
    if not np.array_equal(candidate[years <= 1860], project[years <= 1860]):
        raise RuntimeError("Pre-1861 closure failed")

    metrics = {
        "fitted_on_upper_indus": {
            "project_rmse": rmse(project, train, fit_mask),
            "candidate_rmse": rmse(candidate, train, fit_mask),
            "reference_rmse": rmse(reference, train, fit_mask),
            "project_correlation": corr(project, train, fit_mask),
            "candidate_correlation": corr(candidate, train, fit_mask),
            "reference_correlation": corr(reference, train, fit_mask),
        },
        "heldout_kumrat": {
            "project_rmse": rmse(project, holdout, fit_mask),
            "candidate_rmse": rmse(candidate, holdout, fit_mask),
            "reference_rmse": rmse(reference, holdout, fit_mask),
            "project_correlation": corr(project, holdout, fit_mask),
            "candidate_correlation": corr(candidate, holdout, fit_mask),
            "reference_correlation": corr(reference, holdout, fit_mask),
        },
    }
    for values in metrics.values():
        values["candidate_rmse_improvement_vs_project_fraction"] = 1.0 - values["candidate_rmse"] / values["project_rmse"]

    late_means = {
        "upper_indus": float(np.mean(train[late_mask])),
        "kumrat_holdout": float(np.mean(holdout[late_mask])),
        "project": float(np.mean(project[late_mask])),
        "candidate": float(np.mean(candidate[late_mask])),
        "reference": float(np.mean(reference[late_mask])),
    }
    maximum_correction = float(np.max(np.abs(candidate - project)))
    maximum_correction_step = float(np.max(np.abs(np.diff(candidate - project))))
    gates = {
        "heldout_rmse_improves": metrics["heldout_kumrat"]["candidate_rmse"] < metrics["heldout_kumrat"]["project_rmse"],
        "heldout_late_mean_error_improves": abs(late_means["candidate"] - late_means["kumrat_holdout"]) < abs(late_means["project"] - late_means["kumrat_holdout"]),
        "positive_warming_preserved": late_means["candidate"] > 0.15,
        "pre1861_exact_closure": bool(np.array_equal(candidate[years <= 1860], project[years <= 1860])),
        "maximum_correction_step_below_0p15_sd": maximum_correction_step < 0.15,
    }

    out = frame.copy()
    out["west_transition_ramp"] = ramp
    out["west_diagnostic_candidate_baseline_standardized"] = candidate
    out["candidate_minus_project_baseline_sd"] = candidate - project
    out.to_csv(OUTPUT / "west_transition_amplitude_candidate_1812_1900.csv", index=False)

    metric_rows = []
    for role, values in metrics.items():
        metric_rows.append({"evidence_role": role, **values})
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "fit_holdout_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.plot(years, train, color="#E28C2B", linewidth=2.0, label="Upper Indus (fit only)")
    ax.plot(years, holdout, color="#B92D27", linewidth=2.2, label="Kumrat Hindu Kush (holdout)")
    ax.plot(years, project, color="#777777", linewidth=1.8, label="Original project west")
    ax.plot(years, candidate, color="#276FBF", linewidth=2.5, label="Attenuated west diagnostic")
    ax.plot(years, reference, color="#111111", linewidth=1.5, linestyle="--", label="Reference west")
    ax.axvline(1860, color="#888888", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    ax.set(
        xlim=(1812, 1900),
        xlabel="Year",
        ylabel="Temperature relative to 1812-1860 (baseline SD)",
        title="West-HMA transition amplitude: fit on Upper Indus, validate on Kumrat",
    )
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "west_transition_amplitude_oos_validation.png", dpi=260)
    plt.close(fig)

    summary = {
        "status": "complete",
        "fitted_attenuation_fraction": attenuation,
        "final_post1880_amplitude_scale": final_amplitude_scale,
        "metrics": metrics,
        "late_1881_1900_means_baseline_sd": late_means,
        "maximum_absolute_correction_baseline_sd": maximum_correction,
        "maximum_adjacent_correction_step_baseline_sd": maximum_correction_step,
        "gates": gates,
        "all_regional_diagnostic_gates_pass": bool(all(gates.values())),
        "candidate_status": "regional_diagnostic_only_not_production",
        "reason_not_production": "Only two western proxy reconstructions with different seasons/variables; no full-field validation.",
        "input_hash_unchanged": before == sha256(INPUT),
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"input": str(INPUT), "sha256": before}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
