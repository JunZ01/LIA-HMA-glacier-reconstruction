"""Corrected smoothing-parity western HMA transition comparison."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import digitize_validate_temperature_hindu_kush_transition_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Correct v1 smoothing parity: each series must have exactly one 11-year low-pass operation.\n"
        "- Published Kumrat and Upper Indus red curves are already 11-year low-pass and are not smoothed again.\n"
        "- Project and reference annual curves receive one centered 11-year mean.\n"
        "- Digitize and score only 1812-1900; do not access later phases or the lockbox.\n",
        encoding="utf-8",
    )
    inputs = {"figure5": base.IMAGE, "regional": base.REGIONAL, "upper_indus": base.UPPER_INDUS}
    before = {name: base.sha256(path) for name, path in inputs.items()}
    rgb = np.asarray(Image.open(base.IMAGE).convert("RGB"))
    axes = base.detect_axes(rgb)
    branches: dict[str, np.ndarray] = {}
    missing: dict[str, int] = {}
    for mode in ["strict", "medium", "broad"]:
        branches[mode], missing[mode] = base.digitize(base.red_mask(rgb, mode), axes)
    sensitivity = max(
        float(np.nanmax(np.abs(branches["medium"] - branches["strict"]))),
        float(np.nanmax(np.abs(branches["medium"] - branches["broad"]))),
    )

    regional = base.safe_csv_prefix(base.REGIONAL).set_index("year").loc[base.YEARS]
    project = base.centered_11(regional["west_lt80_project_six_member_degC"].to_numpy(float))
    reference = base.centered_11(regional["west_lt80_reference_family_central_degC"].to_numpy(float))
    upper_frame = base.safe_csv_prefix(base.UPPER_INDUS).set_index("year").loc[base.YEARS]
    upper = upper_frame["upper_indus_medium_z"].to_numpy(float)
    standardized = {
        "kumrat_hindu_kush": base.baseline_standardize(branches["medium"]),
        "upper_indus": base.baseline_standardize(upper),
        "project_west": base.baseline_standardize(project),
        "reference_west": base.baseline_standardize(reference),
    }
    late_changes = {name: base.period_mean(values, base.LATE) for name, values in standardized.items()}
    independent = np.array([late_changes["kumrat_hindu_kush"], late_changes["upper_indus"]])

    output = pd.DataFrame({
        "year": base.YEARS,
        "kumrat_strict_degC": branches["strict"],
        "kumrat_medium_degC": branches["medium"],
        "kumrat_broad_degC": branches["broad"],
        **{f"{name}_baseline_standardized": values for name, values in standardized.items()},
    })
    output.to_csv(OUTPUT / "western_temperature_transition_1812_1900.csv", index=False)
    pd.DataFrame([
        {"series": name, "late_1881_1900_minus_1812_1860_sd": value}
        for name, value in late_changes.items()
    ]).to_csv(OUTPUT / "western_temperature_transition_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    styles = {
        "kumrat_hindu_kush": ("#B92D27", 2.4, "Kumrat Hindu Kush (Mar-Apr Tmin)"),
        "upper_indus": ("#E28C2B", 2.2, "Upper Indus MXD"),
        "project_west": ("#276FBF", 2.1, "Project west (<80E)"),
        "reference_west": ("#111111", 2.0, "Reference west (<80E)"),
    }
    for name, values in standardized.items():
        color, width, label = styles[name]
        ax.plot(base.YEARS, values, color=color, linewidth=width, label=label)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    ax.axvspan(base.LATE[0], base.LATE[1], color="#D9D9D9", alpha=0.35)
    ax.set(xlim=(1812, 1900), xlabel="Year", ylabel="Temperature relative to 1812-1860 (baseline SD)", title="Western HMA transition with one low-pass operation per curve")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "western_temperature_transition_validation.png", dpi=260)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in inputs.items()}
    summary = {
        "status": "complete",
        "supersedes": "climate_reconstruction_temperature_hindu_kush_transition_validation_v1",
        "correction": "Removed unintended second 11-year smoothing of the published Upper Indus red curve.",
        "late_1881_1900_minus_1812_1860_sd": late_changes,
        "independent_consensus": {
            "median_sd": float(np.median(independent)),
            "range_sd": [float(np.min(independent)), float(np.max(independent))],
            "both_support_positive_transition": bool(np.all(independent > 0.15)),
        },
        "maximum_mask_sensitivity_degC": sensitivity,
        "digitization_missing_years_before_interpolation": missing,
        "input_hashes_unchanged": before == after,
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
