"""Test Kathmandu seasonal conflict across published chronology methods."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import validate_temperature_external_low_frequency_v1 as base
import validate_temperature_external_low_frequency_v2 as coverage


OUTPUT = (
    base.RESULTS
    / "climate_reconstruction_temperature_kathmandu_standardization_sensitivity_v1"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    if not contract.exists():
        raise FileNotFoundError("ROUND_CONTRACT.md must predate numeric parsing")

    cook_path = base.SOURCE / "kathmandu_temp-noaa.txt"
    paths = {"candidate": base.CANDIDATE, "cook2003": cook_path, "contract": contract}
    before = {name: base.sha256(path) for name, path in paths.items()}

    columns = [
        "age_CE",
        "FEB-JUN",
        "FEB-JUN-RCS",
        "FEB-JUN-ADJ",
        "OCT-FEB",
        "OCT-FEB-RCS",
        "OCT-FEB-ADJ",
    ]
    candidate = base.read_candidate(base.CANDIDATE)
    cook = base.read_year_rows(cook_path, columns)
    definitions = [
        ("FEB-JUN", "FebJun_hybrid", "Feb-Jun", "Hybrid"),
        ("FEB-JUN-RCS", "FebJun_RCS", "Feb-Jun", "RCS"),
        ("FEB-JUN-ADJ", "FebJun_ADJ", "Feb-Jun", "ADJ"),
        ("OCT-FEB", "OctFeb_hybrid", "Oct-Feb", "Hybrid"),
        ("OCT-FEB-RCS", "OctFeb_RCS", "Oct-Feb", "RCS"),
        ("OCT-FEB-ADJ", "OctFeb_ADJ", "Oct-Feb", "ADJ"),
    ]

    summaries = []
    windows_all = []
    for column, label, season, method in definitions:
        summary, windows, _ = coverage.evaluate_curve(candidate, cook, column, label)
        summary["season"] = season
        summary["standardization"] = method
        summaries.append(summary)
        windows["season"] = season
        windows["standardization"] = method
        windows_all.append(windows)

    metrics = pd.DataFrame(summaries)
    windows = pd.concat(windows_all, ignore_index=True)
    metrics.to_csv(OUTPUT / "kathmandu_standardization_metrics.csv", index=False)
    windows.to_csv(OUTPUT / "kathmandu_standardization_50year_windows.csv", index=False)

    feb = metrics.loc[metrics.season == "Feb-Jun", "rmse_improvement_fraction"].to_numpy(float)
    octfeb = metrics.loc[metrics.season == "Oct-Feb", "rmse_improvement_fraction"].to_numpy(float)
    robust_seasonal = bool(np.all(feb < -0.05) and np.all(octfeb > 0.05))
    mixed_within = bool(
        (np.any(feb > 0) and np.any(feb < 0))
        or (np.any(octfeb > 0) and np.any(octfeb < 0))
    )
    adj_only = bool(
        np.sign(feb[2]) != np.sign(feb[0])
        and np.sign(feb[2]) != np.sign(feb[1])
        and np.sign(octfeb[2]) != np.sign(octfeb[0])
        and np.sign(octfeb[2]) != np.sign(octfeb[1])
    )
    if robust_seasonal:
        classification = "robust_seasonal_conflict"
    elif adj_only:
        classification = "low_frequency_splice_sensitive"
    elif mixed_within:
        classification = "standardization_sensitive"
    else:
        classification = "same_direction_without_frozen_threshold_separation"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, season in zip(axes, ["Feb-Jun", "Oct-Feb"]):
        table = metrics.loc[metrics.season == season]
        values = 100 * table.rmse_improvement_fraction.to_numpy(float)
        axis.bar(table.standardization, values, color=["#0072b2" if v >= 0 else "#d55e00" for v in values])
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.axhline(5, color="#111111", linestyle="--", linewidth=0.8)
        axis.axhline(-5, color="#111111", linestyle="--", linewidth=0.8)
        axis.set_title(f"Kathmandu {season}")
        axis.set_ylabel("Frozen-candidate RMSE improvement (%)")
        axis.grid(axis="y", alpha=0.2)
    figure = OUTPUT / "kathmandu_standardization_sensitivity.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    manifest = {
        "inputs": {
            name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]}
            for name in paths
        },
        "all_input_hashes_unchanged": before == after,
        "kathmandu_supplement_excluded": True,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = {
        "status": "complete",
        "classification": classification,
        "robust_seasonal_conflict": robust_seasonal,
        "standardization_sensitive": mixed_within,
        "adj_only_differs": adj_only,
        "feb_jun_improvements": {
            method: float(value) for method, value in zip(["hybrid", "RCS", "ADJ"], feb)
        },
        "oct_feb_improvements": {
            method: float(value) for method, value in zip(["hybrid", "RCS", "ADJ"], octfeb)
        },
        "locked_1921_1939_numeric_values_parsed": False,
        "phase_1901_1920_used": False,
        "candidate_or_gate_changed": False,
        "production_release_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Kathmandu Standardization Sensitivity\n\n"
        f"- Classification: `{classification}`.\n"
        f"- February-June improvements: `{100*feb[0]:.1f}%`, `{100*feb[1]:.1f}%`, `{100*feb[2]:.1f}%`.\n"
        f"- October-February improvements: `{100*octfeb[0]:.1f}%`, `{100*octfeb[1]:.1f}%`, `{100*octfeb[2]:.1f}%`.\n"
        "- Diagnostic only; no candidate or gate was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
