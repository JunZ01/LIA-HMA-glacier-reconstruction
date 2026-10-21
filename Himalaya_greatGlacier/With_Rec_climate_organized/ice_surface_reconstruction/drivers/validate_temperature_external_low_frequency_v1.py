"""Blind external low-frequency check of the frozen HMA temperature candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = ROOT / "validation_data/sources/noaa_hma_temperature_low_frequency_candidates_v1"
CANDIDATE = (
    RESULTS
    / "climate_reconstruction_temperature_hma_reference_anchored_nested_timescale_v1"
    / "temperature_hma_nested_timescale_candidate_1000_2000.csv"
)
OUTPUT = RESULTS / "climate_reconstruction_temperature_external_low_frequency_validation_v1"

LOCKED_START = 1921
LOCKED_END = 1939
WINDOWS = [(year, year + 49) for year in range(1500, 1900, 50)]
MODERN_START = 1940
MODERN_END = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_or_nan(value: str) -> float:
    value = value.strip()
    if not value or value.upper() in {"NA", "NAN", "-9999"}:
        return np.nan
    return float(value)


def read_year_rows(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Read row-oriented text while rejecting locked years before conversion."""
    header = None
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.startswith("#") or not raw.strip():
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if header is None:
                header = fields
                missing = set(required_columns) - set(header)
                if missing:
                    raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
                continue
            year_token = fields[0].strip()
            try:
                year = int(float(year_token))
            except ValueError:
                continue
            if LOCKED_START <= year <= LOCKED_END:
                continue
            values = dict(zip(header, fields))
            row = {"year": year}
            for column in required_columns:
                if column == header[0]:
                    continue
                row[column] = numeric_or_nan(values.get(column, ""))
            rows.append(row)
    if header is None:
        raise ValueError(f"No data header found in {path}")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def read_candidate(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            year = int(raw["year"])
            if LOCKED_START <= year <= LOCKED_END:
                continue
            rows.append(
                {
                    "year": year,
                    "project": float(raw["project_hma_degC"]),
                    "candidate": float(raw["nested_timescale_candidate_hma_degC"]),
                }
            )
    return pd.DataFrame(rows)


def zscore_modern(frame: pd.DataFrame, column: str) -> pd.Series:
    modern = frame.loc[
        frame["year"].between(MODERN_START, MODERN_END), column
    ].dropna()
    if len(modern) < 30 or modern.std(ddof=0) <= 0:
        raise ValueError(f"Insufficient modern normalization values for {column}")
    return (frame[column] - modern.mean()) / modern.std(ddof=0)


def evaluate_curve(
    candidate: pd.DataFrame,
    external: pd.DataFrame,
    column: str,
    label: str,
) -> tuple[dict[str, float | str | int], pd.DataFrame, pd.DataFrame]:
    merged = candidate.merge(external[["year", column]], on="year", how="inner")
    merged = merged.rename(columns={column: "external"})
    merged["project_z"] = zscore_modern(merged, "project")
    merged["candidate_z"] = zscore_modern(merged, "candidate")
    merged["external_z"] = zscore_modern(merged, "external")

    window_rows = []
    for start, end in WINDOWS:
        subset = merged.loc[merged["year"].between(start, end)]
        counts = subset[["project_z", "candidate_z", "external_z"]].notna().sum()
        if counts.min() < 35:
            raise ValueError(f"{label} has fewer than 35 values in {start}-{end}")
        window_rows.append(
            {
                "record": label,
                "window_start": start,
                "window_end": end,
                "project_z": subset["project_z"].mean(),
                "candidate_z": subset["candidate_z"].mean(),
                "external_z": subset["external_z"].mean(),
                "n_external": int(counts["external_z"]),
            }
        )
    windows = pd.DataFrame(window_rows)
    project_rmse = float(np.sqrt(np.mean((windows.project_z - windows.external_z) ** 2)))
    candidate_rmse = float(
        np.sqrt(np.mean((windows.candidate_z - windows.external_z) ** 2))
    )
    improvement = (project_rmse - candidate_rmse) / project_rmse
    summary = {
        "record": label,
        "modern_start": MODERN_START,
        "modern_end": int(min(MODERN_END, merged.loc[merged.external.notna(), "year"].max())),
        "modern_n": int(
            merged.loc[merged["year"].between(MODERN_START, MODERN_END), "external"].notna().sum()
        ),
        "project_50year_rmse_z": project_rmse,
        "candidate_50year_rmse_z": candidate_rmse,
        "rmse_improvement_fraction": improvement,
        "improves_at_least_10pct": bool(improvement >= 0.10 - 1e-12),
        "worsens_more_than_5pct": bool(improvement < -0.05 - 1e-12),
        "annual_project_correlation_1500_1899": float(
            merged.loc[merged.year.between(1500, 1899), ["project_z", "external_z"]]
            .corr()
            .iloc[0, 1]
        ),
        "annual_candidate_correlation_1500_1899": float(
            merged.loc[merged.year.between(1500, 1899), ["candidate_z", "external_z"]]
            .corr()
            .iloc[0, 1]
        ),
    }
    return summary, windows, merged


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    if not contract.exists():
        raise FileNotFoundError("ROUND_CONTRACT.md must exist before numeric parsing")

    input_paths = {
        "candidate": CANDIDATE,
        "xu2019": SOURCE / "tibet2019temp.txt",
        "cook2003": SOURCE / "kathmandu_temp-noaa.txt",
        "contract": contract,
    }
    before_hashes = {name: sha256(path) for name, path in input_paths.items()}

    candidate = read_candidate(CANDIDATE)
    xu = read_year_rows(
        input_paths["xu2019"], ["age_CE", "temp_NETP", "temp_SETP"]
    )
    cook = read_year_rows(
        input_paths["cook2003"],
        ["age_CE", "FEB-JUN-ADJ", "OCT-FEB-ADJ"],
    )

    definitions = [
        (xu, "temp_NETP", "Xu2019_NETP_annual"),
        (xu, "temp_SETP", "Xu2019_SETP_JJA"),
        (cook, "FEB-JUN-ADJ", "Cook2003_Kathmandu_FebJun_ADJ"),
        (cook, "OCT-FEB-ADJ", "Cook2003_Kathmandu_OctFeb_ADJ"),
    ]

    summaries = []
    window_tables = []
    merged_curves: dict[str, pd.DataFrame] = {}
    for external, column, label in definitions:
        summary, windows, merged = evaluate_curve(candidate, external, column, label)
        summaries.append(summary)
        window_tables.append(windows)
        merged_curves[label] = merged

    summary_table = pd.DataFrame(summaries)
    windows_table = pd.concat(window_tables, ignore_index=True)
    improvements = summary_table["rmse_improvement_fraction"].to_numpy(float)
    gates = {
        "at_least_three_of_four_improve_10pct": bool(np.sum(improvements >= 0.10) >= 3),
        "none_worsen_more_than_5pct": bool(np.all(improvements >= -0.05)),
        "median_improvement_at_least_10pct": bool(np.median(improvements) >= 0.10),
    }
    passes = bool(all(gates.values()))

    summary_table.to_csv(OUTPUT / "external_curve_metrics.csv", index=False)
    windows_table.to_csv(OUTPUT / "external_50year_window_means.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    axes = axes.ravel()
    for axis, (_, _, label) in zip(axes[:4], definitions):
        table = windows_table.loc[windows_table.record == label]
        x = (table.window_start + table.window_end) / 2
        axis.plot(x, table.external_z, color="#111111", marker="o", label="External")
        axis.plot(x, table.project_z, color="#d55e00", marker="s", label="Project")
        axis.plot(x, table.candidate_z, color="#0072b2", marker="^", label="Frozen correction")
        axis.axhline(0, color="#9a9a9a", linewidth=0.8)
        axis.set_title(label)
        axis.set_ylabel("Modern-standardized temperature")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)

    colors = ["#0072b2" if value >= 0 else "#d55e00" for value in improvements]
    axes[4].barh(summary_table.record, 100 * improvements, color=colors)
    axes[4].axvline(10, color="#111111", linestyle="--", linewidth=1)
    axes[4].axvline(0, color="#777777", linewidth=0.8)
    axes[4].set_xlabel("50-year RMSE improvement (%)")
    axes[4].set_title("Frozen correction versus project")
    axes[4].grid(axis="x", alpha=0.2)

    for label, merged in merged_curves.items():
        subset = merged.loc[merged.year.between(1500, 1899)]
        rolling = subset.set_index("year")["external_z"].rolling(31, center=True, min_periods=20).mean()
        axes[5].plot(rolling.index, rolling, linewidth=1.2, label=label)
    axes[5].set_title("External low-frequency curves")
    axes[5].set_xlabel("Year")
    axes[5].set_ylabel("31-year mean standardized temperature")
    axes[5].grid(alpha=0.2)
    axes[5].legend(fontsize=7)

    figure_path = OUTPUT / "temperature_external_low_frequency_validation.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    after_hashes = {name: sha256(path) for name, path in input_paths.items()}
    manifest = {
        "inputs": {
            name: {"path": str(path), "sha256_before": before_hashes[name], "sha256_after": after_hashes[name]}
            for name, path in input_paths.items()
        },
        "all_input_hashes_unchanged": before_hashes == after_hashes,
        "excluded_files": [
            str(SOURCE / "kathmandu_temp-supp.txt"),
            str(SOURCE / "zhang2018easia-temp-grid.txt"),
            str(SOURCE / "zhang2018easia-temp.txt"),
        ],
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    result = {
        "status": "complete",
        "scope": "external_consistency_not_independence",
        "candidate": "31-year low-pass lambda=0.75 frozen before this experiment",
        "paleo_windows": [f"{a}-{b}" for a, b in WINDOWS],
        "modern_normalization_period": f"{MODERN_START}-{MODERN_END}",
        "median_rmse_improvement_fraction": float(np.median(improvements)),
        "curves_improving_at_least_10pct": int(np.sum(improvements >= 0.10)),
        "gates": gates,
        "passes_external_consistency_gates": passes,
        "locked_1921_1939_numeric_values_parsed": False,
        "phase_1901_1920_used_for_tuning_or_scoring": False,
        "precontract_supplemental_display_incident_recorded": True,
        "spatial_field_release_authorized": False,
        "production_release_authorized": False,
        "decision": (
            "retain_candidate_for_independence_review"
            if passes
            else "external_curves_do_not_support_frozen_candidate"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = (
        "# External Low-Frequency Temperature Validation\n\n"
        f"- External-consistency gates passed: `{str(passes).lower()}`.\n"
        f"- Median 50-year RMSE improvement: `{100*np.median(improvements):.2f}%`.\n"
        f"- Curves improving by at least 10%: `{int(np.sum(improvements >= 0.10))}/4`.\n"
        "- This is not yet independent validation; source/proxy overlap remains under audit.\n"
        "- No spatial or production correction is authorized.\n"
        "- The pre-contract locked-period display incident is declared in ROUND_CONTRACT.md.\n"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
