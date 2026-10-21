"""Coverage-corrected wrapper for external low-frequency validation v2."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import validate_temperature_external_low_frequency_v1 as base


base.OUTPUT = (
    base.RESULTS / "climate_reconstruction_temperature_external_low_frequency_validation_v2"
)


def evaluate_curve(
    candidate: pd.DataFrame,
    external: pd.DataFrame,
    column: str,
    label: str,
) -> tuple[dict[str, float | str | int], pd.DataFrame, pd.DataFrame]:
    merged = candidate.merge(external[["year", column]], on="year", how="inner")
    merged = merged.rename(columns={column: "external"})
    merged["project_z"] = base.zscore_modern(merged, "project")
    merged["candidate_z"] = base.zscore_modern(merged, "candidate")
    merged["external_z"] = base.zscore_modern(merged, "external")

    window_rows = []
    for start, end in base.WINDOWS:
        subset = merged.loc[merged["year"].between(start, end)]
        counts = subset[["project_z", "candidate_z", "external_z"]].notna().sum()
        if counts.min() < 35:
            continue
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
    if len(windows) < 6:
        raise ValueError(f"{label} retains only {len(windows)} eligible windows")

    project_rmse = float(np.sqrt(np.mean((windows.project_z - windows.external_z) ** 2)))
    candidate_rmse = float(
        np.sqrt(np.mean((windows.candidate_z - windows.external_z) ** 2))
    )
    improvement = (project_rmse - candidate_rmse) / project_rmse
    paleo = merged.loc[merged.year.between(1500, 1899)]
    summary = {
        "record": label,
        "eligible_window_count": int(len(windows)),
        "first_eligible_window": int(windows.window_start.min()),
        "last_eligible_window": int(windows.window_start.max()),
        "modern_start": base.MODERN_START,
        "modern_end": int(
            min(base.MODERN_END, merged.loc[merged.external.notna(), "year"].max())
        ),
        "modern_n": int(
            merged.loc[
                merged["year"].between(base.MODERN_START, base.MODERN_END), "external"
            ].notna().sum()
        ),
        "project_50year_rmse_z": project_rmse,
        "candidate_50year_rmse_z": candidate_rmse,
        "rmse_improvement_fraction": improvement,
        "improves_at_least_10pct": bool(improvement >= 0.10 - 1e-12),
        "worsens_more_than_5pct": bool(improvement < -0.05 - 1e-12),
        "annual_project_correlation_1500_1899": float(
            paleo[["project_z", "external_z"]].corr().iloc[0, 1]
        ),
        "annual_candidate_correlation_1500_1899": float(
            paleo[["candidate_z", "external_z"]].corr().iloc[0, 1]
        ),
    }
    return summary, windows, merged


if __name__ == "__main__":
    base.evaluate_curve = evaluate_curve
    base.main()
