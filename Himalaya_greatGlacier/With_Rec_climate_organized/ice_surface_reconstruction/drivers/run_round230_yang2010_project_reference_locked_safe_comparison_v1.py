from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_230_yang2010_project_reference_locked_safe_comparison_v1"
ROUND229 = LOOP / "round_229_yang2010_11y_curve_digitization_v1"
YANG_CSV = ROUND229 / "yang2010_red_11y_fft_allowed_digitized.csv"
ROUND216_DRIVER = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
    / "run_round216_haizi_annual_temperature_locked_safe_comparison_v1.py"
)
CONTRACT = Path(__file__).with_name(
    "round230_yang2010_project_reference_locked_safe_comparison_v1_contract.md"
)
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
SITE_LAT = 29.3666667
SITE_LON = 94.2666667
LOCKED_START = 1921
LOCKED_END = 1939


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"size": stat.st_size, "sha256": sha256(path)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def centered_mean_sealed(values: np.ndarray, years: np.ndarray, width: int = 11) -> np.ndarray:
    if width % 2 != 1:
        raise ValueError("Width must be odd")
    output = np.full(values.shape, np.nan, dtype=float)
    half = width // 2
    finite = np.isfinite(values)
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    ends = np.flatnonzero(finite & ~np.r_[finite[1:], False])
    for start, end in zip(starts, ends):
        segment = values[start : end + 1]
        if segment.size < width:
            continue
        kernel = np.ones(width, dtype=float) / width
        smoothed = np.convolve(segment, kernel, mode="valid")
        output[start + half : end - half + 1] = smoothed
    if np.any(np.isfinite(output[(years >= LOCKED_START) & (years <= LOCKED_END)])):
        raise RuntimeError("Smoothing crossed into the sealed interval")
    return output


def recenter(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    if int(np.isfinite(values[baseline]).sum()) != 30:
        raise RuntimeError("Incomplete 1961-1990 baseline")
    return values - float(np.mean(values[baseline]))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def trend(values: np.ndarray, years: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end) & np.isfinite(values)
    if int(selected.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[selected], values[selected], 1)[0] * 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [YANG_CSV, ROUND229 / "summary.json", ROUND216_DRIVER, CONTRACT, FACTS]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    before = {str(path): state(path) for path in required}
    facts_hash = before[str(FACTS)]["sha256"]

    round216 = load_module(ROUND216_DRIVER, "round216_for_round230")
    helper = round216.load_round212()
    round206 = helper.load_round206()
    required.extend([round206.LINEAGE, round206.REFERENCE, round206.HMA_MASK])
    for path in required[-3:]:
        before[str(path)] = state(path)

    years = np.arange(1000, 2001)
    locked = (years >= LOCKED_START) & (years <= LOCKED_END)
    reference, latitudes, longitudes, reference_hma_mask = round206.read_reference_allowed(
        round206.REFERENCE, years
    )
    hma_mask = np.load(round206.HMA_MASK).astype(bool)
    if not np.array_equal(hma_mask, reference_hma_mask):
        raise RuntimeError("HMA mask mismatch")

    lineage = pd.read_csv(round206.LINEAGE)
    members = lineage[lineage.variable == "tas"].copy()
    expected_members = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if len(members) != 6 or set(members.member) != expected_members:
        raise RuntimeError("Temperature member lineage changed")
    round23 = round206.load_round23_module()
    project_members = []
    member_states = {}
    for row in members.itertuples(index=False):
        path = Path(row.source_tif)
        if sha256(path) != row.source_tif_sha256:
            raise RuntimeError(f"Member hash mismatch: {row.member}")
        field, _ = round23.read_allowed_tif(path, years, latitudes, longitudes)
        project_members.append(field)
        member_states[row.member] = state(path)
    project = round23.finite_mean(project_members)
    if np.any(np.isfinite(project[locked])) or np.any(np.isfinite(reference[locked])):
        raise RuntimeError("Sealed project/reference raster bands became visible")

    box_mask = (
        (latitudes[:, None] >= SITE_LAT - 1.0)
        & (latitudes[:, None] <= SITE_LAT + 1.0)
        & (longitudes[None, :] >= SITE_LON - 1.0)
        & (longitudes[None, :] <= SITE_LON + 1.0)
    )
    if int(box_mask.sum()) < 4:
        raise RuntimeError("Source-centered box has too few cells")

    project_raw = round206.area_weighted_series(project, box_mask, latitudes)
    reference_raw = round206.area_weighted_series(reference, box_mask, latitudes)
    project_curve = recenter(centered_mean_sealed(project_raw, years), years)
    reference_curve = recenter(centered_mean_sealed(reference_raw, years), years)

    yang_table = pd.read_csv(YANG_CSV)
    expected_columns = {
        "year",
        "x_pixel",
        "red_y_pixel_median",
        "red_pixel_count",
        "red_y_pixel_iqr",
        "whole_series_departure_degC",
        "anomaly_1961_1990_degC",
    }
    if set(yang_table.columns) != expected_columns:
        raise RuntimeError("Unexpected Round 229 curve columns")
    expected_yang_years = set(range(1377, 1900)) | set(range(1940, 1999))
    if set(yang_table.year.astype(int)) != expected_yang_years:
        raise RuntimeError("Unexpected Yang allowed-year coverage")
    yang_curve = np.full(years.shape, np.nan, dtype=float)
    year_to_index = {int(year): index for index, year in enumerate(years)}
    for row in yang_table.itertuples(index=False):
        yang_curve[year_to_index[int(row.year)]] = float(row.anomaly_1961_1990_degC)
    if np.any(np.isfinite(yang_curve[(years >= 1900) & (years <= 1939)])):
        raise RuntimeError("Yang curve exposed the masked interval")

    common = np.isfinite(yang_curve) & np.isfinite(project_curve) & np.isfinite(reference_curve)
    comparison_rows = []
    for index in np.flatnonzero(common):
        comparison_rows.append(
            {
                "year": int(years[index]),
                "yang_jan_jun_11y_anomaly_degC": yang_curve[index],
                "project_annual_11y_anomaly_degC": project_curve[index],
                "reference_annual_11y_anomaly_degC": reference_curve[index],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(OUT / "common_allowed_11y_curves.csv", index=False)

    window_rows = []
    for start in range(1500, 1900, 50):
        end = start + 49
        selected = (years >= start) & (years <= end)
        curves = {
            "yang_jan_jun_11y_anomaly_degC": yang_curve,
            "project_annual_11y_anomaly_degC": project_curve,
            "reference_annual_11y_anomaly_degC": reference_curve,
        }
        row = {"start_year": start, "end_year": end, "center_year": start + 24.5}
        for name, values in curves.items():
            if int(np.isfinite(values[selected]).sum()) != 50:
                raise RuntimeError(f"Incomplete historical window: {name} {start}-{end}")
            row[name] = float(np.mean(values[selected]))
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(OUT / "temperature_50year_windows_1500_1899.csv", index=False)

    yang_windows = windows["yang_jan_jun_11y_anomaly_degC"].to_numpy(float)
    project_windows = windows["project_annual_11y_anomaly_degC"].to_numpy(float)
    reference_windows = windows["reference_annual_11y_anomaly_degC"].to_numpy(float)
    project_errors = project_windows - yang_windows
    reference_errors = reference_windows - yang_windows
    metrics = {
        "yang_project_window_correlation": correlation(yang_windows, project_windows),
        "yang_reference_window_correlation": correlation(yang_windows, reference_windows),
        "project_window_rmse_degC": float(np.sqrt(np.mean(project_errors**2))),
        "reference_window_rmse_degC": float(np.sqrt(np.mean(reference_errors**2))),
        "project_window_mae_degC": float(np.mean(np.abs(project_errors))),
        "reference_window_mae_degC": float(np.mean(np.abs(reference_errors))),
        "yang_negative_window_count": int(np.sum(yang_windows < 0)),
        "project_negative_window_count": int(np.sum(project_windows < 0)),
        "reference_negative_window_count": int(np.sum(reference_windows < 0)),
        "median_yang_anomaly_1500_1899_degC": float(np.median(yang_windows)),
        "median_project_anomaly_1500_1899_degC": float(np.median(project_windows)),
        "median_reference_anomaly_1500_1899_degC": float(np.median(reference_windows)),
        "median_project_minus_yang_1500_1899_degC": float(np.median(project_errors)),
        "median_reference_minus_yang_1500_1899_degC": float(np.median(reference_errors)),
        "yang_project_adjacent_window_sign_agreement": int(
            np.sum(np.sign(np.diff(yang_windows)) == np.sign(np.diff(project_windows)))
        ),
        "yang_reference_adjacent_window_sign_agreement": int(
            np.sum(np.sign(np.diff(yang_windows)) == np.sign(np.diff(reference_windows)))
        ),
        "yang_trend_1850_1899_degC_per_century": trend(yang_curve, years, 1850, 1899),
        "project_trend_1850_1899_degC_per_century": trend(project_curve, years, 1850, 1899),
        "reference_trend_1850_1899_degC_per_century": trend(reference_curve, years, 1850, 1899),
    }
    supports_warm_bias = (
        metrics["yang_negative_window_count"] >= 6
        and metrics["median_project_minus_yang_1500_1899_degC"] >= 0.15
    )
    closer = (
        "project"
        if metrics["project_window_rmse_degC"] < metrics["reference_window_rmse_degC"]
        else "four_family_reference"
    )
    decision = (
        "yang_supports_frozen_local_warm_bias_gate"
        if supports_warm_bias
        else "yang_does_not_support_frozen_local_warm_bias_gate"
    )

    colors = {"yang": "#202020", "project": "#B43C38", "reference": "#167A72"}
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 8.1),
        sharex=False,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.24},
    )
    axes[0].plot(years, yang_curve, color=colors["yang"], linewidth=2.2, label="Yang 2010 Jan-Jun (11-year FFT)")
    axes[0].plot(years, project_curve, color=colors["project"], linewidth=1.8, label="Project local annual (11-year mean)")
    axes[0].plot(years, reference_curve, color=colors["reference"], linewidth=1.8, label="Four-family local annual (11-year mean)")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].axvspan(1900, 1939, color="#E5E5E5", alpha=0.95, linewidth=0)
    axes[0].set_xlim(1377, 1998)
    axes[0].set_ylabel("Anomaly (degC; 1961-1990)")
    axes[0].set_title("Southeast Tibet: independent Yang trajectory versus project and reference")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.7)

    centers = windows.center_year.to_numpy(float)
    for values, color, label, marker in (
        (yang_windows, colors["yang"], "Yang 2010", "o"),
        (project_windows, colors["project"], "Project local", "s"),
        (reference_windows, colors["reference"], "Four-family local", "^"),
    ):
        axes[1].plot(centers, values, color=color, marker=marker, linewidth=1.9, label=label)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xlim(1490, 1910)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("50-year anomaly (degC)")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[1].legend(frameon=False, ncol=3, loc="lower left")
    figure.text(
        0.99,
        0.01,
        "Gray gap: 1900-1939 excluded. Yang is Jan-Jun and digitized; project/reference are annual.",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.1)
    figure.savefig(OUT / "yang_project_reference_curves_1377_1998_locked_safe.png", dpi=220)
    plt.close(figure)

    after = {str(path): state(path) for path in required}
    if before != after:
        raise RuntimeError("A read-only source changed")
    if sha256(FACTS) != facts_hash:
        raise RuntimeError("Facts file changed")
    summary = {
        "round": 230,
        "decision": decision,
        "closer_historical_trajectory_by_50year_rmse": closer,
        "target_boundary": {
            "yang": "January-June mean temperature, digitized published 11-year FFT low-pass",
            "project_reference": "calendar annual mean temperature, centered 11-year arithmetic mean",
            "native_numeric_series": False,
            "hma_wide_correction_permitted": False,
        },
        "spatial_support": {
            "site_latitude": SITE_LAT,
            "site_longitude": SITE_LON,
            "box_cell_count": int(box_mask.sum()),
            "project_member_count": len(project_members),
        },
        "comparison_support": {
            "common_visible_segments": [[1377, 1899], [1940, 1998]],
            "historical_windows": 8,
            "historical_years": [1500, 1899],
            "excluded_interval": [1900, 1939],
        },
        "metrics": metrics,
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "production_changed": False,
        "facts_hash_unchanged": True,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "project_member_sources": member_states},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = f"""# Round 230 Result

## Decision

`{decision}`

Yang is below its 1961-1990 baseline in
`{metrics['yang_negative_window_count']}/8` historical 50-year windows. The
project and four-family local curves are below zero in
`{metrics['project_negative_window_count']}/8` and
`{metrics['reference_negative_window_count']}/8` windows. Median project-minus-
Yang difference is `{metrics['median_project_minus_yang_1500_1899_degC']:+.3f}`
degC.

The 50-year-window RMSE is `{metrics['project_window_rmse_degC']:.3f}` degC for
the project and `{metrics['reference_window_rmse_degC']:.3f}` degC for the
four-family reference; `{closer}` is closer on this metric. Seasonal-target and
digitization differences prevent this local comparison from authorizing a
production correction. No locked-period or Jionglacuo value was accessed and
no production file changed.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
