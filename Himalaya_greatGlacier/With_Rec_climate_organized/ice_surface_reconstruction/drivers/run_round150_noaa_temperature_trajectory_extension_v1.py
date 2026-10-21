"""Extend the frozen external trajectory benchmark with two NOAA products."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIODS = {
    "prelock_1500_1920": (1500, 1920),
    "early_1901_1920": (1901, 1920),
    "modern_1940_2000": (1940, 2000),
}


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3 or np.nanstd(left[finite]) <= 1e-12 or np.nanstd(right[finite]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(years) & np.isfinite(values)
    if int(finite.sum()) < 3:
        return np.nan
    return float(np.polyfit(years[finite], values[finite], 1)[0] * 100.0)


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=1)


def metrics(name: str, external: pd.Series, current: pd.Series) -> list[dict[str, object]]:
    rows = []
    for period, (start, end) in PERIODS.items():
        frame = pd.concat(
            [external.rename("external"), current.rename("current")], axis=1
        ).loc[start:end].dropna()
        years = frame.index.to_numpy(dtype=float)
        ext = frame.external.to_numpy(dtype=float)
        cur = frame.current.to_numpy(dtype=float)
        lowpass = np.nan
        if len(frame) >= 31:
            smooth = frame.rolling(31, center=True, min_periods=21).mean().dropna()
            lowpass = correlation(smooth.external.to_numpy(), smooth.current.to_numpy())
        ext_slope = slope_per_century(years, ext)
        cur_slope = slope_per_century(years, cur)
        rows.append({
            "benchmark": name,
            "climate_variable": "temperature",
            "region": "east_ge90",
            "period": period,
            "sample_count": int(len(frame)),
            "annual_resolution": True,
            "raw_phase_correlation": correlation(ext, cur),
            "lowpass_31y_correlation": lowpass,
            "benchmark_trend_per_century_native_units": ext_slope,
            "current_trend_per_century_native_units": cur_slope,
            "trend_sign_agreement": bool(np.sign(ext_slope) == np.sign(cur_slope)),
        })
    return rows


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    round122 = validation / "climate_reconstruction_external_low_frequency_benchmark_20260818_v2"
    round149 = validation / "climate_reconstruction_improvement_loop_v2/round_149_noaa_temperature_candidate_identity_v1"
    output = validation / "climate_reconstruction_noaa_temperature_trajectory_extension_20260819_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round150_noaa_temperature_trajectory_extension_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round122_summary": round122 / "summary.json",
        "round122_current": round122 / "current_regional_curves_allowed_years.csv",
        "round122_external": round122 / "external_benchmark_allowed_values.csv",
        "round149_summary": round149 / "summary.json",
        "tibet2015": round149 / "tibet2015_allowed.csv",
        "bhutan2015": round149 / "bhutan2015_allowed.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r149 = json.loads(paths["round149_summary"].read_text(encoding="utf-8"))
    if r149["locked_1921_1939_values_decoded"] != 0 or r149["jionglacuo_external_validation_file_opened"]:
        raise RuntimeError("round149-boundary-failed")

    current_table = pd.read_csv(paths["round122_current"])
    current_table = current_table[
        (current_table.climate_variable == "temperature") & (current_table.region == "east_ge90")
    ]
    current = pd.Series(current_table.value.to_numpy(dtype=float), index=current_table.year.to_numpy(dtype=int)).sort_index()
    if ((current.index >= 1921) & (current.index <= 1939)).any():
        raise RuntimeError("locked-year-in-current-export")

    tibet = pd.read_csv(paths["tibet2015"])
    bhutan = pd.read_csv(paths["bhutan2015"])
    for frame in (tibet, bhutan):
        if ((frame.year >= 1921) & (frame.year <= 1939)).any():
            raise RuntimeError("locked-year-in-candidate-export")
    external = {
        "tibet2015_etp_jja_temperature_anomaly": pd.Series(
            tibet.value_1.replace(-99.99, np.nan).to_numpy(dtype=float), index=tibet.year.to_numpy(dtype=int)
        ).sort_index(),
        "bhutan2015_jja_temperature": pd.Series(
            bhutan.value_1.replace(-99.99, np.nan).to_numpy(dtype=float), index=bhutan.year.to_numpy(dtype=int)
        ).sort_index(),
    }

    metric_rows = []
    for name, series in external.items():
        metric_rows.extend(metrics(name, series, current))
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output / "new_temperature_benchmark_metrics.csv", index=False)

    prior_external = pd.read_csv(paths["round122_external"])
    prior_temperature = prior_external[
        (prior_external.climate_variable == "temperature")
        & (prior_external.region == "east_ge90")
        & prior_external.benchmark.isin([
            "eastern_tibet_814y_jjas_temperature",
            "sun2026_setp_jja_tmax",
            "tibet2019_netp_annual_temperature",
            "tibet2019_setp_jja_temperature",
        ])
    ]
    curves = {"current_east_ge90": current}
    for name, group in prior_temperature.groupby("benchmark"):
        curves[name] = pd.Series(group.value.to_numpy(dtype=float), index=group.year.to_numpy(dtype=int)).sort_index()
    curves.update(external)

    pair_rows = []
    for period, (start, end) in PERIODS.items():
        for left_name, left in curves.items():
            for right_name, right in curves.items():
                if left_name >= right_name:
                    continue
                frame = pd.concat([left.rename("left"), right.rename("right")], axis=1).loc[start:end].dropna()
                smooth = frame.rolling(31, center=True, min_periods=21).mean().dropna()
                pair_rows.append({
                    "period": period,
                    "left": left_name,
                    "right": right_name,
                    "sample_count": int(len(frame)),
                    "raw_correlation": correlation(frame.left.to_numpy(), frame.right.to_numpy()),
                    "lowpass_31y_correlation": correlation(smooth.left.to_numpy(), smooth.right.to_numpy()) if len(smooth) >= 3 else np.nan,
                    "left_trend_per_century": slope_per_century(frame.index.to_numpy(dtype=float), frame.left.to_numpy()),
                    "right_trend_per_century": slope_per_century(frame.index.to_numpy(dtype=float), frame.right.to_numpy()),
                })
    pd.DataFrame(pair_rows).to_csv(output / "temperature_benchmark_pairwise_metrics.csv", index=False)

    trend_rows = []
    for name, curve in curves.items():
        for period, (start, end) in PERIODS.items():
            subset = curve.loc[start:end].dropna()
            trend_rows.append({
                "series": name,
                "period": period,
                "sample_count": int(len(subset)),
                "trend_per_century_native_units": slope_per_century(subset.index.to_numpy(dtype=float), subset.to_numpy(dtype=float)),
            })
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(output / "all_temperature_benchmark_trends.csv", index=False)

    colors = {
        "current_east_ge90": "#111111",
        "tibet2015_etp_jja_temperature_anomaly": "#D1495B",
        "bhutan2015_jja_temperature": "#00798C",
        "sun2026_setp_jja_tmax": "#EDA93B",
        "tibet2019_netp_annual_temperature": "#5E548E",
        "tibet2019_setp_jja_temperature": "#2A9D8F",
        "eastern_tibet_814y_jjas_temperature": "#6C757D",
    }
    labels = {
        "current_east_ge90": "Current east HMA",
        "tibet2015_etp_jja_temperature_anomaly": "ETP 2015 JJA",
        "bhutan2015_jja_temperature": "Bhutan 2015 JJA",
        "sun2026_setp_jja_tmax": "SETP 2026 JJA Tmax",
        "tibet2019_netp_annual_temperature": "NETP 2019 annual",
        "tibet2019_setp_jja_temperature": "SETP 2019 JJA",
        "eastern_tibet_814y_jjas_temperature": "East Tibet 814y JJAS",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    windows = [(1500, 1920, True, "1500-1920, 31-year low pass"), (1890, 1920, False, "1890-1920, annual"), (1940, 2000, True, "1940-2000, 31-year low pass")]
    for axis, (start, end, smooth, title) in zip(axes, windows):
        for name, curve in curves.items():
            subset = curve.loc[start:end]
            if smooth:
                subset = subset.rolling(31, center=True, min_periods=21).mean()
            subset = standardize(subset.dropna())
            axis.plot(subset.index, subset.values, color=colors[name], linewidth=2.2 if name == "current_east_ge90" else 1.25, alpha=1.0 if name in {"current_east_ge90", "tibet2015_etp_jja_temperature_anomaly", "bhutan2015_jja_temperature"} else 0.72, label=labels[name])
        axis.axhline(0, color="#B8B8B8", linewidth=0.8)
        axis.set_xlim(start, end)
        axis.set_ylabel("standardized")
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0].legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Year")
    fig.savefig(output / "temperature_external_trajectory_extension.png", dpi=220)
    plt.close(fig)

    early = trends[trends.period == "early_1901_1920"].copy()
    current_sign = float(np.sign(early.loc[early.series == "current_east_ge90", "trend_per_century_native_units"].iloc[0]))
    external_early = early[early.series != "current_east_ge90"]
    agreement_count = int((np.sign(external_early.trend_per_century_native_units) == current_sign).sum())
    summary = {
        "experiment_id": "round150_noaa_temperature_trajectory_extension_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "new_metrics": metric_frame.to_dict(orient="records"),
        "early_1901_1920_external_series_count": int(len(external_early)),
        "early_1901_1920_trend_sign_agreement_with_current_count": agreement_count,
        "early_1901_1920_trend_sign_agreement_with_current_fraction": float(agreement_count / len(external_early)),
        "scientific_role": "source_overlapping_external_trajectory_benchmarks",
        "locked_1921_1939_accessed": False,
        "jionglacuo_external_validation_file_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    after = {name: fingerprint(path) for name, path in paths.items()}
    summary["source_fingerprints_unchanged"] = before == after
    if not summary["source_fingerprints_unchanged"]:
        raise RuntimeError("source-fingerprint-changed")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    lines = [
        "# Round 150 NOAA Temperature Trajectory Extension",
        "",
        f"- Early 1901--1920 external trend-sign agreement with current: `{agreement_count}/{len(external_early)}`.",
    ]
    for row in metric_rows:
        lines.append(
            f"- `{row['benchmark']}` / `{row['period']}`: raw r=`{row['raw_phase_correlation']:.3f}`, 31y r=`{row['lowpass_31y_correlation']:.3f}`, trend agreement=`{str(row['trend_sign_agreement']).lower()}`."
        )
    lines.extend(["- Locked 1921--1939 accessed: `false`.", "- EnKF/OGGM run: `false`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
