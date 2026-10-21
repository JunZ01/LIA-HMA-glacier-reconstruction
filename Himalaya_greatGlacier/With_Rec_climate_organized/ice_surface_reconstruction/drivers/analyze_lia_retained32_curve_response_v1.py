"""Read-only multivariate response analysis for the retained 32 LIA curves."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SCREEN = RESULTS / "lia_191_accepted_58_curve_shape_screen_v1"
REGISTRY = RESULTS / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
EVIDENCE = RESULTS / "lia_191_accepted55_evidence_index_v1/accepted55_evidence_index.csv"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUTPUT = RESULTS / "lia_191_retained_32_curve_analysis_v1"

TARGET_IDS = [
    130, 214, 252, 518, 991, 1077, 1301, 1326, 1524, 1546, 1616,
    2226, 2259, 2266, 2328, 2337, 2546, 2569, 2595, 2622, 2645,
    2729, 2748, 2806, 2815, 2827, 2902, 2928, 2939, 3010, 4319, 4565,
]

YEARS_SOURCE = np.arange(1499, 2001, dtype=int)
YEARS = np.arange(1500, 2001, dtype=int)
RHO_ICE = 900.0
SMOOTH_YEARS = 21
LAGS = np.arange(0, 61, dtype=int)
EPISODE_MIN_YEARS = 20
EPISODE_DEADBAND_SD = 0.25

# Four preregistered strict forward-run screening cores.
STAGE_MIN_EPISODES = 3
STAGE_MIN_COVERAGE = 0.45
ALIGNMENT_MIN_CORRELATION = 0.30
ALIGNMENT_MIN_SIGN_CONSISTENCY = 0.55
AMPLITUDE_MIN_LENGTH_SEPARATION_RATIO = 0.50
AMPLITUDE_MIN_AREA_FRACTION = 0.01
AMPLITUDE_MIN_VOLUME_FRACTION = 0.05
NEAR_MONOTONIC_MAX_UP_SHARE = 0.15
NEAR_MONOTONIC_MAX_REBOUND_FRACTION = 0.20
TERMINAL_MAX_DOWNWARD_SHARE = 0.55
TERMINAL_MIN_ROBUST_Z = 3.0
HF_MIN_RMS_RANGE = 0.08
HF_MIN_TV_SHARE = 0.50
HF_MIN_ZERO_CROSSINGS = 20

DIRECT_MB_NAMES = {
    "climatic_mb_kg_m2_yr", "surface_mb_kg_m2_yr", "specific_mb_kg_m2_yr",
    "annual_surface_mb_kg_m2_yr", "mass_balance_kg_m2_yr",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def reflect_boxcar(values: np.ndarray, window: int = SMOOTH_YEARS) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("smoothing window must be odd")
    half = window // 2
    padded = np.pad(np.asarray(values, dtype=float), half, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def safe_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source(metric: dict[str, str], evidence: dict[int, dict[str, str]]) -> tuple[Path, str]:
    system_id = int(metric["system_id"])
    if metric["route"] == "frozen-legacy":
        return FROZEN / f"system_{system_id}" / "timeseries.csv", "observation_length_m"
    reproduction_root = metric.get("reproduction_root", "").strip()
    if not reproduction_root:
        reproduction_root = evidence[system_id]["reproduction_root"].strip()
    return ROOT / reproduction_root / "historical_1470_2000.csv", "length_m"


def source_metadata(source: Path, route: str) -> dict[str, object]:
    if route == "frozen-legacy":
        meta = safe_json(source.parent / "summary.json")
    else:
        meta = safe_json(source.parent / "historical_summary.json")
    climate = meta.get("climate", {}) if isinstance(meta.get("climate"), dict) else {}
    return {
        "source_status": str(meta.get("status", "not-explicit")),
        "experiment_label": str(meta.get("experiment", "not-explicit")),
        "oggm_version": str(meta.get("oggm_version", "not-explicit")),
        "climate_source_label": str(climate.get("source", "not-explicit")),
    }


def load_source(source: Path, length_key: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    frame = pd.read_csv(source)
    if "phase" in frame.columns:
        frame = frame[frame["phase"].astype(str) == "forward"].copy()
    frame["year_int"] = np.rint(frame["year"].astype(float)).astype(int)
    frame = frame[(frame["year_int"] >= 1470) & (frame["year_int"] <= 2000)]
    frame = frame.sort_values("year_int").drop_duplicates("year_int", keep="last")
    actual_length_key = length_key if length_key in frame.columns else "length_m"
    needed = [actual_length_key, "area_m2", "volume_m3"]
    if any(name not in frame.columns for name in needed):
        raise RuntimeError(f"missing geometry field in {source}: {needed}")
    direct_mb_columns = sorted(DIRECT_MB_NAMES.intersection(frame.columns))
    original_years = frame["year_int"].to_numpy(float)
    if original_years.size < 2 or original_years[0] > 1499 or original_years[-1] < 2000:
        raise RuntimeError(f"source does not cover 1499-2000: {source}")
    values = frame[needed].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite geometry in {source}")
    arrays = {
        "length_km": np.interp(YEARS_SOURCE, original_years, values[:, 0]) / 1.0e3,
        "area_km2": np.interp(YEARS_SOURCE, original_years, values[:, 1]) / 1.0e6,
        "volume_km3": np.interp(YEARS_SOURCE, original_years, values[:, 2]) / 1.0e9,
    }
    diffs = np.diff(original_years)
    inventory = {
        "source_row_count_1470_2000": int(len(frame)),
        "source_year_start": int(original_years[0]),
        "source_year_end": int(original_years[-1]),
        "source_cadence_median_years": float(np.median(diffs)),
        "source_cadence_min_years": float(np.min(diffs)),
        "source_cadence_max_years": float(np.max(diffs)),
        "interpolated_to_annual": bool(np.any(diffs != 1)),
        "length_source_column": actual_length_key,
        "area_source_column": "area_m2",
        "volume_source_column": "volume_m3",
        "direct_surface_mb_available": bool(direct_mb_columns),
        "direct_surface_mb_columns": ";".join(direct_mb_columns),
    }
    return arrays, inventory


def lag_pair(driver: np.ndarray, response: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag == 0:
        return driver, response
    return driver[:-lag], response[lag:]


def lag_metrics(driver: np.ndarray, response: np.ndarray) -> dict[str, float | int]:
    correlations: list[float] = []
    consistencies: list[float] = []
    threshold = EPISODE_DEADBAND_SD * float(np.std(driver))
    for lag in LAGS:
        x, y = lag_pair(driver, response, int(lag))
        if np.std(x) == 0 or np.std(y) == 0:
            correlation = np.nan
        else:
            correlation = float(np.corrcoef(x, y)[0, 1])
        active = np.abs(x) >= threshold
        consistency = float(np.mean(np.sign(x[active]) == np.sign(y[active]))) if active.any() else np.nan
        correlations.append(correlation)
        consistencies.append(consistency)
    finite = np.isfinite(correlations)
    if not finite.any():
        return {"best_lag_years": -1, "best_correlation": np.nan, "sign_consistency": np.nan}
    best_index = int(np.nanargmax(correlations))
    return {
        "best_lag_years": int(LAGS[best_index]),
        "best_correlation": float(correlations[best_index]),
        "sign_consistency": float(consistencies[best_index]),
    }


def runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            if states[start] != 0:
                result.append((int(states[start]), start, index - 1))
            start = index
    return result


def persistent_episodes(values: np.ndarray) -> list[tuple[int, int, int]]:
    threshold = EPISODE_DEADBAND_SD * float(np.std(values))
    states = np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))
    # Bridge at most five near-zero years between equal signed regimes.
    for state in (-1, 1):
        indices = np.where(states == state)[0]
        for left, right in zip(indices[:-1], indices[1:]):
            if 1 < right - left <= 6 and np.all(states[left + 1 : right] == 0):
                states[left + 1 : right] = state
    return [episode for episode in runs(states) if episode[2] - episode[1] + 1 >= EPISODE_MIN_YEARS]


def total_variation(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def robust_terminal_z(values: np.ndarray, horizon: int = 20) -> float:
    changes = values[horizon:] - values[:-horizon]
    if len(changes) == 0:
        return 0.0
    center = float(np.median(changes))
    mad = float(np.median(np.abs(changes - center)))
    scale = max(1.4826 * mad, 1.0e-9)
    final = float(values[-1] - values[-1 - horizon])
    return (center - final) / scale


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def analyze_one(metric: dict[str, str], evidence_row: dict[str, str]) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    system_id = int(metric["system_id"])
    source, length_key = resolve_source(metric, {system_id: evidence_row})
    arrays, inventory = load_source(source, length_key)
    meta = source_metadata(source, metric["route"])

    area_m2 = arrays["area_km2"] * 1.0e6
    volume_m3 = arrays["volume_km3"] * 1.0e9
    mb = RHO_ICE * np.diff(volume_m3) / (0.5 * (area_m2[1:] + area_m2[:-1]))
    raw = {key: values[1:] for key, values in arrays.items()}
    smooth = {key: reflect_boxcar(values) for key, values in raw.items()}
    mb_smooth = reflect_boxcar(mb)
    rates = {key: np.gradient(np.log(values)) for key, values in smooth.items()}
    lag_a = lag_metrics(mb_smooth, rates["area_km2"])
    lag_l = lag_metrics(mb_smooth, rates["length_km"])
    lag_v = lag_metrics(mb_smooth, rates["volume_km3"])

    episodes = persistent_episodes(mb_smooth)
    episode_coverage = sum(end - start + 1 for _, start, end in episodes) / len(YEARS)
    signs = {sign for sign, _, _ in episodes}
    episode_years_positive = sum(end - start + 1 for sign, start, end in episodes if sign > 0)
    episode_years_negative = sum(end - start + 1 for sign, start, end in episodes if sign < 0)

    length = smooth["length_km"]
    differences = np.diff(length)
    upward = float(np.clip(differences, 0, None).sum())
    downward = float(np.clip(-differences, 0, None).sum())
    variation = upward + downward
    upward_share = upward / variation if variation else 0.0
    running_min = np.minimum.accumulate(length)
    curve_range = float(np.ptp(length))
    rebound = float(np.max(length - running_min))
    rebound_fraction = rebound / curve_range if curve_range else 0.0
    net_change = float(length[-1] - length[0])

    peaks, properties = find_peaks(length, prominence=max(0.10 * curve_range, 1.0e-9), distance=40)
    widths = peak_widths(length, peaks, rel_height=0.5)[0] if len(peaks) else np.array([])
    prominent_peak_count = int(len(peaks))
    broad_peak_years = float(np.max(widths)) if len(widths) else 0.0
    high_excursion_years = int(np.sum(length >= np.min(length) + 0.70 * curve_range)) if curve_range else 0

    residual = raw["length_km"] - length
    residual_rms_range = float(np.sqrt(np.mean(residual**2)) / max(curve_range, 1.0e-9))
    raw_tv = total_variation(raw["length_km"])
    residual_tv_share = total_variation(residual) / max(raw_tv, 1.0e-9)
    residual_zero_crossings = int(np.sum(np.signbit(residual[1:]) != np.signbit(residual[:-1])))
    down_total = float(np.clip(-np.diff(length), 0, None).sum())
    terminal_decline = max(0.0, float(length[-21] - length[-1]))
    terminal_down_share = terminal_decline / max(down_total, 1.0e-9)
    terminal_z = robust_terminal_z(length)

    separation = abs(float(metric["lia_km"]) - float(metric["rgi_km"]))
    length_range_sep_ratio = curve_range / max(separation, 1.0e-9)
    area_range_fraction = float(np.ptp(smooth["area_km2"]) / smooth["area_km2"][0])
    volume_range_fraction = float(np.ptp(smooth["volume_km3"]) / smooth["volume_km3"][0])

    near_monotonic = bool(net_change < 0 and upward_share <= NEAR_MONOTONIC_MAX_UP_SHARE and rebound_fraction <= NEAR_MONOTONIC_MAX_REBOUND_FRACTION)
    terminal_drop = bool(terminal_down_share > TERMINAL_MAX_DOWNWARD_SHARE and terminal_z > TERMINAL_MIN_ROBUST_Z)
    high_frequency = bool(residual_rms_range > HF_MIN_RMS_RANGE and residual_tv_share > HF_MIN_TV_SHARE and residual_zero_crossings >= HF_MIN_ZERO_CROSSINGS)
    stage_pass = bool(len(episodes) >= STAGE_MIN_EPISODES and signs == {-1, 1} and episode_coverage >= STAGE_MIN_COVERAGE)
    correlation_min = float(min(lag_a["best_correlation"], lag_l["best_correlation"]))
    consistency_min = float(min(lag_a["sign_consistency"], lag_l["sign_consistency"]))
    alignment_pass = bool(correlation_min >= ALIGNMENT_MIN_CORRELATION and consistency_min >= ALIGNMENT_MIN_SIGN_CONSISTENCY)
    amplitude_pass = bool(length_range_sep_ratio >= AMPLITUDE_MIN_LENGTH_SEPARATION_RATIO and area_range_fraction >= AMPLITUDE_MIN_AREA_FRACTION and volume_range_fraction >= AMPLITUDE_MIN_VOLUME_FRACTION)
    artifact_pass = bool(not near_monotonic and not terminal_drop and not high_frequency and net_change < 0)
    strict_pass = bool(stage_pass and alignment_pass and amplitude_pass and artifact_pass)
    core_pass_count = int(stage_pass) + int(alignment_pass) + int(amplitude_pass) + int(artifact_pass)
    isolated_peak = bool(prominent_peak_count <= 2 and high_excursion_years < 50 and (not stage_pass or not alignment_pass))
    net_growth = bool(net_change >= 0)

    if high_frequency:
        category = "numerical-high-frequency-oscillation"
    elif terminal_drop:
        category = "terminal-endpoint-drop"
    elif near_monotonic:
        category = "near-monotonic"
    elif net_growth:
        category = "net-growth-or-wrong-direction"
    elif isolated_peak:
        category = "one-or-two-isolated-peaks"
    elif strict_pass:
        category = "robust-multistage-lagged-response"
    elif stage_pass and (alignment_pass or core_pass_count >= 3):
        category = "plausible-multistage-uncertain"
    else:
        category = "mixed-low-confidence"

    uncertainty_reasons: list[str] = []
    if not stage_pass:
        uncertainty_reasons.append("persistent-stage-core-fail")
    if not alignment_pass:
        uncertainty_reasons.append("area-or-length-lag-alignment-fail")
    if not amplitude_pass:
        uncertainty_reasons.append("nontrivial-amplitude-core-fail")
    if not artifact_pass:
        uncertainty_reasons.append("artifact-or-net-direction-core-fail")
    if inventory["interpolated_to_annual"]:
        uncertainty_reasons.append("annual-values-include-interpolation")
    if not inventory["direct_surface_mb_available"]:
        uncertainty_reasons.append("mb-is-volume-change-equivalent-not-direct-surface-mb")

    record: dict[str, object] = {
        "system_id": system_id,
        "rgi_id": metric["rgi_id"],
        "model_family_registry_label": metric["route"],
        "route": metric["route"],
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "source_status": meta["source_status"],
        "experiment_label": meta["experiment_label"],
        "oggm_version": meta["oggm_version"],
        "climate_source_label": meta["climate_source_label"],
        "accepted_evidence_pass": evidence_row["accepted_evidence_pass"],
        "analysis_status": "read-only-provisional-curve-response-classification",
        **inventory,
        "mb_definition": "rho_ice*dV/mean_area; net volume-change-equivalent, not direct climatic surface MB",
        "smoothing_years": SMOOTH_YEARS,
        "lag_search_min_years": int(LAGS.min()),
        "lag_search_max_years": int(LAGS.max()),
        "persistent_episode_count": len(episodes),
        "persistent_positive_episode_count": sum(sign > 0 for sign, _, _ in episodes),
        "persistent_negative_episode_count": sum(sign < 0 for sign, _, _ in episodes),
        "persistent_episode_coverage_fraction": episode_coverage,
        "persistent_positive_years": episode_years_positive,
        "persistent_negative_years": episode_years_negative,
        "area_best_lag_years": lag_a["best_lag_years"],
        "area_best_lag_correlation": lag_a["best_correlation"],
        "area_sign_consistency": lag_a["sign_consistency"],
        "length_best_lag_years": lag_l["best_lag_years"],
        "length_best_lag_correlation": lag_l["best_correlation"],
        "length_sign_consistency": lag_l["sign_consistency"],
        "volume_best_lag_years_definitional": lag_v["best_lag_years"],
        "volume_best_lag_correlation_definitional": lag_v["best_correlation"],
        "response_alignment_correlation_min": correlation_min,
        "response_direction_consistency_min": consistency_min,
        "length_range_target_separation_ratio": length_range_sep_ratio,
        "area_range_fraction_of_1500": area_range_fraction,
        "volume_range_fraction_of_1500": volume_range_fraction,
        "smoothed_net_length_change_km": net_change,
        "smoothed_upward_variation_share": upward_share,
        "smoothed_rebound_range_fraction": rebound_fraction,
        "prominent_peak_count_auxiliary": prominent_peak_count,
        "broadest_peak_width_years_auxiliary": broad_peak_years,
        "high_excursion_years_auxiliary": high_excursion_years,
        "terminal_20yr_downward_variation_share": terminal_down_share,
        "terminal_20yr_robust_drop_z": terminal_z,
        "high_frequency_residual_rms_range_ratio": residual_rms_range,
        "high_frequency_residual_tv_share": residual_tv_share,
        "high_frequency_residual_zero_crossings": residual_zero_crossings,
        "flag_near_monotonic": near_monotonic,
        "flag_one_or_two_isolated_peaks": isolated_peak,
        "flag_terminal_endpoint_drop": terminal_drop,
        "flag_numerical_high_frequency_oscillation": high_frequency,
        "flag_net_growth_or_wrong_direction": net_growth,
        "core_stage_pass": stage_pass,
        "core_lag_alignment_pass": alignment_pass,
        "core_amplitude_pass": amplitude_pass,
        "core_artifact_free_pass": artifact_pass,
        "core_pass_count": core_pass_count,
        "preliminary_strict_forward_pass": strict_pass,
        "primary_classification": category,
        "uncertainty_reasons": ";".join(uncertainty_reasons),
    }

    series = pd.DataFrame({
        "system_id": system_id,
        "year": YEARS,
        "length_km": raw["length_km"],
        "area_km2": raw["area_km2"],
        "volume_km3": raw["volume_km3"],
        "net_volume_change_equivalent_mb_kg_m2_yr": mb,
        "length_21yr_km": smooth["length_km"],
        "area_21yr_km2": smooth["area_km2"],
        "volume_21yr_km3": smooth["volume_km3"],
        "net_mb_21yr_kg_m2_yr": mb_smooth,
        "dln_length_21yr_per_year": rates["length_km"],
        "dln_area_21yr_per_year": rates["area_km2"],
        "dln_volume_21yr_per_year": rates["volume_km3"],
    })
    manifest = {
        "system_id": system_id,
        "rgi_id": metric["rgi_id"],
        "model_family_registry_label": metric["route"],
        "route": metric["route"],
        "source": record["source"],
        "source_sha256": record["source_sha256"],
        "source_status": meta["source_status"],
        "experiment_label": meta["experiment_label"],
        "accepted_evidence_pass": evidence_row["accepted_evidence_pass"],
        "direct_surface_mb_available": inventory["direct_surface_mb_available"],
        "analysis_status": record["analysis_status"],
    }
    return record, series, manifest


def representative_ids(metrics: pd.DataFrame, maximum: int = 8) -> list[int]:
    preferred = [
        "robust-multistage-lagged-response", "plausible-multistage-uncertain",
        "one-or-two-isolated-peaks", "near-monotonic", "terminal-endpoint-drop",
        "numerical-high-frequency-oscillation", "net-growth-or-wrong-direction",
        "mixed-low-confidence",
    ]
    selected: list[int] = []
    keys = [
        "persistent_episode_coverage_fraction", "response_alignment_correlation_min",
        "length_range_target_separation_ratio", "smoothed_upward_variation_share",
    ]
    for category in preferred:
        group = metrics[metrics["primary_classification"] == category]
        if group.empty:
            continue
        z = group[keys].copy()
        center = z.median()
        scale = (z.quantile(0.75) - z.quantile(0.25)).replace(0, 1.0)
        distance = ((z - center) / scale).pow(2).sum(axis=1)
        selected.append(int(group.loc[distance.idxmin(), "system_id"]))
        if len(selected) >= maximum:
            break
    return selected


def plot_distributions(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    items = [
        ("persistent_episode_coverage_fraction", "Persistent stage coverage", STAGE_MIN_COVERAGE),
        ("response_alignment_correlation_min", "Min best-lag correlation (A, L)", ALIGNMENT_MIN_CORRELATION),
        ("response_direction_consistency_min", "Min lagged sign consistency (A, L)", ALIGNMENT_MIN_SIGN_CONSISTENCY),
        ("length_range_target_separation_ratio", "Length range / target separation", AMPLITUDE_MIN_LENGTH_SEPARATION_RATIO),
        ("terminal_20yr_downward_variation_share", "Final-20y share of downward variation", TERMINAL_MAX_DOWNWARD_SHARE),
    ]
    for axis, (key, title, threshold) in zip(axes.ravel()[:5], items):
        axis.hist(metrics[key], bins=10, color="#54778f", alpha=0.82, edgecolor="white")
        axis.axvline(threshold, color="#9b3d35", linestyle="--", linewidth=1.3)
        axis.set_title(title)
        axis.set_ylabel("Systems")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    category_counts = metrics["primary_classification"].value_counts().sort_values()
    axes.ravel()[5].barh(category_counts.index.str.replace("-", " "), category_counts.values, color="#c77755")
    axes.ravel()[5].set_title("Primary classification")
    axes.ravel()[5].set_xlabel("Systems")
    axes.ravel()[5].grid(axis="x", color="#dddddd", linewidth=0.6)
    fig.suptitle("Retained-32 response evidence distributions", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT / "overall_distribution_and_classification.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_typical(metrics: pd.DataFrame, series: pd.DataFrame) -> tuple[Path, list[int]]:
    selected = representative_ids(metrics)
    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(14.5, max(3.0 * rows, 4.0)), sharex=True, squeeze=False)
    by_id = metrics.set_index("system_id")
    colors = {"length_21yr_km": "#287271", "area_21yr_km2": "#d77a61", "volume_21yr_km3": "#3f5f8f"}
    for row, system_id in enumerate(selected):
        group = series[series["system_id"] == system_id].sort_values("year")
        metric = by_id.loc[system_id]
        left, right = axes[row]
        for key, label in [("length_21yr_km", "Length"), ("area_21yr_km2", "Area"), ("volume_21yr_km3", "Volume")]:
            values = group[key].to_numpy(float)
            left.plot(group["year"], 100.0 * (values / values[0] - 1.0), label=label, color=colors[key], linewidth=1.6)
        left.axhline(0, color="#555555", linewidth=0.7)
        left.set_ylabel("Change from 1500 (%)")
        left.set_title(f"{system_id}: {str(metric['primary_classification']).replace('-', ' ')}")
        left.legend(frameon=False, ncol=3, fontsize=8)
        mb = group["net_mb_21yr_kg_m2_yr"].to_numpy(float)
        right.axhline(0, color="#555555", linewidth=0.7)
        right.fill_between(group["year"], 0, mb, where=mb >= 0, color="#4f8b68", alpha=0.45)
        right.fill_between(group["year"], 0, mb, where=mb < 0, color="#b95f55", alpha=0.45)
        right.plot(group["year"], mb, color="#4a4a4a", linewidth=0.9)
        right.set_ylabel("Net volume-change MB\n(kg m$^{-2}$ yr$^{-1}$)")
        right.set_title(f"A lag {metric['area_best_lag_years']} y; L lag {metric['length_best_lag_years']} y")
        for axis in (left, right):
            axis.grid(axis="y", color="#dddddd", linewidth=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    for axis in axes.ravel():
        axis.set_xlim(1500, 2000)
    fig.suptitle("Representative curves by evidence-based primary class", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = OUTPUT / "typical_multivariate_curves.png"
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, selected


def plot_all_grid(metrics: pd.DataFrame, series: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(8, 4, figsize=(16, 24), sharex=True)
    by_id = metrics.set_index("system_id")
    for axis, system_id in zip(axes.ravel(), TARGET_IDS, strict=True):
        group = series[series["system_id"] == system_id].sort_values("year")
        metric = by_id.loc[system_id]
        for key, color in [("length_21yr_km", "#287271"), ("area_21yr_km2", "#d77a61"), ("volume_21yr_km3", "#3f5f8f")]:
            values = group[key].to_numpy(float)
            axis.plot(group["year"], 100.0 * (values / values[0] - 1.0), color=color, linewidth=1.0)
        axis.axhline(0, color="#666666", linewidth=0.5)
        mark = "PASS" if bool(metric["preliminary_strict_forward_pass"]) else "review"
        axis.set_title(f"{system_id} | {mark} | core {metric['core_pass_count']}/4", fontsize=8)
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.45)
        axis.tick_params(labelsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.supxlabel("Year")
    fig.supylabel("21-year-smoothed change from 1500 (%)")
    fig.suptitle("All retained-32 length (green), area (orange), volume (blue) curves", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0.025, 0.02, 1, 0.975))
    path = OUTPUT / "all_32_multivariate_geometry_grid.png"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def markdown_table(metrics: pd.DataFrame) -> str:
    header = "| ID | Route | Primary class | Core | Strict | A lag/corr | L lag/corr | Stage n/coverage | Amp Lsep/A/V | Flags |\n|---:|---|---|---:|---|---|---|---|---|---|\n"
    lines = [header]
    for _, row in metrics.sort_values("system_id").iterrows():
        flags = []
        for column, label in [
            ("flag_near_monotonic", "near-mono"), ("flag_one_or_two_isolated_peaks", "isolated"),
            ("flag_terminal_endpoint_drop", "terminal"), ("flag_numerical_high_frequency_oscillation", "HF"),
            ("flag_net_growth_or_wrong_direction", "net-growth"),
        ]:
            if bool(row[column]):
                flags.append(label)
        lines.append(
            f"| {int(row['system_id'])} | {row['route']} | {row['primary_classification']} | "
            f"{int(row['core_pass_count'])}/4 | {bool_text(bool(row['preliminary_strict_forward_pass']))} | "
            f"{int(row['area_best_lag_years'])}/{row['area_best_lag_correlation']:.2f} | "
            f"{int(row['length_best_lag_years'])}/{row['length_best_lag_correlation']:.2f} | "
            f"{int(row['persistent_episode_count'])}/{row['persistent_episode_coverage_fraction']:.2f} | "
            f"{row['length_range_target_separation_ratio']:.2f}/{row['area_range_fraction_of_1500']:.2%}/{row['volume_range_fraction_of_1500']:.2%} | "
            f"{', '.join(flags) if flags else 'none'} |\n"
        )
    return "".join(lines)


def write_report(metrics: pd.DataFrame, representative: Iterable[int]) -> Path:
    counts = metrics["primary_classification"].value_counts().sort_index()
    passed = metrics[metrics["preliminary_strict_forward_pass"]]["system_id"].astype(int).tolist()
    uncertain = metrics[(~metrics["preliminary_strict_forward_pass"]) & (metrics["core_pass_count"] >= 3)]["system_id"].astype(int).tolist()
    category_text = "\n".join(f"- `{name}`: {count}" for name, count in counts.items())
    report = f"""# Retained-32 multivariate curve-response analysis v1

## Scope and authority

This is a read-only research analysis of the 32 fixed systems listed in the requested shape-screen evidence. It does not change the formal accepted count (58), the controller, canonical state, any parameter matrix, or any simulation. Registry route labels, exact curve sources, upstream statuses, experiments, and hashes are retained in `source_manifest.csv` and `classification_table.csv`.

## Variable inventory

- Length, area, and volume exist for all 32 sources.
- Direct climatic/surface mass balance exists for **0/32** sources.
- `B(t)` here is explicitly the net volume-change-equivalent specific balance, `rho_ice * (V_t - V_t-1) / mean(A_t, A_t-1)`, with `rho_ice=900 kg m-3`. It contains dynamics and is not a directly saved surface-MB diagnostic.
- Consequently, B-to-volume-rate agreement is definitional and is reported only as an audit, not as independent response evidence. Classification uses B-to-area-rate and B-to-length-rate evidence.

## Preregistered common method

All series are placed on 1500-2000 annual coordinates. When a source is coarser than annual, interpolation status and native cadence are disclosed. A 21-year centered boxcar with reflected endpoints is applied uniformly to length, area, volume, and derived B. Lag analysis searches only nonnegative lags 0-60 years and selects the lag maximizing correlation between B(t) and each later geometry log-rate. Direction consistency is measured only where |B| is at least 0.25 of its smoothed standard deviation.

Persistent forcing stages are positive or negative B regimes lasting at least 20 years; up to five near-zero years may bridge equal-sign regimes. Peak counts are auxiliary only.

## Four-core strict forward-run screen

1. **Persistent multi-stage forcing:** at least 3 persistent regimes, both signs present, covering at least 45% of 1500-2000.
2. **Lagged directional response:** for both area and length, best-lag correlation at least 0.30 and sign consistency at least 0.55, with lag constrained to 0-60 years.
3. **Nontrivial amplitude:** smoothed length range at least 0.50 of the LIA-RGI target separation, area range at least 1% of its 1500 value, and volume range at least 5%.
4. **Artifact/direction gate:** net length retreat and no near-monotonic, terminal-endpoint-drop, or numerical high-frequency flag.

This is deliberately a compact prior/forward-screen, not a new acceptance rule. It must be validated prospectively before governing a parameter matrix.

## Result

- Preliminary strict passes: **{len(passed)}/32** — {', '.join(map(str, passed)) if passed else 'none'}.
- Uncertain members with 3/4 cores: **{len(uncertain)}** — {', '.join(map(str, uncertain)) if uncertain else 'none'}.
- Representative systems shown in the typical-curves figure: {', '.join(map(str, representative))}.

Primary-class distribution:

{category_text}

## Interpretation for a one-dimensional parameter matrix

Use strict-pass curves only as qualitative prior support for response timescale, sustained sign changes, and plausible normalized geometry amplitude. Use the uncertain set as held-out sensitivity/validation cases, especially for lag or interpolation dependence. Use artifact and wrong-direction classes as negative controls. Do not fit directly to peak counts, do not treat derived B as an independent volume constraint, and do not infer missing direct climatic MB. A future matrix should freeze this four-core screen before launch, report route/model-family strata separately, and test whether parameter proposals reproduce both the multistage geometry response and endpoint gates without changing the canonical accepted ledger.

## Per-system evidence table

{markdown_table(metrics)}
"""
    path = OUTPUT / "analysis_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    retained = pd.read_csv(SCREEN / "retained_32_systems.csv")
    retained_ids = retained["system_id"].astype(int).tolist()
    if retained_ids != TARGET_IDS:
        raise RuntimeError("requested ID order does not exactly match retained-32 evidence")
    registry = {int(row["system_id"]): row for row in read_csv_rows(REGISTRY)}
    evidence = {int(row["system_id"]): row for row in read_csv_rows(EVIDENCE)}
    if not all(system_id in registry and system_id in evidence for system_id in TARGET_IDS):
        raise RuntimeError("one or more requested systems are absent from registry/evidence index")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    series_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    for system_id in TARGET_IDS:
        record, series, manifest = analyze_one(registry[system_id], evidence[system_id])
        records.append(record)
        series_parts.append(series)
        manifests.append(manifest)
    metrics = pd.DataFrame(records)
    annual = pd.concat(series_parts, ignore_index=True)
    manifest = pd.DataFrame(manifests)

    if len(metrics) != 32 or metrics["system_id"].nunique() != 32:
        raise RuntimeError("classification table is not exactly 32 unique systems")
    if len(annual) != 32 * len(YEARS):
        raise RuntimeError("annual series table has unexpected row count")
    if metrics["direct_surface_mb_available"].any():
        raise RuntimeError("direct MB was detected; method/report must be reviewed before continuing")

    metrics.to_csv(OUTPUT / "classification_table.csv", index=False)
    annual.to_csv(OUTPUT / "annual_smoothed_multivariate_series.csv", index=False)
    manifest.to_csv(OUTPUT / "source_manifest.csv", index=False)
    (OUTPUT / "classification_table.md").write_text(markdown_table(metrics), encoding="utf-8")

    distribution_plot = plot_distributions(metrics)
    typical_plot, representative = plot_typical(metrics, annual)
    grid_plot = plot_all_grid(metrics, annual)
    report = write_report(metrics, representative)

    strict_ids = metrics.loc[metrics["preliminary_strict_forward_pass"], "system_id"].astype(int).tolist()
    uncertain_ids = metrics.loc[(~metrics["preliminary_strict_forward_pass"]) & (metrics["core_pass_count"] >= 3), "system_id"].astype(int).tolist()
    method = {
        "status": "read-only-provisional-curve-response-classification",
        "smoothing": {"kind": "centered-boxcar-reflected-endpoints", "window_years": SMOOTH_YEARS},
        "lag_analysis": {"allowed_lag_years": [0, 60], "direction": "B(t) to later dlnX/dt", "geometry_used_for_independent_evidence": ["area", "length"], "volume_relation": "definitional-audit-only"},
        "persistent_stage": {"deadband_standard_deviations": EPISODE_DEADBAND_SD, "minimum_duration_years": EPISODE_MIN_YEARS, "maximum_bridged_near_zero_gap_years": 5},
        "strict_four_core_thresholds": {
            "stage": {"minimum_episodes": STAGE_MIN_EPISODES, "both_signs": True, "minimum_coverage_fraction": STAGE_MIN_COVERAGE},
            "alignment": {"minimum_best_lag_correlation": ALIGNMENT_MIN_CORRELATION, "minimum_sign_consistency": ALIGNMENT_MIN_SIGN_CONSISTENCY},
            "amplitude": {"minimum_length_range_target_separation_ratio": AMPLITUDE_MIN_LENGTH_SEPARATION_RATIO, "minimum_area_range_fraction": AMPLITUDE_MIN_AREA_FRACTION, "minimum_volume_range_fraction": AMPLITUDE_MIN_VOLUME_FRACTION},
            "artifact_direction": {"net_length_change_must_be_negative": True, "near_monotonic_up_share_max": NEAR_MONOTONIC_MAX_UP_SHARE, "near_monotonic_rebound_fraction_max": NEAR_MONOTONIC_MAX_REBOUND_FRACTION, "terminal_20yr_down_share_max": TERMINAL_MAX_DOWNWARD_SHARE, "terminal_robust_z_min": TERMINAL_MIN_ROBUST_Z, "hf_rms_range_min": HF_MIN_RMS_RANGE, "hf_tv_share_min": HF_MIN_TV_SHARE, "hf_zero_crossings_min": HF_MIN_ZERO_CROSSINGS},
        },
    }
    (OUTPUT / "method_specification.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    summary = {
        "schema": "lia-retained-32-curve-analysis-v1",
        "status": "read-only-analysis-complete",
        "analysis_scope": "fixed retained 32 only",
        "model_family_route_labels": metrics["model_family_registry_label"].value_counts().to_dict(),
        "source_status_labels": metrics["source_status"].value_counts().to_dict(),
        "direct_surface_mass_balance_available_count": int(metrics["direct_surface_mb_available"].sum()),
        "derived_mass_balance_count": 32,
        "preliminary_strict_pass_count": len(strict_ids),
        "preliminary_strict_pass_system_ids": strict_ids,
        "uncertain_three_of_four_core_count": len(uncertain_ids),
        "uncertain_three_of_four_core_system_ids": uncertain_ids,
        "primary_classification_counts": metrics["primary_classification"].value_counts().to_dict(),
        "representative_system_ids": representative,
        "formal_accepted_count_unchanged": 58,
        "registry_controller_parameter_matrix_modified": False,
        "simulation_launched_stopped_resumed_or_modified": False,
        "outputs": {
            "report": report.name,
            "classification_csv": "classification_table.csv",
            "classification_markdown": "classification_table.md",
            "annual_series": "annual_smoothed_multivariate_series.csv",
            "source_manifest": "source_manifest.csv",
            "method_specification": "method_specification.json",
            "overall_distribution_plot": distribution_plot.name,
            "typical_curves_plot": typical_plot.name,
            "all_32_grid_plot": grid_plot.name,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
