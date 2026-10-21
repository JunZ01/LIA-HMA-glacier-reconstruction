"""Audit the spatial and source structure of the 1915-1918 positive-b2 event."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_YEARS = np.arange(1905, 1915)
EVENT_YEARS = np.arange(1915, 1919)
RECOVERY_YEARS = np.arange(1919, 1921)
CALIBRATION_YEARS = np.arange(1940, 2001)
RNG_SEED = 20260814
PERMUTATIONS = 10000


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all" / "python_data" / "proxy_data_6.mat"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values) & pd.notna(labels)
    y = values[valid]
    groups = labels[valid]
    if len(y) < 3 or np.var(y) == 0:
        return float("nan")
    grand = np.mean(y)
    between = sum(np.sum(groups == group) * (np.mean(y[groups == group]) - grand) ** 2 for group in np.unique(groups))
    total = np.sum((y - grand) ** 2)
    return float(between / total) if total > 0 else float("nan")


def permutation_eta(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    observed = eta_squared(values, labels)
    if not np.isfinite(observed) or len(np.unique(labels[pd.notna(labels)])) < 2:
        return observed, float("nan")
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        null[index] = eta_squared(values, rng.permutation(labels))
    return observed, float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1))


def haversine_pairwise_mean(latitude: np.ndarray, longitude: np.ndarray) -> float:
    if len(latitude) < 2:
        return float("nan")
    distances = []
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    for left, right in combinations(range(len(lat)), 2):
        dlat = lat[right] - lat[left]
        dlon = lon[right] - lon[left]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[left]) * np.cos(lat[right]) * np.sin(dlon / 2) ** 2
        distances.append(6371.0 * 2.0 * np.arcsin(np.sqrt(a)))
    return float(np.mean(distances))


def group_summary(table: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for label, group in table.groupby(field, dropna=False):
        rows.append(
            {
                "grouping": field,
                "group": str(label),
                "proxy_count": len(group),
                "unique_site_count": group["site_id"].nunique(),
                "mean_event_drop_z": group["event_drop_z"].mean(),
                "median_event_drop_z": group["event_drop_z"].median(),
                "fraction_event_drop_negative": (group["event_drop_z"] < 0).mean(),
                "mean_recovery_minus_event_z": group["recovery_minus_event_z"].mean(),
                "sum_precipitation_pathway_effect_trend": group["precipitation_pathway_effect_trend_mm_day_per_century"].sum(),
                "fraction_total_negative_pathway_effect": group["negative_effect_magnitude"].sum() / table["negative_effect_magnitude"].sum(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    mechanism = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1/positive_b2_proxy_mechanism_table.csv"
    proxy6 = archive_proxy_path()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_1915_1918_spatial_source_v1"
    if not mechanism.exists() or not proxy6.exists():
        raise FileNotFoundError("missing-input")
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    mechanism_hash_before = digest(mechanism)
    proxy_stat_before = source_stat(proxy6)
    table = pd.read_csv(mechanism)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")
    selected = table["proxy_index"].to_numpy(dtype=int)

    periods = {
        "baseline": (slice(1904, 1914), BASELINE_YEARS),
        "event": (slice(1914, 1918), EVENT_YEARS),
        "recovery": (slice(1918, 1920), RECOVERY_YEARS),
        "calibration": (slice(1939, 2000), CALIBRATION_YEARS),
    }
    raw: dict[str, np.ndarray] = {name: np.full((len(years), len(selected)), np.nan) for name, (_, years) in periods.items()}
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(selected):
            proxy = handle[refs[proxy_index - 1]]
            for name, (year_slice, years) in periods.items():
                ages = np.asarray(proxy["age_yng"][0, year_slice], dtype=float)
                if not np.array_equal(ages, years):
                    raise RuntimeError(f"annual-index-contract-failed:{proxy_index}:{name}")
                raw[name][:, column] = np.asarray(proxy["data"][year_slice, 0], dtype=float)

    calibration_count = np.sum(np.isfinite(raw["calibration"]), axis=0)
    calibration_mean = np.nanmean(raw["calibration"], axis=0)
    calibration_std = np.nanstd(raw["calibration"], axis=0)
    valid = (calibration_count >= 20) & np.isfinite(calibration_std) & (calibration_std > 0)
    if not valid.all():
        raise RuntimeError(f"nonstandardizable-selected-proxies:{int((~valid).sum())}")
    standardized = {name: (values - calibration_mean[None, :]) / calibration_std[None, :] for name, values in raw.items()}
    if not all(np.all(np.isfinite(standardized[name])) for name in ("baseline", "event", "recovery")):
        raise RuntimeError("event-window-missingness-detected")

    baseline_mean = np.mean(standardized["baseline"], axis=0)
    event_mean = np.mean(standardized["event"], axis=0)
    recovery_mean = np.mean(standardized["recovery"], axis=0)
    table["calibration_valid_count"] = calibration_count
    table["calibration_mean"] = calibration_mean
    table["calibration_std"] = calibration_std
    table["baseline_1905_1914_mean_z"] = baseline_mean
    table["event_1915_1918_mean_z"] = event_mean
    table["recovery_1919_1920_mean_z"] = recovery_mean
    table["event_drop_z"] = event_mean - baseline_mean
    table["recovery_minus_event_z"] = recovery_mean - event_mean
    table["negative_effect_magnitude"] = np.maximum(-table["precipitation_pathway_effect_trend_mm_day_per_century"], 0)
    table["site_id"] = table["latitude"].round(4).astype(str) + "," + table["longitude"].round(4).astype(str)
    table["latitude_band"] = pd.cut(table["latitude"], [-np.inf, 30, 35, np.inf], labels=["<30N", "30-35N", ">=35N"], right=False).astype(str)

    group_fields = ["longitude_band", "latitude_band", "site_class", "source_family", "data_name"]
    groups = pd.concat([group_summary(table, field) for field in group_fields], ignore_index=True)
    rng = np.random.default_rng(RNG_SEED)
    tests = []
    for field in group_fields:
        eta, pvalue = permutation_eta(table["event_drop_z"].to_numpy(), table[field].astype(str).to_numpy(), rng)
        tests.append({"grouping": field, "eta_squared": eta, "permutation_pvalue": pvalue, "group_count": table[field].nunique(dropna=False)})
    tests = pd.DataFrame(tests)

    strongest_count = max(3, int(np.ceil(len(table) * 0.25)))
    strongest = table.nsmallest(strongest_count, "event_drop_z")
    observed_distance = haversine_pairwise_mean(strongest["latitude"].to_numpy(), strongest["longitude"].to_numpy())
    null_distance = np.empty(PERMUTATIONS)
    positions = np.arange(len(table))
    for index in range(PERMUTATIONS):
        chosen = rng.choice(positions, size=strongest_count, replace=False)
        null_distance[index] = haversine_pairwise_mean(table.iloc[chosen]["latitude"].to_numpy(), table.iloc[chosen]["longitude"].to_numpy())
    spatial_p_lower = float((1 + np.sum(null_distance <= observed_distance)) / (PERMUTATIONS + 1))

    network_baseline = np.mean(standardized["baseline"], axis=1)
    network_event = np.mean(standardized["event"], axis=1)
    network_recovery = np.mean(standardized["recovery"], axis=1)
    annual = pd.DataFrame(
        {
            "year": np.concatenate([BASELINE_YEARS, EVENT_YEARS, RECOVERY_YEARS]),
            "phase": ["baseline"] * len(BASELINE_YEARS) + ["event"] * len(EVENT_YEARS) + ["recovery"] * len(RECOVERY_YEARS),
            "positive_b2_network_mean_calibration_z": np.concatenate([network_baseline, network_event, network_recovery]),
        }
    )
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_event_1915_1918_spatial_source_v1",
        "analyzed_periods": [[1905, 1920], [1940, 2000]],
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed": False,
        "positive_b2_proxy_count": len(table),
        "unique_site_count": int(table["site_id"].nunique()),
        "event_metrics": {
            "network_event_drop_z": float(table["event_drop_z"].mean()),
            "median_proxy_event_drop_z": float(table["event_drop_z"].median()),
            "negative_proxy_count": int((table["event_drop_z"] < 0).sum()),
            "negative_proxy_fraction": float((table["event_drop_z"] < 0).mean()),
            "network_recovery_minus_event_z": float(table["recovery_minus_event_z"].mean()),
            "event_drop_vs_precipitation_pathway_effect_correlation": correlation(table["event_drop_z"].to_numpy(), table["precipitation_pathway_effect_trend_mm_day_per_century"].to_numpy()),
            "event_drop_vs_pc1_loading_correlation": correlation(table["event_drop_z"].to_numpy(), table["pc1_loading"].to_numpy()),
        },
        "spatial_clustering": {
            "strongest_decline_proxy_count": strongest_count,
            "observed_mean_pairwise_distance_km": observed_distance,
            "random_mean_pairwise_distance_km": float(np.mean(null_distance)),
            "lower_tail_permutation_pvalue": spatial_p_lower,
        },
        "group_tests": tests.to_dict(orient="records"),
        "interpretation_limit": "This diagnostic localizes a stored-proxy event and links it to the frozen EnKF pathway. It cannot by itself distinguish a real regional climate event from a preprocessing common mode shared across source records.",
    }

    output.mkdir(parents=True)
    table.to_csv(output / "positive_b2_event_proxy_table.csv", index=False)
    groups.to_csv(output / "positive_b2_event_group_summary.csv", index=False)
    tests.to_csv(output / "positive_b2_event_group_tests.csv", index=False)
    annual.to_csv(output / "positive_b2_event_network_annual.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    size = 32 + 170 * table["negative_effect_magnitude"] / max(table["negative_effect_magnitude"].max(), 1e-12)
    scatter = axes[0].scatter(table["longitude"], table["latitude"], c=table["event_drop_z"], s=size, cmap="RdBu_r", vmin=-max(abs(table["event_drop_z"]).max(), 0.1), vmax=max(abs(table["event_drop_z"]).max(), 0.1), edgecolor="#333333", linewidth=0.4)
    axes[0].set_xlabel("Longitude (degrees east)"); axes[0].set_ylabel("Latitude (degrees north)"); axes[0].set_title("Proxy event change (1915-1918 minus 1905-1914)")
    figure.colorbar(scatter, ax=axes[0], label="Event change (calibration z)")
    order = ["<80E", "80-90E", "90-100E", ">=100E"]
    for x, label in enumerate(order):
        values = table.loc[table["longitude_band"] == label, "event_drop_z"].to_numpy()
        axes[1].scatter(np.full(len(values), x) + rng.uniform(-0.12, 0.12, len(values)), values, color="#0072B2", alpha=0.7, s=26)
        axes[1].plot([x - 0.22, x + 0.22], [np.median(values), np.median(values)], color="#1b1b1b", linewidth=2.2)
    axes[1].axhline(0, color="#888888", linewidth=0.8); axes[1].set_xticks(range(len(order)), order); axes[1].set_ylabel("Event change (calibration z)"); axes[1].set_title("Decline across longitude bands")
    ranked = table.sort_values("event_drop_z").reset_index(drop=True)
    colors = np.where(ranked["event_drop_z"] < 0, "#D55E00", "#009E73")
    axes[2].bar(np.arange(len(ranked)), ranked["event_drop_z"], color=colors, width=0.85)
    axes[2].axhline(0, color="#555555", linewidth=0.8); axes[2].set_xlabel("Proxies ranked by event change"); axes[2].set_ylabel("Event change (calibration z)"); axes[2].set_title("Network-wide sign and magnitude")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    figure.savefig(output / "positive_b2_event_spatial_source.png", dpi=180)
    plt.close(figure)

    lon_test = tests.loc[tests["grouping"] == "longitude_band"].iloc[0]
    site_test = tests.loc[tests["grouping"] == "site_class"].iloc[0]
    report = [
        "# Positive-b2 1915-1918 Event Spatial/Source Audit",
        "",
        "Only 1905-1920 and 1940-2000 were read. The locked 1921-1939 period was not accessed.",
        "",
        f"- Event decline: `{summary['event_metrics']['network_event_drop_z']:.3f}` calibration-z; `{summary['event_metrics']['negative_proxy_count']}/60` proxies declined.",
        f"- Recovery in 1919-1920: `{summary['event_metrics']['network_recovery_minus_event_z']:.3f}` calibration-z relative to the event mean.",
        f"- Longitude-band eta-squared/p: `{lon_test['eta_squared']:.3f}` / `{lon_test['permutation_pvalue']:.4f}`.",
        f"- Duplicate-site eta-squared/p: `{site_test['eta_squared']:.3f}` / `{site_test['permutation_pvalue']:.4f}`.",
        f"- Strongest-decline quartile mean separation: `{observed_distance:.0f} km`; random expectation `{np.mean(null_distance):.0f} km`; clustering p `{spatial_p_lower:.4f}`.",
        f"- Event change versus direct precipitation-pathway trend correlation: `{summary['event_metrics']['event_drop_vs_precipitation_pathway_effect_correlation']:.3f}`.",
        "",
        "This identifies whether the event is spatially/source concentrated. It does not yet decide whether a broad common mode is climatic or inherited from shared preprocessing.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    mechanism_hash_after = digest(mechanism)
    proxy_stat_after = source_stat(proxy6)
    if mechanism_hash_before != mechanism_hash_after or proxy_stat_before != proxy_stat_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    manifest = {
        "mechanism_sha256_before": mechanism_hash_before,
        "mechanism_sha256_after": mechanism_hash_after,
        "proxy6_stat_before": proxy_stat_before,
        "proxy6_stat_after": proxy_stat_after,
        "analysis_driver_sha256": digest(Path(__file__)),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
