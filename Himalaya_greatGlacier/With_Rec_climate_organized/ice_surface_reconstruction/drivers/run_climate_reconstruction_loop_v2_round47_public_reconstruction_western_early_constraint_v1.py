"""Test fixed public precipitation reconstructions against the western early constraint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EARLY_YEARS = np.arange(1901, 1921)
BASELINE_YEARS = np.arange(1961, 1991)
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
MEMBERS = {
    "HGAO": "pr_rec_ref_HGAO_c05d",
    "LMR": "pr_rec_ref_LMR_c05d",
    "O": "pr_rec_ref_O_c05d",
}
OBSERVATIONS = ("UDel", "CRU", "GPCC")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def inside(value: float, lower: float, upper: float, tolerance: float = 1e-12) -> bool:
    return bool(lower - tolerance <= value <= upper + tolerance)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_47_public_reconstruction_western_early_constraint_v1"
    round45 = loop / "round_45_udel_1901_1920_precipitation_product_audit_v1"
    paths = {
        "public_reference_source": Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "observation_metrics": round45 / "early_product_metrics.csv",
        "observation_series": round45 / "early_regional_series.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    before = {name: state(path) for name, path in paths.items()}
    mask = np.load(paths["hma_mask"]).astype(bool)
    observation_metrics = pd.read_csv(paths["observation_metrics"])
    observation_series = pd.read_csv(paths["observation_series"])
    expected_series_rows = len(EARLY_YEARS) * len(REGIONS)
    if len(observation_series) != expected_series_rows:
        raise RuntimeError("round45-observation-series-row-count-failed")

    with h5py.File(paths["public_reference_source"], "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        early_indices = np.flatnonzero((years >= 1901) & (years <= 1920))
        baseline_indices = np.flatnonzero((years >= 1961) & (years <= 1990))
        if not np.array_equal(years[early_indices], EARLY_YEARS):
            raise RuntimeError("early-year-selection-failed")
        if not np.array_equal(years[baseline_indices], BASELINE_YEARS):
            raise RuntimeError("baseline-year-selection-failed")
        requested_years = np.concatenate((years[early_indices], years[baseline_indices]))
        if np.any((requested_years >= 1921) & (requested_years <= 1939)):
            raise RuntimeError("locked-period-index-requested")
        early_slice = slice(int(early_indices[0]), int(early_indices[-1]) + 1)
        baseline_slice = slice(int(baseline_indices[0]), int(baseline_indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        members: dict[str, np.ndarray] = {}
        for name, variable in MEMBERS.items():
            baseline = np.nanmean(clean(handle[variable], baseline_slice), axis=0)
            members[name] = clean(handle[variable], early_slice) - baseline[None]

    candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean(
            [finite_mean([members["HGAO"], members["O"]]), members["LMR"]]
        ),
    }
    longitude_grid = np.broadcast_to(longitudes[None], mask.shape)
    region_masks = {
        name: mask & (longitude_grid >= lower) & (longitude_grid < upper)
        for name, (lower, upper) in REGIONS.items()
    }

    observation_lookup: dict[tuple[str, str], np.ndarray] = {}
    for region in REGIONS:
        selected = observation_series[observation_series.region == region].sort_values("year")
        if not np.array_equal(selected.year.to_numpy(int), EARLY_YEARS):
            raise RuntimeError(f"round45-year-axis-failed:{region}")
        for product in OBSERVATIONS:
            observation_lookup[(region, product)] = selected[product].to_numpy(float)

    series_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    candidate_series: dict[tuple[str, str], np.ndarray] = {}
    for candidate_name, values in candidates.items():
        for region_name, region_mask in region_masks.items():
            common = region_mask & np.all(np.isfinite(values), axis=0)
            if not np.any(common):
                raise RuntimeError(f"empty-candidate-support:{candidate_name}:{region_name}")
            series = weighted_series(values, common, latitudes)
            candidate_series[(candidate_name, region_name)] = series
            trend = float(np.polyfit(EARLY_YEARS, series, 1)[0] * 100.0)
            endpoint_change = float(np.mean(series[-5:]) - np.mean(series[:5]))
            correlations = {
                product: correlation(series, observation_lookup[(region_name, product)])
                for product in OBSERVATIONS
            }
            metric_rows.append(
                {
                    "candidate": candidate_name,
                    "region": region_name,
                    "finite_cell_count": int(common.sum()),
                    "trend_mm_day_per_century": trend,
                    "change_1916_1920_minus_1901_1905_mm_day": endpoint_change,
                    **{f"correlation_{product}": value for product, value in correlations.items()},
                    "median_observation_correlation": float(np.nanmedian(list(correlations.values()))),
                }
            )
            for index, year in enumerate(EARLY_YEARS):
                series_rows.append(
                    {
                        "year": int(year),
                        "region": region_name,
                        "candidate": candidate_name,
                        "anomaly_mm_day": float(series[index]),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    series_frame = pd.DataFrame(series_rows)
    western_observations = observation_metrics[
        (observation_metrics.region == "west_lt80")
        & observation_metrics["product"].isin(OBSERVATIONS)
    ].set_index("product").reindex(OBSERVATIONS)
    if western_observations.isnull().any().any():
        raise RuntimeError("western-observation-envelope-input-failed")
    trend_envelope = (
        float(western_observations.trend_mm_day_per_century.min()),
        float(western_observations.trend_mm_day_per_century.max()),
    )
    change_column = "change_1916_1920_minus_1901_1905_mm_day"
    change_envelope = (
        float(western_observations[change_column].min()),
        float(western_observations[change_column].max()),
    )

    western_results: dict[str, dict[str, object]] = {}
    for candidate_name in candidates:
        index = metrics.index[(metrics.candidate == candidate_name) & (metrics.region == "west_lt80")]
        if len(index) != 1:
            raise RuntimeError(f"western-candidate-row-failed:{candidate_name}")
        row_index = int(index[0])
        trend = float(metrics.loc[row_index, "trend_mm_day_per_century"])
        endpoint_change = float(metrics.loc[row_index, change_column])
        median_correlation = float(metrics.loc[row_index, "median_observation_correlation"])
        gates = {
            "trend_inside_three_product_envelope": inside(trend, *trend_envelope),
            "endpoint_change_inside_three_product_envelope": inside(endpoint_change, *change_envelope),
            "median_annual_correlation_at_least_0_30": median_correlation >= 0.30,
        }
        passes = bool(all(gates.values()))
        metrics.loc[row_index, "trend_inside_observation_envelope"] = gates[
            "trend_inside_three_product_envelope"
        ]
        metrics.loc[row_index, "endpoint_change_inside_observation_envelope"] = gates[
            "endpoint_change_inside_three_product_envelope"
        ]
        metrics.loc[row_index, "passes_western_early_constraint"] = passes
        western_results[candidate_name] = {
            "trend_mm_day_per_century": trend,
            "endpoint_change_mm_day": endpoint_change,
            "median_observation_correlation": median_correlation,
            "gates": gates,
            "passes": passes,
        }

    metrics.to_csv(output / "candidate_early_constraint_metrics.csv", index=False)
    series_frame.to_csv(output / "candidate_regional_series.csv", index=False)
    passing = [name for name, result in western_results.items() if result["passes"]]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "candidate_count": len(candidates),
        "western_observation_trend_envelope_mm_day_per_century": list(trend_envelope),
        "western_observation_endpoint_change_envelope_mm_day": list(change_envelope),
        "western_candidate_results": western_results,
        "passing_candidates": passing,
        "any_current_reconstruction_matches_western_early_constraint": bool(passing),
        "decision": (
            "retain_passing_candidates_only_as_western_early_constraint_matches"
            if passing
            else "current_public_reconstructions_fail_product_robust_western_early_constraint"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {
        "HGAO": "#0072b2",
        "O": "#cc79a7",
        "LMR": "#e69f00",
        "local_enkf": "#56b4e9",
        "member_equal": "#009e73",
        "family_balanced": "#d55e00",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for product, color in (("UDel", "#111111"), ("CRU", "#666666"), ("GPCC", "#999999")):
        axes[0].plot(
            EARLY_YEARS,
            observation_lookup[("west_lt80", product)],
            color=color,
            linewidth=2.2,
            label=product,
        )
    for candidate_name in candidates:
        axes[0].plot(
            EARLY_YEARS,
            candidate_series[(candidate_name, "west_lt80")],
            color=colors[candidate_name],
            linewidth=1.35,
            alpha=0.88,
            label=candidate_name,
        )
    axes[0].axhline(0, color="#bbbbbb", linewidth=0.8)
    axes[0].set(
        title="Western HMA 1901-1920 precipitation",
        xlabel="Year",
        ylabel="Anomaly (mm/day)",
    )
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3)

    west = metrics[metrics.region == "west_lt80"].set_index("candidate").reindex(candidates)
    x = np.arange(len(candidates))
    axes[1].axhspan(trend_envelope[0], trend_envelope[1], color="#d9d9d9", alpha=0.75)
    axes[1].scatter(
        x,
        west.trend_mm_day_per_century,
        color=[colors[name] for name in candidates],
        s=55,
        zorder=3,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, list(candidates), rotation=25, ha="right")
    axes[1].set(
        title="Western trend against observation envelope",
        ylabel="Trend (mm/day/century)",
    )
    for position, candidate_name in enumerate(candidates):
        axes[1].annotate(
            f"r={western_results[candidate_name]['median_observation_correlation']:.2f}",
            (position, west.loc[candidate_name, "trend_mm_day_per_century"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    figure.savefig(output / "western_early_constraint.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Public Reconstruction Western Early Constraint",
        "",
        f"- Observation trend envelope: `{trend_envelope[0]:.3f}` to `{trend_envelope[1]:.3f}` mm/day/century.",
        f"- Observation endpoint-change envelope: `{change_envelope[0]:.3f}` to `{change_envelope[1]:.3f}` mm/day.",
    ]
    for name, result in western_results.items():
        lines.append(
            f"- {name}: trend `{result['trend_mm_day_per_century']:.3f}`, "
            f"endpoint change `{result['endpoint_change_mm_day']:.3f}`, "
            f"median observation correlation `{result['median_observation_correlation']:.3f}`, "
            f"passes `{str(result['passes']).lower()}`."
        )
    lines.extend(
        [
            "",
            f"Passing candidates: `{passing}`.",
            "",
            f"Decision: `{summary['decision']}`.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
