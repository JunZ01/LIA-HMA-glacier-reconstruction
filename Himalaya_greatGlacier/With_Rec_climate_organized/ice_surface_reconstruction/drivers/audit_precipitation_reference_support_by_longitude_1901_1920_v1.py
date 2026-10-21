"""Compare reconstruction and references within fixed longitude/support masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v1"
YEARS = np.arange(1901, 1921, dtype=int)
LOCKED_PERIOD = (1921, 1939)
HMA_CLIMATE = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
REGIONS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_90_100": (90.0, 100.0),
    "far_east_ge100": (100.0, np.inf),
}
SUPPORT_LEVELS = ("all_joint", "dual_any", "dual_high")
PRODUCTS = ("cru", "gpcc", "rec_result")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path, include_hash: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if include_hash:
        result["sha256"] = digest(path)
    return result


def clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values[(values <= -9000) | (values >= 1e20)] = np.nan
    return values


def weighted_mean(field: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    values = np.where(mask[None, :, :], field, np.nan)
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def event_change(values: np.ndarray) -> float:
    baseline = (YEARS >= 1905) & (YEARS <= 1914)
    event = (YEARS >= 1915) & (YEARS <= 1918)
    return float(np.nanmean(values[event]) - np.nanmean(values[baseline]))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "hma_climate": HMA_CLIMATE,
        "support_grid": result_root / "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1/hma_grid_reference_support_1901_1920.csv",
        "support_summary": result_root / "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1/summary.json",
        "locked_contract": result_root / "climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    contract = json.loads(sources["locked_contract"].read_text(encoding="utf-8"))
    support_summary = json.loads(sources["support_summary"].read_text(encoding="utf-8"))
    if contract["status"] != "locked_not_evaluated" or support_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("locked-period-contract-failed")
    if np.any((YEARS >= LOCKED_PERIOD[0]) & (YEARS <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested")
    before = {
        name: source_state(path, include_hash=name != "hma_climate")
        for name, path in sources.items()
    }

    support = pd.read_csv(sources["support_grid"])
    with h5py.File(HMA_CLIMATE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        fields = {
            "cru": clean(np.asarray(handle["pr_obs_cru_pre_05deg"][indices, :, :], dtype=float)),
            "gpcc": clean(np.asarray(handle["pr_obs_gpcp_05deg"][indices, :, :], dtype=float)),
            "rec_result": clean(np.asarray(handle["pr_ensemble_mean_rec_result"][indices, :, :], dtype=float)),
        }
    expected_rows = len(latitudes) * len(longitudes)
    if len(support) != expected_rows:
        raise RuntimeError(f"support-grid-shape-failed:{len(support)}:{expected_rows}")
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    if not (
        np.allclose(support["latitude"].to_numpy(), lat_grid.reshape(-1))
        and np.allclose(support["longitude"].to_numpy(), lon_grid.reshape(-1))
    ):
        raise RuntimeError("support-grid-coordinate-order-failed")

    joint = support["joint_complete"].to_numpy(dtype=bool).reshape(lat_grid.shape)
    cru_fraction = support["cru_station_month_fraction_1901_1920"].to_numpy(dtype=float).reshape(lat_grid.shape)
    gpcc_fraction = support["gpcc_direct_gauge_month_fraction_1901_1920"].to_numpy(dtype=float).reshape(lat_grid.shape)
    dual_any = joint & (cru_fraction > 0) & (gpcc_fraction > 0)
    dual_high = joint & (cru_fraction >= 0.5) & (gpcc_fraction >= 0.5)
    area_weights = np.cos(np.deg2rad(latitudes))[:, None]
    annual = pd.DataFrame({"year": YEARS})
    score_rows = []

    for region_name, (lower_lon, upper_lon) in REGIONS.items():
        region = (lon_grid >= lower_lon) & (lon_grid < upper_lon)
        region_joint = joint & region
        masks = {
            "all_joint": region_joint,
            "dual_any": dual_any & region,
            "dual_high": dual_high & region,
        }
        region_area = float(np.sum(area_weights * region_joint))
        for support_level, mask in masks.items():
            curves = {product: weighted_mean(fields[product], latitudes, mask) for product in PRODUCTS}
            for product, values in curves.items():
                annual[f"{region_name}_{support_level}_{product}_mm_day"] = values
            cru_trend = trend(curves["cru"])
            gpcc_trend = trend(curves["gpcc"])
            rec_trend = trend(curves["rec_result"])
            lower = np.minimum(curves["cru"], curves["gpcc"])
            upper = np.maximum(curves["cru"], curves["gpcc"])
            score_rows.append({
                "longitude_region": region_name,
                "support_level": support_level,
                "gridcell_count": int(mask.sum()),
                "area_fraction_of_region_joint": float(np.sum(area_weights * mask) / region_area) if region_area > 0 else float("nan"),
                "cru_trend_mm_day_per_century": cru_trend,
                "gpcc_trend_mm_day_per_century": gpcc_trend,
                "rec_result_trend_mm_day_per_century": rec_trend,
                "rec_result_minus_more_negative_product_trend": rec_trend - min(cru_trend, gpcc_trend) if np.isfinite(rec_trend) else float("nan"),
                "cru_event_change_mm_day": event_change(curves["cru"]),
                "gpcc_event_change_mm_day": event_change(curves["gpcc"]),
                "rec_result_event_change_mm_day": event_change(curves["rec_result"]),
                "rec_result_outside_annual_product_bounds_years": int(np.sum((curves["rec_result"] < lower) | (curves["rec_result"] > upper))) if int(mask.sum()) else 0,
                "cru_gpcc_annual_correlation": correlation(curves["cru"], curves["gpcc"]),
                "cru_gpcc_annual_rmse_mm_day": float(np.sqrt(np.nanmean(np.square(curves["cru"] - curves["gpcc"])))) if int(mask.sum()) else float("nan"),
            })
    scores = pd.DataFrame(score_rows)

    eligible = scores[(scores["support_level"] == "dual_any") & (scores["gridcell_count"] >= 5)].copy()
    eligible["opposite_or_more_negative_than_both"] = (
        eligible["rec_result_trend_mm_day_per_century"]
        < eligible[["cru_trend_mm_day_per_century", "gpcc_trend_mm_day_per_century"]].min(axis=1)
    )
    persistence_count = int(eligible["opposite_or_more_negative_than_both"].sum())
    classification = (
        "negative_reconstruction_trend_persists_across_supported_longitude_bands"
        if len(eligible) >= 2 and persistence_count == len(eligible)
        else "supported_longitude_bands_do_not_share_one_reconstruction_conflict"
    )

    output.mkdir(parents=True)
    scores.to_csv(output / "longitude_support_scores.csv", index=False)
    annual.to_csv(output / "longitude_support_hma_curves_1901_1920.csv", index=False)

    colors = {"cru": "#6A3D9A", "gpcc": "#777777", "rec_result": "#D55E00"}
    labels = {"cru": "CRU", "gpcc": "GPCC", "rec_result": "rec_result"}
    titles = {"west_lt80": "<80E", "central_80_90": "80-90E", "east_90_100": "90-100E", "far_east_ge100": ">=100E"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=True, constrained_layout=True)
    for axis, region_name in zip(axes.reshape(-1), REGIONS):
        dual_count = int(scores[(scores["longitude_region"] == region_name) & (scores["support_level"] == "dual_any")]["gridcell_count"].iloc[0])
        all_count = int(scores[(scores["longitude_region"] == region_name) & (scores["support_level"] == "all_joint")]["gridcell_count"].iloc[0])
        for product in PRODUCTS:
            all_values = annual[f"{region_name}_all_joint_{product}_mm_day"]
            axis.plot(YEARS, all_values, color=colors[product], linewidth=1.0, linestyle="--", alpha=0.45)
            if dual_count:
                dual_values = annual[f"{region_name}_dual_any_{product}_mm_day"]
                axis.plot(YEARS, dual_values, color=colors[product], linewidth=2.0 if product == "rec_result" else 1.6, label=labels[product])
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.set_title(f"{titles[region_name]}: dual-any n={dual_count}, all n={all_count}")
        axis.set_ylabel("Precipitation anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=3, title="Solid: dual-any; dashed: all joint")
    fig.savefig(output / "longitude_fixed_support_precipitation_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "longitude_regions": {name: [None if not np.isfinite(low) else low, None if not np.isfinite(high) else high] for name, (low, high) in REGIONS.items()},
        "support_levels": {
            "all_joint": "All region cells complete in CRU, GPCC, and rec_result for 1901-1920.",
            "dual_any": "Fixed subset with at least one direct CRU station month and one direct GPCC gauge month.",
            "dual_high": "Fixed subset with direct support in both products for at least 50% of months.",
        },
        "scores": scores.to_dict(orient="records"),
        "eligible_dual_any_regions_minimum_5_cells": eligible.to_dict(orient="records"),
        "classification": classification,
        "interpretation_boundary": "Longitude bands and support masks were fixed before trend comparison. Small eastern samples limit precision; this is not a production correction or proof that either reference product is truth.",
    }
    after = {
        name: source_state(path, include_hash=name != "hma_climate")
        for name, path in sources.items()
    }
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-longitude-support-audit")
    lines = ["# Fixed-support Longitude Audit v1", ""]
    for _, row in eligible.iterrows():
        lines.append(
            f"- {row['longitude_region']} dual-any n={int(row['gridcell_count'])}: CRU/GPCC/rec_result trends "
            f"`{row['cru_trend_mm_day_per_century']:+.3f}` / `{row['gpcc_trend_mm_day_per_century']:+.3f}` / "
            f"`{row['rec_result_trend_mm_day_per_century']:+.3f} mm/day/century`."
        )
    lines.extend(["", f"Classification: `{classification}`. This is a fixed-mask mechanism audit, not a production correction."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
