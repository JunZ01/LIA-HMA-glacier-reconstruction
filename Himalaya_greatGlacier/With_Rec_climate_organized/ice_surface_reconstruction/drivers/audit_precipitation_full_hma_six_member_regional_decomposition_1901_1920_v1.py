"""Compare production precipitation members on full-HMA and gauge-support masks."""

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


EXPERIMENT_ID = "climate_reconstruction_precipitation_full_hma_six_member_regional_decomposition_1901_1920_v1"
YEARS = np.arange(1901, 1921, dtype=int)
LOCKED = (1921, 1939)
PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
MEMBERS = {
    "AO_6": "pr_rec_result_AO_6_m_c05d",
    "AO_9": "pr_rec_result_AO_9_m_c05d",
    "HGAO_6": "pr_rec_result_HGAO_6_m_c05d",
    "HGAO_9": "pr_rec_result_HGAO_9_m_c05d",
    "O_6": "pr_rec_result_O_6_m_c05d",
    "O_9": "pr_rec_result_O_9_m_c05d",
}
ENSEMBLE = "pr_ensemble_mean_rec_result"
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
LABELS = {
    "hma": "Full HMA",
    "west_lt80": "West (<80E)",
    "central_80_90": "Central (80-90E)",
    "east_ge90": "East (>=90E)",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path, hash_file: bool = True) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_file:
        result["sha256"] = sha256(path)
    return result


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result <= -9000) | (result >= 1e20)] = np.nan
    return result


def weighted_curve(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(field) & mask[None, :, :]
    numerator = np.nansum(np.where(valid, field, np.nan) * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def trend(curve: np.ndarray) -> float:
    valid = np.isfinite(curve)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(YEARS[valid], curve[valid], 1)[0] * 100.0)


def endpoint_change(curve: np.ndarray) -> float:
    return float(np.nanmean(curve[-5:]) - np.nanmean(curve[:5]))


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "public_fields": PUBLIC,
        "hma_mask": result_root / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "support_grid": result_root / "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1/hma_grid_reference_support_1901_1920.csv",
        "dual_support_audit": result_root / "climate_reconstruction_precipitation_rec_result_ensemble_member_decomposition_1901_1920_v1/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.any((YEARS >= LOCKED[0]) & (YEARS <= LOCKED[1])):
        raise RuntimeError("locked-period-requested")
    before = {name: state(path, hash_file=name != "public_fields") for name, path in sources.items()}

    hma_mask = np.load(sources["hma_mask"]).astype(bool)
    with h5py.File(PUBLIC, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.array([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        member_fields = {label: clean(handle[var][indices]) for label, var in MEMBERS.items()}
        archived = clean(handle[ENSEMBLE][indices])
    if hma_mask.shape != (len(lat), len(lon)):
        raise RuntimeError(f"hma-mask-shape-conflict:{hma_mask.shape}:{(len(lat), len(lon))}")

    member_stack = np.stack([member_fields[label] for label in MEMBERS])
    rebuilt = np.nanmean(member_stack, axis=0)
    finite = np.isfinite(archived) & np.isfinite(rebuilt)
    grid_closure = float(np.max(np.abs(archived[finite] - rebuilt[finite])))
    if grid_closure > 2e-6:
        raise RuntimeError(f"ensemble-grid-closure-failed:{grid_closure}")

    support = pd.read_csv(sources["support_grid"])
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    if len(support) != lat_grid.size:
        raise RuntimeError("support-grid-size-conflict")
    if not np.allclose(support["latitude"], lat_grid.ravel()) or not np.allclose(support["longitude"], lon_grid.ravel()):
        raise RuntimeError("support-grid-coordinate-conflict")
    dual = (
        support["joint_complete"].to_numpy(dtype=bool)
        & (support["cru_station_month_fraction_1901_1920"].to_numpy(dtype=float) > 0)
        & (support["gpcc_direct_gauge_month_fraction_1901_1920"].to_numpy(dtype=float) > 0)
    ).reshape(hma_mask.shape)

    annual_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str, str], np.ndarray] = {}
    masks = {"full_hma": hma_mask, "dual_support": hma_mask & dual}
    for mask_name, base_mask in masks.items():
        for region, (lower, upper) in REGIONS.items():
            region_mask = base_mask & (lon_grid >= lower) & (lon_grid < upper)
            cell_count = int(region_mask.sum())
            if cell_count == 0:
                continue
            branch_curves = {label: weighted_curve(field, lat, region_mask) for label, field in member_fields.items()}
            branch_curves["ensemble"] = weighted_curve(archived, lat, region_mask)
            rebuilt_curve = np.mean(np.stack([branch_curves[label] for label in MEMBERS]), axis=0)
            curve_closure = float(np.max(np.abs(branch_curves["ensemble"] - rebuilt_curve)))
            if curve_closure > 2e-6:
                raise RuntimeError(f"ensemble-curve-closure-failed:{mask_name}:{region}:{curve_closure}")
            for label, curve in branch_curves.items():
                curves[(mask_name, region, label)] = curve
                trend_rows.append({
                    "mask": mask_name,
                    "region": region,
                    "field": label,
                    "gridcell_count": cell_count,
                    "trend_mm_day_per_century": trend(curve),
                    "endpoint_change_mm_day": endpoint_change(curve),
                    "curve_closure_max_abs_mm_day": curve_closure if label == "ensemble" else np.nan,
                })
                for year, value in zip(YEARS, curve):
                    annual_rows.append({
                        "mask": mask_name,
                        "region": region,
                        "field": label,
                        "year": int(year),
                        "precipitation_anomaly_mm_day": float(value),
                    })

    trends = pd.DataFrame(trend_rows)
    annual = pd.DataFrame(annual_rows)
    for region in REGIONS:
        for label in [*MEMBERS, "ensemble"]:
            full = trends[(trends["mask"] == "full_hma") & (trends["region"] == region) & (trends["field"] == label)]
            dual_row = trends[(trends["mask"] == "dual_support") & (trends["region"] == region) & (trends["field"] == label)]
            if full.empty or dual_row.empty:
                continue
            full_trend = float(full.iloc[0]["trend_mm_day_per_century"])
            dual_trend = float(dual_row.iloc[0]["trend_mm_day_per_century"])
            contrast_rows.append({
                "region": region,
                "field": label,
                "full_hma_trend_mm_day_per_century": full_trend,
                "dual_support_trend_mm_day_per_century": dual_trend,
                "full_minus_dual_mm_day_per_century": full_trend - dual_trend,
                "trend_sign_agreement": bool(np.sign(full_trend) == np.sign(dual_trend)),
                "full_hma_gridcell_count": int(full.iloc[0]["gridcell_count"]),
                "dual_support_gridcell_count": int(dual_row.iloc[0]["gridcell_count"]),
            })
    contrasts = pd.DataFrame(contrast_rows)

    output.mkdir(parents=True)
    annual.to_csv(output / "full_hma_and_dual_support_member_curves.csv", index=False)
    trends.to_csv(output / "full_hma_and_dual_support_member_trends.csv", index=False)
    contrasts.to_csv(output / "spatial_sampling_trend_contrasts.csv", index=False)

    colors = {
        "AO_6": "#0072B2", "AO_9": "#56B4E9", "HGAO_6": "#009E73",
        "HGAO_9": "#E69F00", "O_6": "#CC79A7", "O_9": "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.ravel(), REGIONS):
        for label, color in colors.items():
            axis.plot(YEARS, curves[("full_hma", region, label)], color=color, linewidth=1.15, alpha=0.8, label=label)
        mean_curve = curves[("full_hma", region, "ensemble")]
        mean_trend = float(trends[(trends["mask"] == "full_hma") & (trends["region"] == region) & (trends["field"] == "ensemble")]["trend_mm_day_per_century"].iloc[0])
        axis.plot(YEARS, mean_curve, color="#111111", linewidth=2.8, label="Six-member mean")
        axis.set_title(f"{LABELS[region]} | mean trend {mean_trend:+.3f} mm/day/century")
        axis.axhline(0, color="#AAAAAA", linewidth=0.7)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.set_ylabel("Precipitation anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=4, fontsize=8)
    fig.savefig(output / "full_hma_six_member_regional_curves.png", dpi=220)
    plt.close(fig)

    ensemble_contrast = contrasts[contrasts["field"] == "ensemble"].copy()
    member_sign_counts = []
    for region in REGIONS:
        subset = trends[(trends["mask"] == "full_hma") & (trends["region"] == region) & trends["field"].isin(MEMBERS)]
        member_sign_counts.append({
            "region": region,
            "negative_member_count": int((subset["trend_mm_day_per_century"] < 0).sum()),
            "positive_member_count": int((subset["trend_mm_day_per_century"] > 0).sum()),
            "minimum_member_trend": float(subset["trend_mm_day_per_century"].min()),
            "maximum_member_trend": float(subset["trend_mm_day_per_century"].max()),
        })
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "grid_closure_max_abs_mm_day": grid_closure,
        "ensemble_spatial_sampling_contrasts": ensemble_contrast.to_dict(orient="records"),
        "full_hma_member_sign_counts": member_sign_counts,
        "interpretation": "Full-HMA and gauge-support masks answer different spatial questions; operator experiment 1 remains a mechanism sensitivity for one ordinary-EnKF/proxy6 realization only.",
    }
    after = {name: state(path, hash_file=name != "public_fields") for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    lines = [
        "# Full-HMA Six-member Regional Precipitation Decomposition",
        "",
        "The six production members were compared on the complete HMA mask and on the much smaller dual-gauge-support subset using identical years and longitude boundaries.",
        "",
    ]
    for row in ensemble_contrast.to_dict(orient="records"):
        lines.append(
            f"- {row['region']}: full-HMA `{row['full_hma_trend_mm_day_per_century']:+.3f}` versus dual-support "
            f"`{row['dual_support_trend_mm_day_per_century']:+.3f} mm/day/century`; sign agreement={str(row['trend_sign_agreement']).lower()}."
        )
    lines.extend([
        "",
        "The full-HMA result is the correct production-product context for the MADA comparison. The dual-support result is only the correct context for direct CRU/GPCC station-support comparison.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if before != after:
        raise RuntimeError("source-changed-during-audit")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
