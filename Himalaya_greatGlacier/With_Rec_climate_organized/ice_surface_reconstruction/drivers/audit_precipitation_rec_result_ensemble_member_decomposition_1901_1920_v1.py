"""Decompose the archived precipitation rec_result ensemble into six members."""

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


EXPERIMENT_ID = "climate_reconstruction_precipitation_rec_result_ensemble_member_decomposition_1901_1920_v1"
YEARS = np.arange(1901, 1921, dtype=int)
LOCKED_PERIOD = (1921, 1939)
HMA_CLIMATE = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
MEMBERS = {
    "AO_6": "pr_rec_result_AO_6_m_c05d",
    "AO_9": "pr_rec_result_AO_9_m_c05d",
    "HGAO_6": "pr_rec_result_HGAO_6_m_c05d",
    "HGAO_9": "pr_rec_result_HGAO_9_m_c05d",
    "O_6": "pr_rec_result_O_6_m_c05d",
    "O_9": "pr_rec_result_O_9_m_c05d",
}
ENSEMBLE_VARIABLE = "pr_ensemble_mean_rec_result"
REGIONS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_90_100": (90.0, 100.0),
}
REGION_LABELS = {"west_lt80": "<80E", "central_80_90": "80-90E", "east_90_100": "90-100E"}


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


def source_state(path: Path, include_hash: bool) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
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
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def event_change(values: np.ndarray) -> float:
    baseline = (YEARS >= 1905) & (YEARS <= 1914)
    event = (YEARS >= 1915) & (YEARS <= 1918)
    return float(np.nanmean(values[event]) - np.nanmean(values[baseline]))


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "hma_climate": HMA_CLIMATE,
        "support_grid": result_root / "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1/hma_grid_reference_support_1901_1920.csv",
        "locked_contract": result_root / "climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "netcdf_builder": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\mat_climate_anomaly_visualizer_v1.py"),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.any((YEARS >= LOCKED_PERIOD[0]) & (YEARS <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested")
    locked = json.loads(sources["locked_contract"].read_text(encoding="utf-8"))
    if locked["status"] != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    before = {
        name: source_state(path, include_hash=name != "hma_climate")
        for name, path in sources.items()
    }

    with h5py.File(HMA_CLIMATE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        missing_variables = [name for name in [*MEMBERS.values(), ENSEMBLE_VARIABLE, "pr_obs_cru_pre_05deg", "pr_obs_gpcp_05deg"] if name not in handle]
        if missing_variables:
            raise KeyError(f"missing-netcdf-variables:{missing_variables}")
        member_fields = {label: clean(np.asarray(handle[name][indices, :, :], dtype=float)) for label, name in MEMBERS.items()}
        archived_ensemble = clean(np.asarray(handle[ENSEMBLE_VARIABLE][indices, :, :], dtype=float))
        references = {
            "CRU": clean(np.asarray(handle["pr_obs_cru_pre_05deg"][indices, :, :], dtype=float)),
            "GPCC": clean(np.asarray(handle["pr_obs_gpcp_05deg"][indices, :, :], dtype=float)),
        }
    member_stack = np.stack([member_fields[label] for label in MEMBERS], axis=0)
    rebuilt_ensemble = np.nanmean(member_stack, axis=0)
    finite = np.isfinite(archived_ensemble) & np.isfinite(rebuilt_ensemble)
    grid_closure = {
        "finite_pair_count": int(finite.sum()),
        "maximum_absolute_error_mm_day": float(np.max(np.abs(archived_ensemble[finite] - rebuilt_ensemble[finite]))),
        "rmse_mm_day": float(np.sqrt(np.mean(np.square(archived_ensemble[finite] - rebuilt_ensemble[finite])))),
    }
    if grid_closure["maximum_absolute_error_mm_day"] > 2e-6:
        raise RuntimeError(f"six-member-grid-closure-failed:{grid_closure}")

    support = pd.read_csv(sources["support_grid"])
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    if len(support) != lat_grid.size or not (
        np.allclose(support["latitude"].to_numpy(), lat_grid.reshape(-1))
        and np.allclose(support["longitude"].to_numpy(), lon_grid.reshape(-1))
    ):
        raise RuntimeError("support-grid-coordinate-contract-failed")
    joint = support["joint_complete"].to_numpy(dtype=bool).reshape(lat_grid.shape)
    cru_fraction = support["cru_station_month_fraction_1901_1920"].to_numpy(dtype=float).reshape(lat_grid.shape)
    gpcc_fraction = support["gpcc_direct_gauge_month_fraction_1901_1920"].to_numpy(dtype=float).reshape(lat_grid.shape)
    dual_any = joint & (cru_fraction > 0) & (gpcc_fraction > 0)

    annual = pd.DataFrame({"year": YEARS})
    score_rows = []
    contribution_rows = []
    group_rows = []
    for region, (lower_lon, upper_lon) in REGIONS.items():
        mask = dual_any & (lon_grid >= lower_lon) & (lon_grid < upper_lon)
        count = int(mask.sum())
        if count < 5:
            raise RuntimeError(f"insufficient-fixed-support:{region}:{count}")
        curves = {label: weighted_mean(field, latitudes, mask) for label, field in member_fields.items()}
        ensemble_curve = weighted_mean(archived_ensemble, latitudes, mask)
        rebuilt_curve = np.mean(np.stack([curves[label] for label in MEMBERS], axis=0), axis=0)
        curve_closure = float(np.max(np.abs(ensemble_curve - rebuilt_curve)))
        if curve_closure > 2e-6:
            raise RuntimeError(f"six-member-curve-closure-failed:{region}:{curve_closure}")
        annual[f"{region}_ensemble_mm_day"] = ensemble_curve
        for label, curve in curves.items():
            annual[f"{region}_{label}_mm_day"] = curve
        for label, field in references.items():
            annual[f"{region}_{label.lower()}_mm_day"] = weighted_mean(field, latitudes, mask)

        member_trends = {label: trend(curve) for label, curve in curves.items()}
        ensemble_trend = trend(ensemble_curve)
        trend_closure = float(np.mean(list(member_trends.values())) - ensemble_trend)
        score_rows.append({
            "longitude_region": region,
            "dual_any_gridcell_count": count,
            "ensemble_trend_mm_day_per_century": ensemble_trend,
            "cru_trend_mm_day_per_century": trend(annual[f"{region}_cru_mm_day"]),
            "gpcc_trend_mm_day_per_century": trend(annual[f"{region}_gpcc_mm_day"]),
            "member_mean_trend_closure_mm_day_per_century": trend_closure,
            "curve_closure_max_abs_error_mm_day": curve_closure,
            "most_negative_member": min(member_trends, key=member_trends.get),
            "most_negative_member_trend_mm_day_per_century": min(member_trends.values()),
            "most_positive_member": max(member_trends, key=member_trends.get),
            "most_positive_member_trend_mm_day_per_century": max(member_trends.values()),
            "negative_member_count": int(sum(value < 0 for value in member_trends.values())),
            "positive_member_count": int(sum(value > 0 for value in member_trends.values())),
        })
        for label, value in member_trends.items():
            contribution_rows.append({
                "longitude_region": region,
                "member": label,
                "method": label.rsplit("_", 1)[0],
                "proxy_network": int(label.rsplit("_", 1)[1]),
                "member_trend_mm_day_per_century": value,
                "equal_weight_contribution_to_ensemble_trend_mm_day_per_century": value / len(MEMBERS),
                "member_event_change_mm_day": event_change(curves[label]),
            })
        for method in ("AO", "HGAO", "O"):
            labels = [f"{method}_6", f"{method}_9"]
            group_curve = np.mean(np.stack([curves[label] for label in labels]), axis=0)
            group_rows.append({
                "longitude_region": region,
                "group_type": "method",
                "group": method,
                "member_count": 2,
                "trend_mm_day_per_century": trend(group_curve),
            })
        for network in (6, 9):
            labels = [f"AO_{network}", f"HGAO_{network}", f"O_{network}"]
            group_curve = np.mean(np.stack([curves[label] for label in labels]), axis=0)
            group_rows.append({
                "longitude_region": region,
                "group_type": "proxy_network",
                "group": str(network),
                "member_count": 3,
                "trend_mm_day_per_century": trend(group_curve),
            })
    scores = pd.DataFrame(score_rows)
    contributions = pd.DataFrame(contribution_rows)
    groups = pd.DataFrame(group_rows)

    output.mkdir(parents=True)
    annual.to_csv(output / "fixed_support_six_member_curves_1901_1920.csv", index=False)
    scores.to_csv(output / "fixed_support_ensemble_scores.csv", index=False)
    contributions.to_csv(output / "six_member_trend_contributions.csv", index=False)
    groups.to_csv(output / "method_and_proxy_network_group_trends.csv", index=False)

    member_colors = {
        "AO_6": "#0072B2", "AO_9": "#56B4E9", "HGAO_6": "#009E73",
        "HGAO_9": "#E69F00", "O_6": "#CC79A7", "O_9": "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.reshape(-1)[:3], REGIONS):
        for label, color in member_colors.items():
            axis.plot(YEARS, annual[f"{region}_{label}_mm_day"], color=color, linewidth=1.2, alpha=0.85, label=label)
        axis.plot(YEARS, annual[f"{region}_ensemble_mm_day"], color="#111111", linewidth=2.8, label="Six-member mean")
        axis.plot(YEARS, annual[f"{region}_cru_mm_day"], color="#6A3D9A", linewidth=1.3, linestyle="--", label="CRU")
        axis.plot(YEARS, annual[f"{region}_gpcc_mm_day"], color="#777777", linewidth=1.3, linestyle="--", label="GPCC")
        count = int(scores.loc[scores["longitude_region"] == region, "dual_any_gridcell_count"].iloc[0])
        axis.set_title(f"{REGION_LABELS[region]} fixed dual support (n={count})")
        axis.set_ylabel("Precipitation anomaly (mm/day)")
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)

    axis = axes[1, 1]
    x = np.arange(len(REGIONS))
    width = 0.11
    for index, (label, color) in enumerate(member_colors.items()):
        values = [float(contributions[(contributions["longitude_region"] == region) & (contributions["member"] == label)]["member_trend_mm_day_per_century"].iloc[0]) for region in REGIONS]
        axis.bar(x + (index - 2.5) * width, values, width=width, color=color, label=label)
    axis.plot(x, scores["ensemble_trend_mm_day_per_century"], color="#111111", marker="o", linewidth=1.8, label="Mean")
    axis.set_xticks(x, [REGION_LABELS[name] for name in REGIONS])
    axis.set_ylabel("Trend (mm/day/century)")
    axis.set_title("Member trend decomposition")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "fixed_support_six_member_ensemble_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "archived_ensemble_definition": "Equal-weight finite-value mean of six rec_result median fields: AO/HGAO/O crossed with proxy networks 6/9.",
        "members": MEMBERS,
        "grid_closure": grid_closure,
        "scores": scores.to_dict(orient="records"),
        "member_contributions": contributions.to_dict(orient="records"),
        "group_trends": groups.to_dict(orient="records"),
        "scope_correction": "A replay of proxy6/EnKF/experiment1 is one native-grid experiment and cannot by itself be used as a prior/update or innovation/gain decomposition of pr_ensemble_mean_rec_result, which averages six median reconstruction fields.",
        "interpretation_boundary": "This experiment decomposes the archived production-facing rec_result ensemble but does not yet recover the 50-experiment distributions or EnKF operators for all six member fields.",
    }
    after = {
        name: source_state(path, include_hash=name != "hma_climate")
        for name, path in sources.items()
    }
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-six-member-audit")
    lines = [
        "# Precipitation rec_result Six-member Decomposition",
        "",
        "The archived rec_result field is the equal-weight mean of six median reconstruction fields, not EnKF experiment 1.",
        "",
    ]
    for row in scores.to_dict(orient="records"):
        lines.append(
            f"- {row['longitude_region']} n={row['dual_any_gridcell_count']}: ensemble trend "
            f"`{row['ensemble_trend_mm_day_per_century']:+.3f}`; member range "
            f"`{row['most_negative_member']} {row['most_negative_member_trend_mm_day_per_century']:+.3f}` to "
            f"`{row['most_positive_member']} {row['most_positive_member_trend_mm_day_per_century']:+.3f} mm/day/century`."
        )
    lines.extend([
        "",
        "The experiment-1 innovation/gain result remains a single-experiment mechanism diagnostic and must not be presented as the decomposition of the six-member archived ensemble.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
