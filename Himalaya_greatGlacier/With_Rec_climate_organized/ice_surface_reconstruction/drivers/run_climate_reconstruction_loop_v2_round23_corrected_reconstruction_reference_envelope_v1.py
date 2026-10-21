"""Compare corrected six-member reconstruction with the reference family envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = ("local_enkf", "lmr", "neukom", "phyda")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat(); digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def read_reference_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as dataset:
        latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    return latitudes, longitudes


def interpolate_to_reference(data: np.ndarray, tif_lat: np.ndarray, tif_lon: np.ndarray, ref_lat: np.ndarray, ref_lon: np.ndarray) -> tuple[np.ndarray, str]:
    if np.any(np.diff(tif_lat) < 0):
        tif_lat = tif_lat[::-1]
        data = data[:, ::-1, :]
    if np.any(np.diff(tif_lon) < 0):
        tif_lon = tif_lon[::-1]
        data = data[:, :, ::-1]
    lat_grid, lon_grid = np.meshgrid(ref_lat, ref_lon, indexing="ij")
    fy = (lat_grid - tif_lat[0]) / (tif_lat[1] - tif_lat[0])
    fx = (lon_grid - tif_lon[0]) / (tif_lon[1] - tif_lon[0])
    inside = (fy >= 0) & (fy <= len(tif_lat) - 1) & (fx >= 0) & (fx <= len(tif_lon) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(tif_lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(tif_lon) - 2)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    a = data[:, y0, x0]
    b = data[:, y0, x0 + 1]
    c = data[:, y0 + 1, x0]
    d = data[:, y0 + 1, x0 + 1]
    result = (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b) + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    result[:, ~inside] = np.nan
    return result, "bilinear_source_center_to_reference_center"

def read_allowed_tif(path: Path, years: np.ndarray, ref_lat: np.ndarray, ref_lon: np.ndarray) -> tuple[np.ndarray, str]:
    early_years = np.arange(1000, 1921)
    late_years = np.arange(1940, 2001)
    early_bands = (early_years - 999).tolist()
    late_bands = (late_years - 999).tolist()
    result = np.full((len(years), len(ref_lat), len(ref_lon)), np.nan, dtype=float)
    with rasterio.open(path) as source:
        if source.count != 1001 or source.height != len(ref_lat) or source.width != len(ref_lon):
            raise RuntimeError(f"tif-shape-contract-failed:{path.name}:{source.count}:{source.height}:{source.width}")
        early = source.read(early_bands).astype(float)
        late = source.read(late_bands).astype(float)
        nodata = source.nodata if source.nodata is not None else -9999.0
        early[np.isclose(early, nodata) | (early < -9000)] = np.nan
        late[np.isclose(late, nodata) | (late < -9000)] = np.nan
        tif_lat = np.asarray([source.xy(row, 0, offset="center")[1] for row in range(source.height)])
        tif_lon = np.asarray([source.xy(0, col, offset="center")[0] for col in range(source.width)])
    early, orientation = interpolate_to_reference(early, tif_lat, tif_lon, ref_lat, ref_lon)
    late, late_orientation = interpolate_to_reference(late, tif_lat, tif_lon, ref_lat, ref_lon)
    if orientation != late_orientation:
        raise RuntimeError("orientation-branch-mismatch")
    result[years <= 1920] = early
    result[years >= 1940] = late
    baseline = (years >= 1961) & (years <= 1990)
    with np.errstate(invalid="ignore"):
        result = result - np.nanmean(result[baseline], axis=0)[None, :, :]
    return result, orientation


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(np.sum(valid * weights[None], axis=(1, 2)) == 0, np.nan, np.sum(valid * weights[None], axis=(1, 2)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_23_corrected_reconstruction_reference_envelope_v1"
    lineage_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
    reference_curves_path = loop / "round_21_temperature_reference_field_v1/hma_temperature_reference_curves.csv"
    reference_netcdf_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    lineage = pd.read_csv(lineage_path)
    tas = lineage[lineage.variable == "tas"].copy()
    if len(tas) != 6 or set(tas.member) != {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}:
        raise RuntimeError("temperature-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}
    paths = {
        "lineage": lineage_path,
        "reference_curves": reference_curves_path,
        "reference_netcdf": reference_netcdf_path,
        "hma_mask": mask_path,
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
        **{f"member_{name}": path for name, path in member_paths.items()},
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for row in tas.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"source-tif-hash-mismatch:{row.member}")

    years = np.arange(1000, 2001)
    locked = (years >= 1921) & (years <= 1939)
    hma_mask = np.load(mask_path).astype(bool)
    ref_lat, ref_lon = read_reference_grid(reference_netcdf_path)
    members: dict[str, np.ndarray] = {}
    orientations: dict[str, str] = {}
    for name, path in member_paths.items():
        members[name], orientations[name] = read_allowed_tif(path, years, ref_lat, ref_lon)
        if np.any(np.isfinite(members[name][locked])):
            raise RuntimeError(f"locked-values-present:{name}")
    reconstruction = finite_mean(list(members.values()))
    reconstruction[locked] = np.nan
    curves = pd.read_csv(reference_curves_path)
    if not np.array_equal(curves.year.to_numpy(int), years):
        raise RuntimeError("reference-year-contract-failed")
    if not curves.loc[locked, [f"{name}_hma_degC" for name in FAMILIES]].isna().all().all():
        raise RuntimeError("reference-locked-values-present")
    reconstruction_series = weighted_series(reconstruction, hma_mask, ref_lat)
    output_curves = pd.DataFrame({"year": years, "corrected_six_member_hma_degC": reconstruction_series})
    for name in members:
        output_curves[f"{name}_hma_degC"] = weighted_series(members[name], hma_mask, ref_lat)
    for column in ["family_balanced_hma_degC", "member_equal_hma_degC", *[f"{name}_hma_degC" for name in FAMILIES]]:
        output_curves[f"reference_{column}"] = curves[column]
    output_curves.to_csv(output / "corrected_reconstruction_reference_curves.csv", index=False)

    windows = [(start, start + 49) for start in range(1000, 1900, 50)] + [(1940, 1989)]
    rows: list[dict[str, object]] = []
    for start, end in windows:
        selection = (years >= start) & (years <= end)
        if int(selection.sum()) != 50:
            raise RuntimeError("window-size-failed")
        family_means = np.asarray([float(np.nanmean(curves.loc[selection, f"{name}_hma_degC"])) for name in FAMILIES])
        rec_mean = float(np.nanmean(reconstruction_series[selection]))
        central = float(np.mean(family_means))
        spread = float(np.std(family_means, ddof=1))
        rows.append({
            "start_year": start,
            "end_year": end,
            "is_lia_window_1400_1899": bool(start >= 1400 and end <= 1899),
            "corrected_reconstruction_mean_degC": rec_mean,
            "reference_family_central_mean_degC": central,
            "reference_family_min_degC": float(np.min(family_means)),
            "reference_family_max_degC": float(np.max(family_means)),
            "reference_family_std_degC": spread,
            "reconstruction_minus_reference_central_degC": rec_mean - central,
            "reconstruction_family_standardized_difference": float((rec_mean - central) / spread) if spread > 0 else np.nan,
            "inside_reference_family_range": bool(np.min(family_means) <= rec_mean <= np.max(family_means)),
        })
    window_table = pd.DataFrame(rows)
    window_table.to_csv(output / "corrected_reconstruction_reference_50year_windows.csv", index=False)
    lia = window_table[window_table.is_lia_window_1400_1899]
    outside_count = int((~lia.inside_reference_family_range).sum())
    median_abs_z = float(np.median(np.abs(lia.reconstruction_family_standardized_difference)))
    material_conflict = bool(outside_count >= 3 or median_abs_z > 1.0)
    member_window = []
    for name in members:
        for row in rows:
            selection = (years >= row["start_year"]) & (years <= row["end_year"])
            member_window.append({"member": name, "start_year": row["start_year"], "end_year": row["end_year"], "mean_degC": float(np.nanmean(output_curves.loc[selection, f"{name}_hma_degC"]))})
    pd.DataFrame(member_window).to_csv(output / "member_50year_means.csv", index=False)

    summary = {
        "status": "complete",
        "locked_1921_1939_values_read": False,
        "member_count": 6,
        "orientations": orientations,
        "lia_window_count": int(len(lia)),
        "lia_outside_reference_family_range_count": outside_count,
        "lia_outside_reference_family_range_fraction": float(outside_count / len(lia)),
        "lia_median_absolute_family_standardized_difference": median_abs_z,
        "lia_median_reconstruction_minus_reference_central_degC": float(lia.reconstruction_minus_reference_central_degC.median()),
        "material_lia_conflict": material_conflict,
        "decision": "close_current_six_member_temperature_reconstruction_as_validated_lia_central_estimate" if material_conflict else "retain_current_reconstruction_within_reference_family_uncertainty",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    centers = (window_table.start_year + window_table.end_year) / 2
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.fill_between(centers, window_table.reference_family_min_degC, window_table.reference_family_max_degC, color="#9ecae1", alpha=0.55, label="Reference family range")
    axis.plot(centers, window_table.reference_family_central_mean_degC, color="#005f73", marker="o", linewidth=2, label="Reference central")
    axis.plot(centers, window_table.corrected_reconstruction_mean_degC, color="#D55E00", marker="s", linewidth=2, label="Corrected six-member reconstruction")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.axvspan(1400, 1900, color="#dddddd", alpha=0.25)
    axis.set(xlabel="50-year window center", ylabel="HMA temperature anomaly (degC)", title="Corrected reconstruction versus public-reference family envelope")
    axis.legend(frameon=False)
    fig.savefig(output / "corrected_reconstruction_reference_envelope.png", dpi=180)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Corrected Reconstruction vs Reference Envelope\n\n"
        f"- LIA windows outside four-family range: `{outside_count}/{len(lia)}`.\n"
        f"- Median absolute family-standardized difference: `{median_abs_z:.3f}`.\n"
        f"- Median reconstruction minus reference central: `{summary['lia_median_reconstruction_minus_reference_central_degC']:+.3f} degC`.\n"
        f"- Material LIA conflict: `{str(material_conflict).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
