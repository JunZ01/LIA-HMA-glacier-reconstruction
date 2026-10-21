"""Build the allowed-period family-balanced annual temperature reference field."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEMBERS = {
    "HGAO": "tas_rec_ref_HGAO_c05d",
    "LMR": "tas_rec_ref_LMR_c05d",
    "Nk_AM": "tas_rec_ref_Nk_AM_c05d",
    "Nk_CCA": "tas_rec_ref_Nk_CCA_c05d",
    "Nk_CPS": "tas_rec_ref_Nk_CPS_c05d",
    "Nk_DA": "tas_rec_ref_Nk_DA_c05d",
    "Nk_GraphEM": "tas_rec_ref_Nk_GraphEM_c05d",
    "Nk_PCR": "tas_rec_ref_Nk_PCR_c05d",
    "O": "tas_rec_ref_O_c05d",
    "PHYDA": "tas_rec_ref_PHYDA_c05d",
}
FILL_VALUE = np.float32(-9999.0)


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


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": sha256(path)}


def clean_slice(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection, :, :], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def read_allowed_member(dataset: h5py.Dataset, years: np.ndarray) -> np.ndarray:
    result = np.full((len(years), dataset.shape[1], dataset.shape[2]), np.nan, dtype=float)
    early = np.flatnonzero(years <= 1920)
    late = np.flatnonzero(years >= 1940)
    if not np.array_equal(early, np.arange(early[0], early[-1] + 1)):
        raise RuntimeError("early-period-not-contiguous")
    if not np.array_equal(late, np.arange(late[0], late[-1] + 1)):
        raise RuntimeError("late-period-not-contiguous")
    result[early] = clean_slice(dataset, slice(int(early[0]), int(early[-1]) + 1))
    result[late] = clean_slice(dataset, slice(int(late[0]), int(late[-1]) + 1))
    baseline = (years >= 1961) & (years <= 1990)
    with np.errstate(invalid="ignore"):
        reference = np.nanmean(result[baseline], axis=0)
    return result - reference[None, :, :]


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def smooth_allowed(years: np.ndarray, values: np.ndarray, width: int = 31) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window_years = years[index - half:index + half + 1]
        window = values[index - half:index + half + 1]
        if np.all(np.diff(window_years) == 1) and np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def write_netcdf(
    path: Path,
    years: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    spread: np.ndarray,
    family_count: np.ndarray,
    hma_mask: np.ndarray,
    source_hash: str,
) -> None:
    with netcdf_file(path, "w", version=2) as dataset:
        dataset.createDimension("year", len(years))
        dataset.createDimension("lat", len(latitudes))
        dataset.createDimension("lon", len(longitudes))
        year_var = dataset.createVariable("year", "i", ("year",))
        lat_var = dataset.createVariable("lat", "f", ("lat",))
        lon_var = dataset.createVariable("lon", "f", ("lon",))
        year_var[:] = years.astype(np.int32)
        lat_var[:] = latitudes.astype(np.float32)
        lon_var[:] = longitudes.astype(np.float32)
        year_var.units = "year CE"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        fields = {
            "tas_reference_family_balanced": candidate,
            "tas_reference_member_equal": control,
            "tas_reference_family_spread_std": spread,
        }
        for name, values in fields.items():
            variable = dataset.createVariable(name, "f", ("year", "lat", "lon"))
            variable._FillValue = FILL_VALUE
            variable.units = "degC"
            variable.reference_period = "1961-1990, independently recentered before aggregation"
            variable[:] = np.where(np.isfinite(values), values, FILL_VALUE).astype(np.float32)
        count_var = dataset.createVariable("tas_reference_valid_family_count", "b", ("year", "lat", "lon"))
        count_var._FillValue = np.int8(-1)
        count_var[:] = np.where(np.isfinite(candidate), family_count, -1).astype(np.int8)
        mask_var = dataset.createVariable("hma_mask", "b", ("lat", "lon"))
        mask_var[:] = hma_mask.astype(np.int8)
        mask_var.flag_values = "0,1"
        mask_var.flag_meanings = "outside_hma inside_hma"

        dataset.title = "HMA public-reference annual temperature, evidence-family balanced, allowed-period v1"
        dataset.source_sha256 = source_hash
        dataset.aggregation = "Equal mean of local-EnKF, LMR, Neukom, and PHYDA family fields"
        dataset.local_enkf_family = "finite mean of HGAO and O"
        dataset.neukom_family = "finite mean of AM, CCA, CPS, DA, GraphEM, and PCR"
        dataset.locked_period = "1921-1939 values were not read and are intentionally stored as FillValue"
        dataset.status = "comparison_reference_candidate_not_project_reconstruction_not_oggm_forcing"


def read_output(path: Path) -> dict[str, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as dataset:
        result = {
            "year": np.asarray(dataset.variables["year"].data, dtype=int).copy(),
            "lat": np.asarray(dataset.variables["lat"].data, dtype=float).copy(),
            "lon": np.asarray(dataset.variables["lon"].data, dtype=float).copy(),
            "candidate_raw": np.asarray(dataset.variables["tas_reference_family_balanced"].data, dtype=float).copy(),
            "control_raw": np.asarray(dataset.variables["tas_reference_member_equal"].data, dtype=float).copy(),
            "spread_raw": np.asarray(dataset.variables["tas_reference_family_spread_std"].data, dtype=float).copy(),
            "count_raw": np.asarray(dataset.variables["tas_reference_valid_family_count"].data, dtype=int).copy(),
            "hma_mask": np.asarray(dataset.variables["hma_mask"].data, dtype=int).copy(),
        }
    return result


def main() -> int:
    project = project_root()
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_21_temperature_reference_field_v1"
    source = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    result_path = output / "temperature_reference_family_balanced_allowed_v1.nc"
    paths = {"source": source, "hma_mask": mask_path, "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if result_path.exists() or (output / "summary.json").exists():
        raise FileExistsError(result_path)
    before = {name: state(path) for name, path in paths.items()}
    hma_mask = np.load(mask_path).astype(bool)

    with h5py.File(source, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(years, np.arange(1000, 2001)):
            raise RuntimeError("source-year-contract-failed")
        if hma_mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        members = {name: read_allowed_member(handle[variable], years) for name, variable in MEMBERS.items()}

    local_enkf = finite_mean([members["HGAO"], members["O"]])
    neukom = finite_mean([members[name] for name in members if name.startswith("Nk_")])
    families = {
        "local_enkf": local_enkf,
        "lmr": members["LMR"],
        "neukom": neukom,
        "phyda": members["PHYDA"],
    }
    candidate = finite_mean(list(families.values()))
    control = finite_mean(list(members.values()))
    family_stack = np.stack(list(families.values()))
    family_count = np.isfinite(family_stack).sum(axis=0).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        family_spread = np.nanstd(family_stack, axis=0, ddof=1)
    locked = (years >= 1921) & (years <= 1939)
    for values in (candidate, control, family_spread):
        values[locked] = np.nan
    family_count[locked] = 0

    write_netcdf(result_path, years, latitudes, longitudes, candidate, control, family_spread, family_count, hma_mask, before["source"]["sha256"])
    written = read_output(result_path)
    written_candidate = written["candidate_raw"]
    written_control = written["control_raw"]
    written_spread = written["spread_raw"]
    for values in (written_candidate, written_control, written_spread):
        values[np.isclose(values, FILL_VALUE)] = np.nan
    allowed = ~locked
    candidate_closure = float(np.nanmax(np.abs(written_candidate[allowed] - candidate[allowed])))
    control_closure = float(np.nanmax(np.abs(written_control[allowed] - control[allowed])))
    spread_closure = float(np.nanmax(np.abs(written_spread[allowed] - family_spread[allowed])))
    baseline = (years >= 1961) & (years <= 1990)
    baseline_mean = np.nanmean(written_candidate[baseline], axis=0)
    baseline_hma_max_abs = float(np.nanmax(np.abs(baseline_mean[hma_mask])))
    locked_all_missing = bool(np.all(~np.isfinite(written_candidate[locked])) and np.all(~np.isfinite(written_control[locked])) and np.all(~np.isfinite(written_spread[locked])) and np.all(written["count_raw"][locked] == -1))
    coordinate_closure = bool(np.array_equal(written["year"], years) and np.allclose(written["lat"], latitudes) and np.allclose(written["lon"], longitudes) and np.array_equal(written["hma_mask"].astype(bool), hma_mask))
    verification_passes = bool(candidate_closure <= 5e-6 and control_closure <= 5e-6 and spread_closure <= 5e-6 and baseline_hma_max_abs <= 5e-6 and locked_all_missing and coordinate_closure)

    curves = pd.DataFrame({"year": years})
    curves["family_balanced_hma_degC"] = weighted_series(candidate, hma_mask, latitudes)
    curves["member_equal_hma_degC"] = weighted_series(control, hma_mask, latitudes)
    curves["family_spread_hma_degC"] = weighted_series(family_spread, hma_mask, latitudes)
    for name, values in families.items():
        curves[f"{name}_hma_degC"] = weighted_series(values, hma_mask, latitudes)
    curves.to_csv(output / "hma_temperature_reference_curves.csv", index=False)

    period_rows: list[dict[str, object]] = []
    periods = [(1000, 1199), (1200, 1399), (1400, 1599), (1600, 1799), (1800, 1920), (1940, 2000)]
    for start, end in periods:
        selection = (years >= start) & (years <= end)
        period_rows.append({
            "start_year": start,
            "end_year": end,
            "family_balanced_mean_degC": float(np.nanmean(curves.loc[selection, "family_balanced_hma_degC"])),
            "member_equal_mean_degC": float(np.nanmean(curves.loc[selection, "member_equal_hma_degC"])),
            "family_balanced_minus_member_equal_degC": float(np.nanmean(curves.loc[selection, "family_balanced_hma_degC"] - curves.loc[selection, "member_equal_hma_degC"])),
            "mean_family_spread_degC": float(np.nanmean(curves.loc[selection, "family_spread_hma_degC"])),
        })
    pd.DataFrame(period_rows).to_csv(output / "period_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    smooth_candidate = smooth_allowed(years, curves.family_balanced_hma_degC.to_numpy(float))
    smooth_control = smooth_allowed(years, curves.member_equal_hma_degC.to_numpy(float))
    axes[0].plot(years, curves.family_balanced_hma_degC, color="#0072B2", linewidth=0.45, alpha=0.45)
    axes[0].plot(years, smooth_candidate, color="#0072B2", linewidth=2.0, label="Family balanced, 31-year")
    axes[0].plot(years, smooth_control, color="#333333", linewidth=1.5, label="Member equal, 31-year")
    axes[0].set(xlim=(1000, 2000), ylabel="HMA anomaly (degC)", title="Allowed-period temperature reference candidate")
    axes[0].legend(frameon=False)
    for name in families:
        axes[1].plot(years, smooth_allowed(years, curves[f"{name}_hma_degC"].to_numpy(float)), linewidth=1.2, label=name)
    axes[1].plot(years, smooth_candidate, color="#111111", linewidth=2.1, label="family balanced")
    axes[1].set(xlim=(1800, 2000), xlabel="Year", ylabel="HMA anomaly (degC)", title="Family influence near the instrumental period")
    axes[1].legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(output / "temperature_reference_family_balanced_allowed.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "complete",
        "locked_1921_1939_values_read": False,
        "output_netcdf": str(result_path),
        "output_sha256": sha256(result_path),
        "output_size_bytes": int(result_path.stat().st_size),
        "source_sha256": before["source"]["sha256"],
        "allowed_year_ranges": [[1000, 1920], [1940, 2000]],
        "locked_year_range": [1921, 1939],
        "verification": {
            "candidate_max_abs_closure": candidate_closure,
            "control_max_abs_closure": control_closure,
            "spread_max_abs_closure": spread_closure,
            "baseline_hma_max_abs_mean": baseline_hma_max_abs,
            "locked_all_missing": locked_all_missing,
            "coordinate_and_mask_closure": coordinate_closure,
            "passes": verification_passes,
        },
        "classification": "versioned_comparison_reference_candidate_not_project_reconstruction_not_oggm_forcing",
        "decision": "advance_to_preinstrumental_family_spread_and_reference_difference_audit" if verification_passes else "halt_output_verification_failed",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Reference Field v1\n\n"
        f"- NetCDF: `{result_path.name}` (`{summary['output_size_bytes']}` bytes).\n"
        f"- Allowed years: `1000-1920`, `1940-2000`; locked `1921-1939` all missing: `{str(locked_all_missing).lower()}`.\n"
        f"- Candidate/control/spread closure: `{candidate_closure:.3e}` / `{control_closure:.3e}` / `{spread_closure:.3e}`.\n"
        f"- 1961-1990 HMA maximum absolute baseline mean: `{baseline_hma_max_abs:.3e} degC`.\n"
        f"- Verification passes: `{str(verification_passes).lower()}`.\n\n"
        "This is a versioned comparison/reference candidate, not the project's reconstructed field and not an OGGM forcing.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    if not verification_passes:
        raise RuntimeError("output-verification-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
