"""Assess a controlled 50-record JJA PSM repair against instrumental temperature."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("controlled_enkf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("baseline-driver-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_grid(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Bilinear values from the 0.5-degree combined file at EnKF 2-degree centres."""
    lat2, lon2 = np.meshgrid(target_lat, target_lon, indexing="ij")
    fy = (lat2 - lat[0]) / (lat[1] - lat[0])
    fx = (lon2 - lon[0]) / (lon[1] - lon[0])
    inside = (fy >= 0) & (fy <= len(lat) - 1) & (fx >= 0) & (fx <= len(lon) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(lon) - 2)
    wy, wx = np.clip(fy - y0, 0, 1), np.clip(fx - x0, 0, 1)
    a, b, c, d = field[:, y0, x0], field[:, y0, x0 + 1], field[:, y0 + 1, x0], field[:, y0 + 1, x0 + 1]
    values = (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b) + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    values[:, ~inside] = np.nan
    return values


def to_cube(analysis: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    return np.asarray(analysis).reshape(nlon, nlat, analysis.shape[1]).transpose(2, 1, 0)


def anomaly(values: np.ndarray, years: np.ndarray, reference: tuple[int, int]) -> np.ndarray:
    period = (years >= reference[0]) & (years <= reference[1])
    return values - np.nanmean(values[period], axis=0)[None]


def rolling(values: np.ndarray, width: int = 15) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    half = width // 2
    for index in range(half, len(values) - half):
        out[index] = np.nanmean(values[index - half:index + half + 1])
    return out


def hma_mean(values: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def metrics(a: np.ndarray, b: np.ndarray, years: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(a) & np.isfinite(b) & mask[None]
    n = valid.sum(axis=0)
    am = np.sum(np.where(valid, a, 0), axis=0) / np.where(n == 0, np.nan, n)
    bm = np.sum(np.where(valid, b, 0), axis=0) / np.where(n == 0, np.nan, n)
    da, db = np.where(valid, a - am, 0), np.where(valid, b - bm, 0)
    corr = np.sum(da * db, axis=0) / np.sqrt(np.sum(da * da, axis=0) * np.sum(db * db, axis=0))
    rmse = np.sqrt(np.sum(np.where(valid, (a - b) ** 2, 0), axis=0) / np.where(n == 0, np.nan, n))
    x = years.astype(float)[:, None, None]
    xm = np.sum(np.where(valid, x, 0), axis=0) / np.where(n == 0, np.nan, n)
    trend_a = 100 * np.sum(np.where(valid, x - xm, 0) * da, axis=0) / np.sum(np.where(valid, x - xm, 0) ** 2, axis=0)
    trend_b = 100 * np.sum(np.where(valid, x - xm, 0) * db, axis=0) / np.sum(np.where(valid, x - xm, 0) ** 2, axis=0)
    sa, sb = np.apply_along_axis(rolling, 0, a), np.apply_along_axis(rolling, 0, b)
    sv = np.isfinite(sa) & np.isfinite(sb) & mask[None]
    sn = sv.sum(axis=0)
    sam = np.sum(np.where(sv, sa, 0), axis=0) / np.where(sn == 0, np.nan, sn)
    sbm = np.sum(np.where(sv, sb, 0), axis=0) / np.where(sn == 0, np.nan, sn)
    sda, sdb = np.where(sv, sa - sam, 0), np.where(sv, sb - sbm, 0)
    low = np.sum(sda * sdb, axis=0) / np.sqrt(np.sum(sda * sda, axis=0) * np.sum(sdb * sdb, axis=0))
    eligible = mask & (n >= 0.7 * len(years))
    return {
        "eligible_pixels": int(eligible.sum()),
        "median_annual_correlation": float(np.nanmedian(corr[eligible])),
        "median_lowpass_correlation_15y": float(np.nanmedian(low[eligible])),
        "median_rmse": float(np.nanmedian(rmse[eligible])),
        "median_abs_trend_difference": float(np.nanmedian(np.abs(trend_a[eligible] - trend_b[eligible]))),
        "same_trend_sign_fraction": float(np.nanmean((trend_a[eligible] * trend_b[eligible]) > 0)),
    }


def plot_curves(years: np.ndarray, curves: dict[str, np.ndarray], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for name, values in curves.items():
        axis.plot(years, rolling(values), linewidth=1.8, label=name)
    axis.axhspan(1901, 1950, color="0.9", zorder=-2)
    axis.axhline(0, color="0.3", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="HMA temperature: controlled JJA network repair versus instruments")
    axis.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_map(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str, path: Path) -> None:
    limit = np.nanpercentile(np.abs(values), 98)
    fig, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    image = axis.pcolormesh(lon, lat, values, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(xlabel="Longitude", ylabel="Latitude", title=title)
    fig.colorbar(image, ax=axis, label="degC")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    project = root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    archive = Path(cfg["archive_root"])
    source = cfg["sources"]
    output = project / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": project / source["network_audit"],
        "baseline_driver": project / source["baseline_driver"],
        "baseline_arrays": project / source["baseline_arrays"],
        "prior": project / source["prior"],
        "indices": archive / source["indices"],
        "proxy9": archive / source["proxy9"],
        "regem": project / source["proxy9_regem"],
        "netcdf": Path(source["reference_netcdf"]),
        "hma_mask": project / source["hma_mask"],
    }
    psm_path = output / "network_jja_psm_delta_anchor.mat"
    if not psm_path.exists():
        raise FileNotFoundError(f"missing MATLAB PSM output: {psm_path}")
    hashes_before = {name: sha256(path) for name, path in {**paths, "psm": psm_path}.items()}
    selection = pd.read_csv(paths["selection"])
    target = selection[(selection.lmr_metadata_available == True) & (selection.lmr_climate_variable == "temperature") & (selection.lmr_relation == "positive") & (selection.lmr_seasonality == "[6, 7, 8]")].copy()
    if len(target) != int(cfg["selection"]["expected_record_count"]):
        raise RuntimeError(f"selection count changed: {len(target)}")
    target.to_csv(output / "selected_records.csv", index=False)
    psm = scipy.io.loadmat(psm_path, squeeze_me=True)
    records = np.asarray(psm["record_ids"], dtype=int).reshape(-1)
    if not np.array_equal(records, target.proxy9_record_1based.to_numpy(dtype=int)):
        raise RuntimeError("MATLAB record IDs differ from independently selected records")
    annual_repeat = np.asarray(psm["annual_repeat_hxb"], dtype=float)
    annual = np.asarray(psm["annual_hxb"], dtype=float)
    preflight = {
        "selected_records": int(len(records)),
        "common_rng_annual_repeat_max_abs": float(np.nanmax(np.abs(annual_repeat - annual))),
        "annual_data_max_abs_median": float(np.nanmedian(np.asarray(psm["annual_data_max_abs"], dtype=float))),
        "annual_data_max_abs_max": float(np.nanmax(np.asarray(psm["annual_data_max_abs"], dtype=float))),
        "jja_delta_rms": float(np.sqrt(np.nanmean(np.asarray(psm["jja_hxb_delta"], dtype=float) ** 2))),
        "jja_delta_max_abs": float(np.nanmax(np.abs(np.asarray(psm["jja_hxb_delta"], dtype=float)))),
    }
    preflight["status"] = "pass" if preflight["common_rng_annual_repeat_max_abs"] <= 1e-12 else "fail"
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if preflight["status"] != "pass":
        raise RuntimeError("common-rng preflight failed")

    base = load_module(paths["baseline_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["regem"])["R0v_filled"], dtype=float)
    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    baseline = base._run_branch(prior, samples, proxy, residuals, target_weights, regional_weights)
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_closure = float(np.max(np.abs(np.asarray(baseline["analysis"]) - saved["proxy9_analysis"])))
    repaired_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    repaired_proxy["hxb"][records - 1] = np.asarray(psm["anchored_jja_hxb"], dtype=float)
    repaired = base._run_branch(prior, samples, repaired_proxy, residuals, target_weights, regional_weights)
    closure = max(float(item[key]) for item in (baseline, repaired) for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c"))
    if baseline_closure > 1e-12 or closure > 1e-10:
        raise RuntimeError(f"EnKF closure failure baseline={baseline_closure} closure={closure}")

    baseline_cube = anomaly(to_cube(np.asarray(baseline["analysis"]), len(grid_lat), len(grid_lon)), years, tuple(cfg["periods"]["reference"]))
    repaired_cube = anomaly(to_cube(np.asarray(repaired["analysis"]), len(grid_lat), len(grid_lon)), years, tuple(cfg["periods"]["reference"]))
    with h5py.File(paths["netcdf"], "r") as handle:
        source_lat, source_lon = np.asarray(handle["lat"], dtype=float), np.asarray(handle["lon"], dtype=float)
        obs = {}
        for name in ("tas_obs_bea_05deg", "tas_obs_cru_tmp_05deg", "tas_obs_GIST_05deg", "tas_obs_had5_05deg", "tas_obs_NOAA_05deg", "tas_ensemble_mean_rec_ref"):
            values = np.asarray(handle[name], dtype=float)
            values[values < -9000] = np.nan
            obs[name] = anomaly(source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, tuple(cfg["periods"]["reference"]))
    mask05 = np.load(paths["hma_mask"])
    mask2 = source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    comparisons = {"baseline_annual": baseline_cube, "repair_50_temperature_JJA": repaired_cube, "rec_ref": obs.pop("tas_ensemble_mean_rec_ref")}
    rows = []
    for label, field in comparisons.items():
        for obs_name, observed in obs.items():
            for period_name in ("early", "late"):
                start, end = cfg["periods"][period_name]
                use = (years >= start) & (years <= end)
                row = {"case": label, "observation": obs_name, "period": f"{start}-{end}"}
                row.update(metrics(field[use], observed[use], years[use], mask2))
                rows.append(row)
    pd.DataFrame(rows).to_csv(output / "instrumental_skill_summary_2deg.csv", index=False)
    regional = {name: hma_mean(field, grid_lat, mask2) for name, field in {**comparisons, **obs}.items()}
    pd.DataFrame({"year": years, **regional}).to_csv(output / "hma_regional_series_2deg.csv", index=False)
    use_plot = (years >= cfg["periods"]["plot"][0]) & (years <= cfg["periods"]["plot"][1])
    plot_curves(years[use_plot], {name: values[use_plot] for name, values in regional.items()}, output / "figure_hma_temperature_curves_1850_2000.png")
    early = (years >= cfg["periods"]["early"][0]) & (years <= cfg["periods"]["early"][1])
    plot_map(np.nanmean(repaired_cube[early] - baseline_cube[early], axis=0), grid_lat, grid_lon, "50-record JJA repair minus annual baseline, 1901-1950 mean", output / "map_repair_minus_baseline_1901_1950_mean.png")
    np.savez_compressed(output / "controlled_repair_arrays.npz", years=years, grid_lat=grid_lat, grid_lon=grid_lon, mask2=mask2, baseline_analysis=np.asarray(baseline["analysis"]), repaired_analysis=np.asarray(repaired["analysis"]), selected_records=records)
    hashes_after = {name: sha256(path) for name, path in {**paths, "psm": psm_path}.items()}
    pure = hashes_before == hashes_after
    summary = {"experiment_id": cfg["experiment_id"], "status": "pass" if pure else "fail", "source_hashes_unchanged": pure, "preflight": preflight, "baseline_state_closure_max_abs": baseline_closure, "enkf_closure_max_abs": closure, "selection": {"records": records.tolist(), "metadata_signs_forced": False}, "interpretation": "A controlled seasonal-operator sensitivity. It is not a final product and does not authorize sign constraints or replacement of the archived reconstruction."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text("# 50-record JJA Temperature Repair v1\n\nControlled read-only seasonal-operator experiment. Compare `instrumental_skill_summary_2deg.csv` across baseline, repair, and rec_ref; acceptance cannot be based on rec_ref alone.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("input purity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
