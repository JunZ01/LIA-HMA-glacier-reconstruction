"""Build and validate eight diagnostic 0.5-degree maps for the fixed PMIP blend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import decompose_temperature_lia_regional_prior_update_v1 as regional_base
import extract_temperature_pmip_gtng15_regional_v1 as pmip_base
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_spatial_preflight_v1"
REFERENCE = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/round_21_temperature_reference_field_v1/"
    "temperature_reference_family_balanced_allowed_v1.nc"
)
CATALOG = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/remote_source_catalog.json"
MODE_SIM = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/round_74_mode_sim_allowed_temperature_acquisition_v3/"
    "mode_sim_temp2_allowed_hma_bbox.nc"
)
REGIONAL_WINDOWS = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_gtng15_v1/"
    "equal_blend_gtng15_50year_curves.csv"
)
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
MODELS = ["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
YEARS = np.r_[np.arange(1500, 1900), np.arange(1961, 1991)]
LOCKED_START = 1901
LOCKED_END = 1939


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_points(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    return np.column_stack([lat2[mask], lon2[mask]])


def remote_segment_values(
    dataset: Dataset,
    indices: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    lat_slice, lon_slice, source_lat, source_lon, lat_order, lon_order = pmip_base.source_subset(
        dataset, target_lat, target_lon
    )
    variable = dataset.variables["tas"]
    indexers: list[Any] = []
    kept = []
    for dimension in variable.dimensions:
        if dimension == "time":
            indexers.append(slice(int(indices[0]), int(indices[-1]) + 1))
            kept.append("time")
        elif dimension == "lat":
            indexers.append(lat_slice)
            kept.append("lat")
        elif dimension == "lon":
            indexers.append(lon_slice)
            kept.append("lon")
        else:
            indexers.append(0)
    fields = np.asarray(variable[tuple(indexers)], dtype=float).squeeze()
    if kept != ["time", "lat", "lon"] or fields.ndim != 3:
        raise RuntimeError(f"Unexpected remote field dimensions: {variable.dimensions}")
    fields = fields[:, lat_order][:, :, lon_order]
    points = target_points(target_lat, target_lon, mask)
    values = np.empty((len(indices), len(points)), dtype=np.float32)
    for index, field in enumerate(fields):
        values[index] = RegularGridInterpolator(
            (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
        )(points)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite remote interpolation inside HMA")
    return values


def remote_model_windows(
    model: str,
    files: list[dict[str, Any]],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    audit = []
    for item in files:
        ranges = [(1500, 1849)] if item["experiment_role"] == "past" else [(1850, 1900), (1940, 2000)]
        with Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"), only_use_cftime_datetimes=True)
            years = np.asarray([int(date.year) + int(item["offset"]) for date in dates])
            allowed = np.zeros(len(years), dtype=bool)
            for start, end in ranges:
                allowed |= (years >= start) & (years <= end)
            indices = np.flatnonzero(allowed)
            if np.any((years[indices] >= LOCKED_START) & (years[indices] <= LOCKED_END)):
                raise RuntimeError("Locked remote climate values requested")
            segments = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            for segment in segments:
                values = remote_segment_values(dataset, segment, target_lat, target_lon, mask)
                for row, source_index in enumerate(segment):
                    year = int(years[source_index])
                    sums[year] = sums.get(year, np.zeros(values.shape[1], dtype=np.float64)) + values[row]
                    counts[year] = counts.get(year, 0) + 1
            audit.append({
                "model": model,
                "url": item["url"],
                "requested_month_count": int(len(indices)),
                "locked_climate_values_requested": 0,
            })
    required = set(np.arange(1500, 1900)) | set(np.arange(1961, 1991))
    if any(counts.get(int(year), 0) != 12 for year in required):
        raise RuntimeError(f"Incomplete annual coverage for {model}")
    annual = {year: sums[year] / counts[year] for year in required}
    baseline = np.mean(np.stack([annual[year] for year in range(1961, 1991)]), axis=0)
    anomalies = {year: annual[year] - baseline for year in range(1500, 1900)}
    windows = np.stack([
        np.mean(np.stack([anomalies[year] for year in range(start, end + 1)]), axis=0)
        for start, end in WINDOWS
    ])
    return windows, audit


def mode_sim_windows(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = target_points(lat, lon, mask)
    annual: dict[int, np.ndarray] = {}
    with Dataset(MODE_SIM) as dataset:
        years = np.asarray(dataset.variables["time"][:], dtype=int)
        source_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        source_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        variable = dataset.variables["temp2_anomaly_1961_1990"]
        for year in np.r_[np.arange(1500, 1900), np.arange(1961, 1991)]:
            source_index = int(np.flatnonzero(years == year)[0])
            annual[int(year)] = RegularGridInterpolator(
                (source_lat, source_lon), np.asarray(variable[source_index], dtype=float),
                bounds_error=False, fill_value=np.nan,
            )(points)
    baseline = np.mean(np.stack([annual[year] for year in range(1961, 1991)]), axis=0)
    return np.stack([
        np.mean(np.stack([annual[year] - baseline for year in range(start, end + 1)]), axis=0)
        for start, end in WINDOWS
    ])


def background_windows(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    archive = regional_base.ARCHIVE
    prior_path = regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
    index_path = archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
    prior = np.load(prior_path)
    with h5py.File(index_path, "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    adaptive = {}
    for network in (6, 9):
        proxy_path = archive / f"python_data/proxy_data_{network}.mat"
        hxb, observations, locked_reads = round25.read_proxy_allowed(proxy_path, YEARS)
        if locked_reads != 0:
            raise RuntimeError("Locked proxy values read")
        adaptive[network], _ = round25.ao_background_mean(prior, experiment_indices, hxb, observations)
    fixed_one = np.mean(np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0)
    fixed = np.repeat(fixed_one[None, :], len(YEARS), axis=0)
    native_lat, native_lon = round25.native_coordinates()
    fields = []
    for method, network in round25.MEMBERS.values():
        flat = fixed if method == "O" else adaptive[network]
        fields.append(round25.interpolate_fields(round25.anomaly_field(flat, YEARS), native_lat, native_lon, lat, lon))
    ensemble = np.nanmean(np.stack(fields), axis=0)
    return np.stack([np.nanmean(ensemble[YEARS >= start][YEARS[YEARS >= start] <= end], axis=0)[mask] for start, end in WINDOWS])


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights2 = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights2[mask]
    return np.sum(values * weights[None], axis=1) / np.sum(weights)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Fixed 50:50 Spatial-Field Preflight Contract\n\n"
        "Generate only eight diagnostic 50-year fields (1500-1899) on the frozen 0.5-degree HMA grid. "
        "Use the already accepted fixed 0.5*background + 0.5*three-model-mean formula without fitting. "
        "Require complete finite HMA coverage, aggregate closure <=1e-6 degC against the accepted HMA/GTN-G "
        "curves, positive HMA spatiotemporal RMSE improvement, >=67% area-weighted HMA cells improving, "
        ">=12/15 GTN-G regions improving in field RMSE, and no key region (13-08, 14-03, 15-01, 15-02) "
        "degrading by more than 10%. No annual production field is generated and no 1901-1939 value is read.\n",
        encoding="utf-8",
    )
    local_paths = {
        "reference": REFERENCE, "catalog": CATALOG, "mode_sim": MODE_SIM,
        "regional_windows": REGIONAL_WINDOWS, "assignment": ASSIGNMENT,
        "support": SUPPORT, "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in local_paths.items()}
    with Dataset(REFERENCE) as dataset:
        all_years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        variable = dataset.variables["tas_reference_family_balanced"]
        reference_windows = np.stack([
            np.nanmean(np.asarray(variable[np.flatnonzero((all_years >= start) & (all_years <= end))], dtype=float), axis=0)[hma_mask]
            for start, end in WINDOWS
        ])

    background = background_windows(lat, lon, hma_mask)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    model_fields = {"ModE-Sim": mode_sim_windows(lat, lon, hma_mask)}
    request_audit = []
    for model in ["MPI-ESM1-2-LR", "MIROC-ES2L"]:
        model_fields[model], audit = remote_model_windows(
            model, [item for item in catalog if item["model"] == model], lat, lon, hma_mask
        )
        request_audit.extend(audit)
    forced = np.mean(np.stack([model_fields[model] for model in MODELS]), axis=0)
    candidate = 0.5 * background + 0.5 * forced
    if not all(np.all(np.isfinite(values)) for values in [reference_windows, background, forced, candidate]):
        raise RuntimeError("Spatial field contains non-finite HMA values")

    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    masks = {"hma": hma_mask}
    for region in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        masks[region.o2region] = mask
    accepted = pd.read_csv(REGIONAL_WINDOWS)
    closure_errors = []
    score_rows = []
    for domain, mask in masks.items():
        point_mask = mask[hma_mask]
        ref_domain = reference_windows[:, point_mask]
        bg_domain = background[:, point_mask]
        candidate_domain = candidate[:, point_mask]
        bg_rmse = float(np.sqrt(np.mean((bg_domain - ref_domain) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate_domain - ref_domain) ** 2)))
        score_rows.append({
            "domain": domain,
            "background_spatiotemporal_rmse_degC": bg_rmse,
            "candidate_spatiotemporal_rmse_degC": candidate_rmse,
            "improvement_vs_background": float(1 - candidate_rmse / bg_rmse),
        })
        for index, (start, end) in enumerate(WINDOWS):
            row = accepted[(accepted.domain.eq(domain)) & (accepted.start_year.eq(start)) & (accepted.end_year.eq(end))].iloc[0]
            closure_errors.extend([
                abs(float(weighted_mean(reference_windows[index:index + 1], mask, lat)[0]) - float(row.reference_degC)),
                abs(float(weighted_mean(background[index:index + 1], mask, lat)[0]) - float(row.background_degC)),
                abs(float(weighted_mean(candidate[index:index + 1], mask, lat)[0]) - float(row.candidate_degC)),
            ])
    scores = pd.DataFrame(score_rows)
    regional_scores = scores[scores.domain.ne("hma")]
    hma_score = scores[scores.domain.eq("hma")].iloc[0]

    bg_cell_rmse = np.sqrt(np.mean((background - reference_windows) ** 2, axis=0))
    candidate_cell_rmse = np.sqrt(np.mean((candidate - reference_windows) ** 2, axis=0))
    cell_improvement = 1 - candidate_cell_rmse / bg_cell_rmse
    point_lat = target_points(lat, lon, hma_mask)[:, 0]
    point_weights = np.cos(np.deg2rad(point_lat))
    area_improving_fraction = float(np.sum(point_weights[cell_improvement > 0]) / np.sum(point_weights))
    key = regional_scores[regional_scores.domain.isin(["13-08", "14-03", "15-01", "15-02"])]
    closure_max = float(max(closure_errors))
    gates = {
        "finite_hma_coverage": True,
        "aggregate_closure_at_most_1e_6_degC": bool(closure_max <= 1e-6),
        "hma_spatiotemporal_rmse_improves": bool(hma_score.improvement_vs_background > 0),
        "area_weighted_improving_cell_fraction_at_least_0p67": bool(area_improving_fraction >= 0.67),
        "at_least_12_of_15_regions_improve_in_field_rmse": bool((regional_scores.improvement_vs_background > 0).sum() >= 12),
        "key_regions_do_not_degrade_more_than_0p10": bool((key.improvement_vs_background >= -0.10).all()),
    }
    pass_all = bool(all(gates.values()))

    np.savez_compressed(
        OUTPUT / "equal_blend_diagnostic_50year_fields_hma_points.npz",
        windows=np.asarray(WINDOWS), lat=lat, lon=lon, hma_mask=hma_mask,
        reference=reference_windows, background=background, forced_mean=forced, candidate=candidate,
    )
    scores.to_csv(OUTPUT / "spatial_field_domain_scores.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)

    maps = []
    for values in [bg_cell_rmse, candidate_cell_rmse, cell_improvement * 100]:
        field = np.full(hma_mask.shape, np.nan)
        field[hma_mask] = values
        maps.append(field)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    specs = [
        (maps[0], "Current background RMSE", "viridis", None, "degC"),
        (maps[1], "50:50 candidate RMSE", "viridis", None, "degC"),
        (maps[2], "Cell-wise RMSE improvement", "RdBu_r", (-100, 100), "%"),
    ]
    for axis, (field, title, cmap, limits, label) in zip(axes, specs, strict=True):
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, **kwargs)
        axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(image, ax=axis, label=label, shrink=0.82)
    figure = OUTPUT / "temperature_pmip_equal_blend_spatial_preflight.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in local_paths.items()}
    unchanged = before == after
    decision = "authorize_provisional_annual_spatial_candidate_build" if pass_all else "reject_fixed_uniform_spatial_blend"
    summary = {
        "status": "complete", "locked_1901_1939_values_read": 0,
        "source_hashes_unchanged": unchanged, "aggregate_closure_max_degC": closure_max,
        "hma_spatiotemporal_rmse_improvement": float(hma_score.improvement_vs_background),
        "area_weighted_improving_cell_fraction": area_improving_fraction,
        "improving_region_count": int((regional_scores.improvement_vs_background > 0).sum()),
        "worst_region": str(regional_scores.loc[regional_scores.improvement_vs_background.idxmin(), "domain"]),
        "worst_region_improvement": float(regional_scores.improvement_vs_background.min()),
        "gates": gates, "passes_all_gates": pass_all, "decision": decision,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Fixed 50:50 Spatial Preflight\n\n"
        f"- HMA spatiotemporal RMSE improvement: `{100 * summary['hma_spatiotemporal_rmse_improvement']:+.1f}%`.\n"
        f"- Area-weighted improving HMA cells: `{100 * area_improving_fraction:.1f}%`.\n"
        f"- Improving GTN-G regions: `{summary['improving_region_count']}/15`.\n"
        f"- Aggregate closure maximum: `{closure_max:.3e} degC`.\n"
        f"- All frozen gates pass: `{str(pass_all).lower()}`.\n\nDecision: `{decision}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A local source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
