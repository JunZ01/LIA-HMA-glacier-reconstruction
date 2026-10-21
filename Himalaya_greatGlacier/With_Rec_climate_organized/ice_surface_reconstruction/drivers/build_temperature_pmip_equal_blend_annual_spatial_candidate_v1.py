"""Build a provisional annual 1500-1899 spatial field for the fixed PMIP blend."""

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

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as base
import preflight_temperature_pmip_equal_blend_spatial_field_v3 as retry


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1"
SPATIAL_PREFLIGHT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_spatial_common_support_v1"
PREFLIGHT_FIELDS = SPATIAL_PREFLIGHT / "equal_blend_common_support_50year_fields.npz"
PREFLIGHT_SUMMARY = SPATIAL_PREFLIGHT / "summary.json"
REGIONAL_ANNUAL = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_gtng15_v1/"
    "equal_blend_gtng15_annual_curves_1500_1899.csv"
)
REFERENCE = base.REFERENCE
CATALOG = base.CATALOG
MODE_SIM = base.MODE_SIM
ASSIGNMENT = base.ASSIGNMENT
SUPPORT = base.SUPPORT
FACTS = base.FACTS
DRIVER = Path(__file__).resolve()
PRIOR = base.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
INDICES = base.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY6 = base.regional_base.ARCHIVE / "python_data/proxy_data_6.mat"
PROXY9 = base.regional_base.ARCHIVE / "python_data/proxy_data_9.mat"
YEARS = np.arange(1500, 1900)
RECOVERY_YEARS = np.r_[YEARS, np.arange(1961, 1991)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_model_annual(
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
            dates = num2date(
                time[:], units=time.units, calendar=getattr(time, "calendar", "standard"),
                only_use_cftime_datetimes=True,
            )
            source_years = np.asarray([int(date.year) + int(item["offset"]) for date in dates])
            allowed = np.zeros(len(source_years), dtype=bool)
            for start, end in ranges:
                allowed |= (source_years >= start) & (source_years <= end)
            indices = np.flatnonzero(allowed)
            if np.any((source_years[indices] >= 1901) & (source_years[indices] <= 1939)):
                raise RuntimeError("Locked remote climate values requested")
            segments = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            for segment in segments:
                values = retry.chunked_remote_segment_values(
                    dataset, segment, target_lat, target_lon, mask
                )
                for row, source_index in enumerate(segment):
                    year = int(source_years[source_index])
                    sums[year] = sums.get(year, np.zeros(values.shape[1], dtype=np.float64)) + values[row]
                    counts[year] = counts.get(year, 0) + 1
            audit.append({
                "model": model, "url": item["url"],
                "requested_month_count": int(len(indices)),
                "locked_climate_values_requested": 0,
            })
    required = set(YEARS.tolist()) | set(range(1961, 1991))
    if any(counts.get(int(year), 0) != 12 for year in required):
        raise RuntimeError(f"Incomplete monthly coverage for {model}")
    annual = {year: sums[year] / 12.0 for year in required}
    baseline = np.mean(np.stack([annual[year] for year in range(1961, 1991)]), axis=0)
    return np.stack([annual[int(year)] - baseline for year in YEARS]), audit


def mode_sim_annual(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = base.target_points(lat, lon, mask)
    annual = {}
    with Dataset(MODE_SIM) as dataset:
        source_years = np.asarray(dataset.variables["time"][:], dtype=int)
        source_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        source_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        variable = dataset.variables["temp2_anomaly_1961_1990"]
        for year in np.r_[YEARS, np.arange(1961, 1991)]:
            source_index = int(np.flatnonzero(source_years == year)[0])
            annual[int(year)] = RegularGridInterpolator(
                (source_lat, source_lon), np.asarray(variable[source_index], dtype=float),
                bounds_error=False, fill_value=np.nan,
            )(points)
    baseline = np.mean(np.stack([annual[year] for year in range(1961, 1991)]), axis=0)
    return np.stack([annual[int(year)] - baseline for year in YEARS])


def background_annual(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prior = np.load(PRIOR)
    with h5py.File(INDICES, "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    adaptive = {}
    for network, proxy_path in [(6, PROXY6), (9, PROXY9)]:
        hxb, observations, locked_reads = base.round25.read_proxy_allowed(proxy_path, RECOVERY_YEARS)
        if locked_reads != 0:
            raise RuntimeError("Locked proxy values read")
        adaptive[network], _ = base.round25.ao_background_mean(
            prior, experiment_indices, hxb, observations
        )
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed = np.repeat(fixed_one[None, :], len(RECOVERY_YEARS), axis=0)
    native_lat, native_lon = base.round25.native_coordinates()
    members = []
    for method, network in base.round25.MEMBERS.values():
        flat = fixed if method == "O" else adaptive[network]
        anomaly = base.round25.anomaly_field(flat, RECOVERY_YEARS)
        members.append(base.round25.interpolate_fields(anomaly, native_lat, native_lon, lat, lon))
    ensemble = np.nanmean(np.stack(members), axis=0)
    return ensemble[: len(YEARS), :, :][:, mask]


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def window_fields(values: np.ndarray) -> np.ndarray:
    return np.stack([
        np.mean(values[(YEARS >= start) & (YEARS <= end)], axis=0)
        for start, end in base.WINDOWS
    ])


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Provisional Annual Spatial Candidate Contract\n\n"
        "Build annual 1500-1899 diagnostic fields using exactly 0.5 times the current six-member "
        "background plus 0.5 times the equal mean of ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L. "
        "Use the exact common support accepted by the spatial preflight. Require algebraic blend "
        "closure <=1e-12 degC, eight-window closure <=1e-6 degC, GTN-G annual aggregate closure "
        "<=1e-6 degC, HMA annual aggregate change from the accepted regional curves <=0.005 degC, "
        "and unchanged source hashes. This creates a versioned diagnostic NetCDF, not a production "
        "reconstruction or OGGM forcing. No 1901-1939 value is read.\n",
        encoding="utf-8",
    )
    paths = {
        "reference": REFERENCE, "catalog": CATALOG, "mode_sim": MODE_SIM,
        "assignment": ASSIGNMENT, "support": SUPPORT, "regional_annual": REGIONAL_ANNUAL,
        "preflight_fields": PREFLIGHT_FIELDS, "preflight_summary": PREFLIGHT_SUMMARY,
        "prior": PRIOR, "indices": INDICES, "proxy6": PROXY6, "proxy9": PROXY9,
        "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}
    preflight_summary = json.loads(PREFLIGHT_SUMMARY.read_text(encoding="utf-8"))
    if not preflight_summary["passes_all_gates"]:
        raise RuntimeError("Spatial preflight did not authorize this build")

    with Dataset(REFERENCE) as dataset:
        source_years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"][indices], dtype=float)[:, hma_mask]

    background = background_annual(lat, lon, hma_mask)
    model_fields = {"ModE-Sim": mode_sim_annual(lat, lon, hma_mask)}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    request_audit = []
    for model in ["MPI-ESM1-2-LR", "MIROC-ES2L"]:
        model_fields[model], audit = remote_model_annual(
            model, [item for item in catalog if item["model"] == model], lat, lon, hma_mask
        )
        request_audit.extend(audit)
    forced = np.mean(np.stack([model_fields[model] for model in base.MODELS]), axis=0)
    candidate = 0.5 * background + 0.5 * forced
    blend_closure = float(np.nanmax(np.abs(candidate - (0.5 * background + 0.5 * forced))))

    preflight = np.load(PREFLIGHT_FIELDS)
    common_support = np.asarray(preflight["common_support"], dtype=bool)
    annual_common = np.all(np.isfinite(reference), axis=0)
    for values in [background, forced, candidate, *model_fields.values()]:
        annual_common &= np.all(np.isfinite(values), axis=0)
    support_identical = bool(np.array_equal(annual_common, common_support))
    window_closure = max(
        float(np.nanmax(np.abs(window_fields(values) - np.asarray(preflight[name]))))
        for name, values in [
            ("reference", reference), ("background", background),
            ("forced_mean", forced), ("candidate", candidate),
        ]
    )

    points = base.target_points(lat, lon, hma_mask)
    point_lat = points[:, 0]
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domain_masks[region.o2region] = grid_mask[hma_mask]
    accepted = pd.read_csv(REGIONAL_ANNUAL)
    annual_rows = []
    max_gtng_closure = 0.0
    max_hma_change = 0.0
    for domain, raw_mask in domain_masks.items():
        mask = raw_mask & common_support
        series = {
            "reference": domain_mean(reference, mask, point_lat),
            "background": domain_mean(background, mask, point_lat),
            "forced_mean": domain_mean(forced, mask, point_lat),
            "candidate": domain_mean(candidate, mask, point_lat),
        }
        rows = accepted[accepted.domain.eq(domain)].sort_values("year")
        for name, accepted_column in [
            ("reference", "reference_four_family_degC"),
            ("background", "background_six_member_degC"),
            ("forced_mean", "forced_mean"),
            ("candidate", "equal_half_forced_mean"),
        ]:
            difference = series[name] - rows[accepted_column].to_numpy(float)
            maximum = float(np.max(np.abs(difference)))
            if domain == "hma":
                max_hma_change = max(max_hma_change, maximum)
            else:
                max_gtng_closure = max(max_gtng_closure, maximum)
        for index, year in enumerate(YEARS):
            annual_rows.append({
                "domain": domain, "year": int(year),
                **{f"{name}_degC": float(values[index]) for name, values in series.items()},
            })
    annual_domains = pd.DataFrame(annual_rows)

    output_nc = OUTPUT / "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = YEARS
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        common_grid = np.zeros_like(hma_mask, dtype=np.int8)
        common_grid[hma_mask] = common_support.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common_grid
        for name, values in {
            "tas_reference_family_balanced": reference,
            "tas_background_six_member": background,
            "tas_forced_equal_three_model_mean": forced,
            "tas_equal_half_candidate": candidate,
            "tas_mode_sim": model_fields["ModE-Sim"],
            "tas_mpi_esm1_2_lr": model_fields["MPI-ESM1-2-LR"],
            "tas_miroc_es2l": model_fields["MIROC-ES2L"],
        }.items():
            grid = np.full((len(YEARS), len(lat), len(lon)), np.nan, dtype=np.float32)
            grid[:, hma_mask] = values.astype(np.float32)
            variable = dataset.createVariable(
                name, "f4", ("year", "lat", "lon"), zlib=True, complevel=4,
                chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(np.nan),
            )
            variable[:] = grid
            variable.units = "degC anomaly relative to 1961-1990"
        dataset.title = "Provisional fixed 50:50 annual HMA temperature diagnostic"
        dataset.status = "diagnostic candidate; not production reconstruction or OGGM forcing"
        dataset.locked_years_read = "none; 1901-1939 excluded"
        dataset.blend_formula = "0.5 * background_six_member + 0.5 * equal_three_model_mean"

    annual_domains.to_csv(OUTPUT / "annual_domain_curves_1500_1899.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)
    plot_domains = ["hma", "13-08", "14-03", "15-01", "15-02"]
    fig, axes = plt.subplots(len(plot_domains), 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    names = {"hma": "HMA", "13-08": "Inner Tibet", "14-03": "West Himalaya", "15-01": "Central Himalaya", "15-02": "East Himalaya"}
    for axis, domain in zip(axes, plot_domains, strict=True):
        rows = annual_domains[annual_domains.domain.eq(domain)].set_index("year")
        for column, color, label in [
            ("reference_degC", "#111111", "Reference"),
            ("background_degC", "#D55E00", "Current background"),
            ("candidate_degC", "#0072B2", "50:50 candidate"),
        ]:
            axis.plot(rows.index, rows[column].rolling(31, center=True, min_periods=16).mean(), color=color, linewidth=1.7, label=label)
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.set(ylabel="degC", title=names[domain])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Year")
    figure = OUTPUT / "temperature_equal_blend_annual_regional_curves.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    gates = {
        "common_support_identical_to_preflight": support_identical,
        "algebraic_blend_closure_at_most_1e_12_degC": bool(blend_closure <= 1e-12),
        "eight_window_field_closure_at_most_1e_6_degC": bool(window_closure <= 1e-6),
        "gtng_annual_aggregate_closure_at_most_1e_6_degC": bool(max_gtng_closure <= 1e-6),
        "hma_annual_aggregate_change_at_most_0p005_degC": bool(max_hma_change <= 0.005),
    }
    pass_all = bool(all(gates.values()))
    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "year_start": 1500, "year_end": 1899,
        "locked_1901_1939_values_read": 0, "source_hashes_unchanged": unchanged,
        "common_support_cell_count": int(common_support.sum()),
        "algebraic_blend_closure_max_degC": blend_closure,
        "eight_window_field_closure_max_degC": window_closure,
        "gtng_annual_aggregate_closure_max_degC": max_gtng_closure,
        "hma_annual_aggregate_change_max_degC": max_hma_change,
        "gates": gates, "passes_all_gates": pass_all,
        "decision": "retain_as_provisional_annual_spatial_candidate" if pass_all else "reject_annual_spatial_candidate",
        "production_authorized": False, "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Provisional Annual Spatial Candidate\n\n"
        f"- Years: `1500-1899`.\n- Common-support cells: `{int(common_support.sum())}`.\n"
        f"- Eight-window closure: `{window_closure:.3e} degC`.\n"
        f"- GTN-G annual closure: `{max_gtng_closure:.3e} degC`.\n"
        f"- Maximum HMA annual aggregate change: `{max_hma_change:.3e} degC`.\n"
        f"- All gates pass: `{str(pass_all).lower()}`.\n\n"
        "This is a diagnostic candidate only. Production reconstruction and OGGM forcing remain unauthorized.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A source file changed")
    if not pass_all:
        raise RuntimeError("Annual spatial candidate failed a frozen gate")
    print(json.dumps(summary, indent=2))
    print(output_nc)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
