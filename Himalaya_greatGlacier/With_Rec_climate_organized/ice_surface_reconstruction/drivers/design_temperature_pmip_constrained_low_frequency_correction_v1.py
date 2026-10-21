"""Design a minimum-curvature PMIP correction constrained by accepted windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
BUILD = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1"
ANNUAL = BUILD / "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
FREQUENCY_AUDIT = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_frequency_audit_v1/summary.json"
)
PREFLIGHT = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_spatial_common_support_v1/"
    "equal_blend_common_support_50year_fields.npz"
)
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_constrained_low_frequency_correction_v1"
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]
KEY_REGIONS = ["13-08", "14-03", "15-01", "15-02"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def split_frequency(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = pd.Series(values).rolling(31, center=True, min_periods=31).mean().to_numpy(float)
    return low, values - low


def corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def correction_operator(years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(years)
    constraint = np.zeros((len(WINDOWS), count), dtype=float)
    for row, (start, end) in enumerate(WINDOWS):
        selected = (years >= start) & (years <= end)
        constraint[row, selected] = 1.0 / int(selected.sum())
    second = np.zeros((count - 2, count), dtype=float)
    for row in range(count - 2):
        second[row, row : row + 3] = [1.0, -2.0, 1.0]
    penalty = second.T @ second + 1e-10 * np.eye(count)
    kkt = np.block([
        [penalty, constraint.T],
        [constraint, np.zeros((len(WINDOWS), len(WINDOWS)))],
    ])
    rhs = np.vstack([np.zeros((count, len(WINDOWS))), np.eye(len(WINDOWS))])
    operator = np.linalg.solve(kkt, rhs)[:count]
    return operator, constraint


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract_file = OUTPUT / "ROUND_CONTRACT.md"
    contract_file.write_text(
        "# Constrained Low-Frequency Correction Contract\n\n"
        "The direct annual blend is rejected. Construct a correction independently at every common-"
        "support grid cell by minimizing the sum of squared annual second differences, subject to "
        "eight exact constraints: each 50-year correction mean equals 0.5 times the accepted forced-"
        "minus-background window mean. Add this smooth correction to the original annual background. "
        "No parameter is fitted to reference data. Require constraint closure <=1e-10 degC; stored "
        "50-year field closure <=2e-6 degC; HMA corrected/background high-pass correlation >=0.95; "
        "median regional corrected/background high-pass correlation >=0.95 and every key region >=0.90; "
        "median regional high-pass correlation change versus reference >=-0.02 with at least 12/15 "
        "regions losing no more than 0.05; and HMA/key 1849-1850 jumps <=3 sigma. No 1901-1939 value "
        "is read. Passing retains a low-frequency diagnostic candidate only.\n",
        encoding="utf-8",
    )
    numerical_paths = {
        "annual": ANNUAL, "frequency_audit": FREQUENCY_AUDIT,
        "preflight": PREFLIGHT, "assignment": ASSIGNMENT,
        "support": SUPPORT, "driver": DRIVER, "contract": contract_file,
    }
    before = {name: sha256(path) for name, path in numerical_paths.items()}
    facts_snapshot = sha256(FACTS)
    frequency = json.loads(FREQUENCY_AUDIT.read_text(encoding="utf-8"))
    if frequency["passes_all_direct_annual_gates"]:
        raise RuntimeError("Direct annual blend was not rejected")

    with Dataset(ANNUAL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)[hma_mask]
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"][:], dtype=float)[:, hma_mask]
        background = np.asarray(dataset.variables["tas_background_six_member"][:], dtype=float)[:, hma_mask]
        forced = np.asarray(dataset.variables["tas_forced_equal_three_model_mean"][:], dtype=float)[:, hma_mask]
    if not np.array_equal(years, np.arange(1500, 1900)):
        raise RuntimeError("Unexpected years")

    operator, constraint = correction_operator(years)
    background_windows = constraint @ background[:, common]
    forced_windows = constraint @ forced[:, common]
    target_window_correction = 0.5 * (forced_windows - background_windows)
    correction_common = operator @ target_window_correction
    constraint_closure = float(np.max(np.abs(constraint @ correction_common - target_window_correction)))
    correction = np.full_like(background, np.nan)
    correction[:, common] = correction_common
    corrected = background + correction

    preflight = np.load(PREFLIGHT)
    window_closure = max(
        float(np.nanmax(np.abs(constraint @ values - np.asarray(preflight[name])[:, common])))
        for name, values in [
            ("background", background[:, common]),
            ("candidate", corrected[:, common]),
        ]
    )

    lat2, _ = np.meshgrid(lat, lon, indexing="ij")
    point_lat = lat2[hma_mask]
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    domain_names = {"hma": "High Mountain Asia"}
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domain_masks[region.o2region] = grid_mask[hma_mask]
        domain_names[region.o2region] = region.full_name

    metric_rows = []
    curve_rows = []
    for domain, raw_mask in domain_masks.items():
        mask = raw_mask & common
        series = {
            "reference": domain_mean(reference, mask, point_lat),
            "background": domain_mean(background, mask, point_lat),
            "correction": domain_mean(correction, mask, point_lat),
            "corrected": domain_mean(corrected, mask, point_lat),
        }
        frequencies = {name: split_frequency(values) for name, values in series.items() if name != "correction"}
        for index, year in enumerate(years):
            curve_rows.append({
                "domain": domain, "domain_name": domain_names[domain], "year": int(year),
                **{f"{name}_degC": float(values[index]) for name, values in series.items()},
                **{f"{name}_low31_degC": float(frequencies[name][0][index]) for name in frequencies},
                **{f"{name}_high31_degC": float(frequencies[name][1][index]) for name in frequencies},
            })
        reference_high = frequencies["reference"][1]
        background_high = frequencies["background"][1]
        corrected_high = frequencies["corrected"][1]
        background_ref_corr = corr(background_high, reference_high)
        corrected_ref_corr = corr(corrected_high, reference_high)
        differences = np.diff(series["corrected"])
        splice_index = int(np.flatnonzero(years[:-1] == 1849)[0])
        typical = np.delete(differences, splice_index)
        splice_sigma = float(abs(differences[splice_index] - np.mean(typical)) / np.std(typical, ddof=1))
        metric_rows.append({
            "domain": domain, "domain_name": domain_names[domain],
            "corrected_background_high31_correlation": corr(corrected_high, background_high),
            "background_reference_high31_correlation": background_ref_corr,
            "corrected_reference_high31_correlation": corrected_ref_corr,
            "corrected_minus_background_reference_high31_correlation": corrected_ref_corr - background_ref_corr,
            "corrected_1849_1850_jump_sigma": splice_sigma,
            "correction_annual_increment_p99_degC": float(np.percentile(np.abs(np.diff(series["correction"])), 99)),
        })
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    regional = metrics[metrics.domain.ne("hma")]
    hma = metrics[metrics.domain.eq("hma")].iloc[0]
    key = metrics[metrics.domain.isin(KEY_REGIONS)]
    gates = {
        "constraint_closure_at_most_1e_10_degC": bool(constraint_closure <= 1e-10),
        "fifty_year_field_closure_at_most_2e_6_degC": bool(window_closure <= 2e-6),
        "hma_corrected_background_highpass_correlation_at_least_0p95": bool(
            hma.corrected_background_high31_correlation >= 0.95
        ),
        "median_regional_corrected_background_highpass_correlation_at_least_0p95": bool(
            regional.corrected_background_high31_correlation.median() >= 0.95
        ),
        "all_key_regions_corrected_background_highpass_correlation_at_least_0p90": bool(
            (key.corrected_background_high31_correlation >= 0.90).all()
        ),
        "median_regional_reference_highpass_correlation_change_at_least_minus_0p02": bool(
            regional.corrected_minus_background_reference_high31_correlation.median() >= -0.02
        ),
        "at_least_12_regions_lose_no_more_than_0p05_reference_highpass_correlation": bool(
            (regional.corrected_minus_background_reference_high31_correlation >= -0.05).sum() >= 12
        ),
        "hma_and_key_splice_jumps_at_most_3_sigma": bool(
            hma.corrected_1849_1850_jump_sigma <= 3 and (key.corrected_1849_1850_jump_sigma <= 3).all()
        ),
    }
    pass_all = bool(all(gates.values()))

    output_nc = OUTPUT / "temperature_equal_blend_constrained_low_frequency_candidate_1500_1899.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(years))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = years
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        common_grid = np.zeros_like(hma_mask, dtype=np.int8)
        common_grid[hma_mask] = common.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common_grid
        for name, values in {
            "tas_reference_family_balanced": reference,
            "tas_background_six_member": background,
            "tas_low_frequency_correction": correction,
            "tas_corrected_candidate": corrected,
        }.items():
            grid = np.full((len(years), len(lat), len(lon)), np.nan, dtype=np.float32)
            grid[:, hma_mask] = values.astype(np.float32)
            variable = dataset.createVariable(
                name, "f4", ("year", "lat", "lon"), zlib=True, complevel=4,
                chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(np.nan),
            )
            variable[:] = grid
            variable.units = "degC anomaly relative to 1961-1990"
        dataset.title = "Constrained low-frequency PMIP correction diagnostic"
        dataset.status = "provisional diagnostic; not production or OGGM forcing"
        dataset.method = "minimum annual second-difference curvature under eight exact 50-year mean constraints"
        dataset.locked_years_read = "none; 1901-1939 excluded"

    metrics.to_csv(OUTPUT / "low_frequency_domain_metrics.csv", index=False)
    curves.to_csv(OUTPUT / "low_frequency_domain_curves.csv", index=False)
    hma_curves = curves[curves.domain.eq("hma")].set_index("year")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for column, color, label in [
        ("reference_low31_degC", "#111111", "Reference"),
        ("background_low31_degC", "#D55E00", "Current background"),
        ("corrected_low31_degC", "#0072B2", "Constrained low-frequency candidate"),
    ]:
        axes[0].plot(hma_curves.index, hma_curves[column], color=color, linewidth=1.8, label=label)
    axes[0].set(title="HMA low-frequency temperature", ylabel="degC")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(hma_curves.index, hma_curves.background_high31_degC, color="#D55E00", linewidth=1.0, label="Background high-pass")
    axes[1].plot(hma_curves.index, hma_curves.corrected_high31_degC, color="#0072B2", linewidth=1.0, alpha=0.85, label="Corrected high-pass")
    axes[1].set(title="Annual signal retained after low-frequency correction", ylabel="degC", xlabel="Year")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    figure = OUTPUT / "temperature_constrained_low_frequency_correction.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in numerical_paths.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "locked_1901_1939_values_read": 0,
        "numerical_sources_unchanged": unchanged, "facts_snapshot_sha256": facts_snapshot,
        "constraint_closure_max_degC": constraint_closure,
        "fifty_year_field_closure_max_degC": window_closure,
        "hma_corrected_background_highpass_correlation": float(hma.corrected_background_high31_correlation),
        "median_regional_corrected_background_highpass_correlation": float(regional.corrected_background_high31_correlation.median()),
        "median_regional_reference_highpass_correlation_change": float(regional.corrected_minus_background_reference_high31_correlation.median()),
        "regions_losing_more_than_0p05_reference_highpass_correlation": int((regional.corrected_minus_background_reference_high31_correlation < -0.05).sum()),
        "hma_splice_jump_sigma": float(hma.corrected_1849_1850_jump_sigma),
        "max_key_splice_jump_sigma": float(key.corrected_1849_1850_jump_sigma.max()),
        "gates": gates, "passes_all_gates": pass_all,
        "decision": "retain_constrained_low_frequency_candidate" if pass_all else "reject_constrained_low_frequency_candidate",
        "production_authorized": False, "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "facts_snapshot_sha256": facts_snapshot}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Constrained Low-Frequency Correction\n\n"
        f"- Constraint closure: `{constraint_closure:.3e} degC`.\n"
        f"- 50-year field closure: `{window_closure:.3e} degC`.\n"
        f"- HMA corrected/background high-pass correlation: `{summary['hma_corrected_background_highpass_correlation']:.3f}`.\n"
        f"- Median regional corrected/background high-pass correlation: `{summary['median_regional_corrected_background_highpass_correlation']:.3f}`.\n"
        f"- Median regional reference high-pass correlation change: `{summary['median_regional_reference_highpass_correlation_change']:+.3f}`.\n"
        f"- All gates pass: `{str(pass_all).lower()}`.\n\nDecision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A numerical source changed")
    print(json.dumps(summary, indent=2))
    print(output_nc)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
