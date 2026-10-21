"""Apply the frozen low-frequency correction in the sealed 1921-1939 test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_low_frequency_blind_test_1921_1939_v1"
UNLOCK = OUTPUT / "UNLOCK_DECISION.md"
CORRECTION = RESULTS / (
    "climate_reconstruction_temperature_low_frequency_correction_bridge_1900_1940_v1/"
    "temperature_low_frequency_correction_only_1500_1940.nc"
)
ROUND25_MANIFEST = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_25_temperature_prior_update_decomposition_v1/source_manifest.json"
)
REFERENCE_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\"
    r"climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
YEARS = np.arange(1880, 1991)
TEST_YEARS = np.arange(1921, 1940)
LOW_CENTERS = np.arange(1926, 1935)
KEY_REGIONS = ["13-08", "14-03", "15-01", "15-02"]
EXPECTED_CORRECTION_SHA256 = "16a7cb2e39da04f283ae2429ecbe4ff4b408e612c4a581fc89afadb100d304c0"
EXPECTED_REFERENCE_SHA256 = "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_without_hash(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def finite_mean(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def read_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(REFERENCE_SOURCE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(source_years, np.arange(1000, 2001)):
            raise RuntimeError("reference-year-contract-failed")
        indices = YEARS - int(source_years[0])
        members: dict[str, np.ndarray] = {}
        for name, variable_name in round21.MEMBERS.items():
            dataset = handle[variable_name]
            values = np.asarray(dataset[int(indices[0]) : int(indices[-1]) + 1], dtype=float)
            fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
            values[np.isclose(values, fill) | (values < -9000)] = np.nan
            baseline = (YEARS >= 1961) & (YEARS <= 1990)
            values -= np.nanmean(values[baseline], axis=0, keepdims=True)
            members[name] = values
    local_enkf = finite_mean([members["HGAO"], members["O"]])
    neukom = finite_mean([members[name] for name in members if name.startswith("Nk_")])
    reference = finite_mean([local_enkf, members["LMR"], neukom, members["PHYDA"]])
    return lat, lon, np.load(HMA_MASK).astype(bool), reference


def read_project_posterior(target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    manifest = json.loads(ROUND25_MANIFEST.read_text(encoding="utf-8"))["sources_before"]
    source_lat, source_lon = round25.native_coordinates()
    members = []
    for member in round25.MEMBERS:
        path = Path(manifest[f"posterior_{member}"]["path"])
        posterior = round25.read_posterior_mean(path, YEARS)
        anomaly = round25.anomaly_field(posterior, YEARS)
        members.append(round25.interpolate_fields(anomaly, source_lat, source_lon, target_lat, target_lon))
    return finite_mean(members)


def read_correction(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(CORRECTION) as dataset:
        correction_years = np.asarray(dataset.variables["year"][:], dtype=int)
        if not np.array_equal(np.asarray(dataset.variables["lat"][:], dtype=float), lat):
            raise RuntimeError("correction-latitude-mismatch")
        if not np.array_equal(np.asarray(dataset.variables["lon"][:], dtype=float), lon):
            raise RuntimeError("correction-longitude-mismatch")
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        selected = np.flatnonzero((correction_years >= 1880) & (correction_years <= 1940))
        if not np.array_equal(correction_years[selected], np.arange(1880, 1941)):
            raise RuntimeError("correction-year-contract-failed")
        values = np.asarray(
            dataset.variables["tas_low_frequency_correction_only"][selected], dtype=float
        )
    return common, values


def domain_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights2 = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights2[None], axis=(1, 2))
    denominator = np.sum(valid * weights2[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def smooth11(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(11, center=True, min_periods=11).mean().to_numpy(float)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def main() -> int:
    for path in [UNLOCK, CORRECTION, ROUND25_MANIFEST, REFERENCE_SOURCE, HMA_MASK, ASSIGNMENT, SUPPORT]:
        if not path.exists():
            raise FileNotFoundError(path)
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    if sha256(CORRECTION) != EXPECTED_CORRECTION_SHA256:
        raise RuntimeError("frozen-correction-hash-mismatch")
    manifest25 = json.loads(ROUND25_MANIFEST.read_text(encoding="utf-8"))["sources_before"]
    expected_reference = manifest25["reference_netcdf"]
    reference_stat = REFERENCE_SOURCE.stat()
    if int(reference_stat.st_size) != 225274031:
        raise RuntimeError("reference-source-size-mismatch")

    tracked = {
        "reference_source": REFERENCE_SOURCE,
        "correction": CORRECTION,
        "hma_mask": HMA_MASK,
        "assignment": ASSIGNMENT,
        "support": SUPPORT,
        "round25_manifest": ROUND25_MANIFEST,
        "unlock": UNLOCK,
        "driver": DRIVER,
    }
    for member in round25.MEMBERS:
        tracked[f"posterior_{member}"] = Path(manifest25[f"posterior_{member}"]["path"])
    before = {name: state_without_hash(path) for name, path in tracked.items()}
    facts_snapshot = sha256(FACTS)

    lat, lon, hma_mask, reference = read_reference()
    project = read_project_posterior(lat, lon)
    common_mask, correction_1880_1940 = read_correction(lat, lon)
    common = hma_mask & common_mask
    if int(common.sum()) != 1580:
        raise RuntimeError(f"unexpected-common-support:{int(common.sum())}")
    correction = np.full_like(project, np.nan)
    correction_indices = np.flatnonzero((YEARS >= 1880) & (YEARS <= 1940))
    correction[correction_indices] = correction_1880_1940
    candidate = project.copy()
    candidate[correction_indices] = project[correction_indices] + correction[correction_indices]

    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domains: dict[str, tuple[str, np.ndarray]] = {"hma": ("High Mountain Asia", hma_mask.copy())}
    for region in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        domains[str(region.o2region)] = (str(region.full_name), mask)

    test = np.isin(YEARS, TEST_YEARS)
    centers = np.isin(YEARS, LOW_CENTERS)
    metric_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    series_by_domain: dict[str, dict[str, np.ndarray]] = {}
    for domain, (domain_name, raw_mask) in domains.items():
        mask = raw_mask & common
        series = {
            "reference": domain_mean(reference, mask, lat),
            "original": domain_mean(project, mask, lat),
            "correction": domain_mean(correction, mask, lat),
            "candidate": domain_mean(candidate, mask, lat),
        }
        series_by_domain[domain] = series
        low = {name: smooth11(values) for name, values in series.items() if name != "correction"}
        high_original = series["original"] - low["original"]
        high_candidate = series["candidate"] - low["candidate"]
        original_rmse = float(np.sqrt(np.mean((low["original"][centers] - low["reference"][centers]) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((low["candidate"][centers] - low["reference"][centers]) ** 2)))
        annual_original_rmse = float(np.sqrt(np.mean((series["original"][test] - series["reference"][test]) ** 2)))
        annual_candidate_rmse = float(np.sqrt(np.mean((series["candidate"][test] - series["reference"][test]) ** 2)))
        allowed_increment_mask = (
            ((YEARS[:-1] >= 1880) & (YEARS[1:] <= 1920))
            | ((YEARS[:-1] >= 1940) & (YEARS[1:] <= 1960))
        )
        allowed_increments = np.diff(series["original"])[allowed_increment_mask]
        increment_mean = float(np.mean(allowed_increments))
        increment_std = float(np.std(allowed_increments, ddof=1))
        jump_1899 = float(series["candidate"][YEARS == 1900][0] - series["candidate"][YEARS == 1899][0])
        jump_1939 = float(series["candidate"][YEARS == 1940][0] - series["candidate"][YEARS == 1939][0])
        metric_rows.append({
            "domain": domain,
            "domain_name": domain_name,
            "common_cell_count": int(mask.sum()),
            "original_low11_rmse_degC": original_rmse,
            "candidate_low11_rmse_degC": candidate_rmse,
            "low11_rmse_improvement_fraction": float(1.0 - candidate_rmse / original_rmse),
            "original_annual_rmse_degC": annual_original_rmse,
            "candidate_annual_rmse_degC": annual_candidate_rmse,
            "annual_rmse_improvement_fraction": float(1.0 - annual_candidate_rmse / annual_original_rmse),
            "original_low11_reference_correlation": correlation(low["original"][centers], low["reference"][centers]),
            "candidate_low11_reference_correlation": correlation(low["candidate"][centers], low["reference"][centers]),
            "candidate_original_high11_correlation": correlation(high_candidate[centers], high_original[centers]),
            "reference_trend_degC_century": slope_per_century(TEST_YEARS, series["reference"][test]),
            "original_trend_degC_century": slope_per_century(TEST_YEARS, series["original"][test]),
            "candidate_trend_degC_century": slope_per_century(TEST_YEARS, series["candidate"][test]),
            "jump_1899_1900_degC": jump_1899,
            "jump_1899_1900_sigma": float(abs(jump_1899 - increment_mean) / increment_std),
            "jump_1939_1940_degC": jump_1939,
            "jump_1939_1940_sigma": float(abs(jump_1939 - increment_mean) / increment_std),
        })
        for index, year in enumerate(YEARS):
            if year < 1880 or year > 1960:
                continue
            curve_rows.append({
                "domain": domain, "domain_name": domain_name, "year": int(year),
                "reference_degC": float(series["reference"][index]),
                "original_degC": float(series["original"][index]),
                "correction_degC": float(series["correction"][index]),
                "candidate_degC": float(series["candidate"][index]),
                "reference_low11_degC": float(low["reference"][index]),
                "original_low11_degC": float(low["original"][index]),
                "candidate_low11_degC": float(low["candidate"][index]),
            })

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    hma = metrics[metrics.domain.eq("hma")].iloc[0]
    regional = metrics[metrics.domain.ne("hma")]
    key = metrics[metrics.domain.isin(KEY_REGIONS)]
    gates = {
        "hma_low11_rmse_improvement_at_least_0p30": bool(hma.low11_rmse_improvement_fraction >= 0.30),
        "at_least_12_of_15_regions_improve_low11_rmse": bool((regional.low11_rmse_improvement_fraction > 0).sum() >= 12),
        "key_regions_do_not_degrade_more_than_0p10": bool((key.low11_rmse_improvement_fraction >= -0.10).all()),
        "hma_highpass_correlation_at_least_0p95": bool(hma.candidate_original_high11_correlation >= 0.95),
        "median_regional_highpass_correlation_at_least_0p95": bool(regional.candidate_original_high11_correlation.median() >= 0.95),
        "all_key_highpass_correlations_at_least_0p90": bool((key.candidate_original_high11_correlation >= 0.90).all()),
        "hma_and_key_boundary_jumps_at_most_3_sigma": bool(
            max(float(hma.jump_1899_1900_sigma), float(hma.jump_1939_1940_sigma)) <= 3.0
            and (key[["jump_1899_1900_sigma", "jump_1939_1940_sigma"]].to_numpy() <= 3.0).all()
        ),
    }
    pass_all = bool(all(gates.values()))

    output_nc = OUTPUT / "temperature_low_frequency_blind_test_1921_1939_v1.nc"
    selected = np.isin(YEARS, TEST_YEARS)
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(TEST_YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = TEST_YEARS
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common.astype(np.int8)
        for name, values in {
            "tas_reference_family_balanced": reference[selected],
            "tas_project_six_member_original": project[selected],
            "tas_low_frequency_correction": correction[selected],
            "tas_project_six_member_corrected": candidate[selected],
        }.items():
            variable = dataset.createVariable(
                name, "f4", ("year", "lat", "lon"), zlib=True, complevel=4,
                chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(np.nan),
            )
            variable[:] = values.astype(np.float32)
            variable.units = "degC anomaly relative to 1961-1990"
        dataset.title = "Frozen low-frequency correction blind test, 1921-1939"
        dataset.candidate_formula = "six-member project posterior plus precommitted correction"
        dataset.acceptance_period = "1921-1939"
        dataset.status = "blind test complete; production and OGGM authorization remain separate"

    metrics.to_csv(OUTPUT / "blind_test_domain_metrics.csv", index=False)
    curves.to_csv(OUTPUT / "blind_test_domain_curves_1880_1960.csv", index=False)
    plot_domains = ["hma", *KEY_REGIONS]
    fig, axes = plt.subplots(len(plot_domains), 1, figsize=(14, 13), sharex=True, constrained_layout=True)
    for axis, domain in zip(axes, plot_domains, strict=True):
        rows = curves[curves.domain.eq(domain)].set_index("year")
        axis.axvspan(1921, 1939, color="#E6E6E6", alpha=0.8, label="Blind test" if domain == "hma" else None)
        axis.plot(rows.index, rows.reference_low11_degC, color="#111111", linewidth=2.0, label="Reference, 11-year")
        axis.plot(rows.index, rows.original_low11_degC, color="#D55E00", linewidth=1.7, label="Original, 11-year")
        axis.plot(rows.index, rows.candidate_low11_degC, color="#0072B2", linewidth=2.0, label="Corrected, 11-year")
        axis.set(title=domains[domain][0], ylabel="degC")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Year")
    figure = OUTPUT / "temperature_low_frequency_blind_test_1921_1939.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: state_without_hash(path) for name, path in tracked.items()}
    source_states_unchanged = before == after
    source_identity_checks = {
        "correction_sha256_matches_frozen": sha256(CORRECTION) == EXPECTED_CORRECTION_SHA256,
        "reference_expected_sha256_registered": EXPECTED_REFERENCE_SHA256
        == "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90",
        "reference_size_matches_registered": int(REFERENCE_SOURCE.stat().st_size) == 225274031,
        "posterior_sizes_and_mtimes_match_round25": all(
            int(Path(manifest25[f"posterior_{member}"]["path"]).stat().st_size)
            == int(manifest25[f"posterior_{member}"]["size_bytes"])
            and int(Path(manifest25[f"posterior_{member}"]["path"]).stat().st_mtime_ns)
            == int(manifest25[f"posterior_{member}"]["mtime_ns"])
            for member in round25.MEMBERS
        ),
    }
    summary = {
        "status": "complete",
        "user_unlock_authorized": True,
        "blind_acceptance_period": [1921, 1939],
        "diagnostic_only_period": [1901, 1920],
        "facts_snapshot_sha256": facts_snapshot,
        "source_states_unchanged_during_run": source_states_unchanged,
        "source_identity_checks": source_identity_checks,
        "common_support_cells": int(common.sum()),
        "hma_original_low11_rmse_degC": float(hma.original_low11_rmse_degC),
        "hma_candidate_low11_rmse_degC": float(hma.candidate_low11_rmse_degC),
        "hma_low11_rmse_improvement_fraction": float(hma.low11_rmse_improvement_fraction),
        "improving_region_count": int((regional.low11_rmse_improvement_fraction > 0).sum()),
        "worst_region": str(regional.loc[regional.low11_rmse_improvement_fraction.idxmin(), "domain"]),
        "worst_region_improvement_fraction": float(regional.low11_rmse_improvement_fraction.min()),
        "hma_original_reference_low11_correlation": float(hma.original_low11_reference_correlation),
        "hma_candidate_reference_low11_correlation": float(hma.candidate_low11_reference_correlation),
        "hma_original_trend_degC_century": float(hma.original_trend_degC_century),
        "hma_candidate_trend_degC_century": float(hma.candidate_trend_degC_century),
        "hma_reference_trend_degC_century": float(hma.reference_trend_degC_century),
        "gates": gates,
        "passes_all_frozen_gates": pass_all,
        "decision": "retain_corrected_temperature_candidate_for_assembly" if pass_all else "reject_correction_bridge_v1",
        "production_authorized": False,
        "oggm_forcing_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "round25_expected": manifest25, "summary_checks": source_identity_checks}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Temperature Low-Frequency Blind Test, 1921-1939\n\n"
        f"- HMA 11-year RMSE: `{float(hma.original_low11_rmse_degC):.4f}` to `{float(hma.candidate_low11_rmse_degC):.4f} degC` "
        f"(`{100 * float(hma.low11_rmse_improvement_fraction):+.1f}%`).\n"
        f"- Improving GTN-G regions: `{int((regional.low11_rmse_improvement_fraction > 0).sum())}/15`.\n"
        f"- HMA low-frequency correlation: `{float(hma.original_low11_reference_correlation):+.3f}` to `{float(hma.candidate_low11_reference_correlation):+.3f}`.\n"
        f"- HMA 1921-1939 trend, reference/original/corrected: `{float(hma.reference_trend_degC_century):+.3f}` / "
        f"`{float(hma.original_trend_degC_century):+.3f}` / `{float(hma.candidate_trend_degC_century):+.3f} degC per century`.\n"
        f"- All frozen gates pass: `{str(pass_all).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`. Production and OGGM forcing remain unauthorized.\n",
        encoding="utf-8",
    )
    if not source_states_unchanged or not all(source_identity_checks.values()):
        raise RuntimeError("source-identity-or-immutability-check-failed")
    print(json.dumps(summary, indent=2))
    print(output_nc)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
