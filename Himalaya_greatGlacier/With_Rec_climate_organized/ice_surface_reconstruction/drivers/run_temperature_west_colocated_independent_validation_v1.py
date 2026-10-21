"""Run frozen co-located western-temperature branch diagnostics through 1900."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import scipy.io
import scipy.linalg
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v2 as amplitude
import run_climate_reconstruction_loop_v2_round23_corrected_reconstruction_reference_envelope_v1 as round23


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_colocated_independent_validation_v1"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"
LINEAGE = RESULTS / "climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
REFERENCE = LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
COMPONENTS = RESULTS / "climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1"
HIRPORA = RESULTS / "climate_reconstruction_temperature_hirpora_independent_pre1901_validation_v1/hirpora_digitized_and_frozen_branches_1812_1900.csv"
WESTERN = RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2/western_temperature_transition_1812_1900.csv"
MANIFEST27 = LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"

YEARS = np.arange(1812, 1901)
OPERATOR_YEARS = np.arange(1861, 1901)
ALPHAS = np.asarray([0.0, 0.25, 1.0])
SITES = {
    "hirpora": {"latitude": 33.066, "longitude": 74.068, "season": "JJ", "smoother": 15},
    "kumrat": {"latitude": 35.570, "longitude": 72.207, "season": "MA", "smoother": 11},
    "upper_indus": {"latitude": 35.34, "longitude": 74.79, "season": "MAMJJAS", "smoother": 11},
}
RADIUS_KM = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_prefix(path: Path, start: int = 1812, end: int = 1900) -> pd.DataFrame:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle)
        lines.append(header)
        year_column = header.rstrip().split(",").index("year")
        for line in handle:
            year = int(float(line.split(",")[year_column]))
            if year > end:
                break
            if year >= start:
                lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(lines)))
    if not np.array_equal(frame.year.to_numpy(int), np.arange(start, end + 1)):
        raise RuntimeError(f"safe-prefix-year-closure-failed:{path}")
    return frame


def distance_km(lat_grid: np.ndarray, lon_grid: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    return np.sqrt(
        ((lat_grid - latitude) * 111.0) ** 2
        + ((lon_grid - longitude) * 111.0 * np.cos(np.deg2rad(latitude))) ** 2
    )


def weighted_target(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def read_tif_prefix(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    bands = (YEARS - 999).tolist()
    with rasterio.open(path) as source:
        if source.count != 1001:
            raise RuntimeError(f"unexpected-tif-band-count:{path}")
        data = source.read(bands).astype(float)
        nodata = source.nodata if source.nodata is not None else -9999.0
        data[np.isclose(data, nodata) | (data < -9000)] = np.nan
        tif_lat = np.asarray([source.xy(row, 0, offset="center")[1] for row in range(source.height)])
        tif_lon = np.asarray([source.xy(0, column, offset="center")[0] for column in range(source.width)])
    projected, _ = round23.interpolate_to_reference(data, tif_lat, tif_lon, lat, lon)
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    projected -= np.nanmean(projected[baseline], axis=0)[None]
    return projected


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    minimum = (width + 1) // 2
    return pd.Series(values).rolling(width, center=True, min_periods=minimum).mean().to_numpy(float)


def standardize_smoothed(values: np.ndarray) -> np.ndarray:
    baseline = YEARS <= 1860
    mean = float(np.nanmean(values[baseline]))
    scale = float(np.nanstd(values[baseline], ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid-smoothed-baseline-scale")
    return (values - mean) / scale


def transform_project(values: np.ndarray, width: int) -> np.ndarray:
    return standardize_smoothed(smooth(values, width))


def transform_published(values: np.ndarray) -> np.ndarray:
    return standardize_smoothed(values)


def correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(left) & np.isfinite(right)
    return float(np.sqrt(np.mean((left[valid] - right[valid]) ** 2)))


def trend(values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(values)
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def raw_increments_multi(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    curves = np.zeros((len(targets), len(OPERATOR_YEARS)), dtype=float)
    closure = 0.0
    for pool in pools:
        target_pool = targets[:, pool]
        target_prime = target_pool - np.mean(target_pool, axis=1, keepdims=True)
        for year_index in range(len(OPERATOR_YEARS)):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator = hxb[np.ix_(active, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            covariance = operator_prime @ operator_prime.T / (len(pool) - 1) + np.diag(error_variance[active])
            rhs = (target_prime @ operator_prime.T / (len(pool) - 1)).T
            try:
                factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
                gains = scipy.linalg.cho_solve(factor, rhs, check_finite=False).T
            except np.linalg.LinAlgError:
                gains = scipy.linalg.solve(covariance, rhs, assume_a="sym", check_finite=False).T
            innovation = observations[active, year_index] - np.mean(operator, axis=1)
            local = gains @ innovation
            curves[:, year_index] += local / len(pools)
            closure = max(closure, float(np.max(np.abs(local - gains @ innovation))))
    return curves, closure


def main() -> int:
    if not CONTRACT.exists():
        raise FileNotFoundError(CONTRACT)
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError(OUTPUT / "summary.json")
    recovered = json.loads(MANIFEST27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "contract": CONTRACT,
        "lineage": LINEAGE,
        "reference": REFERENCE,
        "hma_mask": HMA_MASK,
        "hirpora": HIRPORA,
        "western": WESTERN,
        "manifest27": MANIFEST27,
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "components6": COMPONENTS / "proxy6_bivariate_operator_components.npz",
        "components9": COMPONENTS / "proxy9_bivariate_operator_components.npz",
    }
    lineage = pd.read_csv(LINEAGE)
    temperature = lineage[lineage.variable.eq("tas")].copy()
    expected_members = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if set(temperature.member) != expected_members:
        raise RuntimeError("temperature-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in temperature.itertuples(index=False)}
    paths.update({f"member_{name}": path for name, path in member_paths.items()})
    before = {name: sha256(path) for name, path in paths.items()}
    for row in temperature.itertuples(index=False):
        if before[f"member_{row.member}"] != row.source_tif_sha256:
            raise RuntimeError(f"member-hash-mismatch:{row.member}")

    with netcdf_file(REFERENCE, "r", mmap=False) as dataset:
        ref_year = np.asarray(dataset.variables["year"].data, dtype=int).copy()
        indices = np.flatnonzero((ref_year >= YEARS[0]) & (ref_year <= YEARS[-1]))
        if not np.array_equal(ref_year[indices], YEARS):
            raise RuntimeError("reference-prefix-year-closure-failed")
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"].data[indices], dtype=float).copy()
    reference[reference < -9000] = np.nan
    hma = np.load(HMA_MASK).astype(bool)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    masks: dict[str, np.ndarray] = {}
    spatial_rows = []
    for site, config in SITES.items():
        distances = distance_km(lat_grid, lon_grid, config["latitude"], config["longitude"])
        valid_distance = np.where(hma, distances, np.inf)
        nearest_flat = int(np.argmin(valid_distance))
        nearest = np.zeros_like(hma)
        nearest.flat[nearest_flat] = True
        radius = hma & (distances <= RADIUS_KM)
        nearest_distance = float(valid_distance.flat[nearest_flat])
        if nearest_distance > 35.0 or int(radius.sum()) < 3:
            raise RuntimeError(f"spatial-gate-failed:{site}:{nearest_distance}:{radius.sum()}")
        for scale, mask in (("nearest", nearest), ("radius100km", radius)):
            key = f"{site}__{scale}"
            masks[key] = mask
            spatial_rows.append({
                "target": key,
                "site": site,
                "scale": scale,
                "source_latitude": config["latitude"],
                "source_longitude": config["longitude"],
                "season": config["season"],
                "cell_count": int(mask.sum()),
                "nearest_cell_latitude": float(lat_grid.flat[nearest_flat]),
                "nearest_cell_longitude": float(lon_grid.flat[nearest_flat]),
                "nearest_cell_distance_km": nearest_distance,
            })
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(OUTPUT / "site_spatial_contract.csv", index=False)
    target_names = list(masks)

    member_fields = [read_tif_prefix(path, lat, lon) for path in member_paths.values()]
    project_field = np.nanmean(np.stack(member_fields), axis=0)
    project_local = np.stack([weighted_target(project_field, masks[name], lat) for name in target_names])
    reference_local = np.stack([weighted_target(reference, masks[name], lat) for name in target_names])

    prior = np.load(paths["prior"])
    native_lat, native_lon = amplitude.v1.base.round25.native_coordinates()
    prior_fields = prior.T.reshape(prior.shape[1], len(native_lat), len(native_lon), order="F")
    projected_prior = amplitude.v1.base.round25.interpolate_fields(prior_fields, native_lat, native_lon, lat, lon)
    targets = np.stack([weighted_target(projected_prior, masks[name], lat) for name in target_names])
    if not np.isfinite(targets).all():
        raise RuntimeError("local-prior-target-not-finite")
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1

    raw: dict[int, dict[float, np.ndarray]] = {}
    closure_values = []
    alpha_one_errors = []
    for network in (6, 9):
        hxb, observations, locked_reads = amplitude.v1.base.round25.read_proxy_allowed(paths[f"proxy{network}"], OPERATOR_YEARS)
        if locked_reads:
            raise RuntimeError("locked-proxy-values-read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        with np.load(paths[f"components{network}"], allow_pickle=False) as archive:
            selected = archive["selected_indices_0based"].astype(int)
            temperature_component = archive["temperature_anomaly"]
            precipitation_component = archive["precipitation_anomaly"]
        means = np.mean(hxb[selected], axis=1, keepdims=True)
        raw[network] = {}
        for alpha in ALPHAS:
            operator = hxb.copy()
            operator[selected] = means + alpha * temperature_component + precipitation_component
            if alpha == 1.0:
                alpha_one_errors.append(float(np.max(np.abs(operator - hxb))))
            raw[network][float(alpha)], closure = raw_increments_multi(
                operator, observations, error_variance, pools, targets
            )
            closure_values.append(closure)
    if max(closure_values) > 1e-12 or max(alpha_one_errors) > 1e-12:
        raise RuntimeError("operator-closure-failed")

    hirpora = safe_prefix(HIRPORA)
    western = safe_prefix(WESTERN)
    independent_raw = {
        "hirpora": hirpora.hirpora_medium_degC.to_numpy(float),
        "kumrat": western.kumrat_medium_degC.to_numpy(float),
        "upper_indus": western.upper_indus_baseline_standardized.to_numpy(float),
    }
    independent = {
        "hirpora": transform_project(independent_raw["hirpora"], 15),
        "kumrat": transform_published(independent_raw["kumrat"]),
        "upper_indus": transform_published(independent_raw["upper_indus"]),
    }
    compare = (YEARS >= 1861) & (YEARS <= 1900)
    late = (YEARS >= 1881) & (YEARS <= 1900)
    branch_curves: dict[tuple[str, float], np.ndarray] = {}
    metric_rows = []
    curve_rows = []
    transition = YEARS >= 1861
    for target_index, target_name in enumerate(target_names):
        site = target_name.split("__", 1)[0]
        width = int(SITES[site]["smoother"])
        independent_curve = independent[site]
        reference_curve = transform_project(reference_local[target_index], width)
        baseline_raw = np.mean(
            np.stack([raw[6][1.0][target_index], raw[9][1.0][target_index]]), axis=0
        )
        for alpha in ALPHAS:
            branch_raw = np.mean(
                np.stack([raw[6][float(alpha)][target_index], raw[9][float(alpha)][target_index]]), axis=0
            )
            candidate = project_local[target_index].copy()
            candidate[transition] += branch_raw - baseline_raw
            curve = transform_project(candidate, width)
            branch_curves[(target_name, float(alpha))] = curve
            metric_rows.append({
                "target": target_name,
                "site": site,
                "scale": target_name.split("__", 1)[1],
                "season": SITES[site]["season"],
                "alpha": float(alpha),
                "rmse_1861_1900": rmse(curve, independent_curve, compare),
                "correlation_1861_1900": correlation(curve, independent_curve, compare),
                "absolute_late_mean_error_1881_1900": float(abs(np.mean(curve[late]) - np.mean(independent_curve[late]))),
                "branch_trend_1861_1900_sd_century": trend(curve, compare),
                "independent_trend_1861_1900_sd_century": trend(independent_curve, compare),
                "reference_correlation_1861_1900": correlation(reference_curve, independent_curve, compare),
            })
            for year, value, evidence, reference_value in zip(YEARS, curve, independent_curve, reference_curve):
                curve_rows.append({
                    "target": target_name,
                    "site": site,
                    "scale": target_name.split("__", 1)[1],
                    "year": int(year),
                    "alpha": float(alpha),
                    "branch_baseline_sd": float(value),
                    "independent_baseline_sd": float(evidence),
                    "reference_baseline_sd": float(reference_value),
                })
    metrics = pd.DataFrame(metric_rows)
    metrics["best_rmse_for_target"] = metrics.groupby("target").rmse_1861_1900.transform("min")
    metrics["descriptive_best_branch"] = np.isclose(metrics.rmse_1861_1900, metrics.best_rmse_for_target)
    metrics.to_csv(OUTPUT / "colocated_branch_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUTPUT / "colocated_branch_curves_1812_1900.csv", index=False)

    alpha1 = metrics[metrics.alpha.eq(1.0)].set_index("target")
    uniform_support = {}
    sensitivity_rows = []
    for alpha in (0.0, 0.25):
        candidate = metrics[metrics.alpha.eq(alpha)].set_index("target")
        rmse_gain = (alpha1.rmse_1861_1900 - candidate.rmse_1861_1900) / alpha1.rmse_1861_1900
        correlation_change = candidate.correlation_1861_1900 - alpha1.correlation_1861_1900
        late_gain = alpha1.absolute_late_mean_error_1881_1900 - candidate.absolute_late_mean_error_1881_1900
        passed = (rmse_gain >= 0.05) & (correlation_change >= -0.05) & (late_gain > 0)
        uniform_support[str(alpha)] = bool(passed.all())
        for target_name in target_names:
            sensitivity_rows.append({
                "target": target_name,
                "alpha": alpha,
                "rmse_improvement_fraction": float(rmse_gain[target_name]),
                "correlation_change": float(correlation_change[target_name]),
                "late_mean_error_improvement_sd": float(late_gain[target_name]),
                "all_target_gates_pass": bool(passed[target_name]),
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT / "uniform_branch_gate_audit.csv", index=False)
    winners = metrics.loc[metrics.descriptive_best_branch].groupby("target").alpha.first().to_dict()
    independent_trends = metrics.groupby("site").independent_trend_1861_1900_sd_century.first().to_dict()
    trend_signs = {site: int(np.sign(value)) for site, value in independent_trends.items()}
    mixed_winners = len(set(winners.values())) > 1
    mixed_independent_trends = len(set(trend_signs.values())) > 1
    regional_distortion_supported = any(uniform_support.values())
    heterogeneity_prevents_closure = (not regional_distortion_supported) and (mixed_winners or mixed_independent_trends)
    if regional_distortion_supported:
        decision = "uniform_regional_temperature_operator_distortion_supported"
    elif heterogeneity_prevents_closure:
        decision = "spatial_or_seasonal_heterogeneity_prevents_uniform_regional_causal_closure"
    else:
        decision = "colocated_western_evidence_inconclusive"

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 11.0), sharex=True, constrained_layout=True)
    colors = {1.0: "#B43A35", 0.25: "#207A6E", 0.0: "#276FBF"}
    for axis, target_name in zip(axes.flat, target_names):
        site, scale = target_name.split("__", 1)
        evidence = independent[site]
        axis.plot(YEARS, evidence, color="#111111", linewidth=2.5, label=f"{site} {SITES[site]['season']}")
        for alpha in (1.0, 0.25, 0.0):
            axis.plot(YEARS, branch_curves[(target_name, alpha)], color=colors[alpha], linewidth=1.8, label=f"alpha={alpha:g}")
        axis.axvline(1860, color="#888888", linestyle=":", linewidth=0.8)
        axis.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
        axis.set_title(f"{site} | {scale}")
        axis.set_ylabel("Matched low-pass baseline SD")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("Year")
    fig.savefig(OUTPUT / "temperature_west_colocated_independent_validation.png", dpi=240)
    plt.close(fig)

    summary = {
        "status": "complete",
        "site_coordinates": SITES,
        "spatial_scales": ["nearest_valid_0p5_degree_hma_cell", "hma_cells_within_100km_area_weighted"],
        "maximum_nearest_cell_distance_km": float(spatial.nearest_cell_distance_km.max()),
        "minimum_radius_cell_count": int(spatial.loc[spatial.scale.eq("radius100km"), "cell_count"].min()),
        "operator_algebraic_closure_max": max(closure_values),
        "alpha_one_operator_max_abs_error": max(alpha_one_errors),
        "descriptive_best_branch_by_target": winners,
        "independent_trends_sd_per_century": independent_trends,
        "independent_trend_signs": trend_signs,
        "uniform_attenuated_branch_support": uniform_support,
        "regional_operator_distortion_supported": regional_distortion_supported,
        "heterogeneity_prevents_uniform_closure": heterogeneity_prevents_closure,
        "decision": decision,
        "seasonal_field_limitation": "project reconstructs annual anomalies and cannot explicitly represent JJ, MA, and MAMJJAS separately",
        "production_authorized": False,
        "enkf_production_replay_authorized": False,
        "oggm_authorized": False,
        "maximum_year_read": 1900,
        "locked_or_post1900_values_read": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Co-located western-temperature validation

- Maximum site-to-nearest-grid distance: `{summary['maximum_nearest_cell_distance_km']:.1f} km`.
- Descriptive best frozen branches: `{winners}`.
- Independent 1861-1900 trend signs: `{trend_signs}`.
- Uniform alpha=0 support across all six targets: `{uniform_support['0.0']}`.
- Uniform alpha=0.25 support across all six targets: `{uniform_support['0.25']}`.

Decision: `{decision}`.

The project is an annual anomaly field, while the independent targets use JJ, MA, and MAMJJAS seasons. The experiment can test co-located low-frequency consistency but cannot uniquely separate spatial from seasonal heterogeneity. No correction or production replay is authorized.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    after = {name: sha256(path) for name, path in paths.items()}
    manifest = {"before": before, "after": after, "source_hashes_unchanged": before == after}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
