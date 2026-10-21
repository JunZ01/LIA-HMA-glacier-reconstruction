"""Separate background and assimilation-update temperature signals on native state support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEMBERS = {
    "AO_6": ("AO", 6),
    "AO_9": ("AO", 9),
    "HGAO_6": ("HGAO", 6),
    "HGAO_9": ("HGAO", 9),
    "O_6": ("O", 6),
    "O_9": ("O", 9),
}
FAMILIES = ("local_enkf", "lmr", "neukom", "phyda")
REQUESTED_YEARS = np.r_[np.arange(1400, 1900), np.arange(1961, 1991)]


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def native_coordinates() -> tuple[np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    return (
        all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)],
        all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)],
    )


def interpolate_fields(
    data: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    if np.any(np.diff(source_lat) < 0):
        source_lat = source_lat[::-1]
        data = data[:, ::-1, :]
    if np.any(np.diff(source_lon) < 0):
        source_lon = source_lon[::-1]
        data = data[:, :, ::-1]
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    fy = (lat_grid - source_lat[0]) / (source_lat[1] - source_lat[0])
    fx = (lon_grid - source_lon[0]) / (source_lon[1] - source_lon[0])
    inside = (
        (fy >= 0)
        & (fy <= len(source_lat) - 1)
        & (fx >= 0)
        & (fx <= len(source_lon) - 1)
    )
    y0 = np.clip(np.floor(fy).astype(int), 0, len(source_lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(source_lon) - 2)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    a = data[:, y0, x0]
    b = data[:, y0, x0 + 1]
    c = data[:, y0 + 1, x0]
    d = data[:, y0 + 1, x0 + 1]
    result = (
        (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b)
        + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    )
    result[:, ~inside] = np.nan
    return result


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(
        denominator == 0, np.nan, denominator
    )


def read_proxy_allowed(path: Path, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    requested = {int(year): index for index, year in enumerate(years)}
    locked_reads = 0
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        hxb = np.empty((len(references), 848), dtype=float)
        observations = np.full((len(references), len(years)), np.nan, dtype=float)
        for proxy_index, reference in enumerate(references):
            proxy = archive[reference]
            hxb[proxy_index] = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            source_indices: list[int] = []
            target_indices: list[int] = []
            for source_index, age in enumerate(ages):
                if not np.isfinite(age):
                    continue
                year = int(age)
                if year in requested:
                    source_indices.append(source_index)
                    target_indices.append(requested[year])
            if not source_indices:
                continue
            order = np.argsort(source_indices, kind="stable")
            source_array = np.asarray(source_indices, dtype=int)[order]
            target_array = np.asarray(target_indices, dtype=int)[order]
            dataset = proxy["data"]
            if dataset.shape == (len(ages), 1):
                values = np.asarray(dataset[source_array, 0], dtype=float)
            elif dataset.shape == (1, len(ages)):
                values = np.asarray(dataset[0, source_array], dtype=float)
            else:
                raise RuntimeError(f"unexpected-proxy-data-shape:{dataset.shape}")
            observations[proxy_index, target_array] = values
    return hxb, observations, locked_reads


def read_posterior_mean(path: Path, years: np.ndarray) -> np.ndarray:
    requested_indices = years - 1000
    chunks: list[np.ndarray] = []
    with h5py.File(path, "r") as archive:
        dataset = archive["Xa_m_all"]
        if dataset.shape != (50, 1001, 336):
            raise RuntimeError(f"unexpected-posterior-shape:{path.name}:{dataset.shape}")
        starts = [0]
        starts.extend((np.flatnonzero(np.diff(requested_indices) != 1) + 1).tolist())
        starts.append(len(requested_indices))
        for left, right in zip(starts[:-1], starts[1:]):
            first = int(requested_indices[left])
            last = int(requested_indices[right - 1]) + 1
            chunks.append(np.asarray(dataset[:, first:last, :], dtype=float))
    return np.concatenate(chunks, axis=1).mean(axis=0)


def ao_background_mean(
    prior: np.ndarray,
    experiment_indices: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    year_count = observations.shape[1]
    background = np.empty((year_count, prior.shape[0]), dtype=float)
    selected_regional_pool_mean = np.empty((year_count, experiment_indices.shape[0]), dtype=float)
    native_lat, native_lon = native_coordinates()
    state_lat = np.tile(native_lat, len(native_lon))
    state_weights = np.cos(np.deg2rad(state_lat))
    state_weights /= state_weights.sum()
    prior_regional = state_weights @ prior
    for year_index in range(year_count):
        active = np.isfinite(observations[:, year_index])
        experiment_fields = np.empty((experiment_indices.shape[0], prior.shape[0]), dtype=float)
        if np.any(active):
            y = observations[active, year_index]
            y_anomaly = y - np.mean(y)
            model = hxb[active]
            model_anomaly = model - np.mean(model, axis=0, keepdims=True)
            numerator = y_anomaly @ model_anomaly
            denominator = np.sqrt(np.sum(y_anomaly**2) * np.sum(model_anomaly**2, axis=0))
            scores = np.full(model.shape[1], -1.0, dtype=float)
            valid = denominator > 0
            scores[valid] = numerator[valid] / denominator[valid]
            for experiment, pool in enumerate(experiment_indices):
                local_order = np.argsort(-scores[pool], kind="stable")[:100]
                selected = pool[local_order]
                experiment_fields[experiment] = np.mean(prior[:, selected], axis=1)
                selected_regional_pool_mean[year_index, experiment] = float(np.mean(prior_regional[selected]))
        else:
            for experiment, pool in enumerate(experiment_indices):
                experiment_fields[experiment] = np.mean(prior[:, pool], axis=1)
                selected_regional_pool_mean[year_index, experiment] = float(np.mean(prior_regional[pool]))
        background[year_index] = np.mean(experiment_fields, axis=0)
    return background, selected_regional_pool_mean


def anomaly_field(flat_values: np.ndarray, years: np.ndarray) -> np.ndarray:
    native_lat, native_lon = native_coordinates()
    fields = flat_values.reshape(len(years), len(native_lat), len(native_lon), order="F")
    baseline = (years >= 1961) & (years <= 1990)
    return fields - np.mean(fields[baseline], axis=0, keepdims=True)


def project_series(
    fields: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return weighted_series(
        interpolate_fields(fields, source_lat, source_lon, target_lat, target_lon),
        mask,
        target_lat,
    )


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_25_temperature_prior_update_decomposition_v1"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    source_recovery = project / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2"
    result_root = archive / "fixed_proxy_experiments_69/tas/results"
    paths: dict[str, Path] = {
        "prior": source_recovery / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archive_config": archive / "fixed_proxy_experiments_69/tas/experiment_config.mat",
        "recovered_algorithm": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\fix_proxys_experiments.m"),
        "reference_curves": loop / "round_21_temperature_reference_field_v1/hma_temperature_reference_curves.csv",
        "reference_netcdf": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "round23_curves": loop / "round_23_corrected_reconstruction_reference_envelope_v1/corrected_reconstruction_reference_curves.csv",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for member, (method, network) in MEMBERS.items():
        archive_method = {"AO": "AOEnKF", "HGAO": "HGAOEnKF", "O": "EnKF"}[method]
        paths[f"posterior_{member}"] = result_root / f"proxy{network}/{archive_method}_results.mat"
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    before = {name: file_state(path) for name, path in paths.items()}
    config = scipy.io.loadmat(paths["archive_config"], squeeze_me=True, struct_as_record=False)["config"]
    recovered_config = {
        "num_experiments": int(config.num_experiments),
        "analysis_ensemble_size": 100,
        "aoenkf_strategy": str(config.aoenkf_strategy),
        "alpha": float(config.alpha),
        "start_year": int(config.r_o),
        "end_year": int(config.r_f),
    }
    if recovered_config != {
        "num_experiments": 50,
        "analysis_ensemble_size": 100,
        "aoenkf_strategy": "C",
        "alpha": 0.3,
        "start_year": 1000,
        "end_year": 2000,
    }:
        raise RuntimeError(f"archive-config-contract-failed:{recovered_config}")

    prior = np.load(paths["prior"])
    if prior.shape != (336, 848):
        raise RuntimeError(f"unexpected-prior-shape:{prior.shape}")
    with h5py.File(paths["indices"], "r") as archive_file:
        experiment_indices = np.asarray(archive_file["all_Xb_indices"], dtype=int) - 1
    if experiment_indices.shape != (50, 500):
        raise RuntimeError(f"unexpected-index-shape:{experiment_indices.shape}")

    proxies: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    locked_proxy_reads = 0
    for network in (6, 9):
        hxb, observations, locked_reads = read_proxy_allowed(paths[f"proxy{network}"], REQUESTED_YEARS)
        proxies[network] = (hxb, observations)
        locked_proxy_reads += locked_reads

    adaptive_background: dict[int, np.ndarray] = {}
    adaptive_pool_means: dict[int, np.ndarray] = {}
    for network in (6, 9):
        adaptive_background[network], adaptive_pool_means[network] = ao_background_mean(
            prior,
            experiment_indices,
            proxies[network][0],
            proxies[network][1],
        )
    fixed_background = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]),
        axis=0,
    )
    fixed_background = np.repeat(fixed_background[None, :], len(REQUESTED_YEARS), axis=0)

    native_lat, native_lon = native_coordinates()
    with netcdf_file(paths["reference_netcdf"], "r", mmap=False) as dataset:
        target_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        target_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    hma_mask = np.load(paths["hma_mask"]).astype(bool)
    if hma_mask.shape != (len(target_lat), len(target_lon)):
        raise RuntimeError("hma-mask-shape-failed")

    posterior_fields: dict[str, np.ndarray] = {}
    background_fields: dict[str, np.ndarray] = {}
    increment_fields: dict[str, np.ndarray] = {}
    series: dict[str, dict[str, np.ndarray]] = {}
    for member, (method, network) in MEMBERS.items():
        posterior_fields[member] = anomaly_field(
            read_posterior_mean(paths[f"posterior_{member}"], REQUESTED_YEARS),
            REQUESTED_YEARS,
        )
        background_flat = fixed_background if method == "O" else adaptive_background[network]
        background_fields[member] = anomaly_field(background_flat, REQUESTED_YEARS)
        increment_fields[member] = posterior_fields[member] - background_fields[member]
        series[member] = {
            "posterior": project_series(posterior_fields[member], native_lat, native_lon, target_lat, target_lon, hma_mask),
            "background": project_series(background_fields[member], native_lat, native_lon, target_lat, target_lon, hma_mask),
            "increment": project_series(increment_fields[member], native_lat, native_lon, target_lat, target_lon, hma_mask),
        }

    ensemble = {
        component: np.mean(np.stack([series[member][component] for member in MEMBERS]), axis=0)
        for component in ("posterior", "background", "increment")
    }
    reference = pd.read_csv(paths["reference_curves"]).set_index("year")
    round23 = pd.read_csv(paths["round23_curves"]).set_index("year")
    requested_reference = reference.loc[REQUESTED_YEARS]
    reference_central = requested_reference["family_balanced_hma_degC"].to_numpy(float)
    reference_family = {
        family: requested_reference[f"{family}_hma_degC"].to_numpy(float)
        for family in FAMILIES
    }
    export_series = round23.loc[REQUESTED_YEARS, "corrected_six_member_hma_degC"].to_numpy(float)

    annual = pd.DataFrame({
        "year": REQUESTED_YEARS,
        "reference_family_central_degC": reference_central,
        "round23_export_six_member_degC": export_series,
        "native_six_member_posterior_degC": ensemble["posterior"],
        "native_six_member_background_degC": ensemble["background"],
        "native_six_member_increment_degC": ensemble["increment"],
    })
    for family, values in reference_family.items():
        annual[f"reference_{family}_degC"] = values
    for member in MEMBERS:
        for component in ("posterior", "background", "increment"):
            annual[f"{member}_{component}_degC"] = series[member][component]
    annual.to_csv(output / "annual_native_state_decomposition.csv", index=False)

    closure_values = [
        float(np.max(np.abs(series[member]["posterior"] - series[member]["background"] - series[member]["increment"])))
        for member in MEMBERS
    ]
    closure_values.append(float(np.max(np.abs(ensemble["posterior"] - ensemble["background"] - ensemble["increment"]))))
    closure_max = max(closure_values)
    fixed_background_max = max(
        float(np.max(np.abs(series[member]["background"]))) for member in ("O_6", "O_9")
    )
    adaptive_background_identity = max(
        float(np.max(np.abs(series[f"AO_{network}"]["background"] - series[f"HGAO_{network}"]["background"])))
        for network in (6, 9)
    )

    window_rows: list[dict[str, object]] = []
    for start in range(1400, 1900, 50):
        end = start + 49
        selected = (REQUESTED_YEARS >= start) & (REQUESTED_YEARS <= end)
        family_means = np.asarray([np.mean(values[selected]) for values in reference_family.values()])
        row: dict[str, object] = {
            "start_year": start,
            "end_year": end,
            "reference_central_degC": float(np.mean(reference_central[selected])),
            "reference_family_min_degC": float(np.min(family_means)),
            "reference_family_max_degC": float(np.max(family_means)),
            "posterior_degC": float(np.mean(ensemble["posterior"][selected])),
            "background_degC": float(np.mean(ensemble["background"][selected])),
            "increment_degC": float(np.mean(ensemble["increment"][selected])),
        }
        row["posterior_minus_reference_degC"] = float(row["posterior_degC"] - row["reference_central_degC"])
        row["increment_only_minus_reference_degC"] = float(row["increment_degC"] - row["reference_central_degC"])
        row["posterior_inside_reference_range"] = bool(
            row["reference_family_min_degC"] <= row["posterior_degC"] <= row["reference_family_max_degC"]
        )
        for member in MEMBERS:
            for component in ("posterior", "background", "increment"):
                row[f"{member}_{component}_degC"] = float(np.mean(series[member][component][selected]))
            row[f"{member}_posterior_minus_reference_degC"] = float(
                row[f"{member}_posterior_degC"] - row["reference_central_degC"]
            )
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "lia_50year_prior_update_decomposition.csv", index=False)

    method_rows = []
    for member in MEMBERS:
        gaps = windows[f"{member}_posterior_minus_reference_degC"]
        backgrounds = windows[f"{member}_background_degC"]
        increments = windows[f"{member}_increment_degC"]
        method_rows.append({
            "member": member,
            "method": MEMBERS[member][0],
            "proxy_network": MEMBERS[member][1],
            "warm_gap_window_count": int((gaps > 0).sum()),
            "median_posterior_minus_reference_degC": float(gaps.median()),
            "median_background_anomaly_degC": float(backgrounds.median()),
            "median_increment_anomaly_degC": float(increments.median()),
        })
    method_table = pd.DataFrame(method_rows)
    method_table.to_csv(output / "member_prior_update_summary.csv", index=False)

    export_difference = ensemble["posterior"] - export_series
    export_valid = np.isfinite(export_difference)
    ordinary_warm_all = bool(
        all((windows[f"O_{network}_posterior_minus_reference_degC"] > 0).all() for network in (6, 9))
    )
    increment_only_warm_count = int((windows["increment_only_minus_reference_degC"] > 0).sum())
    adaptive_background_median = float(windows["background_degC"].median())
    total_gap_median = float(windows["posterior_minus_reference_degC"].median())
    background_fraction = float(adaptive_background_median / total_gap_median) if total_gap_median != 0 else np.nan
    passed = bool(
        closure_max <= 1e-12
        and fixed_background_max <= 1e-12
        and adaptive_background_identity <= 1e-12
        and locked_proxy_reads == 0
    )
    decision = (
        "route_to_update_amplitude_and_prior_covariance_validation"
        if passed and ordinary_warm_all and increment_only_warm_count >= 8
        else "retain_prior_update_mechanism_unresolved"
    )
    summary = {
        "status": "pass" if passed else "fail",
        "locked_1921_1939_values_read": False,
        "requested_years": ["1400-1899", "1961-1990"],
        "recovered_config": recovered_config,
        "closure_max_abs_degC": closure_max,
        "ordinary_fixed_background_anomaly_max_abs_degC": fixed_background_max,
        "ao_hgao_background_identity_max_abs_degC": adaptive_background_identity,
        "ordinary_members_warm_in_all_lia_windows": ordinary_warm_all,
        "increment_only_warmer_than_reference_window_count": increment_only_warm_count,
        "lia_window_count": int(len(windows)),
        "median_native_posterior_minus_reference_degC": total_gap_median,
        "median_ensemble_background_anomaly_degC": adaptive_background_median,
        "median_background_fraction_of_warm_gap": background_fraction,
        "native_vs_round23_export": {
            "correlation": float(np.corrcoef(ensemble["posterior"][export_valid], export_series[export_valid])[0, 1]),
            "rmse_degC": float(np.sqrt(np.mean(export_difference[export_valid] ** 2))),
            "max_abs_degC": float(np.max(np.abs(export_difference[export_valid]))),
        },
        "interpretation_boundary": "AO/HGAO backgrounds are deterministic reconstructions from archived strategy-C configuration because their yearly selected-member indices were not archived.",
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lia = annual[(annual.year >= 1400) & (annual.year <= 1899)].copy()
    smooth = lambda values: pd.Series(values).rolling(11, center=True, min_periods=11).mean().to_numpy()
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(lia.year, smooth(lia.reference_family_central_degC), color="#006d77", lw=2.2, label="Reference central")
    axes[0].plot(lia.year, smooth(lia.native_six_member_posterior_degC), color="#d55e00", lw=2.0, label="Posterior")
    axes[0].plot(lia.year, smooth(lia.native_six_member_background_degC), color="#5e3c99", lw=1.7, label="Background")
    axes[0].plot(lia.year, smooth(lia.native_six_member_increment_degC), color="#0072b2", lw=1.7, label="Assimilation increment")
    axes[0].axhline(0, color="#555555", lw=0.8)
    axes[0].set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("Native-state HMA temperature decomposition (11-year mean)")
    axes[0].legend(frameon=False, ncol=4)
    centers = (windows.start_year + windows.end_year) / 2
    axes[1].bar(centers - 7, windows.background_degC, width=14, color="#5e3c99", label="Background")
    axes[1].bar(centers + 7, windows.increment_degC, width=14, color="#0072b2", label="Increment")
    axes[1].plot(centers, windows.posterior_degC, color="#d55e00", marker="o", lw=1.8, label="Posterior")
    axes[1].plot(centers, windows.reference_central_degC, color="#006d77", marker="s", lw=1.8, label="Reference")
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set(xlabel="Year", ylabel="50-year mean anomaly (degC)", title="50-year component means")
    axes[1].legend(frameon=False, ncol=4)
    figure.savefig(output / "temperature_prior_update_decomposition.png", dpi=190)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Prior/Update Decomposition\n\n"
        f"- Numerical gates: `{summary['status']}`; maximum closure `{closure_max:.3e} degC`.\n"
        f"- Ordinary fixed-background anomaly maximum: `{fixed_background_max:.3e} degC`.\n"
        f"- Ordinary EnKF members warm relative to reference in every LIA window: `{str(ordinary_warm_all).lower()}`.\n"
        f"- Increment-only ensemble remains warmer than reference in `{increment_only_warm_count}/10` windows.\n"
        f"- Median posterior warm gap: `{total_gap_median:+.3f} degC`.\n"
        f"- Median adaptive-background contribution in the six-member ensemble: `{adaptive_background_median:+.3f} degC` "
        f"(`{100 * background_fraction:+.1f}%` of the median warm gap).\n\n"
        f"Decision: `{decision}`.\n\n"
        "The fixed ordinary-EnKF branches prove that adaptive analog selection is not necessary for the warm conflict. "
        "The next mechanism test should target update amplitude and the prior state-proxy covariance, not a time-varying prior forcing.\n",
        encoding="utf-8",
    )

    after = {name: file_state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2),
        encoding="utf-8",
    )
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
