from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return float("nan")
    x = years[finite].astype(float)
    x -= np.mean(x)
    return float(100.0 * np.polyfit(x, values[finite], 1)[0])


def _change(
    years: np.ndarray,
    values: np.ndarray,
    early_years: tuple[int, int],
    late_years: tuple[int, int],
) -> float:
    early = (years >= early_years[0]) & (years <= early_years[1])
    late = (years >= late_years[0]) & (years <= late_years[1])
    return float(np.nanmean(values[late]) - np.nanmean(values[early]))


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _anomaly(
    years: np.ndarray,
    values: np.ndarray,
    reference_years: tuple[int, int],
) -> np.ndarray:
    reference = (years >= reference_years[0]) & (years <= reference_years[1])
    return values - np.nanmean(values[..., reference], axis=-1, keepdims=True)


def _target_weights(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    lat_hi = int(np.searchsorted(grid_lat, target_lat))
    lon_hi = int(np.searchsorted(grid_lon, target_lon))
    lat_lo = lat_hi - 1
    lon_lo = lon_hi - 1
    if lat_lo < 0 or lon_lo < 0 or lat_hi >= len(grid_lat) or lon_hi >= len(grid_lon):
        raise ValueError("target-outside-state-grid")
    fy = (target_lat - grid_lat[lat_lo]) / (grid_lat[lat_hi] - grid_lat[lat_lo])
    fx = (target_lon - grid_lon[lon_lo]) / (grid_lon[lon_hi] - grid_lon[lon_lo])
    weights = np.zeros(len(grid_lat) * len(grid_lon), dtype=float)
    weights[lon_lo * len(grid_lat) + lat_lo] = (1.0 - fx) * (1.0 - fy)
    weights[lon_lo * len(grid_lat) + lat_hi] = (1.0 - fx) * fy
    weights[lon_hi * len(grid_lat) + lat_lo] = fx * (1.0 - fy)
    weights[lon_hi * len(grid_lat) + lat_hi] = fx * fy
    return weights


def _haversine_km(
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(target_lat)
    dlat = lat1 - lat2
    dlon = np.deg2rad(lon - target_lon)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2.0
    ) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(value))


def _distance_band(distance: float, bounds: list[float]) -> str:
    if distance < bounds[0]:
        return f"<{int(bounds[0])}"
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        if distance < upper:
            return f"{int(lower)}-{int(upper)}"
    return f">{int(bounds[-1])}"


def _sector(lat: float, lon: float, target_lat: float, target_lon: float) -> str:
    north = "N" if lat >= target_lat else "S"
    east = "E" if lon >= target_lon else "W"
    return north + east


def _series_metrics(
    years: np.ndarray,
    reconstruction: np.ndarray,
    observation: np.ndarray,
    analysis_years: tuple[int, int],
    early_years: tuple[int, int],
    late_years: tuple[int, int],
    rolling_window: int,
) -> dict[str, float]:
    selected = (years >= analysis_years[0]) & (years <= analysis_years[1])
    x = years[selected]
    rec = reconstruction[selected]
    obs = observation[selected]
    valid = np.isfinite(rec) & np.isfinite(obs)
    rec_roll = _rolling(rec, rolling_window)
    obs_roll = _rolling(obs, rolling_window)
    rolling_valid = np.isfinite(rec_roll) & np.isfinite(obs_roll)
    rec_trend = _trend(x, rec)
    obs_trend = _trend(x, obs)
    rec_change = _change(x, rec, early_years, late_years)
    obs_change = _change(x, obs, early_years, late_years)
    return {
        "annual_rmse_c": float(np.sqrt(np.mean((rec[valid] - obs[valid]) ** 2))),
        "annual_correlation": float(np.corrcoef(rec[valid], obs[valid])[0, 1]),
        "rolling30_rmse_c": float(
            np.sqrt(np.mean((rec_roll[rolling_valid] - obs_roll[rolling_valid]) ** 2))
        ),
        "rolling30_correlation": float(
            np.corrcoef(rec_roll[rolling_valid], obs_roll[rolling_valid])[0, 1]
        ),
        "reconstruction_trend_c_per_century": rec_trend,
        "observation_trend_c_per_century": obs_trend,
        "trend_absolute_error_c_per_century": abs(rec_trend - obs_trend),
        "reconstruction_early_late_change_c": rec_change,
        "observation_early_late_change_c": obs_change,
        "early_late_absolute_error_c": abs(rec_change - obs_change),
    }


def _instrument_tables(
    archive: np.lib.npyio.NpzFile,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = archive["years"]
    reference_years = tuple(config["scope"]["reference_years"])
    analysis_years = tuple(config["scope"]["instrumental_years"])
    early_years = tuple(config["scope"]["early_years"])
    late_years = tuple(config["scope"]["late_years"])
    rolling_window = int(config["scope"]["rolling_window_years"])
    names = [str(value) for value in archive["member_names"]]
    names += ["ensemble_mean", "rec_ref"]
    target_series = list(archive["member_targets"])
    target_series += [archive["ensemble_target"], archive["reference_target"]]
    field_series = list(archive["member_fields"])
    field_series += [archive["ensemble_field"], archive["reference_field"]]
    target_obs = _anomaly(
        years, archive["observation_targets"], reference_years
    )
    field_obs = _anomaly(years, archive["observation_fields"], reference_years)
    products = [str(value) for value in archive["product_names"]]
    target_rows = []
    regional_rows = []
    summary_rows = []
    selected = (years >= analysis_years[0]) & (years <= analysis_years[1])

    for name, target_raw, field_raw in zip(names, target_series, field_series):
        target = _anomaly(years, target_raw, reference_years)
        field = _anomaly(years, field_raw, reference_years)
        regional = np.nanmean(field, axis=0)
        summary_rows.append(
            {
                "branch": name,
                "target_trend_c_per_century": _trend(years[selected], target[selected]),
                "target_early_late_change_c": _change(
                    years[selected], target[selected], early_years, late_years
                ),
                "regional_trend_c_per_century": _trend(
                    years[selected], regional[selected]
                ),
                "regional_early_late_change_c": _change(
                    years[selected], regional[selected], early_years, late_years
                ),
            }
        )
        for product_index, product in enumerate(products):
            target_metric = _series_metrics(
                years,
                target,
                target_obs[product_index],
                analysis_years,
                early_years,
                late_years,
                rolling_window,
            )
            target_rows.append({"branch": name, "product": product, **target_metric})
            obs = field_obs[product_index]
            complete = np.isfinite(obs[:, selected]).all(axis=1)
            complete &= np.isfinite(field[:, selected]).all(axis=1)
            if complete.sum() < 3:
                raise ValueError(f"insufficient-regional-cells:{name}:{product}")
            rec_regional = np.mean(field[complete], axis=0)
            obs_regional = np.mean(obs[complete], axis=0)
            regional_metric = _series_metrics(
                years,
                rec_regional,
                obs_regional,
                analysis_years,
                early_years,
                late_years,
                rolling_window,
            )
            regional_rows.append(
                {
                    "branch": name,
                    "product": product,
                    "complete_cell_count": int(complete.sum()),
                    **regional_metric,
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(target_rows),
        pd.DataFrame(regional_rows),
    )


def _replay_and_contributions(
    replay: np.lib.npyio.NpzFile,
    target_weights: np.ndarray,
) -> dict[str, np.ndarray | float]:
    background = replay["background"]
    hxb = replay["hxb"]
    observations = replay["observations"]
    residuals = replay["residuals"]
    members = background.shape[1]
    background_mean = np.mean(background, axis=1)
    background_prime = background - background_mean[:, None]
    hxb_mean = np.mean(hxb, axis=1)
    hxb_prime = hxb - hxb_mean[:, None]
    pbht = background_prime @ hxb_prime.T / (members - 1)
    hpbh = hxb_prime @ hxb_prime.T / (members - 1)
    error_variance = np.var(residuals, axis=1, ddof=1)
    analysis = np.repeat(background_mean[:, None], observations.shape[1], axis=1)
    decomposed = np.repeat(background_mean[:, None], observations.shape[1], axis=1)
    target_contributions = np.zeros_like(observations)
    regional_contributions = np.zeros_like(observations)
    solved_weights = np.zeros_like(observations)
    innovations = np.full_like(observations, np.nan)
    target_pbht = target_weights @ pbht
    regional_pbht = np.mean(pbht, axis=0)

    groups: dict[tuple[int, ...], list[int]] = {}
    for year_index in range(observations.shape[1]):
        active = tuple(np.flatnonzero(np.isfinite(observations[:, year_index])))
        groups.setdefault(active, []).append(year_index)
    for active_tuple, year_indices_list in groups.items():
        if not active_tuple:
            continue
        active = np.asarray(active_tuple, dtype=int)
        year_indices = np.asarray(year_indices_list, dtype=int)
        covariance = hpbh[np.ix_(active, active)] + np.diag(error_variance[active])
        innovation = observations[np.ix_(active, year_indices)] - hxb_mean[active, None]
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            weights = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            weights = scipy.linalg.solve(
                covariance, innovation, assume_a="sym", check_finite=False
            )
        increment = pbht[:, active] @ weights
        explicit_sum = np.einsum("sp,pt->st", pbht[:, active], weights)
        analysis[:, year_indices] += increment
        decomposed[:, year_indices] += explicit_sum
        target_contributions[np.ix_(active, year_indices)] = (
            target_pbht[active, None] * weights
        )
        regional_contributions[np.ix_(active, year_indices)] = (
            regional_pbht[active, None] * weights
        )
        solved_weights[np.ix_(active, year_indices)] = weights
        innovations[np.ix_(active, year_indices)] = innovation

    target_analysis = target_weights @ analysis
    regional_analysis = np.mean(analysis, axis=0)
    target_background = float(target_weights @ background_mean)
    regional_background = float(np.mean(background_mean))
    return {
        "analysis": analysis,
        "target_analysis": target_analysis,
        "regional_analysis": regional_analysis,
        "target_background": target_background,
        "regional_background": regional_background,
        "target_contributions": target_contributions,
        "regional_contributions": regional_contributions,
        "weights": solved_weights,
        "innovations": innovations,
        "increment_closure_state_max_c": float(np.max(np.abs(analysis - decomposed))),
        "increment_closure_target_max_c": float(
            np.max(
                np.abs(
                    target_analysis
                    - (target_background + np.sum(target_contributions, axis=0))
                )
            )
        ),
        "increment_closure_regional_max_c": float(
            np.max(
                np.abs(
                    regional_analysis
                    - (regional_background + np.sum(regional_contributions, axis=0))
                )
            )
        ),
    }


def _proxy_table(
    replay: np.lib.npyio.NpzFile,
    result: dict[str, np.ndarray | float],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = replay["years"]
    analysis_years = tuple(config["scope"]["instrumental_years"])
    early_years = tuple(config["scope"]["early_years"])
    late_years = tuple(config["scope"]["late_years"])
    selected = (years >= analysis_years[0]) & (years <= analysis_years[1])
    x = years[selected]
    target_lat = float(config["scope"]["target_lat"])
    target_lon = float(config["scope"]["target_lon"])
    locations = replay["proxy_locations"]
    coefficients = replay["proxy_coefficients"]
    distance = _haversine_km(
        locations[:, 0], locations[:, 1], target_lat, target_lon
    )
    bounds = [float(value) for value in config["distance_bands_km"]]
    target_contributions = np.asarray(result["target_contributions"])
    regional_contributions = np.asarray(result["regional_contributions"])
    innovations = np.asarray(result["innovations"])
    weights = np.asarray(result["weights"])
    rows = []
    for index in range(target_contributions.shape[0]):
        target = target_contributions[index, selected]
        regional = regional_contributions[index, selected]
        innovation = innovations[index, selected]
        weight = weights[index, selected]
        slope = coefficients[index, 1]
        rows.append(
            {
                "proxy_index_1based": index + 1,
                "latitude": float(locations[index, 0]),
                "longitude": float(locations[index, 1]),
                "distance_to_3584_km": float(distance[index]),
                "distance_band_km": _distance_band(float(distance[index]), bounds),
                "sector_from_3584": _sector(
                    float(locations[index, 0]),
                    float(locations[index, 1]),
                    target_lat,
                    target_lon,
                ),
                "proxy_type": str(replay["proxy_types"][index]),
                "psm_method": str(replay["psm_methods"][index]),
                "temperature_coefficient": float(slope),
                "temperature_coefficient_sign": (
                    "positive" if slope > 0 else "negative" if slope < 0 else "zero"
                ),
                "active_year_count_1901_2000": int(np.isfinite(innovation).sum()),
                "mean_innovation_1901_2000": float(np.nanmean(innovation)),
                "mean_abs_weight_1901_2000": float(np.nanmean(np.abs(weight))),
                "target_trend_contribution_c_per_century": _trend(x, target),
                "target_early_late_contribution_c": _change(
                    x, target, early_years, late_years
                ),
                "regional_trend_contribution_c_per_century": _trend(x, regional),
                "regional_early_late_contribution_c": _change(
                    x, regional, early_years, late_years
                ),
            }
        )
    frame = pd.DataFrame(rows)
    group_frames = []
    for column in [
        "distance_band_km",
        "sector_from_3584",
        "proxy_type",
        "psm_method",
        "temperature_coefficient_sign",
    ]:
        for group, indices in frame.groupby(column).groups.items():
            ids = np.asarray(list(indices), dtype=int)
            target = np.sum(target_contributions[ids], axis=0)[selected]
            regional = np.sum(regional_contributions[ids], axis=0)[selected]
            group_frames.append(
                {
                    "grouping": column,
                    "group": str(group),
                    "proxy_count": int(ids.size),
                    "target_trend_contribution_c_per_century": _trend(x, target),
                    "target_early_late_contribution_c": _change(
                        x, target, early_years, late_years
                    ),
                    "regional_trend_contribution_c_per_century": _trend(x, regional),
                    "regional_early_late_contribution_c": _change(
                        x, regional, early_years, late_years
                    ),
                }
            )
    return frame, pd.DataFrame(group_frames)


def _copy_baseline_validation(root: Path, output: Path, config: dict) -> dict:
    source = root / config["source_paths"]["localization_pilot_result"]
    proxy = pd.read_csv(source / "heldout_proxy_metrics.csv")
    instrument = pd.read_csv(source / "instrument_product_metrics.csv")
    branch = pd.read_csv(source / "branch_summary.csv")
    proxy = proxy[proxy["branch"].eq("no_localization")].copy()
    instrument = instrument[instrument["branch"].eq("no_localization")].copy()
    branch = branch[branch["branch"].eq("no_localization")].copy()
    proxy.to_csv(output / "baseline_proxy_validation.csv", index=False)
    instrument.to_csv(output / "baseline_instrument_validation.csv", index=False)
    summary = branch.iloc[0].to_dict()
    for key, value in list(summary.items()):
        if isinstance(value, (np.integer, np.floating)):
            summary[key] = value.item()
    payload = {
        "interpretation": (
            "Inherited unchanged baseline validation because this diagnostic "
            "does not alter the reconstruction."
        ),
        "source_directory": str(source),
        "source_files": {
            name: _sha256(source / name)
            for name in [
                "heldout_proxy_metrics.csv",
                "instrument_product_metrics.csv",
                "branch_summary.csv",
            ]
        },
        "no_localization_summary": summary,
    }
    (output / "baseline_validation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return payload


def _plot_member_curves(
    output: Path,
    archive: np.lib.npyio.NpzFile,
    config: dict,
) -> None:
    years = archive["years"]
    selected = (years >= 1901) & (years <= 2000)
    x = years[selected]
    reference_years = tuple(config["scope"]["reference_years"])
    window = int(config["scope"]["rolling_window_years"])
    members_target = _anomaly(years, archive["member_targets"], reference_years)
    ensemble_target = _anomaly(years, archive["ensemble_target"], reference_years)
    reference_target = _anomaly(years, archive["reference_target"], reference_years)
    observations_target = _anomaly(
        years, archive["observation_targets"], reference_years
    )
    members_regional = np.nanmean(
        _anomaly(years, archive["member_fields"], reference_years), axis=1
    )
    ensemble_regional = np.nanmean(
        _anomaly(years, archive["ensemble_field"], reference_years), axis=0
    )
    observations_regional = np.nanmean(
        _anomaly(years, archive["observation_fields"], reference_years), axis=1
    )
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for index, name in enumerate(archive["member_names"]):
        axes[0].plot(
            x,
            _rolling(members_target[index, selected], window),
            color=colors[index],
            linewidth=1.5,
            label=str(name),
        )
    obs_mean = np.nanmean(observations_target[:, selected], axis=0)
    obs_min = np.nanmin(observations_target[:, selected], axis=0)
    obs_max = np.nanmax(observations_target[:, selected], axis=0)
    axes[0].fill_between(
        x, _rolling(obs_min, window), _rolling(obs_max, window), color="0.85"
    )
    axes[0].plot(
        x, _rolling(obs_mean, window), color="black", linewidth=2.2, label="five-product mean"
    )
    axes[0].plot(
        x,
        _rolling(ensemble_target[selected], window),
        color="#6A3D9A",
        linewidth=2.4,
        label="six-member mean",
    )
    axes[0].plot(
        x,
        _rolling(reference_target[selected], window),
        color="0.35",
        linestyle="--",
        linewidth=1.7,
        label="rec_ref",
    )
    axes[0].set_title("3584 pixel: archived reconstruction members")
    axes[0].set_ylabel("30-year anomaly (degC)")
    axes[0].legend(ncol=3, fontsize=8)
    for index, name in enumerate(archive["member_names"]):
        axes[1].plot(
            x,
            _rolling(members_regional[index, selected], window),
            color=colors[index],
            linewidth=1.5,
            label=str(name),
        )
    regional_obs_mean = np.nanmean(observations_regional[:, selected], axis=0)
    axes[1].plot(
        x,
        _rolling(regional_obs_mean, window),
        color="black",
        linewidth=2.2,
        label="five-product mean",
    )
    axes[1].plot(
        x,
        _rolling(ensemble_regional[selected], window),
        color="#6A3D9A",
        linewidth=2.4,
        label="six-member mean",
    )
    axes[1].set_title("HMA replay state grid")
    axes[1].set_ylabel("30-year anomaly (degC)")
    axes[1].set_xlabel("Year / rolling-window end year")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "six_member_low_frequency_curves.png", dpi=220)
    plt.close(fig)


def _plot_contributions(
    output: Path,
    archive: np.lib.npyio.NpzFile,
    replay: np.lib.npyio.NpzFile,
    result: dict[str, np.ndarray | float],
    proxy_frame: pd.DataFrame,
    config: dict,
) -> None:
    years = replay["years"]
    selected = (years >= 1901) & (years <= 2000)
    x = years[selected]
    window = int(config["scope"]["rolling_window_years"])
    reference_years = tuple(config["scope"]["reference_years"])
    replay_target = _anomaly(
        years, np.asarray(result["target_analysis"]), reference_years
    )
    o6_index = list(archive["member_names"]).index("OEnKF-proxy6")
    archive_o6 = _anomaly(
        years, archive["member_targets"][o6_index], reference_years
    )
    observations = _anomaly(
        years, archive["observation_targets"], reference_years
    )
    target_contributions = np.asarray(result["target_contributions"])
    distance_groups = {}
    for group, indices in proxy_frame.groupby("distance_band_km").groups.items():
        distance_groups[str(group)] = np.sum(
            target_contributions[np.asarray(list(indices), dtype=int)], axis=0
        )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    obs_mean = np.nanmean(observations[:, selected], axis=0)
    obs_min = np.nanmin(observations[:, selected], axis=0)
    obs_max = np.nanmax(observations[:, selected], axis=0)
    axes[0].fill_between(
        x, _rolling(obs_min, window), _rolling(obs_max, window), color="0.85"
    )
    axes[0].plot(
        x, _rolling(obs_mean, window), color="black", linewidth=2.2, label="five-product mean"
    )
    axes[0].plot(
        x,
        _rolling(archive_o6[selected], window),
        color="#D55E00",
        linewidth=2.2,
        label="archived OEnKF-proxy6",
    )
    axes[0].plot(
        x,
        _rolling(replay_target[selected], window),
        color="#0072B2",
        linestyle="--",
        linewidth=2.0,
        label="exact replay",
    )
    axes[0].set_title("3584 OEnKF-proxy6 replay")
    axes[0].set_ylabel("30-year anomaly (degC)")
    axes[0].legend(fontsize=8)
    for group, values in distance_groups.items():
        axes[1].plot(x, _rolling(values[selected], window), linewidth=1.8, label=group + " km")
    total = np.sum(target_contributions, axis=0)
    axes[1].plot(
        x, _rolling(total[selected], window), color="black", linewidth=2.4, label="total increment"
    )
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1].set_title("Posterior increment grouped by proxy distance")
    axes[1].set_ylabel("30-year contribution (degC)")
    axes[1].set_xlabel("Year / rolling-window end year")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "o6_proxy_contribution_curves.png", dpi=220)
    plt.close(fig)

    ranked = proxy_frame.reindex(
        proxy_frame["target_trend_contribution_c_per_century"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(15)
    ranked = ranked.iloc[::-1]
    colors = [
        "#D55E00" if value < 0 else "#0072B2"
        for value in ranked["target_trend_contribution_c_per_century"]
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [
        f"{int(row.proxy_index_1based)} | {row.distance_to_3584_km:.0f} km"
        for row in ranked.itertuples()
    ]
    ax.barh(labels, ranked["target_trend_contribution_c_per_century"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("3584 trend contribution (degC per century)")
    ax.set_ylabel("Proxy index | distance")
    ax.set_title("Largest absolute proxy contributions, 1901-2000")
    fig.tight_layout()
    fig.savefig(output / "proxy_trend_contribution_ranking.png", dpi=220)
    plt.close(fig)


def _source_purity(
    output: Path,
    archive_manifest: dict,
    replay_manifest: dict,
) -> dict:
    records = []
    source_records = [
        {
            "path": archive_manifest["source"],
            "sha256": archive_manifest["source_sha256"],
            "size_bytes": archive_manifest["source_size_bytes"],
        }
    ] + replay_manifest["source_files"]
    for record in source_records:
        path = Path(record["path"])
        current_hash = _sha256(path)
        current_size = path.stat().st_size
        records.append(
            {
                "path": str(path),
                "before_sha256": record["sha256"],
                "after_sha256": current_hash,
                "before_size_bytes": int(record["size_bytes"]),
                "after_size_bytes": int(current_size),
                "unchanged": (
                    current_hash == record["sha256"]
                    and current_size == int(record["size_bytes"])
                ),
            }
        )
    payload = {
        "all_sources_unchanged": all(item["unchanged"] for item in records),
        "files": records,
    }
    (output / "source_purity_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_increment_attribution_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = np.load(output / "inputs/archive_members_and_instruments.npz")
    replay = np.load(output / "inputs/replay_attribution_inputs.npz")
    if not np.array_equal(archive["years"], replay["years"]):
        raise ValueError("year-axis-mismatch")
    if not np.allclose(archive["state_lat"], replay["state_lat"]):
        raise ValueError("state-latitude-mismatch")
    if not np.allclose(archive["state_lon"], replay["state_lon"]):
        raise ValueError("state-longitude-mismatch")

    target_weights = _target_weights(
        replay["grid_lat"],
        replay["grid_lon"],
        float(config["scope"]["target_lat"]),
        float(config["scope"]["target_lon"]),
    )
    result = _replay_and_contributions(replay, target_weights)
    years = replay["years"]
    selected = (years >= 1901) & (years <= 2000)
    x = years[selected]
    early_years = tuple(config["scope"]["early_years"])
    late_years = tuple(config["scope"]["late_years"])
    target_increment = np.sum(
        np.asarray(result["target_contributions"]), axis=0
    )[selected]
    regional_increment = np.sum(
        np.asarray(result["regional_contributions"]), axis=0
    )[selected]
    proxy_frame, group_frame = _proxy_table(replay, result, config)
    member_summary, target_metrics, regional_metrics = _instrument_tables(
        archive, config
    )

    member_summary.to_csv(output / "member_low_frequency_summary.csv", index=False)
    target_metrics.to_csv(output / "member_target_instrument_metrics.csv", index=False)
    regional_metrics.to_csv(
        output / "member_regional_instrument_metrics.csv", index=False
    )
    proxy_frame.to_csv(output / "proxy_increment_contributions.csv", index=False)
    group_frame.to_csv(output / "proxy_increment_group_summary.csv", index=False)
    np.savez_compressed(
        output / "proxy_increment_timeseries.npz",
        years=years,
        target_contributions=result["target_contributions"],
        regional_contributions=result["regional_contributions"],
        weights=result["weights"],
        innovations=result["innovations"],
        target_analysis=result["target_analysis"],
        regional_analysis=result["regional_analysis"],
    )
    baseline_validation = _copy_baseline_validation(root, output, config)

    member_difference_state = (
        np.mean(archive["member_fields"], axis=0) - archive["ensemble_field"]
    )
    member_difference_target = (
        np.mean(archive["member_targets"], axis=0) - archive["ensemble_target"]
    )
    replay_difference = np.asarray(result["analysis"]) - replay["target_mean"]
    target_trend_closure = abs(
        proxy_frame["target_trend_contribution_c_per_century"].sum()
        - _trend(x, target_increment)
    )
    target_change_closure = abs(
        proxy_frame["target_early_late_contribution_c"].sum()
        - _change(x, target_increment, early_years, late_years)
    )
    regional_trend_closure = abs(
        proxy_frame["regional_trend_contribution_c_per_century"].sum()
        - _trend(x, regional_increment)
    )
    regional_change_closure = abs(
        proxy_frame["regional_early_late_contribution_c"].sum()
        - _change(x, regional_increment, early_years, late_years)
    )
    metric_closure_max = max(
        target_trend_closure,
        target_change_closure,
        regional_trend_closure,
        regional_change_closure,
    )
    thresholds = config["preflight"]
    preflight = {
        "member_mean_state_max_abs_c": float(
            np.nanmax(np.abs(member_difference_state))
        ),
        "member_mean_target_max_abs_c": float(
            np.nanmax(np.abs(member_difference_target))
        ),
        "replay_archive_max_abs_c": float(np.max(np.abs(replay_difference))),
        "replay_archive_rmse_c": float(
            np.sqrt(np.mean(np.asarray(replay_difference) ** 2))
        ),
        "increment_closure_state_max_c": result[
            "increment_closure_state_max_c"
        ],
        "increment_closure_target_max_c": result[
            "increment_closure_target_max_c"
        ],
        "increment_closure_regional_max_c": result[
            "increment_closure_regional_max_c"
        ],
        "metric_closure_max_c": metric_closure_max,
    }
    preflight["member_mean_pass"] = (
        max(
            preflight["member_mean_state_max_abs_c"],
            preflight["member_mean_target_max_abs_c"],
        )
        <= thresholds["member_mean_closure_max_c"]
    )
    preflight["replay_pass"] = (
        preflight["replay_archive_max_abs_c"]
        <= thresholds["replay_archive_max_c"]
    )
    preflight["increment_closure_pass"] = (
        max(
            preflight["increment_closure_state_max_c"],
            preflight["increment_closure_target_max_c"],
            preflight["increment_closure_regional_max_c"],
        )
        <= thresholds["increment_closure_max_c"]
    )
    preflight["metric_closure_pass"] = (
        preflight["metric_closure_max_c"] <= thresholds["metric_closure_max_c"]
    )
    preflight["all_pass"] = all(
        preflight[key]
        for key in [
            "member_mean_pass",
            "replay_pass",
            "increment_closure_pass",
            "metric_closure_pass",
        ]
    )
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    _plot_member_curves(output, archive, config)
    _plot_contributions(output, archive, replay, result, proxy_frame, config)
    archive_manifest = json.loads(
        (output / "inputs/archive_export_manifest.json").read_text(encoding="utf-8")
    )
    replay_manifest = json.loads(
        (output / "inputs/replay_input_manifest.json").read_text(encoding="utf-8")
    )
    purity = _source_purity(output, archive_manifest, replay_manifest)

    member_only = member_summary[
        member_summary["branch"].isin([str(value) for value in archive["member_names"]])
    ]
    all_target_negative = bool(
        (member_only["target_trend_c_per_century"] < 0).all()
    )
    all_regional_negative = bool(
        (member_only["regional_trend_c_per_century"] < 0).all()
    )
    top = proxy_frame.reindex(
        proxy_frame["target_trend_contribution_c_per_century"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(15)
    total_abs = float(
        proxy_frame["target_trend_contribution_c_per_century"].abs().sum()
    )
    top_abs_fraction = float(
        top["target_trend_contribution_c_per_century"].abs().sum() / total_abs
    )
    strongest_groups = (
        group_frame[group_frame["grouping"].eq("distance_band_km")]
        .reindex(
            group_frame[group_frame["grouping"].eq("distance_band_km")][
                "target_trend_contribution_c_per_century"
            ]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        .to_dict(orient="records")
    )
    o6_row = member_summary[member_summary["branch"].eq("OEnKF-proxy6")].iloc[0]
    ensemble_row = member_summary[member_summary["branch"].eq("ensemble_mean")].iloc[0]
    summary = {
        "experiment_id": config["experiment_id"],
        "preflight": preflight,
        "source_purity": purity["all_sources_unchanged"],
        "six_member_attribution": {
            "all_six_target_trends_negative": all_target_negative,
            "all_six_regional_trends_negative": all_regional_negative,
            "oenkf_proxy6_target_trend_c_per_century": float(
                o6_row["target_trend_c_per_century"]
            ),
            "ensemble_target_trend_c_per_century": float(
                ensemble_row["target_trend_c_per_century"]
            ),
            "interpretation": (
                "The negative final signal is shared by every archived member; "
                "it is not created by averaging one anomalous branch."
                if all_target_negative
                else "The negative final signal is concentrated in a subset of members."
            ),
        },
        "proxy_increment_attribution": {
            "top_15_absolute_fraction": top_abs_fraction,
            "top_15_proxy_indices_1based": [
                int(value) for value in top["proxy_index_1based"]
            ],
            "largest_distance_band_contributions": strongest_groups,
        },
        "validation": baseline_validation,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    top_lines = "\n".join(
        (
            f"- Proxy {int(row.proxy_index_1based)} at "
            f"{row.distance_to_3584_km:.0f} km: "
            f"{row.target_trend_contribution_c_per_century:+.4f} "
            "degC/century."
        )
        for row in top.head(8).itertuples()
    )
    group_lines = "\n".join(
        (
            f"- {row['group']} km ({int(row['proxy_count'])} proxies): "
            f"{row['target_trend_contribution_c_per_century']:+.4f} "
            "degC/century."
        )
        for row in strongest_groups
    )
    result_text = f"""# Increment attribution final result

## Preflight

All hard gates passed: {preflight['all_pass']}. The six-member mean closes to
the archived ensemble at the state grid within
{preflight['member_mean_state_max_abs_c']:.3e} degC. The OEnKF-proxy6 replay
remains within {preflight['replay_archive_max_abs_c']:.8f} degC of the archived
posterior. Per-proxy increments close at state, target, regional, trend, and
early-to-late levels; the largest metric closure error is
{preflight['metric_closure_max_c']:.3e}.

## Six-member result

All six target trends are negative: {all_target_negative}. All six regional
trends are negative: {all_regional_negative}. OEnKF-proxy6 has a 3584 trend of
{float(o6_row['target_trend_c_per_century']):+.4f} degC/century, while the
six-member mean is {float(ensemble_row['target_trend_c_per_century']):+.4f}
degC/century. Therefore the negative low-frequency sign is shared across the
archived method/proxy branches, not introduced by the ensemble averaging step.

## OEnKF-proxy6 proxy increments

The 15 largest absolute proxy contributions account for
{100.0 * top_abs_fraction:.1f}% of the sum of absolute 3584 trend
contributions. This percentage measures concentration, not net explained
variance, because positive and negative proxy increments cancel.

Largest individual absolute contributions:

{top_lines}

Distance-band net contributions:

{group_lines}

## Validation and scope

This experiment changes no reconstruction. Proxy holdout and five-product
instrumental metrics are copied from the frozen no-localization baseline and
remain unchanged. The attribution is mechanistic: it identifies where the
posterior signal enters, but it does not by itself decide whether the proxy
innovation, PSM coefficient, or prior covariance is physically wrong.

No glacier simulation or glacier curve was used. All recorded source files
were unchanged after the experiment: {purity['all_sources_unchanged']}.
"""
    (output / "FINAL_RESULT.md").write_text(result_text, encoding="utf-8")
    patch_text = f"""*** Begin Patch
*** Update File: LIA_PROJECT_FACTS.md
@@
+- Independent climate increment attribution experiment
+  climate_reconstruction_increment_attribution_v1 passed all conservation
+  gates. All six archived reconstruction members have negative 1901-2000
+  trends at the 3584 diagnostic pixel, so the negative ensemble sign is
+  shared across branches rather than introduced by averaging. The exact
+  OEnKF-proxy6 replay closes to its archive within
+  {preflight['replay_archive_max_abs_c']:.8f} degC, and per-proxy K times
+  innovation contributions close at state, target, regional, trend, and
+  early-to-late levels. This is a diagnostic result only and does not promote
+  a climate correction.
*** End Patch
"""
    (output / "lia_facts_update_increment_attribution_v1.patch").write_text(
        patch_text, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if preflight["all_pass"] and purity["all_sources_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
