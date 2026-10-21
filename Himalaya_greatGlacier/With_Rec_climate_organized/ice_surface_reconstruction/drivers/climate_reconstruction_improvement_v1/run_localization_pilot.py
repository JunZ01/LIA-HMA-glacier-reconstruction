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
from scipy.interpolate import RegularGridInterpolator


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


def _gaspari_cohn(
    state_lat: np.ndarray,
    state_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    radius_km: float | None,
) -> np.ndarray:
    if radius_km is None:
        return np.ones((state_lat.size, obs_lat.size), dtype=float)
    dlat = state_lat[:, None] - obs_lat[None, :]
    dlon = state_lon[:, None] - obs_lon[None, :]
    distance = 111.32 * np.sqrt(
        dlat**2 + (dlon * np.cos(np.deg2rad(state_lat[:, None]))) ** 2
    )
    z = distance / float(radius_km)
    rho = np.zeros_like(z)
    inner = z < 1.0
    middle = (z >= 1.0) & (z < 2.0)
    zi = z[inner]
    zm = z[middle]
    rho[inner] = -0.25 * zi**5 + 0.5 * zi**4 + 0.625 * zi**3 - (5.0 / 3.0) * zi**2 + 1.0
    rho[middle] = (
        (1.0 / 12.0) * zm**5
        - 0.5 * zm**4
        + 0.625 * zm**3
        + (5.0 / 3.0) * zm**2
        - 5.0 * zm
        + 4.0
        - 2.0 / (3.0 * zm)
    )
    return rho


def _analysis_mean(
    background: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
    residuals: np.ndarray,
    proxy_indices: np.ndarray,
    localization: np.ndarray,
) -> np.ndarray:
    members = background.shape[1]
    background_mean = background.mean(axis=1)
    background_prime = background - background_mean[:, None]
    hxb_mean = hxb.mean(axis=1)
    hxb_prime = hxb - hxb_mean[:, None]
    pbht = background_prime @ hxb_prime.T / (members - 1)
    obs_covariance = hxb_prime @ hxb_prime.T / (members - 1)
    error_variance = np.var(residuals, axis=1, ddof=1)
    output = np.repeat(background_mean[:, None], observations.shape[1], axis=1)

    groups: dict[tuple[int, ...], list[int]] = {}
    allowed = np.zeros(hxb.shape[0], dtype=bool)
    allowed[proxy_indices] = True
    for year_index in range(observations.shape[1]):
        active = tuple(np.flatnonzero(allowed & np.isfinite(observations[:, year_index])))
        groups.setdefault(active, []).append(year_index)

    for active_tuple, year_indices in groups.items():
        if not active_tuple:
            continue
        active = np.asarray(active_tuple, dtype=int)
        covariance = obs_covariance[np.ix_(active, active)] + np.diag(error_variance[active])
        innovation = observations[np.ix_(active, year_indices)] - hxb_mean[active, None]
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            weights = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            weights = scipy.linalg.solve(covariance, innovation, assume_a="sym")
        localized_pbht = pbht[:, active] * localization[:, active]
        output[:, year_indices] += localized_pbht @ weights
    return output


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        scale = np.std(row[finite], ddof=1)
        if scale > 0:
            output[row_index, finite] = (row[finite] - np.mean(row[finite])) / scale
        else:
            output[row_index, finite] = 0.0
    return output


def _proxy_metrics(
    reconstruction: np.ndarray,
    precipitation: np.ndarray,
    holdout_indices: np.ndarray,
    observations: np.ndarray,
    state_index: np.ndarray,
    coefficients: np.ndarray,
    bivariate: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    predicted = np.full((holdout_indices.size, reconstruction.shape[1]), np.nan)
    actual = observations[holdout_indices]
    for row, proxy_index in enumerate(holdout_indices):
        grid_index = int(state_index[proxy_index])
        beta = coefficients[proxy_index]
        if not np.isfinite(beta[:2]).all():
            continue
        predicted[row] = beta[0] + beta[1] * reconstruction[grid_index]
        if bivariate[proxy_index] and np.isfinite(beta[2]):
            predicted[row] += beta[2] * precipitation[grid_index]

    predicted_std = _standardize_rows(predicted)
    actual_std = _standardize_rows(actual)
    rows = []
    for row, proxy_index in enumerate(holdout_indices):
        valid = np.isfinite(predicted_std[row]) & np.isfinite(actual_std[row])
        if valid.sum() < 5:
            continue
        pred = predicted_std[row, valid]
        obs = actual_std[row, valid]
        residual = pred - obs
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        rows.append(
            {
                "proxy_index_1based": int(proxy_index + 1),
                "valid_year_count": int(valid.sum()),
                "correlation": float(np.corrcoef(pred, obs)[0, 1]),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "ce": float(1.0 - np.sum(residual**2) / denominator),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "proxy_count": int(len(frame)),
        "mean_correlation": float(frame["correlation"].mean()),
        "mean_rmse": float(frame["rmse"].mean()),
        "mean_ce": float(frame["ce"].mean()),
        "median_ce": float(frame["ce"].median()),
    }
    return summary, rows


def _trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    centered = years[finite].astype(float) - float(np.mean(years[finite]))
    return float(100.0 * np.polyfit(centered, values[finite], 1)[0])


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _instrument_metrics(
    reconstruction: np.ndarray,
    instrument: np.ndarray,
    product_names: np.ndarray,
    years: np.ndarray,
    reference_years: tuple[int, int],
    analysis_years: tuple[int, int],
    rolling_window: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    reference = (years >= reference_years[0]) & (years <= reference_years[1])
    analysis = (years >= analysis_years[0]) & (years <= analysis_years[1])
    analysis_year_values = years[analysis]
    rec_anomaly = reconstruction - np.nanmean(reconstruction[:, reference], axis=1)[:, None]
    rows = []
    for product_index, product_name in enumerate(product_names):
        obs = instrument[product_index]
        obs_anomaly = obs - np.nanmean(obs[:, reference], axis=1)[:, None]
        complete = (
            np.isfinite(obs_anomaly[:, analysis]).all(axis=1)
            & np.isfinite(rec_anomaly[:, analysis]).all(axis=1)
        )
        rec = rec_anomaly[complete][:, analysis]
        target = obs_anomaly[complete][:, analysis]
        if rec.shape[0] < 3:
            raise ValueError(f"insufficient-instrument-cells:{product_name}:{rec.shape[0]}")
        difference = rec - target
        temporal_correlations = [
            np.corrcoef(rec[index], target[index])[0, 1] for index in range(rec.shape[0])
        ]
        spatial_correlations = [
            np.corrcoef(rec[:, index], target[:, index])[0, 1]
            for index in range(rec.shape[1])
        ]
        rec_mean = np.mean(rec, axis=0)
        target_mean = np.mean(target, axis=0)
        rec_rolling = _rolling_mean(rec_mean, rolling_window)
        target_rolling = _rolling_mean(target_mean, rolling_window)
        rolling_valid = np.isfinite(rec_rolling) & np.isfinite(target_rolling)
        early = (analysis_year_values >= 1901) & (analysis_year_values <= 1930)
        late = (analysis_year_values >= 1971) & (analysis_year_values <= 2000)
        rec_change = float(np.mean(rec_mean[late]) - np.mean(rec_mean[early]))
        target_change = float(np.mean(target_mean[late]) - np.mean(target_mean[early]))
        rec_trend = _trend_per_century(analysis_year_values, rec_mean)
        target_trend = _trend_per_century(analysis_year_values, target_mean)
        rows.append(
            {
                "product": str(product_name),
                "complete_cell_count": int(rec.shape[0]),
                "spatiotemporal_rmse_c": float(np.sqrt(np.mean(difference**2))),
                "mean_cell_temporal_correlation": float(np.nanmean(temporal_correlations)),
                "mean_year_spatial_correlation": float(np.nanmean(spatial_correlations)),
                "regional_rolling30_correlation": float(
                    np.corrcoef(rec_rolling[rolling_valid], target_rolling[rolling_valid])[0, 1]
                ),
                "regional_rolling30_rmse_c": float(
                    np.sqrt(np.mean((rec_rolling[rolling_valid] - target_rolling[rolling_valid]) ** 2))
                ),
                "reconstruction_trend_c_per_century": rec_trend,
                "observation_trend_c_per_century": target_trend,
                "trend_absolute_error_c_per_century": abs(rec_trend - target_trend),
                "reconstruction_early_late_change_c": rec_change,
                "observation_early_late_change_c": target_change,
                "early_late_absolute_error_c": abs(rec_change - target_change),
            }
        )
    frame = pd.DataFrame(rows)
    aggregate = {
        "product_count": int(len(frame)),
        "median_rolling30_correlation": float(frame["regional_rolling30_correlation"].median()),
        "median_rolling30_rmse_c": float(frame["regional_rolling30_rmse_c"].median()),
        "median_trend_absolute_error_c_per_century": float(
            frame["trend_absolute_error_c_per_century"].median()
        ),
        "median_early_late_absolute_error_c": float(
            frame["early_late_absolute_error_c"].median()
        ),
    }
    return rows, aggregate


def _interpolate_target(
    field: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    cube = np.empty((len(grid_lat), len(grid_lon), field.shape[1]), dtype=float)
    for year_index in range(field.shape[1]):
        cube[:, :, year_index] = field[:, year_index].reshape(
            (len(grid_lat), len(grid_lon)), order="F"
        )
    interpolator = RegularGridInterpolator((grid_lat, grid_lon), cube)
    return np.asarray(interpolator([[target_lat, target_lon]])[0], dtype=float)


def _plot_curves(
    output: Path,
    years: np.ndarray,
    regional_curves: dict[str, np.ndarray],
    target_curves: dict[str, np.ndarray],
    target_observations: np.ndarray,
    product_names: np.ndarray,
    rolling_window: int,
) -> None:
    analysis = (years >= 1901) & (years <= 2000)
    x = years[analysis]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, values in regional_curves.items():
        axes[0].plot(x, _rolling_mean(values[analysis], rolling_window), label=name, linewidth=2)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Regional anomaly (degC)\n30-year mean")
    axes[0].set_title("HMA state-grid instrumental comparison")
    axes[0].legend(ncol=2, fontsize=8)

    observation_mean = np.nanmean(target_observations[:, analysis], axis=0)
    observation_min = np.nanmin(target_observations[:, analysis], axis=0)
    observation_max = np.nanmax(target_observations[:, analysis], axis=0)
    axes[1].fill_between(
        x,
        _rolling_mean(observation_min, rolling_window),
        _rolling_mean(observation_max, rolling_window),
        color="0.8",
        label="five-product range",
    )
    axes[1].plot(
        x,
        _rolling_mean(observation_mean, rolling_window),
        color="black",
        linewidth=2.2,
        label="five-product mean",
    )
    for name, values in target_curves.items():
        axes[1].plot(x, _rolling_mean(values[analysis], rolling_window), label=name, linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("3584 anomaly (degC)\n30-year mean")
    axes[1].set_xlabel("Year / rolling-window end year")
    axes[1].set_title("3584 pixel diagnostic")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "localization_low_frequency_curves.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_localization_pilot_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = np.load(output / "inputs/localization_pilot_inputs.npz")
    instrumental = np.load(output / "inputs/instrumental_targets.npz")

    years = inputs["years"]
    all_proxy_indices = np.arange(inputs["hxb"].shape[0], dtype=int)
    full_localization = np.ones((inputs["background"].shape[0], all_proxy_indices.size))
    full_baseline = _analysis_mean(
        inputs["background"],
        inputs["hxb"],
        inputs["observations"],
        inputs["residuals"],
        all_proxy_indices,
        full_localization,
    )
    replay_difference = np.abs(full_baseline - inputs["target_mean"])
    preflight = {
        "full_proxy_baseline_max_abs_difference_c": float(np.max(replay_difference)),
        "full_proxy_baseline_rmse_c": float(np.sqrt(np.mean(replay_difference**2))),
        "declared_replay_bound_c": 0.024,
        "passed": bool(np.max(replay_difference) <= 0.024),
    }
    (output / "preflight_summary.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if not preflight["passed"]:
        raise RuntimeError(f"baseline-replay-preflight-failed:{preflight}")

    branch_fields: dict[str, np.ndarray] = {}
    branch_rows = []
    proxy_rows = []
    instrument_rows = []
    regional_curves: dict[str, np.ndarray] = {}
    target_curves: dict[str, np.ndarray] = {}
    reference_years = tuple(config["scope"]["reference_years"])
    analysis_years = tuple(config["scope"]["instrumental_years"])
    rolling_window = int(config["scope"]["rolling_window_years"])

    instrument_values = instrumental["observations"]
    reference = (years >= reference_years[0]) & (years <= reference_years[1])
    for branch in config["branches"]:
        name = branch["name"]
        radius = branch["radius_km"]
        localization = _gaspari_cohn(
            inputs["state_lat"],
            inputs["state_lon"],
            inputs["proxy_locations"][:, 0],
            inputs["proxy_locations"][:, 1],
            radius,
        )
        reconstruction = _analysis_mean(
            inputs["background"],
            inputs["hxb"],
            inputs["observations"],
            inputs["residuals"],
            inputs["train_indices"],
            localization,
        )
        branch_fields[name] = reconstruction
        proxy_summary, per_proxy = _proxy_metrics(
            reconstruction,
            inputs["precipitation"],
            inputs["holdout_indices"],
            inputs["observations"],
            inputs["proxy_state_index"],
            inputs["proxy_coefficients"],
            inputs["proxy_bivariate"],
        )
        per_product, instrument_summary = _instrument_metrics(
            reconstruction,
            instrument_values,
            instrumental["product_names"],
            years,
            reference_years,
            analysis_years,
            rolling_window,
        )
        for row in per_proxy:
            proxy_rows.append({"branch": name, **row})
        for row in per_product:
            instrument_rows.append({"branch": name, **row})
        branch_rows.append(
            {
                "branch": name,
                "radius_km": np.nan if radius is None else float(radius),
                **proxy_summary,
                **instrument_summary,
            }
        )

        rec_anomaly = reconstruction - np.nanmean(reconstruction[:, reference], axis=1)[:, None]
        regional_curves[name] = np.nanmean(rec_anomaly, axis=0)
        target = _interpolate_target(
            reconstruction,
            inputs["grid_lat"],
            inputs["grid_lon"],
            33.75,
            75.75,
        )
        target_curves[name] = target - np.nanmean(target[reference])

    branch_frame = pd.DataFrame(branch_rows)
    proxy_frame = pd.DataFrame(proxy_rows)
    instrument_frame = pd.DataFrame(instrument_rows)
    baseline = branch_frame[branch_frame["branch"].eq("no_localization")].iloc[0]
    gates = config["acceptance_gates"]
    decisions = []
    for _, row in branch_frame.iterrows():
        if row["branch"] == "no_localization":
            continue
        product_candidate = instrument_frame[instrument_frame["branch"].eq(row["branch"])]
        product_baseline = instrument_frame[
            instrument_frame["branch"].eq("no_localization")
        ].set_index("product")
        product_candidate = product_candidate.set_index("product")
        better_products = int(
            (
                product_candidate["regional_rolling30_rmse_c"]
                < product_baseline["regional_rolling30_rmse_c"]
            ).sum()
        )
        checks = {
            "proxy_ce": bool(
                row["mean_ce"] - baseline["mean_ce"]
                >= gates["proxy_mean_ce_min_change"]
            ),
            "proxy_rmse": bool(
                row["mean_rmse"] - baseline["mean_rmse"]
                <= gates["proxy_mean_rmse_max_change"]
            ),
            "rolling30_correlation": bool(
                row["median_rolling30_correlation"]
                - baseline["median_rolling30_correlation"]
                >= gates["instrument_median_rolling30_correlation_min_change"]
            ),
            "rolling30_rmse": bool(
                row["median_rolling30_rmse_c"]
                < baseline["median_rolling30_rmse_c"]
            ),
            "trend_error": bool(
                row["median_trend_absolute_error_c_per_century"]
                < baseline["median_trend_absolute_error_c_per_century"]
            ),
            "early_late_error": bool(
                row["median_early_late_absolute_error_c"]
                < baseline["median_early_late_absolute_error_c"]
            ),
            "product_count": bool(
                better_products >= gates["minimum_products_with_better_rolling30_rmse"]
            ),
        }
        decisions.append(
            {
                "branch": row["branch"],
                "better_rolling30_rmse_product_count": better_products,
                "checks": checks,
                "accepted": bool(all(checks.values())),
            }
        )

    accepted = [row["branch"] for row in decisions if row["accepted"]]
    decision = {
        "experiment_id": config["experiment_id"],
        "preflight": preflight,
        "branch_decisions": decisions,
        "accepted_branches": accepted,
        "interpretation": (
            "At least one finite localization radius passes every predeclared gate."
            if accepted
            else "No finite localization radius passes every predeclared gate; do not promote localization from this pilot."
        ),
    }

    branch_frame.to_csv(output / "branch_summary.csv", index=False)
    proxy_frame.to_csv(output / "heldout_proxy_metrics.csv", index=False)
    instrument_frame.to_csv(output / "instrument_product_metrics.csv", index=False)
    np.savez_compressed(output / "reconstruction_means.npz", years=years, **branch_fields)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    target_observations = instrumental["target_3584"].copy()
    target_observations -= np.nanmean(target_observations[:, reference], axis=1)[:, None]
    _plot_curves(
        output,
        years,
        regional_curves,
        target_curves,
        target_observations,
        instrumental["product_names"],
        rolling_window,
    )

    lines = [
        "# Localization pilot result",
        "",
        f"Baseline replay preflight passed: `{preflight['passed']}`; maximum difference "
        f"`{preflight['full_proxy_baseline_max_abs_difference_c']:.6f} degC`.",
        "",
        "No glacier output was used in selection. Original D- and E-drive sources were read only.",
        "",
        "## Decision",
        "",
        decision["interpretation"],
        "",
        "Accepted finite-radius branches: " + (", ".join(accepted) if accepted else "none"),
        "",
        "See `branch_summary.csv`, `heldout_proxy_metrics.csv`, "
        "`instrument_product_metrics.csv`, and `localization_low_frequency_curves.png`.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    generated = [
        "branch_summary.csv",
        "heldout_proxy_metrics.csv",
        "instrument_product_metrics.csv",
        "reconstruction_means.npz",
        "decision.json",
        "localization_low_frequency_curves.png",
        "RESULT.md",
        "preflight_summary.json",
    ]
    manifest = {
        "generated_files": [
            {
                "path": name,
                "sha256": _sha256(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in generated
        ]
    }
    (output / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
