from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_cook_summer_temperature_only_psm_v1"
EVENT_YEARS = np.arange(1905, 1921, dtype=int)
CALIBRATION_YEARS = np.arange(1941, 2001, dtype=int)
SEASONS = {"JJA": (6, 7, 8), "AMJJAS": (4, 5, 6, 7, 8, 9)}
PRODUCTS = ("Berkeley Earth", "CRU TS")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def fit_temperature(response: np.ndarray, temperature: np.ndarray) -> tuple[float, float] | None:
    valid = np.isfinite(response) & np.isfinite(temperature)
    if int(valid.sum()) < 20:
        return None
    design = np.column_stack([np.ones(valid.sum()), temperature[valid]])
    b0, b1 = np.linalg.lstsq(design, response[valid], rcond=None)[0]
    return float(b0), float(b1)


def metrics(response: np.ndarray, prediction: np.ndarray, scale: float) -> tuple[int, float, float, float]:
    valid = np.isfinite(response) & np.isfinite(prediction)
    if int(valid.sum()) < 3:
        return int(valid.sum()), np.nan, np.nan, np.nan
    residual = response[valid] - prediction[valid]
    return (
        int(valid.sum()),
        float(np.corrcoef(response[valid], prediction[valid])[0, 1]),
        float(np.sqrt(np.mean(residual ** 2)) / scale),
        float(np.mean(residual) / scale),
    )


def period_change(values: np.ndarray) -> np.ndarray:
    baseline = (EVENT_YEARS >= 1905) & (EVENT_YEARS <= 1914)
    event = (EVENT_YEARS >= 1915) & (EVENT_YEARS <= 1918)
    return np.nanmean(values[event], axis=0) - np.nanmean(values[baseline], axis=0)


def extract_proxy_series(proxy_path: Path, proxy_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event = np.full((len(EVENT_YEARS), len(proxy_indices)), np.nan)
    calibration = np.full((len(CALIBRATION_YEARS), len(proxy_indices)), np.nan)
    with h5py.File(proxy_path, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(proxy_indices):
            proxy = handle[refs[int(proxy_index) - 1]]
            years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {int(year): float(value) for year, value in zip(years, values)}
            event[:, column] = [lookup.get(int(year), np.nan) for year in EVENT_YEARS]
            calibration[:, column] = [lookup.get(int(year), np.nan) for year in CALIBRATION_YEARS]
    return event, calibration


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    drivers = Path(__file__).resolve().parent
    rolling = load_module(drivers / "run_clean_gpcc_psm_rolling_pseudo_event_v1.py", "cook_summer_monthly")
    base = load_module(drivers / "audit_positive_b2_event_seasonal_psm_v1.py", "cook_summer_sources")
    study_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_pages2k_fixed_network_study_batch_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    proxy_path = Path(base.archive_proxy_path())
    sources = {
        "study_metadata": study_root / "fixed_network_study_metadata.csv",
        "proxy_archive": proxy_path,
        "berkeley_monthly": Path(base.BERKELEY_TEMPERATURE),
        "cru_monthly": Path(base.CRU_TEMPERATURE),
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    metadata = pd.read_csv(sources["study_metadata"])
    cook = metadata[metadata["data_author"] == "Cook, E.R."].sort_values("proxy_index").reset_index(drop=True)
    if len(cook) != 15:
        raise RuntimeError(f"cook-count-contract-failed:{len(cook)}")
    proxy_indices = cook["proxy_index"].to_numpy(dtype=int)
    latitudes = cook["latitude"].to_numpy(dtype=float)
    longitudes = cook["longitude"].to_numpy(dtype=float)
    proxy_event, proxy_calibration = extract_proxy_series(proxy_path, proxy_indices)
    proxy_scale = np.nanstd(proxy_calibration, axis=0)
    actual_event_z = period_change(proxy_event) / proxy_scale
    if float(np.max(np.abs(actual_event_z - cook["event_change_z"].to_numpy(dtype=float)))) > 1e-10:
        raise RuntimeError("cook-target-event-closure-failed")

    cru = rolling.read_netcdf3_points(base.CRU_TEMPERATURE, ["tmp", "stn"], 1901, latitudes, longitudes)
    berkeley = rolling.read_berkeley_points(base.BERKELEY_TEMPERATURE, latitudes, longitudes)
    event_labels = rolling.monthly_labels(1905, 1920)
    calibration_labels = rolling.monthly_labels(1941, 2000)
    climate: dict[tuple[str, str, str], np.ndarray] = {}
    for season, months in SEASONS.items():
        climate[("Berkeley Earth", "event", season)] = rolling.aggregate_temperature(berkeley["event"], event_labels, EVENT_YEARS, months)
        climate[("Berkeley Earth", "calibration", season)] = rolling.aggregate_temperature(berkeley["calibration"], calibration_labels, CALIBRATION_YEARS, months)
        climate[("CRU TS", "event", season)] = rolling.aggregate_temperature(cru["event"]["tmp"], event_labels, EVENT_YEARS, months)
        climate[("CRU TS", "calibration", season)] = rolling.aggregate_temperature(cru["calibration"]["tmp"], calibration_labels, CALIBRATION_YEARS, months)

    rows = []
    early = CALIBRATION_YEARS <= 1970
    late = CALIBRATION_YEARS >= 1971
    for product in PRODUCTS:
        for season in SEASONS:
            calibration_temperature = climate[(product, "calibration", season)]
            event_temperature = climate[(product, "event", season)]
            temperature_change = period_change(event_temperature)
            for column, proxy_index in enumerate(proxy_indices):
                response = proxy_calibration[:, column]
                full_fit = fit_temperature(response, calibration_temperature[:, column])
                early_fit = fit_temperature(response[early], calibration_temperature[early, column])
                late_fit = fit_temperature(response[late], calibration_temperature[late, column])
                if full_fit is None or early_fit is None or late_fit is None:
                    raise RuntimeError(f"temperature-only-fit-failed:{product}:{season}:{proxy_index}")
                full_prediction = full_fit[0] + full_fit[1] * calibration_temperature[:, column]
                early_prediction = early_fit[0] + early_fit[1] * calibration_temperature[:, column]
                late_prediction = late_fit[0] + late_fit[1] * calibration_temperature[:, column]
                full_n, full_r, full_rmse, full_bias = metrics(response, full_prediction, proxy_scale[column])
                early_test_n, early_test_r, early_test_rmse, early_test_bias = metrics(response[late], early_prediction[late], proxy_scale[column])
                late_test_n, late_test_r, late_test_rmse, late_test_bias = metrics(response[early], late_prediction[early], proxy_scale[column])
                predicted_event_z = full_fit[1] * temperature_change[column] / proxy_scale[column]
                rows.append({
                    "proxy_index": int(proxy_index),
                    "product": product,
                    "season": season,
                    "latitude": float(latitudes[column]),
                    "longitude": float(longitudes[column]),
                    "proxy_scale_1941_2000": float(proxy_scale[column]),
                    "b1_full": full_fit[1],
                    "b1_early_1941_1970": early_fit[1],
                    "b1_late_1971_2000": late_fit[1],
                    "b1_early_late_sign_agreement": bool(np.sign(early_fit[1]) == np.sign(late_fit[1])),
                    "b1_positive_full": bool(full_fit[1] > 0),
                    "b1_positive_both_halves": bool(early_fit[1] > 0 and late_fit[1] > 0),
                    "full_fit_n": full_n,
                    "full_fit_r": full_r,
                    "full_fit_nrmse": full_rmse,
                    "full_fit_bias_z": full_bias,
                    "early_fit_late_test_n": early_test_n,
                    "early_fit_late_test_r": early_test_r,
                    "early_fit_late_test_nrmse": early_test_rmse,
                    "early_fit_late_test_bias_z": early_test_bias,
                    "late_fit_early_test_n": late_test_n,
                    "late_fit_early_test_r": late_test_r,
                    "late_fit_early_test_nrmse": late_test_rmse,
                    "late_fit_early_test_bias_z": late_test_bias,
                    "temperature_event_change_c": float(temperature_change[column]),
                    "predicted_event_change_z": float(predicted_event_z),
                    "actual_proxy_event_change_z": float(actual_event_z[column]),
                    "event_residual_actual_minus_predicted_z": float(actual_event_z[column] - predicted_event_z),
                })
    fits = pd.DataFrame(rows)
    group_summary = fits.groupby(["product", "season"], as_index=False).agg(
        record_count=("proxy_index", "size"),
        positive_full_coefficient_fraction=("b1_positive_full", "mean"),
        positive_both_halves_fraction=("b1_positive_both_halves", "mean"),
        early_late_sign_agreement_fraction=("b1_early_late_sign_agreement", "mean"),
        median_full_fit_r=("full_fit_r", "median"),
        median_early_fit_late_test_r=("early_fit_late_test_r", "median"),
        median_late_fit_early_test_r=("late_fit_early_test_r", "median"),
        mean_temperature_event_change_c=("temperature_event_change_c", "mean"),
        mean_predicted_event_change_z=("predicted_event_change_z", "mean"),
        mean_actual_proxy_event_change_z=("actual_proxy_event_change_z", "mean"),
        mean_event_residual_z=("event_residual_actual_minus_predicted_z", "mean"),
    )
    group_summary["signed_amplitude_transfer"] = group_summary["mean_predicted_event_change_z"] / group_summary["mean_actual_proxy_event_change_z"]

    output.mkdir(parents=True)
    fits.to_csv(output / "cook_temperature_only_psm_proxy_level.csv", index=False)
    group_summary.to_csv(output / "cook_temperature_only_psm_group_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    colors = {"JJA": "#247BA0", "AMJJAS": "#E07A1F"}
    for season, subset in fits[fits["product"] == "Berkeley Earth"].groupby("season"):
        axes[0].scatter(subset["b1_early_1941_1970"], subset["b1_late_1971_2000"], s=55, color=colors[season], label=season, alpha=0.82)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Early temperature coefficient")
    axes[0].set_ylabel("Late temperature coefficient")
    axes[0].set_title("Berkeley temperature-only stability")
    axes[0].legend(frameon=False)

    for season, subset in fits[fits["product"] == "CRU TS"].groupby("season"):
        axes[1].scatter(subset["b1_early_1941_1970"], subset["b1_late_1971_2000"], s=55, color=colors[season], label=season, alpha=0.82)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Early temperature coefficient")
    axes[1].set_ylabel("Late temperature coefficient")
    axes[1].set_title("CRU temperature-only stability")

    x = np.arange(len(group_summary))
    axes[2].bar(x - 0.18, group_summary["mean_actual_proxy_event_change_z"], 0.36, color="#1B1B1B", label="proxy actual")
    axes[2].bar(x + 0.18, group_summary["mean_predicted_event_change_z"], 0.36, color="#009E73", label="temperature-only prediction")
    labels = [f"{row.product.replace('Berkeley Earth', 'BEA')}\n{row.season}" for row in group_summary.itertuples(index=False)]
    axes[2].set_xticks(x, labels)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Cook mean event change (z)")
    axes[2].set_title("1915-1918 minus 1905-1914")
    axes[2].legend(frameon=False)
    fig.savefig(output / "cook_summer_temperature_only_psm.png", dpi=220)
    plt.close(fig)

    no_branch_reproduces = bool((group_summary["mean_predicted_event_change_z"] > -0.5).all())
    summary = {
        "cook_record_count": int(len(cook)),
        "event_definition": "mean(1915-1918) minus mean(1905-1914), proxy scaled by 1941-2000 standard deviation",
        "calibration_definition": "unconstrained temperature-only OLS; full 1941-2000 and 1941-1970/1971-2000 split fits",
        "metadata_context": "PAGES TS identifies these records as positive temperature-response TRW, mostly Summer; JJA and AMJJAS are tested as explicit brackets",
        "group_summary": group_summary.to_dict(orient="records"),
        "all_temperature_only_branches_fail_to_reproduce_negative_one_z_scale_event": no_branch_reproduces,
        "classification": (
            "metadata_season_temperature_only_psms_do_not_reproduce_the_cook_negative_event"
            if no_branch_reproduces
            else "at_least_one_temperature_only_branch_reproduces_substantial_cook_event_amplitude"
        ),
        "interpretation": (
            "Failure of unconstrained summer temperature-only fits means replacing the bivariate PSM with a simple metadata-season temperature operator is not validated. "
            "It leaves chronology nonstationarity and early instrumental representativeness as competing explanations."
        ),
        "read_periods": [[1905, 1920], [1941, 2000]],
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-cook-summer-audit")
    (output / "FINAL_RESULT.md").write_text(
        "# Cook Summer Temperature-only PSM Audit v1\n\n"
        + "\n".join(
            f"- {row.product} {row.season}: positive full coefficient fraction {row.positive_full_coefficient_fraction:.3f}, "
            f"predicted event {row.mean_predicted_event_change_z:+.3f} z versus actual {row.mean_actual_proxy_event_change_z:+.3f} z."
            for row in group_summary.itertuples(index=False)
        )
        + f"\n\nClassification: `{summary['classification']}`. No production operator is changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
