from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = ROOT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_psm_modern_stability_v1"
)
MANIFEST = ROOT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_improvement_loop_v2"
    r"\round_27_temperature_innovation_gain_sign_decomposition_v1"
    r"\source_manifest.json"
)
CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
YEARS = np.arange(1940, 2001)
SPLITS = {
    "early_fit_late_test": ((1940, 1969), (1970, 2000)),
    "late_fit_early_test": ((1970, 2000), (1940, 1969)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def fit_psm(
    actual: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray
) -> tuple[np.ndarray, int]:
    valid = np.isfinite(actual) & np.isfinite(temperature) & np.isfinite(precipitation)
    count = int(np.count_nonzero(valid))
    if count < 20:
        return np.full(3, np.nan), count
    design = np.column_stack(
        [np.ones(count), temperature[valid], precipitation[valid]]
    )
    if np.linalg.matrix_rank(design) != 3:
        return np.full(3, np.nan), count
    coefficients, _, _, _ = np.linalg.lstsq(design, actual[valid], rcond=None)
    return coefficients.astype(float), count


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    count = int(np.count_nonzero(valid))
    if count < 20:
        return {"n": count, "r": np.nan, "nrmse": np.nan}
    observed = actual[valid]
    scale = float(np.std(observed, ddof=0))
    correlation = float(np.corrcoef(observed, predicted[valid])[0, 1])
    nrmse = float(np.sqrt(np.mean((predicted[valid] - observed) ** 2)) / scale)
    return {"n": count, "r": correlation, "nrmse": nrmse}


def chronology(proxy: h5py.Group) -> np.ndarray:
    proxy_years = np.rint(
        np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
    ).astype(int)
    proxy_values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
    lookup = {
        int(year): float(value)
        for year, value in zip(proxy_years, proxy_values)
        if np.isfinite(value)
    }
    return np.asarray([lookup.get(int(year), np.nan) for year in YEARS], dtype=float)


def same_nonzero_sign(*values: float) -> bool:
    signs = np.sign(np.asarray(values, dtype=float))
    return bool(np.all(np.isfinite(signs)) and np.all(signs != 0) and np.all(signs == signs[0]))


def audit_network(
    network: int,
    proxy_path: Path,
    climate_years: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with h5py.File(proxy_path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(references):
            proxy = archive[reference]
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            if len(coefficients) != 3 or decode(proxy["psm_mthd"]) != "bivariate":
                continue
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            latitude = float(location[0])
            longitude = float(location[1])
            lat_index = int(np.argmin(np.abs(latitudes - latitude)))
            lon_index = int(np.argmin(np.abs(longitudes - longitude)))
            local_temperature = temperature[:, lat_index, lon_index]
            local_precipitation = precipitation[:, lat_index, lon_index]
            actual = chronology(proxy)
            row: dict[str, object] = {
                "network": network,
                "record_index_1based": index + 1,
                "latitude": latitude,
                "longitude": longitude,
                "grid_latitude": float(latitudes[lat_index]),
                "grid_longitude": float(longitudes[lon_index]),
                "production_b1_temperature": float(coefficients[1]),
                "production_b2_precipitation": float(coefficients[2]),
            }
            split_coefficients = {}
            for split_name, (fit_period, test_period) in SPLITS.items():
                fit_mask = (climate_years >= fit_period[0]) & (climate_years <= fit_period[1])
                test_mask = (climate_years >= test_period[0]) & (climate_years <= test_period[1])
                fitted, fit_n = fit_psm(
                    actual[fit_mask],
                    local_temperature[fit_mask],
                    local_precipitation[fit_mask],
                )
                predicted = (
                    fitted[0]
                    + fitted[1] * local_temperature[test_mask]
                    + fitted[2] * local_precipitation[test_mask]
                )
                score = metrics(actual[test_mask], predicted)
                split_coefficients[split_name] = fitted
                row.update(
                    {
                        f"{split_name}_fit_n": fit_n,
                        f"{split_name}_b1_temperature": float(fitted[1]),
                        f"{split_name}_b2_precipitation": float(fitted[2]),
                        f"{split_name}_test_n": score["n"],
                        f"{split_name}_test_r": score["r"],
                        f"{split_name}_test_nrmse": score["nrmse"],
                    }
                )
            early = split_coefficients["early_fit_late_test"]
            late = split_coefficients["late_fit_early_test"]
            sign_stable = same_nonzero_sign(coefficients[1], early[1], late[1])
            positive_oof = bool(
                row["early_fit_late_test_test_r"] > 0
                and row["late_fit_early_test_test_r"] > 0
            )
            stable = bool(sign_stable and positive_oof)
            row["b1_sign_stable_with_production"] = sign_stable
            row["both_oof_correlations_positive"] = positive_oof
            row["stable_temperature_operator"] = stable
            row["temperature_operator_group"] = "stable" if stable else "unstable"
            rows.append(row)
    table = pd.DataFrame(rows)
    expected = 201 if network == 6 else 290
    if len(table) != expected:
        raise RuntimeError(f"Unexpected proxy{network} bivariate count: {len(table)}")
    return table


def main() -> None:
    output_files = [
        OUTPUT / "temperature_psm_modern_stability_proxy_level.csv",
        OUTPUT / "temperature_psm_modern_stability_network_summary.csv",
        OUTPUT / "temperature_psm_modern_stability.png",
        OUTPUT / "summary.json",
        OUTPUT / "source_manifest.json",
        OUTPUT / "FINAL_RESULT.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    recovered = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "climate": CLIMATE,
        "round27_manifest": MANIFEST,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    before = {name: sha256(path) for name, path in paths.items()}

    with h5py.File(CLIMATE, "r") as archive:
        all_years = np.asarray(archive["time"], dtype=int)
        indices = np.flatnonzero((all_years >= YEARS[0]) & (all_years <= YEARS[-1]))
        climate_years = all_years[indices]
        if not np.array_equal(climate_years, YEARS):
            raise RuntimeError("Modern climate year axis mismatch")
        latitudes = np.asarray(archive["lat"], dtype=float)
        longitudes = np.asarray(archive["lon"], dtype=float)
        temperature = np.asarray(archive["tas_obs_bea_05deg"][indices], dtype=float)
        precipitation = np.asarray(archive["pr_obs_gpcp_05deg"][indices], dtype=float)
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    tables = [
        audit_network(
            network,
            paths[f"proxy{network}"],
            climate_years,
            latitudes,
            longitudes,
            temperature,
            precipitation,
        )
        for network in (6, 9)
    ]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output_files[0], index=False)

    summaries = []
    for network, subset in table.groupby("network"):
        stable = subset["stable_temperature_operator"]
        summaries.append(
            {
                "network": int(network),
                "assessed_bivariate_count": int(len(subset)),
                "stable_count": int(stable.sum()),
                "unstable_count": int((~stable).sum()),
                "stable_fraction": float(stable.mean()),
                "b1_sign_stable_fraction": float(
                    subset["b1_sign_stable_with_production"].mean()
                ),
                "both_oof_correlations_positive_fraction": float(
                    subset["both_oof_correlations_positive"].mean()
                ),
                "median_early_to_late_oof_r": float(
                    subset["early_fit_late_test_test_r"].median()
                ),
                "median_late_to_early_oof_r": float(
                    subset["late_fit_early_test_test_r"].median()
                ),
            }
        )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output_files[1], index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    colors = {True: "#187A68", False: "#C8503E"}
    for stable, subset in table.groupby("stable_temperature_operator"):
        axes[0].scatter(
            subset["early_fit_late_test_b1_temperature"],
            subset["late_fit_early_test_b1_temperature"],
            s=24,
            alpha=0.72,
            color=colors[bool(stable)],
            label="Stable" if stable else "Unstable",
        )
        axes[1].scatter(
            subset["early_fit_late_test_test_r"],
            subset["late_fit_early_test_test_r"],
            s=24,
            alpha=0.72,
            color=colors[bool(stable)],
            label="Stable" if stable else "Unstable",
        )
    coefficient_limit = float(
        np.nanquantile(
            np.abs(
                table[
                    [
                        "early_fit_late_test_b1_temperature",
                        "late_fit_early_test_b1_temperature",
                    ]
                ].to_numpy(float)
            ),
            0.99,
        )
    )
    axes[0].plot(
        [-coefficient_limit, coefficient_limit],
        [-coefficient_limit, coefficient_limit],
        color="#555555",
        linewidth=1.0,
    )
    axes[0].axhline(0, color="#888888", linewidth=0.8)
    axes[0].axvline(0, color="#888888", linewidth=0.8)
    axes[0].set_xlim(-coefficient_limit, coefficient_limit)
    axes[0].set_ylim(-coefficient_limit, coefficient_limit)
    axes[0].set(
        xlabel="b1 fitted on 1940-1969",
        ylabel="b1 fitted on 1970-2000",
        title="Temperature-coefficient stability",
    )
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].axvline(0, color="#888888", linewidth=0.8)
    axes[1].set(
        xlabel="Early fit to late test correlation",
        ylabel="Late fit to early test correlation",
        title="Cross-period predictive skill",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.savefig(output_files[2], dpi=240)
    plt.close(figure)

    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("An input changed during the audit")
    summary = {
        "status": "complete",
        "read_period": [1940, 2000],
        "splits": {
            name: [list(periods[0]), list(periods[1])]
            for name, periods in SPLITS.items()
        },
        "classification": (
            "same_nonzero_b1_sign_across_production_and_both_fits"
            "_and_both_cross_period_correlations_positive"
        ),
        "network_summary": summaries,
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
        "production_modified": False,
    }
    output_files[3].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[4].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    lines = [
        "# Final result",
        "",
        "Modern-only split validation froze the following operator groups:",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- proxy{row['network']}: {row['stable_count']}/{row['assessed_bivariate_count']} "
            f"stable ({100 * row['stable_fraction']:.1f}%); median cross-period r "
            f"{row['median_early_to_late_oof_r']:.3f}/{row['median_late_to_early_oof_r']:.3f}."
        )
    lines.extend(
        [
            "",
            "The classification used only 1940-2000 calibration data and is now frozen for the independent 1861-1900 operator leave-group-out replay.",
            "No production or source file was modified.",
            "",
        ]
    )
    output_files[5].write_text("\n".join(lines), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
