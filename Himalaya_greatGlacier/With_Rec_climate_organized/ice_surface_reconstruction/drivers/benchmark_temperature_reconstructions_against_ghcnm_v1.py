from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_temperature_four_background_instrumental_metrics_v1 as metrics


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
SOURCE_EXPERIMENT = LOOP / "temperature_four_background_scheme_evaluation_v2"
STATION_VALUES = SOURCE_EXPERIMENT / "modern_station_year_scheme_values.csv"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
OUT = LOOP / "temperature_reconstruction_external_instrumental_benchmark_v1"

MEMBERS = {
    "HGAO": "tas_rec_ref_HGAO_c05d",
    "LMR": "tas_rec_ref_LMR_c05d",
    "Nk_AM": "tas_rec_ref_Nk_AM_c05d",
    "Nk_CCA": "tas_rec_ref_Nk_CCA_c05d",
    "Nk_CPS": "tas_rec_ref_Nk_CPS_c05d",
    "Nk_DA": "tas_rec_ref_Nk_DA_c05d",
    "Nk_GraphEM": "tas_rec_ref_Nk_GraphEM_c05d",
    "Nk_PCR": "tas_rec_ref_Nk_PCR_c05d",
    "O": "tas_rec_ref_O_c05d",
    "PHYDA": "tas_rec_ref_PHYDA_c05d",
}
PRODUCTS = {
    "project_current": "current_CESM_degC",
    "local_EnKF_family": "reference_local_enkf_degC",
    "LMR": "reference_lmr_degC",
    "Neukom_family": "reference_neukom_degC",
    "PHYDA": "reference_phyda_degC",
    "ref_rec_family_equal": "reference_family_equal_degC",
}
OBSERVATIONS = {"qcf": "qcf_station_degC", "qcu": "qcu_station_degC"}
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260825


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result < -9000) | ~np.isfinite(result)] = np.nan
    return result


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def extract_reference_families(stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        selected_indices = np.flatnonzero((years >= 1951) & (years <= 2000))
        selected_years = years[selected_indices]
        if not np.array_equal(selected_years, np.arange(1951, 2001)):
            raise RuntimeError("Reference reconstruction 1951-2000 year contract failed")
        baseline = (selected_years >= 1961) & (selected_years <= 1990)
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
        station_rows = []
        location_rows = []
        for station in stations.itertuples(index=False):
            row = int(np.argmin(np.abs(lat - float(station.latitude))))
            col = int(np.argmin(np.abs(lon - float(station.longitude))))
            distance = float(np.hypot(lat[row] - float(station.latitude), lon[col] - float(station.longitude)))
            if distance > 2.0:
                raise RuntimeError(f"Reference grid farther than 2 degrees for {station.station_id}")
            member_values = {}
            for member, variable in MEMBERS.items():
                values = finite(handle[variable][selected_indices, row, col])
                if int(np.isfinite(values[baseline]).sum()) < 20:
                    raise RuntimeError(f"Insufficient 1961-1990 support for {member} at {station.station_id}")
                member_values[member] = values - np.nanmean(values[baseline])
            local_enkf = finite_mean([member_values["HGAO"], member_values["O"]])
            neukom = finite_mean([values for name, values in member_values.items() if name.startswith("Nk_")])
            families = {
                "reference_local_enkf_degC": local_enkf,
                "reference_lmr_degC": member_values["LMR"],
                "reference_neukom_degC": neukom,
                "reference_phyda_degC": member_values["PHYDA"],
            }
            families["reference_family_equal_degC"] = finite_mean(list(families.values()))
            location_rows.append({
                "station_id": station.station_id,
                "station_name": station.station_name,
                "station_latitude": float(station.latitude),
                "station_longitude": float(station.longitude),
                "source_latitude": float(lat[row]),
                "source_longitude": float(lon[col]),
                "source_distance_deg": distance,
            })
            for index, year in enumerate(selected_years):
                station_rows.append({
                    "station_id": station.station_id,
                    "year": int(year),
                    **{name: float(values[index]) for name, values in families.items()},
                })
    return pd.DataFrame(station_rows), pd.DataFrame(location_rows)


def bootstrap_product_differences(frame: pd.DataFrame, observation: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + (0 if observation.startswith("qcf") else 1))
    station_ids = np.array(sorted(frame["station_id"].unique()))
    annual: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    decadal: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for product, column in PRODUCTS.items():
        annual[product] = {}
        decadal[product] = {}
        for station_id in station_ids:
            group = frame[frame["station_id"].eq(station_id)]
            valid = group[[observation, column]].notna().all(axis=1)
            annual[product][station_id] = (
                group.loc[valid, observation].to_numpy(float),
                group.loc[valid, column].to_numpy(float),
            )
            blocks = metrics.decadal_pairs(group, observation, column)
            decadal[product][station_id] = (
                blocks["observed_degC"].to_numpy(float),
                blocks["predicted_degC"].to_numpy(float),
            )
    metric_names = ("annual_rmse", "annual_r", "decadal_rmse", "decadal_r")
    distributions = {
        product: {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in metric_names}
        for product in PRODUCTS
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(station_ids, size=len(station_ids), replace=True)
        calculated = {}
        for product in PRODUCTS:
            annual_obs = np.concatenate([annual[product][station][0] for station in sampled])
            annual_pred = np.concatenate([annual[product][station][1] for station in sampled])
            decade_obs = np.concatenate([decadal[product][station][0] for station in sampled])
            decade_pred = np.concatenate([decadal[product][station][1] for station in sampled])
            calculated[product] = (
                metrics.paired_metrics(annual_obs, annual_pred),
                metrics.paired_metrics(decade_obs, decade_pred),
            )
        control_annual, control_decadal = calculated["project_current"]
        for product, (annual_result, decadal_result) in calculated.items():
            distributions[product]["annual_rmse"][replicate] = (
                annual_result["rmse_degC"] - control_annual["rmse_degC"]
            )
            distributions[product]["annual_r"][replicate] = (
                annual_result["correlation_r"] - control_annual["correlation_r"]
            )
            distributions[product]["decadal_rmse"][replicate] = (
                decadal_result["rmse_degC"] - control_decadal["rmse_degC"]
            )
            distributions[product]["decadal_r"][replicate] = (
                decadal_result["correlation_r"] - control_decadal["correlation_r"]
            )
    rows = []
    for product, metric_values in distributions.items():
        for metric, values in metric_values.items():
            finite_values = values[np.isfinite(values)]
            lower, median, upper = np.quantile(finite_values, [0.025, 0.5, 0.975])
            lower_is_better = metric.endswith("rmse")
            rows.append({
                "observation": observation.removesuffix("_station_degC"),
                "product": product,
                "metric_delta_vs_project": metric,
                "replicate_count": int(len(finite_values)),
                "delta_median": float(median),
                "delta_ci95_lower": float(lower),
                "delta_ci95_upper": float(upper),
                "probability_better_than_project": (
                    float(np.mean(finite_values < 0)) if lower_is_better
                    else float(np.mean(finite_values > 0))
                ),
            })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, station: pd.DataFrame) -> None:
    qcf = summary[summary["observation"].eq("qcf")].set_index("branch").loc[list(PRODUCTS)]
    labels = ["Project", "Local-EnKF", "LMR", "Neukom", "PHYDA", "ref_rec"]
    colors = ["#1f4e79", "#c55a11", "#548235", "#8064a2", "#bf9000", "#666666"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(x, qcf["annual_pooled_rmse_degC"], color=colors)
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel("RMSE (degC)")
    axes[0, 0].set_title("Annual GHCN-M qcf error")
    axes[0, 1].bar(x - 0.18, qcf["annual_pooled_correlation_r"], 0.36, color=colors, alpha=0.65)
    axes[0, 1].bar(x + 0.18, qcf["decadal_correlation_r"], 0.36, color=colors)
    axes[0, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Pearson r")
    axes[0, 1].set_title("Annual (light) and decadal (solid) correlation")
    axes[0, 1].axhline(0, color="#222222", linewidth=0.8)
    axes[1, 0].bar(x, qcf["decadal_rmse_degC"], color=colors)
    axes[1, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("RMSE (degC)")
    axes[1, 0].set_title("Decadal GHCN-M qcf error")
    heat = station[station["observation"].eq("qcf")].pivot(
        index="station_name", columns="branch", values="correlation_r"
    ).reindex(columns=list(PRODUCTS))
    image = axes[1, 1].imshow(heat.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[1, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 1].set_yticks(np.arange(len(heat)), heat.index)
    axes[1, 1].set_title("Station-level annual correlation")
    figure.colorbar(image, ax=axes[1, 1], label="Pearson r", shrink=0.85)
    figure.suptitle("Project reconstruction versus public reconstruction families", fontsize=15)
    figure.savefig(OUT / "external_reconstruction_instrumental_benchmark.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    states_before = {"station_values": file_state(STATION_VALUES), "public_reconstructions": file_state(PUBLIC)}
    frame = pd.read_csv(STATION_VALUES)
    if not frame["year"].between(1951, 2000).all():
        raise RuntimeError("Station pairing table contains values outside 1951-2000")
    stations = frame[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    reference_values, locations = extract_reference_families(stations)
    frame = frame.merge(reference_values, on=["station_id", "year"], how="left", validate="one_to_one")

    station_tables = []
    decade_tables = []
    summary_rows = []
    bootstrap_tables = []
    for observation_name, observation in OBSERVATIONS.items():
        for product, column in PRODUCTS.items():
            station = metrics.station_metrics(frame, observation, product, column)
            decades = metrics.decadal_pairs(frame, observation, column)
            station_tables.append(station)
            saved_decades = decades.copy()
            saved_decades.insert(0, "product", product)
            saved_decades.insert(0, "observation", observation_name)
            decade_tables.append(saved_decades)
            summary_rows.append(metrics.aggregate_metrics(frame, station, decades, observation, product, column))
        bootstrap_tables.append(bootstrap_product_differences(frame, observation))
    station_all = pd.concat(station_tables, ignore_index=True)
    decades_all = pd.concat(decade_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)

    frame.to_csv(OUT / "station_year_reconstruction_values.csv", index=False)
    locations.to_csv(OUT / "station_reference_grid_mapping.csv", index=False)
    station_all.to_csv(OUT / "station_level_annual_metrics.csv", index=False)
    decades_all.to_csv(OUT / "station_decadal_pairs.csv", index=False)
    summary.to_csv(OUT / "reconstruction_instrumental_metrics.csv", index=False)
    bootstrap.to_csv(OUT / "station_bootstrap_differences_vs_project.csv", index=False)
    plot_summary(summary, station_all)

    states_after = {"station_values": file_state(STATION_VALUES), "public_reconstructions": file_state(PUBLIC)}
    if states_before != states_after:
        raise RuntimeError("A read-only source changed during the benchmark")
    qcf = summary[summary["observation"].eq("qcf")].set_index("branch")
    result = {
        "experiment": OUT.name,
        "status": "complete",
        "period": [1951, 2000],
        "station_count": int(frame["station_id"].nunique()),
        "products": list(PRODUCTS),
        "qcf_metrics": summary[summary["observation"].eq("qcf")].to_dict(orient="records"),
        "qcu_metrics": summary[summary["observation"].eq("qcu")].to_dict(orient="records"),
        "best_qcf_annual_rmse": str(qcf["annual_pooled_rmse_degC"].idxmin()),
        "best_qcf_annual_correlation": str(qcf["annual_pooled_correlation_r"].idxmax()),
        "best_qcf_decadal_rmse": str(qcf["decadal_rmse_degC"].idxmin()),
        "best_qcf_decadal_correlation": str(qcf["decadal_correlation_r"].idxmax()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_states_before": states_before,
        "source_states_after": states_after,
        "source_unchanged": True,
        "locked_period_access": "none; reconstruction values read only for 1951-2000",
        "independence_boundary": (
            "The public reconstruction families can share proxy and instrumental calibration sources. "
            "Their scores are comparative reconstruction benchmarks, not independent instrumental votes."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({
        "sources": states_before,
        "member_variables": MEMBERS,
        "family_definitions": {
            "local_EnKF_family": ["HGAO", "O"],
            "LMR": ["LMR"],
            "Neukom_family": [name for name in MEMBERS if name.startswith("Nk_")],
            "PHYDA": ["PHYDA"],
            "ref_rec_family_equal": ["local_EnKF_family", "LMR", "Neukom_family", "PHYDA"],
        },
    }, indent=2), encoding="utf-8")
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        "# Experiment contract\n\n"
        "- Compare the current project reconstruction with local-EnKF, LMR, Neukom, PHYDA, and their family-equal reference.\n"
        "- Use the same nine stations, observations, 1961-1990 anomalies, annual and decadal metrics as the four-scheme benchmark.\n"
        "- Read reconstruction values only for 1951-2000; do not access 1921-1939.\n"
        "- Do not modify original files, production reconstruction, EnKF, or OGGM.\n"
        "- Treat public reconstructions as comparative products with potentially overlapping source lineages.\n",
        encoding="utf-8",
    )
    report = [
        "# Final result", "", "## GHCN-M qcf comparison", "",
        "| Product | Annual RMSE | Annual r | Median station r | Decadal RMSE | Decadal r | Median RE | Median CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for product in PRODUCTS:
        row = qcf.loc[product]
        report.append(
            f"| {product} | {row['annual_pooled_rmse_degC']:.3f} | {row['annual_pooled_correlation_r']:.3f} | "
            f"{row['median_station_correlation_r']:.3f} | {row['decadal_rmse_degC']:.3f} | "
            f"{row['decadal_correlation_r']:.3f} | {row['median_station_re']:.3f} | {row['median_station_ce']:.3f} |"
        )
    report.extend([
        "", "## Interpretation boundary", "",
        "All products are evaluated with identical station observations and metrics. Public reconstructions may share "
        "instrumental calibration data and proxies, so this is a fair comparative benchmark but not a count of independent evidence sources.",
    ])
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
