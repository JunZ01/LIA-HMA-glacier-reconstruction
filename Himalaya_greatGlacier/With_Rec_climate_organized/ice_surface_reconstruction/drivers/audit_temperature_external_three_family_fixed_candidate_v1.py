"""Audit the frozen low-frequency candidate without the local-EnKF anchor."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_temperature_regional_reference_field_nested_cv_v1 as spatial
import run_temperature_station_annual_anomaly_benchmark_v1 as station_base
import validate_temperature_external_low_frequency_v1 as record_base


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v1"
DRIVER = Path(__file__).resolve()
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
ASSIGNMENT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_grid_assignment.csv"
SUPPORT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
STATION = LOOP / "temperature_station_annual_anomaly_benchmark_v4"
ROUND277 = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
RECENT = ROOT / "validation_data/sources/external_hma_temperature_candidates_round2_v1"
LAKE_ICE = ROOT / "validation_data/sources/external_hma_temperature_candidates_lake_ice_v1"

YEARS = np.arange(1000, 2001)
EXTERNAL_FAMILIES = ("lmr", "neukom", "phyda")
WIDTH = 31
LAMBDA = 0.75
SEGMENTS = ((1850, 1879), (1880, 1900), (1901, 1920), (1940, 1950))
OBSERVATIONS = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}


def allowed_field(dataset: h5py.Dataset) -> np.ndarray:
    result = np.full(dataset.shape, np.nan, dtype=float)
    for start, end in ((0, 921), (940, 1001)):
        values = np.asarray(dataset[start:end], dtype=float)
        values[(values < -9000) | (values > 9000)] = np.nan
        result[start:end] = values
    return result


def recenter_field(field: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(field[baseline], axis=0)
    return field - center[None, :, :]


def make_candidate(project: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    candidate = project + LAMBDA * spatial.lowpass(anchor - project, WIDTH)
    return recenter_field(candidate)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return math.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(years) & np.isfinite(values)
    if int(valid.sum()) < 5:
        return math.nan
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def station_sample(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    sampled = station_base.bilinear_points(
        field,
        lat,
        lon,
        stations.latitude.to_numpy(float),
        stations.longitude.to_numpy(float),
    )
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    for column in range(sampled.shape[1]):
        values = sampled[baseline, column]
        if int(np.isfinite(values).sum()) >= 20:
            sampled[:, column] -= float(np.nanmean(values))
        else:
            sampled[:, column] = np.nan
    return sampled


def station_metrics(curves: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for station_id, frame in curves.groupby("station_id"):
        for start, end in SEGMENTS:
            part = frame[frame.year.between(start, end)]
            valid = (
                np.isfinite(part[target])
                & np.isfinite(part["project_grid_degC"])
                & np.isfinite(part["candidate_grid_degC"])
            )
            part = part[valid]
            if len(part) < 5:
                continue
            observed = part[target].to_numpy(float)
            project = part.project_grid_degC.to_numpy(float)
            candidate = part.candidate_grid_degC.to_numpy(float)
            years = part.year.to_numpy(float)
            project_rmse = float(np.sqrt(np.mean((project - observed) ** 2)))
            candidate_rmse = float(np.sqrt(np.mean((candidate - observed) ** 2)))
            observed_trend = trend(years, observed)
            project_trend = trend(years, project)
            candidate_trend = trend(years, candidate)
            project_trend_error = abs(project_trend - observed_trend)
            candidate_trend_error = abs(candidate_trend - observed_trend)
            rows.append(
                {
                    "target": target.removesuffix("_station_degC"),
                    "station_id": station_id,
                    "station_name": frame.station_name.iloc[0],
                    "region_group": frame.region_group.iloc[0],
                    "segment": f"{start}-{end}",
                    "year_count": len(part),
                    "project_rmse_degC": project_rmse,
                    "candidate_rmse_degC": candidate_rmse,
                    "rmse_improvement_fraction": 1.0 - candidate_rmse / project_rmse,
                    "project_bias_degC": float(np.mean(project - observed)),
                    "candidate_bias_degC": float(np.mean(candidate - observed)),
                    "project_correlation": correlation(project, observed),
                    "candidate_correlation": correlation(candidate, observed),
                    "project_trend_degC_century": project_trend,
                    "candidate_trend_degC_century": candidate_trend,
                    "station_trend_degC_century": observed_trend,
                    "project_abs_trend_error": project_trend_error,
                    "candidate_abs_trend_error": candidate_trend_error,
                }
            )
    return pd.DataFrame(rows)


def aggregate_station_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, segment), frame in metrics.groupby(["target", "segment"]):
        project_trend = float(frame.project_abs_trend_error.median())
        candidate_trend = float(frame.candidate_abs_trend_error.median())
        rows.append(
            {
                "target": target,
                "segment": segment,
                "station_count": len(frame),
                "median_rmse_improvement_fraction": float(frame.rmse_improvement_fraction.median()),
                "fraction_stations_rmse_improves": float((frame.rmse_improvement_fraction > 0).mean()),
                "median_project_abs_trend_error": project_trend,
                "median_candidate_abs_trend_error": candidate_trend,
                "median_trend_error_change_fraction": (
                    candidate_trend / project_trend - 1.0 if project_trend > 0 else math.nan
                ),
                "equal_station_project_bias_degC": float(frame.project_bias_degC.mean()),
                "equal_station_candidate_bias_degC": float(frame.candidate_bias_degC.mean()),
                "median_correlation_change": float(
                    (frame.candidate_correlation - frame.project_correlation).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    required = {
        "public": PUBLIC,
        "hma_mask": HMA_MASK,
        "assignment": ASSIGNMENT,
        "support": SUPPORT,
        "station_curves": STATION / "station_annual_anomalies_and_products.csv",
        "station_coverage": STATION / "station_baseline_coverage.csv",
        "station_summary": STATION / "summary.json",
        "round277_panel": ROUND277 / "monthly_temperature_panel.csv",
        "chongce": LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "jiang": LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "haizi": RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": RECENT / "Temperature reconstruction.xlsx",
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "driver": DRIVER,
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: record_base.file_state(path) for name, path in required.items()}
    station_summary = json.loads(required["station_summary"].read_text(encoding="utf-8"))
    unit = station_summary.get("unit_preflight", {})
    if not unit.get("passed") or not np.isclose(unit.get("integer_scale", math.nan), 0.01):
        raise RuntimeError("GHCN-M /100 unit preflight failed")

    temp = Path(tempfile.mkdtemp(prefix=f"{OUTPUT.name}_", dir=RESULTS))
    try:
        (temp / "EXPERIMENT_CONTRACT.md").write_text(
            "# External Three-Family Fixed Candidate Audit\n\n"
            "Freeze width=31 years and lambda=0.75. Exclude local-EnKF from the anchor. "
            "Use LMR, Neukom, and PHYDA as three external reconstruction families, one family one vote. "
            "Do not tune any parameter. Test family-held-out 1500-1899 regional 50-year means, qcf-primary "
            "direct stations in allowed historical segments, qcu homogenization sensitivity, four local "
            "records, and five modern gridded products over 1940-2000. The 1921-1939 interval remains unread. "
            "Acceptance requires all held-out external families to improve, median held-out improvement >=20%, "
            "positive median qcf RMSE improvement in every segment supported by at least two stations, at least "
            "60% of qcf station-segments improving, qcf median trend-error regression <=10%, at least three of "
            "four local records improving with no record worsening by more than 20%, modern median regional RMSE "
            "not worsening, worst modern regional-product RMSE regression <=5%, and median modern correlation "
            "drop <=0.02. Failure rejects a unified HMA spatial correction; thresholds are not revised after run.\n",
            encoding="utf-8",
        )

        hma_mask = np.load(HMA_MASK).astype(bool)
        assignment = pd.read_csv(ASSIGNMENT)
        support = pd.read_csv(SUPPORT).sort_values("o2region")
        region_names = support.o2region.tolist()
        masks = {"hma": hma_mask}
        for row in support.itertuples(index=False):
            mask = np.zeros_like(hma_mask)
            cells = assignment[assignment.o2region.eq(row.o2region)]
            mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
            masks[row.o2region] = mask
        all_domains = ["hma"] + region_names

        with h5py.File(PUBLIC, "r") as handle:
            years = np.asarray(handle["time"], dtype=int).ravel()
            lat = np.asarray(handle["lat"], dtype=float).ravel()
            lon = np.asarray(handle["lon"], dtype=float).ravel()
            if not np.array_equal(years, YEARS):
                raise RuntimeError("year contract failed")
            project_field = recenter_field(allowed_field(handle["tas_ensemble_mean_rec_result"]))
            family_fields = {
                "lmr": recenter_field(allowed_field(handle["tas_rec_ref_LMR_c05d"])),
                "neukom": recenter_field(
                    spatial.finite_mean(
                        [
                            allowed_field(handle[name])
                            for name in (
                                "tas_rec_ref_Nk_AM_c05d",
                                "tas_rec_ref_Nk_CCA_c05d",
                                "tas_rec_ref_Nk_CPS_c05d",
                                "tas_rec_ref_Nk_DA_c05d",
                                "tas_rec_ref_Nk_GraphEM_c05d",
                                "tas_rec_ref_Nk_PCR_c05d",
                            )
                        ]
                    )
                ),
                "phyda": recenter_field(allowed_field(handle["tas_rec_ref_PHYDA_c05d"])),
            }
            observation_fields = {
                name: recenter_field(allowed_field(handle[variable]))
                for name, variable in OBSERVATIONS.items()
            }

        project_series = spatial.regional_series(project_field, masks, lat)
        family_series = {
            name: spatial.regional_series(field, masks, lat)
            for name, field in family_fields.items()
        }
        external_rows = []
        for held_out in EXTERNAL_FAMILIES:
            anchor_names = [name for name in EXTERNAL_FAMILIES if name != held_out]
            anchor = spatial.finite_mean([family_fields[name] for name in anchor_names])
            candidate = make_candidate(project_field, anchor)
            candidate_series = spatial.regional_series(candidate, masks, lat)
            project_error = spatial.rmse(
                spatial.window_vector(project_series, region_names),
                spatial.window_vector(family_series[held_out], region_names),
            )
            candidate_error = spatial.rmse(
                spatial.window_vector(candidate_series, region_names),
                spatial.window_vector(family_series[held_out], region_names),
            )
            hma_project_error = spatial.rmse(
                spatial.window_vector(project_series, ["hma"]),
                spatial.window_vector(family_series[held_out], ["hma"]),
            )
            hma_candidate_error = spatial.rmse(
                spatial.window_vector(candidate_series, ["hma"]),
                spatial.window_vector(family_series[held_out], ["hma"]),
            )
            external_rows.append(
                {
                    "held_out_family": held_out,
                    "anchor_families": "+".join(anchor_names),
                    "width_years": WIDTH,
                    "lambda": LAMBDA,
                    "regional_project_rmse_degC": project_error,
                    "regional_candidate_rmse_degC": candidate_error,
                    "regional_improvement_fraction": 1.0 - candidate_error / project_error,
                    "hma_project_rmse_degC": hma_project_error,
                    "hma_candidate_rmse_degC": hma_candidate_error,
                    "hma_improvement_fraction": 1.0 - hma_candidate_error / hma_project_error,
                }
            )
        external_table = pd.DataFrame(external_rows)
        external_table.to_csv(temp / "external_family_held_out_validation.csv", index=False)

        final_anchor = spatial.finite_mean([family_fields[name] for name in EXTERNAL_FAMILIES])
        candidate_field = make_candidate(project_field, final_anchor)
        correction_field = candidate_field - project_field
        candidate_series = spatial.regional_series(candidate_field, masks, lat)

        modern_rows = []
        modern = (YEARS >= 1940) & (YEARS <= 2000)
        for product, observed_field in observation_fields.items():
            observed_series = spatial.regional_series(observed_field, masks, lat)
            for domain in all_domains:
                project = project_series[domain][modern]
                candidate = candidate_series[domain][modern]
                observed = observed_series[domain][modern]
                project_rmse = spatial.rmse(project, observed)
                candidate_rmse = spatial.rmse(candidate, observed)
                modern_rows.append(
                    {
                        "product": product,
                        "domain": domain,
                        "project_rmse_degC": project_rmse,
                        "candidate_rmse_degC": candidate_rmse,
                        "rmse_improvement_fraction": 1.0 - candidate_rmse / project_rmse,
                        "project_correlation": correlation(project, observed),
                        "candidate_correlation": correlation(candidate, observed),
                    }
                )
        modern_table = pd.DataFrame(modern_rows)
        modern_table["correlation_change"] = (
            modern_table.candidate_correlation - modern_table.project_correlation
        )
        modern_table.to_csv(temp / "modern_five_product_regional_validation.csv", index=False)

        station_curves = pd.read_csv(required["station_curves"])
        coverage = pd.read_csv(required["station_coverage"])
        eligible = set(
            coverage[
                coverage.series.eq("qcf") & coverage.baseline_eligible.astype(bool)
            ].station_id
        )
        stations = (
            station_curves[
                ["station_id", "station_name", "region_group", "latitude", "longitude"]
            ]
            .drop_duplicates("station_id")
            .sort_values("station_id")
            .reset_index(drop=True)
        )
        project_at_station = station_sample(project_field, lat, lon, stations)
        candidate_at_station = station_sample(candidate_field, lat, lon, stations)
        model_rows = []
        for column, station in enumerate(stations.itertuples(index=False)):
            for index, year in enumerate(YEARS):
                if year <= 1920 or year >= 1940:
                    model_rows.append(
                        {
                            "station_id": station.station_id,
                            "year": int(year),
                            "project_grid_degC": float(project_at_station[index, column]),
                            "candidate_grid_degC": float(candidate_at_station[index, column]),
                        }
                    )
        model_table = pd.DataFrame(model_rows)
        station_curves = station_curves.merge(
            model_table, on=["station_id", "year"], how="left", validate="many_to_one"
        )
        station_curves["qcf_eligible"] = station_curves.station_id.isin(eligible)
        closure = station_curves[
            station_curves.project_six_member_degC.notna()
            & station_curves.project_grid_degC.notna()
        ]
        station_project_closure = float(
            np.max(np.abs(closure.project_six_member_degC - closure.project_grid_degC))
        )
        eligible_curves = station_curves[station_curves.qcf_eligible].copy()
        station_qcf = station_metrics(eligible_curves, "qcf_station_degC")
        station_qcu = station_metrics(eligible_curves, "qcu_station_degC")
        station_metrics_all = pd.concat([station_qcf, station_qcu], ignore_index=True)
        station_aggregate = aggregate_station_metrics(station_metrics_all)
        station_curves.to_csv(temp / "station_candidate_curves.csv", index=False)
        station_metrics_all.to_csv(temp / "station_candidate_metrics.csv", index=False)
        station_aggregate.to_csv(temp / "station_segment_summary.csv", index=False)

        boxes = {
            "Chongce": (80.75, 81.75, 34.75, 35.75),
            "Haizi": (99.75, 101.25, 28.75, 30.25),
            "Jionglacuo": (93.5, 95.5, 30.0, 32.0),
            "Jiang": (90.25, 91.25, 31.0, 32.0),
        }
        local_project = {
            name: spatial.local_box(project_field, lat, lon, box) for name, box in boxes.items()
        }
        local_candidate = {
            name: spatial.local_box(candidate_field, lat, lon, box) for name, box in boxes.items()
        }
        chongce = spatial.read_xlsx(required["chongce"], 1, 3, 6, 1524, 2012)
        haizi = spatial.read_xlsx(required["haizi"], 1, 3, 2, 1115, 2017)
        jionglacuo = spatial.read_xlsx(required["jionglacuo"], 1, 2, 1, 1210, 2023)
        jiang = spatial.read_xlsx(required["jiang"], 2, 7, 12, 86, 2013.1)
        independent = {
            "Chongce_annual": spatial.irregular_metric(
                local_project["Chongce"], local_candidate["Chongce"], chongce, 1940, 2000, 1550
            ),
            "Haizi_annual": spatial.regular_record_metric(
                local_project["Haizi"], local_candidate["Haizi"], haizi, 1959, 2000
            ),
            "Jionglacuo_JJAS": spatial.regular_record_metric(
                local_project["Jionglacuo"], local_candidate["Jionglacuo"], jionglacuo, 1970, 2000
            ),
            "Jiang_summer": spatial.irregular_metric(
                local_project["Jiang"], local_candidate["Jiang"], jiang, 1940, 2000, 1550
            ),
        }
        independent_table = pd.DataFrame(
            [{"record": name, **values} for name, values in independent.items()]
        )
        independent_table.to_csv(temp / "independent_local_record_validation.csv", index=False)

        candidate_path = temp / "temperature_external_three_family_fixed_candidate.nc"
        with h5py.File(candidate_path, "w") as out:
            out.create_dataset("time", data=YEARS)
            out.create_dataset("lat", data=lat)
            out.create_dataset("lon", data=lon)
            out.create_dataset(
                "tas_candidate", data=candidate_field.astype("f4"), compression="gzip", compression_opts=4
            )
            out.create_dataset(
                "tas_correction", data=correction_field.astype("f4"), compression="gzip", compression_opts=4
            )
            out.attrs["anchor_families"] = "LMR, Neukom-family-mean, PHYDA"
            out.attrs["excluded_anchor_family"] = "local-EnKF"
            out.attrs["width_years"] = WIDTH
            out.attrs["lambda"] = LAMBDA
            out.attrs["release_status"] = "pending_fixed_gates"
            out.attrs["locked_1921_1939_values_read"] = 0

        external_improvements = external_table.regional_improvement_fraction.to_numpy(float)
        qcf_supported = station_aggregate[
            station_aggregate.target.eq("qcf") & (station_aggregate.station_count >= 2)
        ]
        qcf_rows = station_qcf.copy()
        qcf_project_trend = float(qcf_rows.project_abs_trend_error.median())
        qcf_candidate_trend = float(qcf_rows.candidate_abs_trend_error.median())
        local_improvements = independent_table.improvement.to_numpy(float)
        modern_improvements = modern_table.rmse_improvement_fraction.to_numpy(float)
        modern_correlation_changes = modern_table.correlation_change.to_numpy(float)
        gates = {
            "all_three_external_held_out_families_improve": bool(np.all(external_improvements > 0)),
            "median_external_held_out_improvement_at_least_20pct": bool(
                np.median(external_improvements) >= 0.20
            ),
            "all_supported_qcf_segments_median_rmse_improve": bool(
                len(qcf_supported) > 0
                and np.all(qcf_supported.median_rmse_improvement_fraction > 0)
            ),
            "qcf_station_segment_improvement_fraction_at_least_60pct": bool(
                len(qcf_rows) > 0 and (qcf_rows.rmse_improvement_fraction > 0).mean() >= 0.60
            ),
            "qcf_median_trend_error_regression_at_most_10pct": bool(
                qcf_project_trend > 0 and qcf_candidate_trend <= 1.10 * qcf_project_trend
            ),
            "at_least_three_of_four_local_records_improve": bool(
                int((local_improvements > 0).sum()) >= 3
            ),
            "no_local_record_worsens_more_than_20pct": bool(np.min(local_improvements) >= -0.20),
            "modern_median_regional_rmse_no_regression": bool(np.median(modern_improvements) >= 0),
            "modern_worst_regional_product_rmse_regression_at_most_5pct": bool(
                np.min(modern_improvements) >= -0.05
            ),
            "modern_median_correlation_drop_at_most_0p02": bool(
                np.nanmedian(modern_correlation_changes) >= -0.02
            ),
        }
        passes = bool(all(gates.values()))
        decision = (
            "accept_external_three_family_fixed_candidate_v2"
            if passes
            else "reject_unified_hma_low_frequency_correction"
        )
        with h5py.File(candidate_path, "r+") as out:
            out.attrs["release_status"] = "accepted_candidate_v2" if passes else "diagnostic_only"

        hma_anchor = spatial.weighted_series(final_anchor, hma_mask, lat)
        hma_project = project_series["hma"]
        hma_candidate = candidate_series["hma"]
        hma_project_low = spatial.lowpass(hma_project[:, None], WIDTH)[:, 0]
        hma_anchor_low = spatial.lowpass(hma_anchor[:, None], WIDTH)[:, 0]
        hma_candidate_low = spatial.lowpass(hma_candidate[:, None], WIDTH)[:, 0]
        plot = (YEARS >= 1500) & (YEARS <= 1920)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        axes[0, 0].plot(YEARS[plot], hma_project_low[plot], color="#D55E00", lw=2, label="Project")
        axes[0, 0].plot(YEARS[plot], hma_anchor_low[plot], color="#111111", lw=1.8, ls="--", label="External anchor")
        axes[0, 0].plot(YEARS[plot], hma_candidate_low[plot], color="#0072B2", lw=2.2, label="Fixed candidate")
        axes[0, 0].set(title="HMA 31-year low-frequency temperature", ylabel="degC", xlabel="Year")
        axes[0, 0].legend(frameon=False)
        axes[0, 1].bar(
            external_table.held_out_family,
            100 * external_table.regional_improvement_fraction,
            color="#0072B2",
        )
        axes[0, 1].axhline(0, color="#777777", lw=0.8)
        axes[0, 1].set(title="Held-out external-family test", ylabel="RMSE improvement (%)")
        qcf_plot = qcf_supported.set_index("segment")
        axes[1, 0].bar(
            qcf_plot.index,
            100 * qcf_plot.median_rmse_improvement_fraction,
            color="#009E73",
        )
        axes[1, 0].axhline(0, color="#777777", lw=0.8)
        axes[1, 0].tick_params(axis="x", rotation=20)
        axes[1, 0].set(title="Direct GHCN-M qcf test", ylabel="Median station RMSE improvement (%)")
        axes[1, 1].bar(
            independent_table.record,
            100 * independent_table.improvement,
            color="#B23A48",
        )
        axes[1, 1].axhline(0, color="#777777", lw=0.8)
        axes[1, 1].tick_params(axis="x", rotation=20)
        axes[1, 1].set(title="Independent local-record test", ylabel="RMSE improvement (%)")
        for axis in axes.flat:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
        fig.savefig(temp / "external_three_family_fixed_candidate_audit.png", dpi=220)
        plt.close(fig)

        summary = {
            "status": "complete",
            "decision": decision,
            "anchor_families": list(EXTERNAL_FAMILIES),
            "excluded_anchor_family": "local_enkf",
            "width_years": WIDTH,
            "lambda": LAMBDA,
            "parameters_retuned": False,
            "external_held_out_regional_improvements": {
                row.held_out_family: float(row.regional_improvement_fraction)
                for row in external_table.itertuples(index=False)
            },
            "median_external_held_out_regional_improvement": float(
                np.median(external_improvements)
            ),
            "qcf_eligible_station_count": len(eligible),
            "qcf_station_segment_count": len(qcf_rows),
            "qcf_station_segment_improving_fraction": float(
                (qcf_rows.rmse_improvement_fraction > 0).mean()
            ),
            "qcf_median_project_abs_trend_error": qcf_project_trend,
            "qcf_median_candidate_abs_trend_error": qcf_candidate_trend,
            "independent_local_record_improvements": {
                row.record: float(row.improvement)
                for row in independent_table.itertuples(index=False)
            },
            "modern_median_regional_rmse_improvement": float(np.median(modern_improvements)),
            "modern_worst_regional_product_rmse_improvement": float(np.min(modern_improvements)),
            "modern_median_correlation_change": float(np.nanmedian(modern_correlation_changes)),
            "station_project_grid_closure_max_degC": station_project_closure,
            "gates": gates,
            "passes_all_gates": passes,
            "candidate_release_status": "accepted_candidate_v2" if passes else "diagnostic_only",
            "locked_1921_1939_values_accessed": 0,
            "original_sources_modified": False,
            "production_authorized": False,
            "enkf_run": False,
            "oggm_run": False,
        }
        (temp / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        failed = [name for name, value in gates.items() if not value]
        (temp / "FINAL_RESULT.md").write_text(
            "# External Three-Family Fixed Candidate Audit\n\n"
            f"Decision: `{decision}`.\n\n"
            f"- Frozen method: `{WIDTH}-year low-pass`, `lambda={LAMBDA}`; no parameter was retuned.\n"
            "- Anchor: equal family mean of LMR, Neukom, and PHYDA; local-EnKF excluded.\n"
            f"- Median held-out external-family regional improvement: `{100*np.median(external_improvements):+.2f}%`.\n"
            f"- qcf station-segment improving fraction: `{100*summary['qcf_station_segment_improving_fraction']:.1f}%`.\n"
            f"- Local-record improvements: `{summary['independent_local_record_improvements']}`.\n"
            f"- Modern median regional RMSE improvement: `{100*summary['modern_median_regional_rmse_improvement']:+.2f}%`.\n"
            f"- Failed frozen gates: `{', '.join(failed) if failed else 'none'}`.\n"
            "- Locked 1921-1939 values accessed: `0`.\n"
            "- Production, EnKF, and OGGM runs: `none`.\n",
            encoding="utf-8",
        )

        after = {name: record_base.file_state(path) for name, path in required.items()}
        unchanged = before == after
        (temp / "source_manifest.json").write_text(
            json.dumps(
                {
                    "inputs_before": before,
                    "inputs_after": after,
                    "all_inputs_unchanged": unchanged,
                    "round277_temperature_unit": "degC",
                    "ghcn_integer_scale": 0.01,
                    "locked_1921_1939_values_accessed": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not unchanged:
            raise RuntimeError("input source changed")
        temp.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
