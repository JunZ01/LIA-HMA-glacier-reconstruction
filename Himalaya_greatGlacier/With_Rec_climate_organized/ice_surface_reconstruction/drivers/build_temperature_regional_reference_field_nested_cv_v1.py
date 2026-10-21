"""Nested-CV spatial low-frequency correction with independent record gates."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook

import validate_temperature_external_low_frequency_v1 as base


PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
RESULTS = base.RESULTS
OUTPUT = RESULTS / "climate_reconstruction_temperature_regional_reference_field_nested_cv_v1"
HMA_MASK = RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
ASSIGNMENT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_grid_assignment.csv"
SUPPORT = RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
RECENT = base.ROOT / "validation_data/sources/external_hma_temperature_candidates_round2_v1"
LAKE_ICE = base.ROOT / "validation_data/sources/external_hma_temperature_candidates_lake_ice_v1"
LEDGER = RESULTS / "climate_reconstruction_improvement_loop_v2/round_107_proxy_evidence_master_ledger_v1/proxy_evidence_master_ledger.csv"

YEARS = np.arange(1000, 2001)
WIDTHS = [31, 51, 101]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOWS = [(year, year + 49) for year in range(1500, 1900, 50)]
FAMILY_ORDER = ["local_enkf", "lmr", "neukom", "phyda"]
OBSERVATIONS = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}


def allowed_field(dataset: h5py.Dataset) -> np.ndarray:
    result = np.full(dataset.shape, np.nan, dtype=float)
    result[:921] = np.asarray(dataset[:921], dtype=float)
    result[940:] = np.asarray(dataset[940:], dtype=float)
    return result


def finite_mean(fields: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(fields)
    count = np.sum(np.isfinite(stack), axis=0)
    return np.divide(np.nansum(stack, axis=0), count, out=np.full(stack.shape[1:], np.nan), where=count > 0)


def lowpass(values: np.ndarray, width: int) -> np.ndarray:
    shape = values.shape
    flat = values.reshape(shape[0], -1)
    result = np.full_like(flat, np.nan, dtype=float)
    for start, end in [(0, 921), (940, 1001)]:
        frame = pd.DataFrame(flat[start:end])
        result[start:end] = frame.rolling(width, center=True, min_periods=max(3, width // 2)).mean().to_numpy()
    return result.reshape(shape)


def weighted_series(field: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(field) & mask[None, :, :]
    numerator = np.nansum(np.where(valid, field, 0.0) * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def regional_series(field: np.ndarray, masks: dict[str, np.ndarray], lat: np.ndarray) -> dict[str, np.ndarray]:
    return {name: weighted_series(field, mask, lat) for name, mask in masks.items()}


def window_vector(series: dict[str, np.ndarray], regions: list[str]) -> np.ndarray:
    values = []
    for region in regions:
        for start, end in WINDOWS:
            selected = (YEARS >= start) & (YEARS <= end)
            values.append(float(np.nanmean(series[region][selected])))
    return np.asarray(values)


def candidate_series(project: dict[str, np.ndarray], anchor: dict[str, np.ndarray], width: int, lam: float) -> dict[str, np.ndarray]:
    return {
        region: project[region] + lam * lowpass((anchor[region] - project[region])[:, None], width)[:, 0]
        for region in project
    }


def mean_series(members: list[dict[str, np.ndarray]], regions: list[str]) -> dict[str, np.ndarray]:
    return {region: finite_mean([member[region] for member in members]) for region in regions}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def local_box(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lon0, lon1, lat0, lat1 = bounds
    mask = (lat[:, None] >= lat0) & (lat[:, None] <= lat1) & (lon[None, :] >= lon0) & (lon[None, :] <= lon1)
    return weighted_series(field, mask, lat)


def read_xlsx(path: Path, year_col: int, value_col: int, first_row: int, start: float, end: float) -> pd.DataFrame:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Sheet1"]
    rows = []
    for row in sheet.iter_rows(min_row=first_row):
        try:
            year = float(row[year_col - 1].value)
        except (TypeError, ValueError):
            continue
        if 1921 <= year < 1940 or not start <= year <= end:
            continue
        value = row[value_col - 1].value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rows.append({"year": year, "external": float(value)})
    workbook.close()
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def regular_record_metric(project: np.ndarray, candidate: np.ndarray, external: pd.DataFrame, modern_start: int, modern_end: int) -> dict[str, float]:
    frame = pd.DataFrame({"year": YEARS, "project": project, "candidate": candidate})
    ext = external.copy(); ext["year"] = ext.year.astype(int)
    merged = frame.merge(ext, on="year")
    for column in ["project", "candidate", "external"]:
        modern = merged.loc[merged.year.between(modern_start, modern_end), column].dropna()
        merged[f"{column}_z"] = (merged[column] - modern.mean()) / modern.std(ddof=0)
    rows = []
    for start, end in WINDOWS:
        sub = merged.loc[merged.year.between(start, end)]
        if sub.external_z.notna().sum() < 35:
            continue
        rows.append([sub.project_z.mean(), sub.candidate_z.mean(), sub.external_z.mean()])
    values = np.asarray(rows)
    p = rmse(values[:, 0], values[:, 2]); c = rmse(values[:, 1], values[:, 2])
    return {"project_rmse": p, "candidate_rmse": c, "improvement": (p - c) / p}


def irregular_metric(project: np.ndarray, candidate: np.ndarray, external: pd.DataFrame, modern_start: int, modern_end: int, first_window: int) -> dict[str, float]:
    ext = external.copy()
    ext["project"] = np.interp(ext.year, YEARS, project)
    ext["candidate"] = np.interp(ext.year, YEARS, candidate)
    modern = ext.year.between(modern_start, modern_end)
    for column in ["project", "candidate", "external"]:
        ext[f"{column}_a"] = ext[column] - ext.loc[modern, column].mean()
    rows = []
    for start in range(first_window, 1900, 50):
        sub = ext.loc[ext.year.between(start, start + 49)]
        if len(sub) < 5:
            continue
        rows.append([sub.project_a.mean(), sub.candidate_a.mean(), sub.external_a.mean()])
    values = np.asarray(rows)
    p_err = values[:, 0] - values[:, 2]; c_err = values[:, 1] - values[:, 2]
    p = float(np.sqrt(np.mean(p_err**2))); c = float(np.sqrt(np.mean(c_err**2)))
    return {
        "project_rmse": p, "candidate_rmse": c, "improvement": (p - c) / p,
        "project_bias": float(p_err.mean()), "candidate_bias": float(c_err.mean()),
        "project_corr": float(np.corrcoef(values[:, 0], values[:, 2])[0, 1]),
        "candidate_corr": float(np.corrcoef(values[:, 1], values[:, 2])[0, 1]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {
        "public": PUBLIC, "hma_mask": HMA_MASK, "assignment": ASSIGNMENT, "support": SUPPORT,
        "chongce": LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "jiang": LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "haizi": RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": RECENT / "Temperature reconstruction.xlsx", "ledger": LEDGER, "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}

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
        years = np.asarray(handle["time"], dtype=int).ravel(); lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float)
        if not np.array_equal(years, YEARS): raise ValueError("Year contract failed")
        project_field = allowed_field(handle["tas_ensemble_mean_rec_result"])
        family_fields = {
            "local_enkf": finite_mean([allowed_field(handle["tas_rec_ref_HGAO_c05d"]), allowed_field(handle["tas_rec_ref_O_c05d"])]),
            "lmr": allowed_field(handle["tas_rec_ref_LMR_c05d"]),
            "neukom": finite_mean([allowed_field(handle[name]) for name in ["tas_rec_ref_Nk_AM_c05d","tas_rec_ref_Nk_CCA_c05d","tas_rec_ref_Nk_CPS_c05d","tas_rec_ref_Nk_DA_c05d","tas_rec_ref_Nk_GraphEM_c05d","tas_rec_ref_Nk_PCR_c05d"]]),
            "phyda": allowed_field(handle["tas_rec_ref_PHYDA_c05d"]),
        }
        observation_fields = {name: allowed_field(handle[var]) for name, var in OBSERVATIONS.items()}

    project_series = regional_series(project_field, masks, lat)
    family_series = {name: regional_series(field, masks, lat) for name, field in family_fields.items()}
    outer_rows = []
    selected_pairs = []
    for outer in FAMILY_ORDER:
        train = [name for name in FAMILY_ORDER if name != outer]
        pair_scores = []
        for width in WIDTHS:
            for lam in LAMBDAS:
                inner_scores = []
                for inner in train:
                    anchor_names = [name for name in train if name != inner]
                    anchor = mean_series([family_series[name] for name in anchor_names], all_domains)
                    current = candidate_series(project_series, anchor, width, lam)
                    inner_scores.append(rmse(window_vector(current, region_names), window_vector(family_series[inner], region_names)))
                pair_scores.append((float(np.median(inner_scores)), lam, width))
        _, selected_lam, selected_width = min(pair_scores, key=lambda item: (item[0], item[1], item[2]))
        selected_pairs.append((selected_width, selected_lam))
        outer_anchor = mean_series([family_series[name] for name in train], all_domains)
        outer_candidate = candidate_series(project_series, outer_anchor, selected_width, selected_lam)
        control_score = rmse(window_vector(project_series, region_names), window_vector(family_series[outer], region_names))
        candidate_score = rmse(window_vector(outer_candidate, region_names), window_vector(family_series[outer], region_names))
        outer_rows.append({"held_out_family": outer, "selected_width": selected_width, "selected_lambda": selected_lam, "project_rmse": control_score, "candidate_rmse": candidate_score, "improvement": 1 - candidate_score / control_score})
    final_width = sorted(pair[0] for pair in selected_pairs)[1]
    final_lambda = sorted(pair[1] for pair in selected_pairs)[1]
    outer_table = pd.DataFrame(outer_rows); outer_table.to_csv(OUTPUT / "outer_family_validation.csv", index=False)

    anchor_field = finite_mean([family_fields[name] for name in FAMILY_ORDER])
    correction = final_lambda * lowpass(anchor_field - project_field, final_width)
    candidate_field = project_field + correction
    candidate_path = OUTPUT / "temperature_regional_reference_field_candidate.nc"
    with h5py.File(candidate_path, "w") as out:
        out.create_dataset("time", data=YEARS); out.create_dataset("lat", data=lat); out.create_dataset("lon", data=lon)
        out.create_dataset("tas_candidate", data=candidate_field.astype("f4"), compression="gzip", compression_opts=4)
        out.create_dataset("tas_correction", data=correction.astype("f4"), compression="gzip", compression_opts=4)
        out.attrs["selected_width_years"] = final_width; out.attrs["selected_lambda"] = final_lambda
        out.attrs["release_status"] = "pending_gates"

    candidate_series_all = regional_series(candidate_field, masks, lat)
    observation_series = {name: regional_series(field, masks, lat) for name, field in observation_fields.items()}
    modern_rows = []
    modern = (YEARS >= 1940) & (YEARS <= 2000)
    for product in OBSERVATIONS:
        for domain in all_domains:
            obs = observation_series[product][domain]
            p = rmse(project_series[domain][modern], obs[modern]); c = rmse(candidate_series_all[domain][modern], obs[modern])
            valid = modern & np.isfinite(obs) & np.isfinite(project_series[domain]) & np.isfinite(candidate_series_all[domain])
            pc = float(np.corrcoef(project_series[domain][valid], obs[valid])[0, 1]); cc = float(np.corrcoef(candidate_series_all[domain][valid], obs[valid])[0, 1])
            modern_rows.append({"product": product, "domain": domain, "rmse_improvement": 1-c/p, "correlation_change": cc-pc})
    modern_table = pd.DataFrame(modern_rows); modern_table.to_csv(OUTPUT / "modern_regional_validation.csv", index=False)

    boxes = {
        "Chongce": (80.75,81.75,34.75,35.75), "Haizi": (99.75,101.25,28.75,30.25),
        "Jionglacuo": (93.5,95.5,30.0,32.0), "Jiang": (90.25,91.25,31.0,32.0),
    }
    local_project = {name: local_box(project_field, lat, lon, box) for name, box in boxes.items()}
    local_candidate = {name: local_box(candidate_field, lat, lon, box) for name, box in boxes.items()}
    chongce = read_xlsx(paths["chongce"],1,3,6,1524,2012)
    haizi = read_xlsx(paths["haizi"],1,3,2,1115,2017)
    jion = read_xlsx(paths["jionglacuo"],1,2,1,1210,2023)
    jiang = read_xlsx(paths["jiang"],2,7,12,86,2013.1)
    independent = {
        "Chongce_annual": irregular_metric(local_project["Chongce"], local_candidate["Chongce"], chongce, 1940, 2000, 1550),
        "Haizi_annual": regular_record_metric(local_project["Haizi"], local_candidate["Haizi"], haizi, 1959, 2000),
        "Jionglacuo_JJAS": regular_record_metric(local_project["Jionglacuo"], local_candidate["Jionglacuo"], jion, 1970, 2000),
        "Jiang_summer": irregular_metric(local_project["Jiang"], local_candidate["Jiang"], jiang, 1940, 2000, 1550),
    }
    independent_rows = []
    for name, values in independent.items(): independent_rows.append({"record": name, **values})
    pd.DataFrame(independent_rows).to_csv(OUTPUT / "independent_record_validation.csv", index=False)

    outer_imp = outer_table.improvement.to_numpy(float); modern_imp = modern_table.rmse_improvement.to_numpy(float); modern_corr = modern_table.correlation_change.to_numpy(float)
    gates = {
        "all_outer_family_improvements_at_least_20pct": bool(np.all(outer_imp >= .20)),
        "median_outer_family_improvement_at_least_50pct": bool(np.median(outer_imp) >= .50),
        "modern_median_rmse_no_regression": bool(np.median(modern_imp) >= 0),
        "modern_worst_rmse_regression_at_most_5pct": bool(np.min(modern_imp) >= -.05),
        "modern_median_correlation_drop_at_most_0p02": bool(np.median(modern_corr) >= -.02),
        "chongce_annual_all_gates": bool(independent["Chongce_annual"]["improvement"] >= .10 and abs(independent["Chongce_annual"]["candidate_bias"]) < abs(independent["Chongce_annual"]["project_bias"]) and independent["Chongce_annual"]["candidate_corr"]-independent["Chongce_annual"]["project_corr"] >= -.10),
        "haizi_annual_worsening_at_most_5pct": bool(independent["Haizi_annual"]["improvement"] >= -.05),
        "jionglacuo_summer_worsening_at_most_10pct": bool(independent["Jionglacuo_JJAS"]["improvement"] >= -.10),
        "jiang_summer_worsening_at_most_10pct": bool(independent["Jiang_summer"]["improvement"] >= -.10),
    }
    passes = bool(all(gates.values()))
    with h5py.File(candidate_path, "r+") as out: out.attrs["release_status"] = "accepted" if passes else "diagnostic_only"

    median_correction = np.nanmedian(correction[(YEARS>=1500)&(YEARS<=1899)], axis=0)
    fig, axes = plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    image = axes[0].imshow(median_correction, origin="upper", cmap="RdBu_r", vmin=-1, vmax=1); axes[0].set_title("Median correction 1500-1899 (degC)"); fig.colorbar(image, ax=axes[0])
    axes[1].bar(outer_table.held_out_family, 100*outer_table.improvement); axes[1].axhline(20,color="#111",ls="--"); axes[1].set_title("Outer-family improvement"); axes[1].set_ylabel("Percent")
    indep_table = pd.DataFrame(independent_rows); axes[2].bar(indep_table.record,100*indep_table.improvement); axes[2].axhline(0,color="#777"); axes[2].tick_params(axis="x",rotation=25); axes[2].set_title("Independent-record RMSE improvement"); axes[2].set_ylabel("Percent")
    fig.savefig(OUTPUT / "regional_reference_field_validation.png",dpi=220); plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"inputs":{name:{"path":str(paths[name]),"sha256_before":before[name],"sha256_after":after[name]} for name in paths},"all_input_hashes_unchanged":before==after},indent=2),encoding="utf-8")
    summary = {
        "status":"complete","selected_width_years":final_width,"selected_lambda":final_lambda,
        "outer_selected_pairs":[{"width":w,"lambda":l} for w,l in selected_pairs],
        "median_outer_improvement":float(np.median(outer_imp)),"modern_median_rmse_improvement":float(np.median(modern_imp)),"modern_worst_rmse_improvement":float(np.min(modern_imp)),
        "independent_record_improvements":{name:float(values["improvement"]) for name,values in independent.items()},
        "gates":gates,"passes_all_release_gates":passes,"candidate_file":str(candidate_path),"candidate_release_status":"accepted" if passes else "diagnostic_only",
        "locked_1921_1939_accessed":False,"phase_1901_1920_used":False,"original_sources_modified":False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text("# Regional Reference-Field Nested CV\n\n"+f"- Selected: width `{final_width}`, lambda `{final_lambda}`.\n"+f"- Median outer-family improvement: `{100*np.median(outer_imp):.2f}%`.\n"+f"- All release gates pass: `{str(passes).lower()}`.\n"+f"- Candidate status: `{'accepted' if passes else 'diagnostic_only'}`.\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
