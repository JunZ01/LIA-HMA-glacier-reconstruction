"""Build and validate a west-east annual mean-bias correction field."""

from __future__ import annotations

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_two_anchor_mean_bias_field_v1"
ASSIGNMENT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
SUPPORT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
HMA_MASK = tools.HMA_MASK
WEST_LON = 81.1144
EAST_LON = 100.65


def aligned_window_bias(project: np.ndarray, external: pd.DataFrame, modern_start: int, modern_end: int) -> float:
    frame = pd.DataFrame({"year": tools.YEARS, "project": project})
    ext = external.copy(); ext["year"] = ext.year.astype(int)
    merged = frame.merge(ext, on="year")
    modern = merged.year.between(modern_start, modern_end)
    merged["p"] = merged.project - merged.loc[modern, "project"].mean()
    merged["e"] = merged.external - merged.loc[modern, "external"].mean()
    errors = []
    for start in range(1550, 1900, 50):
        sub = merged.loc[merged.year.between(start, start + 49)]
        if len(sub) >= 35:
            errors.append(float(sub.p.mean() - sub.e.mean()))
    return float(np.mean(errors))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {
        "public": tools.PUBLIC, "hma_mask": HMA_MASK, "assignment": ASSIGNMENT, "support": SUPPORT,
        "chongce": tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "haizi": tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": tools.RECENT / "Temperature reconstruction.xlsx",
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx", "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}
    hma_mask = np.load(HMA_MASK).astype(bool)
    assignment = pd.read_csv(ASSIGNMENT); support = pd.read_csv(SUPPORT).sort_values("o2region")
    region_names = support.o2region.tolist(); masks = {}
    for row in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask); cells = assignment[assignment.o2region.eq(row.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True; masks[row.o2region] = mask

    with h5py.File(tools.PUBLIC, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float)
        project = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
        families = {
            "local_enkf": tools.finite_mean([tools.allowed_field(handle["tas_rec_ref_HGAO_c05d"]),tools.allowed_field(handle["tas_rec_ref_O_c05d"])]),
            "lmr": tools.allowed_field(handle["tas_rec_ref_LMR_c05d"]),
            "neukom": tools.finite_mean([tools.allowed_field(handle[name]) for name in ["tas_rec_ref_Nk_AM_c05d","tas_rec_ref_Nk_CCA_c05d","tas_rec_ref_Nk_CPS_c05d","tas_rec_ref_Nk_DA_c05d","tas_rec_ref_Nk_GraphEM_c05d","tas_rec_ref_Nk_PCR_c05d"]]),
            "phyda": tools.allowed_field(handle["tas_rec_ref_PHYDA_c05d"]),
        }

    chongce = tools.read_xlsx(paths["chongce"],1,3,6,1524,2012)
    haizi = tools.read_xlsx(paths["haizi"],1,3,2,1115,2017)
    west_project = tools.local_box(project,lat,lon,(80.75,81.75,34.75,35.75))
    east_project = tools.local_box(project,lat,lon,(99.75,101.25,28.75,30.25))
    west_bias = aligned_window_bias(west_project,chongce,1940,2000)
    east_bias = aligned_window_bias(east_project,haizi,1959,2000)
    west_correction = -west_bias; east_correction = -east_bias

    fraction = np.clip((lon - WEST_LON) / (EAST_LON - WEST_LON),0,1)
    spatial = west_correction + fraction * (east_correction - west_correction)
    spatial_field = np.broadcast_to(spatial[None,:],project.shape[1:])
    time_weight = np.zeros(len(tools.YEARS),dtype=float)
    time_weight[(tools.YEARS>=1500)&(tools.YEARS<=1899)] = 1.0
    ramp = (tools.YEARS>=1900)&(tools.YEARS<1940); time_weight[ramp] = (1940-tools.YEARS[ramp])/40.0
    correction = time_weight[:,None,None] * spatial_field[None,:,:]
    correction[:,~hma_mask] = 0.0
    candidate = project + correction

    project_series = tools.regional_series(project,masks,lat)
    candidate_series = tools.regional_series(candidate,masks,lat)
    family_series = {name:tools.regional_series(field,masks,lat) for name,field in families.items()}
    family_rows = []
    for name in tools.FAMILY_ORDER:
        p = tools.rmse(tools.window_vector(project_series,region_names),tools.window_vector(family_series[name],region_names))
        c = tools.rmse(tools.window_vector(candidate_series,region_names),tools.window_vector(family_series[name],region_names))
        family_rows.append({"family":name,"project_rmse":p,"candidate_rmse":c,"improvement":1-c/p})
    family_table = pd.DataFrame(family_rows); family_table.to_csv(OUTPUT/"reference_family_validation.csv",index=False)

    jion = tools.read_xlsx(paths["jionglacuo"],1,2,1,1210,2023)
    jiang = tools.read_xlsx(paths["jiang"],2,7,12,86,2013.1)
    local = {
        "Jionglacuo_project":tools.local_box(project,lat,lon,(93.5,95.5,30,32)),
        "Jionglacuo_candidate":tools.local_box(candidate,lat,lon,(93.5,95.5,30,32)),
        "Jiang_project":tools.local_box(project,lat,lon,(90.25,91.25,31,32)),
        "Jiang_candidate":tools.local_box(candidate,lat,lon,(90.25,91.25,31,32)),
    }
    jion_metric = tools.regular_record_metric(local["Jionglacuo_project"],local["Jionglacuo_candidate"],jion,1970,2000)
    jiang_metric = tools.irregular_metric(local["Jiang_project"],local["Jiang_candidate"],jiang,1940,2000,1550)
    pd.DataFrame([{"record":"Jionglacuo_JJAS",**jion_metric},{"record":"Jiang_summer",**jiang_metric}]).to_csv(OUTPUT/"heldout_seasonal_validation.csv",index=False)

    modern = (tools.YEARS>=1940)&(tools.YEARS<=2000)
    modern_closure = float(np.nanmax(np.abs(candidate[modern]-project[modern])))
    hma_spatial = np.where(hma_mask,spatial_field,np.nan)
    zonal_steps = np.abs(np.diff(hma_spatial,axis=1)); max_step = float(np.nanmax(zonal_steps))
    max_abs = float(np.nanmax(np.abs(hma_spatial)))
    improvements = family_table.improvement.to_numpy(float)
    gates = {
        "all_four_reference_families_improve":bool(np.all(improvements>0)),
        "median_reference_family_improvement_at_least_20pct":bool(np.median(improvements)>=.20),
        "jionglacuo_jjas_improves_at_least_10pct":bool(jion_metric["improvement"]>=.10),
        "jiang_summer_improves_at_least_10pct":bool(jiang_metric["improvement"]>=.10),
        "modern_closure_at_most_1e12":bool(modern_closure<=1e-12),
        "maximum_adjacent_zonal_step_at_most_0p10C":bool(max_step<=.10),
        "maximum_absolute_correction_at_most_1p5C":bool(max_abs<=1.5),
    }
    passes = bool(all(gates.values()))
    outpath = OUTPUT/"temperature_two_anchor_mean_bias_candidate_1500_2000.nc"
    selected = tools.YEARS>=1500
    with h5py.File(outpath,"w") as out:
        out.create_dataset("time",data=tools.YEARS[selected]); out.create_dataset("lat",data=lat); out.create_dataset("lon",data=lon)
        out.create_dataset("tas_candidate",data=candidate[selected].astype("f4"),compression="gzip",compression_opts=4)
        out.create_dataset("tas_correction",data=correction[selected].astype("f4"),compression="gzip",compression_opts=4)
        out.attrs["status"] = "provisional_validated_diagnostic" if passes else "diagnostic_only"
        out.attrs["west_anchor_correction_degC"] = west_correction; out.attrs["east_anchor_correction_degC"] = east_correction

    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    im=axes[0].imshow(hma_spatial,origin="upper",cmap="RdBu_r",vmin=-1.5,vmax=1.5);axes[0].set_title("LIA annual mean correction (degC)");fig.colorbar(im,ax=axes[0])
    axes[1].bar(family_table.family,100*family_table.improvement);axes[1].axhline(20,color="#111",ls="--");axes[1].set_title("Reference-family improvement");axes[1].set_ylabel("Percent")
    axes[2].bar(["Jionglacuo","Jiang"],[100*jion_metric["improvement"],100*jiang_metric["improvement"]]);axes[2].axhline(10,color="#111",ls="--");axes[2].set_title("Held-out seasonal improvement");axes[2].set_ylabel("Percent")
    fig.savefig(OUTPUT/"two_anchor_mean_bias_validation.png",dpi=220);plt.close(fig)

    after={name:base.sha256(path) for name,path in paths.items()}
    (OUTPUT/"source_manifest.json").write_text(json.dumps({"inputs":{name:{"path":str(paths[name]),"sha256_before":before[name],"sha256_after":after[name]} for name in paths},"all_input_hashes_unchanged":before==after},indent=2),encoding="utf-8")
    result={
        "status":"complete","west_chongce_correction_degC":west_correction,"east_haizi_correction_degC":east_correction,
        "median_reference_family_improvement":float(np.median(improvements)),"jionglacuo_improvement":float(jion_metric["improvement"]),"jiang_improvement":float(jiang_metric["improvement"]),
        "modern_closure_degC":modern_closure,"maximum_adjacent_zonal_step_degC":max_step,"maximum_absolute_correction_degC":max_abs,
        "gates":gates,"passes_provisional_gates":passes,"candidate_status":"provisional_validated_diagnostic" if passes else "diagnostic_only","candidate_file":str(outpath),
        "production_release_authorized":False,"locked_1921_1939_accessed":False,"phase_1901_1920_used_for_fitting_or_validation":False,"original_sources_modified":False,
    }
    (OUTPUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (OUTPUT/"FINAL_RESULT.md").write_text("# Two-Anchor Annual Mean-Bias Field\n\n"+f"- West correction: `{west_correction:+.3f} degC`; east correction: `{east_correction:+.3f} degC`.\n"+f"- Median reference-family improvement: `{100*np.median(improvements):.2f}%`.\n"+f"- Held-out Jionglacuo/Jiang improvements: `{100*jion_metric['improvement']:.2f}%` / `{100*jiang_metric['improvement']:.2f}%`.\n"+f"- Provisional gates pass: `{str(passes).lower()}`.\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
