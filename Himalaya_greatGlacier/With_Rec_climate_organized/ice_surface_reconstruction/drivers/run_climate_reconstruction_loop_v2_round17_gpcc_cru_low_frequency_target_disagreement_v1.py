"""Audit GPCC/CRU target disagreement at expanded moisture sites."""

from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import h5py
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS=[(1945,1954),(1965,1974),(1985,1995)];HALF=5


def root()->Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists():return parent
    raise RuntimeError("project-root-not-found")


def state(path:Path)->dict[str,object]:
    stat=path.stat();digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def smooth(values:np.ndarray)->np.ndarray:
    result=np.full(values.shape,np.nan)
    for index in range(HALF,len(values)-HALF):result[index]=float(np.mean(values[index-HALF:index+HALF+1]))
    return result


def trend(years:np.ndarray,values:np.ndarray,mask:np.ndarray)->float:
    valid=mask&np.isfinite(values)
    return float(np.polyfit(years[valid],values[valid],1)[0]*100)


def main()->int:
    project=root();base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2";output=base/"round_17_gpcc_cru_low_frequency_target_disagreement_v1"
    paths={"export":base/"round_12_expanded_moisture_nested_cross_product_v1/expanded_moisture_jja_climate_v1.mat","manifest":base/"round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv","round16_curves":base/"round_16_purged_low_frequency_common_mode_v1/low_frequency_oof_curves.csv","contract":output/"ROUND_CONTRACT.md","driver":Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():raise FileNotFoundError(path)
    if (output/"summary.json").exists():raise FileExistsError(output/"summary.json")
    before={name:state(path) for name,path in paths.items()};manifest=pd.read_csv(paths["manifest"])
    with h5py.File(paths["export"],"r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel();locations=np.asarray(handle["locations"],dtype=float).T;gpcc=np.asarray(handle["precipitation_gpcc_jja"],dtype=float).T;cru=np.asarray(handle["precipitation_cru_jja"],dtype=float).T
    if not np.array_equal(years,np.arange(1940,2001)) or np.any((years>=1921)&(years<=1939)):raise RuntimeError("year-contract-failed")
    unique=[];seen=set()
    for index,(lat,lon) in enumerate(locations):
        key=(round(float(lat),6),round(float(lon),6))
        if key not in seen:seen.add(key);unique.append(index)
    if len(unique)!=21:raise RuntimeError(f"unique-site-count-failed:{len(unique)}")
    gpcc_sites=gpcc[:,unique];cru_sites=cru[:,unique];gpcc_region=np.nanmean(gpcc_sites,axis=1);cru_region=np.nanmean(cru_sites,axis=1);gpcc_low=smooth(gpcc_region);cru_low=smooth(cru_region)
    annual_valid=np.isfinite(gpcc_region)&np.isfinite(cru_region);low_valid=np.isfinite(gpcc_low)&np.isfinite(cru_low)
    annual_correlation=float(np.corrcoef(gpcc_region[annual_valid],cru_region[annual_valid])[0,1]);low_correlation=float(np.corrcoef(gpcc_low[low_valid],cru_low[low_valid])[0,1])
    site_rows=[]
    for local,index in enumerate(unique):
        left=gpcc_sites[:,local];right=cru_sites[:,local];left_low=smooth(left);right_low=smooth(right);valid=np.isfinite(left)&np.isfinite(right);valid_low=np.isfinite(left_low)&np.isfinite(right_low)
        site_rows.append({"site_index_1based":local+1,"latitude":locations[index,0],"longitude":locations[index,1],"annual_correlation":float(np.corrcoef(left[valid],right[valid])[0,1]),"low_frequency_correlation":float(np.corrcoef(left_low[valid_low],right_low[valid_low])[0,1])})
    with (output/"site_target_agreement.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(site_rows[0]));writer.writeheader();writer.writerows(site_rows)
    block_rows=[];trend_sign_disagreement=0
    for start,end in BLOCKS:
        mask=(years>=start)&(years<=end);left=trend(years,gpcc_low,mask);right=trend(years,cru_low,mask);same=bool(np.sign(left)==np.sign(right));trend_sign_disagreement+=int(not same);block_rows.append({"start_year":start,"end_year":end,"gpcc_trend_mm_day_per_century":left,"cru_trend_mm_day_per_century":right,"trend_sign_agreement":same})
    with (output/"block_low_frequency_trends.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(block_rows[0]));writer.writeheader();writer.writerows(block_rows)
    median_site_low=float(np.nanmedian([row["low_frequency_correlation"] for row in site_rows]));product_limited=bool(low_correlation<.50 or median_site_low<.50 or trend_sign_disagreement>=2)
    summary={"status":"complete","locked_1921_1939_accessed":False,"unique_site_count":21,"regional_annual_correlation":annual_correlation,"regional_11year_correlation":low_correlation,"median_site_annual_correlation":float(np.nanmedian([row["annual_correlation"] for row in site_rows])),"median_site_11year_correlation":median_site_low,"block_trend_sign_disagreement_count":trend_sign_disagreement,"block_trends":block_rows,"materially_product_limited":product_limited,"decision":"advance_raw_gpcc_cru_station_support_audit" if product_limited else "classify_gpcc_only_pass_as_proxy_model_instability"};(output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    pd.DataFrame({"year":years,"gpcc_jja":gpcc_region,"cru_jja":cru_region,"gpcc_jja_11year":gpcc_low,"cru_jja_11year":cru_low}).to_csv(output/"regional_target_curves.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True);axes[0].plot(years,gpcc_region,label="GPCC",color="#005f73");axes[0].plot(years,cru_region,label="CRU",color="#ee9b00");axes[0].set(title="Annual unique-site JJA target",xlabel="Year",ylabel="Anomaly (mm/day)");axes[0].legend(frameon=False)
    axes[1].plot(years,gpcc_low,label="GPCC 11-year",color="#005f73",linewidth=2);axes[1].plot(years,cru_low,label="CRU 11-year",color="#ee9b00",linewidth=2);axes[1].set(title="Low-frequency target",xlabel="Center year",ylabel="Anomaly (mm/day)");axes[1].legend(frameon=False);fig.suptitle("GPCC-CRU target disagreement",fontsize=14);fig.savefig(output/"gpcc_cru_low_frequency_target_disagreement.png",dpi=180);plt.close(fig)
    (output/"FINAL_RESULT.md").write_text("# GPCC-CRU Low-Frequency Target Disagreement\n\n"+f"- Regional annual / 11-year correlation: `{annual_correlation:.3f}` / `{low_correlation:.3f}`.\n- Median site annual / 11-year correlation: `{summary['median_site_annual_correlation']:.3f}` / `{median_site_low:.3f}`.\n- Block trend-sign disagreements: `{trend_sign_disagreement}/3`.\n- Materially product-limited: `{str(product_limited).lower()}`.\n\nDecision: `{summary['decision']}`.\n",encoding="utf-8")
    after={name:state(path) for name,path in paths.items()};(output/"source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after},indent=2),encoding="utf-8")
    if before!=after:raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
