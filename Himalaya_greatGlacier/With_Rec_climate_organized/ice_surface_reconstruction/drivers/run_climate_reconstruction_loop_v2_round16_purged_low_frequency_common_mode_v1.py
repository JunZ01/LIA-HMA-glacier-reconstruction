"""Purged 11-year proxy common-mode validation."""

from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import h5py
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS=[(1940,1959),(1960,1979),(1980,2000)];HALF=5


def root()->Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists():return parent
    raise RuntimeError("project-root-not-found")


def state(path:Path)->dict[str,object]:
    stat=path.stat();digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def unique_site_mean(field:np.ndarray,locations:np.ndarray,indices:np.ndarray)->np.ndarray:
    chosen=[];seen=set()
    for index in indices:
        key=(round(float(locations[index,0]),6),round(float(locations[index,1]),6))
        if key not in seen:seen.add(key);chosen.append(int(index))
    return np.nanmean(field[:,chosen],axis=1)


def smooth(values:np.ndarray)->np.ndarray:
    result=np.full(values.shape,np.nan)
    for index in range(HALF,len(values)-HALF):result[index]=float(np.mean(values[index-HALF:index+HALF+1]))
    return result


def rmse(y:np.ndarray,p:np.ndarray)->float:return float(np.sqrt(np.mean((y-p)**2)))


def purged_oof(years:np.ndarray,X:np.ndarray,target:np.ndarray)->tuple[np.ndarray,list[dict[str,object]]]:
    prediction=np.full(target.shape,np.nan);rows=[]
    for fold,(start,end) in enumerate(BLOCKS,start=1):
        test_centers=(years>=start+HALF)&(years<=end-HALF)
        train_centers=((years+HALF<start)|(years-HALF>end))&(np.arange(len(years))>=HALF)&(np.arange(len(years))<len(years)-HALF)
        support=np.zeros(len(years),dtype=bool)
        for center in np.flatnonzero(train_centers):support[center-HALF:center+HALF+1]=True
        counts=np.sum(np.isfinite(X[support]),axis=0);means=np.nanmean(X[support],axis=0);stds=np.nanstd(X[support],axis=0)
        keep=(counts>=10)&np.isfinite(means)&np.isfinite(stds)&(stds>0)
        if keep.sum()<5:raise RuntimeError(f"insufficient-features:{fold}:{keep.sum()}")
        standardized=(X[:,keep]-means[keep])/stds[keep];standardized=np.where(np.isfinite(standardized),standardized,0.0);composite=np.mean(standardized,axis=1)
        low_composite=smooth(composite);low_target=smooth(target)
        train=train_centers&np.isfinite(low_target)&np.isfinite(low_composite);test=test_centers&np.isfinite(low_target)&np.isfinite(low_composite)
        coefficient=np.linalg.lstsq(np.column_stack([np.ones(train.sum()),low_composite[train]]),low_target[train],rcond=None)[0];pred=coefficient[0]+coefficient[1]*low_composite[test];prediction[test]=pred
        baseline=np.full(test.sum(),float(np.mean(low_target[train])));model_rmse=rmse(low_target[test],pred);baseline_rmse=rmse(low_target[test],baseline)
        rows.append({"outer_fold":fold,"start_year":start,"end_year":end,"training_center_count":int(train.sum()),"test_center_count":int(test.sum()),"retained_proxy_count":int(keep.sum()),"composite_slope":float(coefficient[1]),"model_rmse":model_rmse,"baseline_rmse":baseline_rmse,"rmse_improvement":float(1-model_rmse/baseline_rmse)})
    return prediction,rows


def write_csv(path:Path,rows:list[dict[str,object]])->None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main()->int:
    project=root();base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2";output=base/"round_16_purged_low_frequency_common_mode_v1"
    paths={"temperature_manifest":base/"round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv","temperature_products":base/"round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat","moisture_export":base/"round_12_expanded_moisture_nested_cross_product_v1/expanded_moisture_jja_climate_v1.mat","moisture_manifest":base/"round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv","contract":output/"ROUND_CONTRACT.md","driver":Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():raise FileNotFoundError(path)
    if (output/"summary.json").exists():raise FileExistsError(output/"summary.json")
    before={name:state(path) for name,path in paths.items()};tmeta=pd.read_csv(paths["temperature_manifest"])
    with h5py.File(paths["temperature_products"],"r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel();Xt=np.asarray(handle["observations"],dtype=float).T;tloc=np.asarray(handle["locations"],dtype=float).T;tfield=np.asarray(handle["temperature"],dtype=float).transpose(3,2,1,0)
    with h5py.File(paths["moisture_export"],"r") as handle:
        years_p=np.asarray(handle["years"],dtype=int).ravel();Xp=np.asarray(handle["observations"],dtype=float).T;ploc=np.asarray(handle["locations"],dtype=float).T;gpcc=np.asarray(handle["precipitation_gpcc_jja"],dtype=float).T;cru_p=np.asarray(handle["precipitation_cru_jja"],dtype=float).T
    if not np.array_equal(years,np.arange(1940,2001)) or not np.array_equal(years,years_p) or np.any((years>=1921)&(years<=1939)):raise RuntimeError("year-contract-failed")
    jja=np.flatnonzero(tmeta.season_code.to_numpy()=="JJA");ann=np.flatnonzero(tmeta.season_code.to_numpy()=="ANN");all_p=np.arange(22)
    branches={"temperature_JJA_Berkeley":(Xt[:,jja],unique_site_mean(tfield[:,:,1,0],tloc,jja),"temperature_JJA"),"temperature_JJA_CRU":(Xt[:,jja],unique_site_mean(tfield[:,:,1,1],tloc,jja),"temperature_JJA"),"temperature_ANN_Berkeley":(Xt[:,ann],unique_site_mean(tfield[:,:,0,0],tloc,ann),"temperature_ANN"),"temperature_ANN_CRU":(Xt[:,ann],unique_site_mean(tfield[:,:,0,1],tloc,ann),"temperature_ANN"),"precipitation_JJA_GPCC":(Xp,unique_site_mean(gpcc,ploc,all_p),"precipitation_JJA"),"precipitation_JJA_CRU":(Xp,unique_site_mean(cru_p,ploc,all_p),"precipitation_JJA")}
    fold_rows=[];curve={"year":years};branch_summary={}
    for name,(X,target,family) in branches.items():
        prediction,rows=purged_oof(years,X,target);low_target=smooth(target);valid=np.isfinite(prediction)&np.isfinite(low_target);baseline=np.full(valid.sum(),float(np.mean(low_target[valid])));improvement=float(1-rmse(low_target[valid],prediction[valid])/rmse(low_target[valid],baseline));correlation=float(np.corrcoef(low_target[valid],prediction[valid])[0,1]);improved=int(sum(row["rmse_improvement"]>0 for row in rows));gates={"pooled_rmse_improvement_at_least_5pct":improvement>=.05,"oof_correlation_at_least_0p30":correlation>=.30,"at_least_two_improved_blocks":improved>=2};branch_summary[name]={"family":family,"predictor_count":int(X.shape[1]),"pooled_oof_rmse_improvement":improvement,"oof_correlation":correlation,"improved_block_count":improved,"gates":gates,"passes":bool(all(gates.values()))};curve[f"{name}_lowpass_target"]=low_target;curve[f"{name}_oof_prediction"]=prediction
        for row in rows:fold_rows.append({"branch":name,**row})
    write_csv(output/"purged_block_skill.csv",fold_rows);pd.DataFrame(curve).to_csv(output/"low_frequency_oof_curves.csv",index=False)
    families={item["family"] for item in branch_summary.values()};family_pass={family:bool(all(item["passes"] for item in branch_summary.values() if item["family"]==family)) for family in families};advancing=[family for family,value in family_pass.items() if value]
    summary={"status":"complete","locked_1921_1939_accessed":False,"filter_window_years":11,"outer_blocks":BLOCKS,"branch_summary":branch_summary,"family_both_product_pass":family_pass,"advancing_families":advancing,"decision":"permit_low_frequency_spatial_field_proof" if advancing else "purged_low_frequency_common_mode_failed"};(output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(3,1,figsize=(11.5,9),sharex=True,constrained_layout=True)
    groups=[("temperature_JJA",["temperature_JJA_Berkeley","temperature_JJA_CRU"]),("temperature_ANN",["temperature_ANN_Berkeley","temperature_ANN_CRU"]),("precipitation_JJA",["precipitation_JJA_GPCC","precipitation_JJA_CRU"])]
    for axis,(family,names) in zip(axes,groups):
        for name in names:axis.plot(years,curve[f"{name}_lowpass_target"],linewidth=1.2,label=f"{name.split('_')[-1]} target");axis.plot(years,curve[f"{name}_oof_prediction"],linestyle="--",linewidth=1.7,label=f"{name.split('_')[-1]} OOF")
        axis.set_title(family);axis.legend(ncol=2,fontsize=8,frameon=False)
    axes[-1].set_xlabel("Center year");fig.suptitle("Purged 11-year proxy common-mode test",fontsize=14);fig.savefig(output/"purged_low_frequency_common_mode.png",dpi=180);plt.close(fig)
    lines=["# Purged Low-Frequency Common-Mode Test",""]
    for name,item in branch_summary.items():lines.append(f"- {name}: improvement `{item['pooled_oof_rmse_improvement']:.3%}`, r `{item['oof_correlation']:.3f}`, improved blocks `{item['improved_block_count']}/3`, pass `{str(item['passes']).lower()}`.")
    lines.extend(["",f"Advancing families: `{json.dumps(advancing)}`.","",f"Decision: `{summary['decision']}`."]);(output/"FINAL_RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    after={name:state(path) for name,path in paths.items()};(output/"source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after},indent=2),encoding="utf-8")
    if before!=after:raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
