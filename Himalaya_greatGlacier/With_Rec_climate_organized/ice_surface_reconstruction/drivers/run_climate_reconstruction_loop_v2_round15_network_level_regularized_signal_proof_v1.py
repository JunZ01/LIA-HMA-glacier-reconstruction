"""Nested ridge proof for regional same-season proxy-network signals."""

from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import h5py
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS=[(1940,1951),(1952,1963),(1964,1975),(1976,1987),(1988,2000)]
ALPHAS=[0.01,0.1,1.0,10.0,100.0,1000.0]


def root()->Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists():return parent
    raise RuntimeError("project-root-not-found")


def state(path:Path)->dict[str,object]:
    stat=path.stat();digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def model(alpha:float)->Pipeline:
    return Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler()),("ridge",Ridge(alpha=alpha))])


def rmse(y:np.ndarray,p:np.ndarray)->float:return float(np.sqrt(np.mean((y-p)**2)))


def unique_site_mean(field:np.ndarray,locations:np.ndarray,indices:np.ndarray)->np.ndarray:
    chosen=[];seen=set()
    for index in indices:
        key=(round(float(locations[index,0]),6),round(float(locations[index,1]),6))
        if key not in seen:seen.add(key);chosen.append(int(index))
    return np.nanmean(field[:,chosen],axis=1)


def nested_oof(years:np.ndarray,X:np.ndarray,y:np.ndarray)->tuple[np.ndarray,list[dict[str,object]]]:
    predictions=np.full(y.shape,np.nan);rows=[]
    for outer,(start,end) in enumerate(BLOCKS,start=1):
        test=(years>=start)&(years<=end)&np.isfinite(y);train=~((years>=start)&(years<=end))&np.isfinite(y)
        alpha_scores=[]
        for alpha in ALPHAS:
            scores=[]
            for inner_start,inner_end in BLOCKS:
                inner_test=train&(years>=inner_start)&(years<=inner_end);inner_train=train&~inner_test
                if inner_test.sum()<3 or inner_train.sum()<20:continue
                fitted=model(alpha).fit(X[inner_train],y[inner_train]);scores.append(rmse(y[inner_test],fitted.predict(X[inner_test])))
            alpha_scores.append(float(np.mean(scores)) if scores else np.inf)
        best=float(ALPHAS[int(np.argmin(alpha_scores))]);fitted=model(best).fit(X[train],y[train]);prediction=fitted.predict(X[test]);predictions[test]=prediction
        baseline=np.full(test.sum(),float(np.mean(y[train])));model_rmse=rmse(y[test],prediction);baseline_rmse=rmse(y[test],baseline)
        rows.append({"outer_fold":outer,"start_year":start,"end_year":end,"selected_alpha":best,"inner_mean_rmse":float(np.min(alpha_scores)),"test_count":int(test.sum()),"model_rmse":model_rmse,"baseline_rmse":baseline_rmse,"rmse_improvement":float(1-model_rmse/baseline_rmse)})
    return predictions,rows


def write_csv(path:Path,rows:list[dict[str,object]])->None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main()->int:
    project=root();base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2";output=base/"round_15_network_level_regularized_signal_proof_v1"
    paths={"temperature_network":base/"round_13_new_lmr_temperature_network_validation_v2/new_lmr_temperature_network_v1.mat","temperature_manifest":base/"round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv","temperature_products":base/"round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat","moisture_export":base/"round_12_expanded_moisture_nested_cross_product_v1/expanded_moisture_jja_climate_v1.mat","moisture_manifest":base/"round_12_expanded_moisture_nested_cross_product_v1/network_manifest.csv","contract":output/"ROUND_CONTRACT.md","driver":Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():raise FileNotFoundError(path)
    if (output/"summary.json").exists():raise FileExistsError(output/"summary.json")
    before={name:state(path) for name,path in paths.items()};tmeta=pd.read_csv(paths["temperature_manifest"]);pmeta=pd.read_csv(paths["moisture_manifest"])
    with h5py.File(paths["temperature_products"],"r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel();Xt=np.asarray(handle["observations"],dtype=float).T;tloc=np.asarray(handle["locations"],dtype=float).T;tfield=np.asarray(handle["temperature"],dtype=float).transpose(3,2,1,0)
    with h5py.File(paths["moisture_export"],"r") as handle:
        years_p=np.asarray(handle["years"],dtype=int).ravel();Xp=np.asarray(handle["observations"],dtype=float).T;ploc=np.asarray(handle["locations"],dtype=float).T;gpcc=np.asarray(handle["precipitation_gpcc_jja"],dtype=float).T;cru_p=np.asarray(handle["precipitation_cru_jja"],dtype=float).T
    if not np.array_equal(years,np.arange(1940,2001)) or not np.array_equal(years,years_p) or np.any((years>=1921)&(years<=1939)):raise RuntimeError("year-contract-failed")
    jja=np.flatnonzero(tmeta.season_code.to_numpy()=="JJA");ann=np.flatnonzero(tmeta.season_code.to_numpy()=="ANN");all_p=np.arange(22)
    branches={
        "temperature_JJA_Berkeley":(Xt[:,jja],unique_site_mean(tfield[:,:,1,0],tloc,jja),"temperature_JJA"),
        "temperature_JJA_CRU":(Xt[:,jja],unique_site_mean(tfield[:,:,1,1],tloc,jja),"temperature_JJA"),
        "temperature_ANN_Berkeley":(Xt[:,ann],unique_site_mean(tfield[:,:,0,0],tloc,ann),"temperature_ANN"),
        "temperature_ANN_CRU":(Xt[:,ann],unique_site_mean(tfield[:,:,0,1],tloc,ann),"temperature_ANN"),
        "precipitation_JJA_GPCC":(Xp,unique_site_mean(gpcc,ploc,all_p),"precipitation_JJA"),
        "precipitation_JJA_CRU":(Xp,unique_site_mean(cru_p,ploc,all_p),"precipitation_JJA"),
    }
    fold_rows=[];curve={"year":years};branch_summary={}
    for name,(X,target,family) in branches.items():
        prediction,rows=nested_oof(years,X,target);valid=np.isfinite(target)&np.isfinite(prediction);baseline=np.full(valid.sum(),float(np.mean(target[valid])));pooled_improvement=float(1-rmse(target[valid],prediction[valid])/rmse(target[valid],baseline));correlation=float(np.corrcoef(target[valid],prediction[valid])[0,1]);positive_folds=int(sum(row["rmse_improvement"]>0 for row in rows));gates={"pooled_rmse_improvement_at_least_5pct":pooled_improvement>=.05,"oof_correlation_at_least_0p30":correlation>=.30,"at_least_three_improved_folds":positive_folds>=3};branch_summary[name]={"family":family,"predictor_count":int(X.shape[1]),"pooled_oof_rmse_improvement":pooled_improvement,"oof_correlation":correlation,"improved_fold_count":positive_folds,"gates":gates,"passes":bool(all(gates.values()))};curve[f"{name}_target"]=target;curve[f"{name}_prediction"]=prediction
        for row in rows:fold_rows.append({"branch":name,**row})
    write_csv(output/"nested_fold_skill.csv",fold_rows);pd.DataFrame(curve).to_csv(output/"annual_oof_regional_signals.csv",index=False)
    family_pass={family:bool(all(branch_summary[name]["passes"] for name in branch_summary if branch_summary[name]["family"]==family)) for family in {item["family"] for item in branch_summary.values()}}
    advancing=[family for family,value in family_pass.items() if value];summary={"status":"complete","locked_1921_1939_accessed":False,"alpha_grid":ALPHAS,"branch_summary":branch_summary,"family_both_product_pass":family_pass,"advancing_families":advancing,"decision":"permit_separate_spatial_field_test_for_advancing_families" if advancing else "regional_regularized_signal_proof_failed"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(3,1,figsize=(11.5,9),sharex=True,constrained_layout=True)
    for axis,(family,names) in zip(axes,[ ("temperature_JJA",["temperature_JJA_Berkeley","temperature_JJA_CRU"]),("temperature_ANN",["temperature_ANN_Berkeley","temperature_ANN_CRU"]),("precipitation_JJA",["precipitation_JJA_GPCC","precipitation_JJA_CRU"]) ]):
        for name in names:axis.plot(years,curve[f"{name}_target"],linewidth=1.2,label=f"{name.split('_')[-1]} target");axis.plot(years,curve[f"{name}_prediction"],linewidth=1.6,linestyle="--",label=f"{name.split('_')[-1]} OOF")
        axis.set_title(family);axis.legend(ncol=2,fontsize=8,frameon=False)
    axes[-1].set_xlabel("Year");fig.suptitle("Nested regional proxy-network signal proof",fontsize=14);fig.savefig(output/"network_level_regularized_signal_proof.png",dpi=180);plt.close(fig)
    lines=["# Network-Level Regularized Signal Proof",""]
    for name,item in branch_summary.items():lines.append(f"- {name}: improvement `{item['pooled_oof_rmse_improvement']:.3%}`, r `{item['oof_correlation']:.3f}`, improved folds `{item['improved_fold_count']}/5`, pass `{str(item['passes']).lower()}`.")
    lines.extend(["",f"Advancing families: `{json.dumps(advancing)}`.","",f"Decision: `{summary['decision']}`."]);(output/"FINAL_RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    after={name:state(path) for name,path in paths.items()};(output/"source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after,"sklearn_pipeline":"SimpleImputer(mean)->StandardScaler->Ridge"},indent=2),encoding="utf-8")
    if before!=after:raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
