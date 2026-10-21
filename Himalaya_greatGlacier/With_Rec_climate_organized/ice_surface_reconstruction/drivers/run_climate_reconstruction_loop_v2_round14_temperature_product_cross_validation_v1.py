"""Cross-validate Berkeley Earth and CRU temperature PSMs."""

from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import h5py
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS=[(1940,1951),(1952,1963),(1964,1975),(1976,1987),(1988,2000)]
PRODUCTS=("Berkeley","CRU"); SEASONS={"ANN":0,"JJA":1,"AMJ":2,"APR":3}


def root()->Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path:Path)->dict[str,object]:
    stat=path.stat();digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def bootstrap(values:np.ndarray,seed:int,draws:int=5000)->tuple[float,float]:
    values=values[np.isfinite(values)];rng=np.random.default_rng(seed);result=np.empty(draws)
    for draw in range(draws):result[draw]=np.median(values[rng.integers(0,len(values),size=len(values))])
    return float(np.percentile(result,2.5)),float(np.percentile(result,97.5))


def sign_agrees(a:float,b:float)->float:
    if not np.isfinite(a) or not np.isfinite(b) or a==0 or b==0:return np.nan
    return float(np.sign(a)==np.sign(b))


def write_csv(path:Path,rows:list[dict[str,object]])->None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def slope(response:np.ndarray,predictor:np.ndarray,mask:np.ndarray)->float:
    valid=mask&np.isfinite(response)&np.isfinite(predictor)
    if valid.sum()<10:return np.nan
    return float(np.linalg.lstsq(np.column_stack([np.ones(valid.sum()),predictor[valid]]),response[valid],rcond=None)[0][1])


def main()->int:
    project=root();drivers=Path(__file__).resolve().parent;base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output=base/"round_14_temperature_product_cross_validation_v1"
    paths={"export":output/"temperature_products_v1.mat","contract":output/"ROUND_CONTRACT.md","network":base/"round_13_new_lmr_temperature_network_validation_v2/new_lmr_temperature_network_v1.mat","manifest":base/"round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv","round13":base/"round_13_new_lmr_temperature_network_validation_v2/summary.json","matlab_driver":drivers/"export_round14_temperature_products_v1.m","analysis_driver":Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():raise FileNotFoundError(path)
    if (output/"summary.json").exists():raise FileExistsError(output/"summary.json")
    before={name:state(path) for name,path in paths.items()};manifest=pd.read_csv(paths["manifest"])
    with h5py.File(paths["export"],"r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel();observations=np.asarray(handle["observations"],dtype=float).T;temperature=np.asarray(handle["temperature"],dtype=float).transpose(3,2,1,0)
    if observations.shape!=(61,214) or temperature.shape!=(61,214,4,2) or not np.array_equal(years,np.arange(1940,2001)):raise RuntimeError("array-contract-failed")
    if np.any((years>=1921)&(years<=1939)):raise RuntimeError("locked-period-accessed")
    predictors=np.full((61,214,2),np.nan)
    for record in range(214):predictors[:,record,:]=temperature[:,record,SEASONS[str(manifest.iloc[record].season_code)],:]
    early=(years>=1940)&(years<=1969);late=(years>=1970)&(years<=2000);all_years=np.ones(years.shape,dtype=bool)
    fold_rows=[];stability_rows=[];target_correlations=[]
    for record in range(214):
        y=observations[:,record]
        valid_targets=np.isfinite(predictors[:,record,0])&np.isfinite(predictors[:,record,1])
        target_correlations.append(float(np.corrcoef(predictors[valid_targets,record,0],predictors[valid_targets,record,1])[0,1]))
        full_slopes=[]
        for fit_index,fit_product in enumerate(PRODUCTS):
            xfit=predictors[:,record,fit_index];full=slope(y,xfit,all_years);first=slope(y,xfit,early);second=slope(y,xfit,late);full_slopes.append(full)
            stability_rows.append({"network_index_1based":record+1,"formal_identity":manifest.iloc[record].formal_identity,"fit_product":fit_product,"full_slope":full,"early_slope":first,"late_slope":second,"early_late_sign_agreement":sign_agrees(first,second),"positive_direction_agreement":sign_agrees(full,1.0)})
            for fold,(start,end) in enumerate(BLOCKS,start=1):
                test=(years>=start)&(years<=end);train=~test;valid_train=train&np.isfinite(y)&np.isfinite(xfit)
                if valid_train.sum()<20:continue
                coef=np.linalg.lstsq(np.column_stack([np.ones(valid_train.sum()),xfit[valid_train]]),y[valid_train],rcond=None)[0];mean=float(np.mean(y[valid_train]))
                for eval_index,eval_product in enumerate(PRODUCTS):
                    xeval=predictors[:,record,eval_index];valid_test=test&np.isfinite(y)&np.isfinite(xeval)
                    if valid_test.sum()<3:continue
                    prediction=coef[0]+coef[1]*xeval[valid_test];rmse_model=float(np.sqrt(np.mean((y[valid_test]-prediction)**2)));rmse_intercept=float(np.sqrt(np.mean((y[valid_test]-mean)**2)))
                    fold_rows.append({"network_index_1based":record+1,"formal_identity":manifest.iloc[record].formal_identity,"fold":fold,"fit_product":fit_product,"eval_product":eval_product,"test_count":int(valid_test.sum()),"slope":float(coef[1]),"model_rmse":rmse_model,"intercept_rmse":rmse_intercept,"rmse_improvement":float(1-rmse_model/rmse_intercept)})
    write_csv(output/"fold_temperature_product_skill.csv",fold_rows);write_csv(output/"record_temperature_product_stability.csv",stability_rows)
    folds=pd.DataFrame(fold_rows);stability=pd.DataFrame(stability_rows);record_rows=[]
    for (index,identity,fit_product,eval_product),group in folds.groupby(["network_index_1based","formal_identity","fit_product","eval_product"]):record_rows.append({"network_index_1based":int(index),"formal_identity":identity,"fit_product":fit_product,"eval_product":eval_product,"finite_fold_count":int(len(group)),"median_rmse_improvement":float(group.rmse_improvement.median())})
    write_csv(output/"record_temperature_product_skill.csv",record_rows);records=pd.DataFrame(record_rows)
    path_summary={};seed=1401
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key=f"{fit_product}_fit_{eval_product}_eval";selected=records[(records.fit_product==fit_product)&(records.eval_product==eval_product)].sort_values("network_index_1based")
            values=selected.median_rmse_improvement.to_numpy(float);ci=bootstrap(values,seed);seed+=1
            path_summary[key]={"median_blocked_rmse_improvement":float(np.median(values)),"bootstrap_95_lower":ci[0],"bootstrap_95_upper":ci[1],"positive_identity_fraction":float(np.mean(values>0))}
    product_gates={}
    for product in PRODUCTS:
        stable=stability[stability.fit_product==product];within=path_summary[f"{product}_fit_{product}_eval"]
        metrics={"early_late_sign_agreement":float(stable.early_late_sign_agreement.mean()),"positive_direction_agreement":float(stable.positive_direction_agreement.mean())}
        gates={"median_at_least_1pct":within["median_blocked_rmse_improvement"]>=.01,"bootstrap_lower_positive":within["bootstrap_95_lower"]>0,"positive_identity_fraction_over_50pct":within["positive_identity_fraction"]>.50,"early_late_sign_agreement_at_least_70pct":metrics["early_late_sign_agreement"]>=.70,"positive_direction_agreement_at_least_70pct":metrics["positive_direction_agreement"]>=.70}
        product_gates[product]={"metrics":metrics,"gates":gates,"passes":bool(all(gates.values()))}
    pivot=folds.pivot_table(index=["network_index_1based","fold","eval_product"],columns="fit_product",values="slope",aggfunc="first").reset_index();paired=pivot[np.isfinite(pivot.Berkeley)&np.isfinite(pivot.CRU)&(pivot.Berkeley!=0)&(pivot.CRU!=0)];coefficient_sign=float(np.mean(np.sign(paired.Berkeley)==np.sign(paired.CRU)))
    rescued=any(item["passes"] for item in product_gates.values());summary={"status":"complete","locked_1921_1939_accessed":False,"identity_count":214,"median_berkeley_cru_target_correlation":float(np.nanmedian(target_correlations)),"paired_fit_coefficient_sign_agreement":coefficient_sign,"product_paths":path_summary,"within_product_rescue":product_gates,"any_temperature_product_rescues_local_linear_psm":rescued,"decision":"permit_product_specific_temperature_operator_test" if rescued else "close_local_linear_temperature_psm_family"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True);keys=list(path_summary);values=[records[(records.fit_product==key.split("_fit_")[0])&(records.eval_product==key.split("_fit_")[1].replace("_eval",""))].median_rmse_improvement.to_numpy(float)*100 for key in keys]
    axes[0].boxplot(values,labels=[key.replace("_fit_","->").replace("_eval","") for key in keys],showfliers=False);axes[0].axhline(0,color="#444444");axes[0].axhline(1,color="#9b2226",linestyle="--");axes[0].set(title="Blocked temperature-product skill",ylabel="RMSE improvement (%)")
    axes[1].scatter(paired.Berkeley,paired.CRU,s=18,alpha=.65,color="#005f73");axes[1].axhline(0,color="#777777");axes[1].axvline(0,color="#777777");axes[1].set(title="Paired fit slopes",xlabel="Berkeley",ylabel="CRU");fig.suptitle("Temperature product cross-validation",fontsize=14);fig.savefig(output/"temperature_product_cross_validation.png",dpi=180);plt.close(fig)
    lines=["# Temperature Product Cross-Validation","",f"- Median Berkeley-CRU target correlation: `{summary['median_berkeley_cru_target_correlation']:.3f}`.",f"- Paired coefficient-sign agreement: `{coefficient_sign:.3%}`."]
    for key,item in path_summary.items():lines.append(f"- {key}: `{item['median_blocked_rmse_improvement']:.3%}` (95% `{item['bootstrap_95_lower']:.3%}` to `{item['bootstrap_95_upper']:.3%}`), positive `{item['positive_identity_fraction']:.3%}`.")
    lines.extend(["",f"Any product rescues the family: `{str(rescued).lower()}`.","",f"Decision: `{summary['decision']}`."]);(output/"FINAL_RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    after={name:state(path) for name,path in paths.items()};(output/"source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after,"bootstrap_draws":5000},indent=2),encoding="utf-8")
    if before!=after:raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
