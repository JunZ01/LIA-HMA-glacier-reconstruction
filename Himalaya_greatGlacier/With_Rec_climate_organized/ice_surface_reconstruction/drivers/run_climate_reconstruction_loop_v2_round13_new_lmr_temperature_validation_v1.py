"""Validate the new source-native LMR documented-temperature network."""

from __future__ import annotations
import csv, hashlib, importlib.util, json
from pathlib import Path
import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat=path.stat(); digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def load_helper(path: Path):
    spec=importlib.util.spec_from_file_location("round13_round09_helper",path)
    if spec is None or spec.loader is None: raise RuntimeError("helper-load-failed")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def write_csv(path: Path, rows: list[dict[str,object]]) -> None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main() -> int:
    project=root();drivers=Path(__file__).resolve().parent
    base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output=base/"round_13_new_lmr_temperature_network_validation_v2"
    paths={
        "network":output/"new_lmr_temperature_network_v1.mat","manifest":output/"network_manifest.csv",
        "season_export":output/"temperature_seasons_v1.mat","contract":output/"ROUND_CONTRACT.md",
        "preparation_manifest":output/"preparation_source_manifest.json",
        "round09_summary":base/"round_09_temperature_proxy_association_stability_v1/summary.json",
        "helper":drivers/"run_climate_reconstruction_loop_v2_round09_temperature_proxy_association_stability_v1.py",
        "matlab_exporter":drivers/"export_round08_temp_seasons_v2.m","analysis_driver":Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output/"summary.json").exists(): raise FileExistsError(output/"summary.json")
    before={name:state(path) for name,path in paths.items()}; helper=load_helper(paths["helper"])
    manifest=pd.read_csv(paths["manifest"])
    observations=np.asarray(scipy.io.loadmat(paths["network"],variable_names=["observations"])["observations"],dtype=float)
    with h5py.File(paths["season_export"],"r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel()
        seasonal={"ANN":np.asarray(handle["annual"],dtype=float).T,"JJA":np.asarray(handle["jja"],dtype=float).T,"AMJ":np.asarray(handle["amj"],dtype=float).T,"APR":np.asarray(handle["april"],dtype=float).T}
    if len(manifest)!=214 or observations.shape!=(61,214) or not np.array_equal(years,np.arange(1940,2001)): raise RuntimeError("array-contract-failed")
    if np.any((years>=1921)&(years<=1939)): raise RuntimeError("locked-period-accessed")
    early=(years>=1940)&(years<=1969);late=(years>=1970)&(years<=2000)
    rows=[]
    for record in manifest.itertuples(index=False):
        index=int(record.network_index_1based)-1; predictor=seasonal[str(record.season_code)][:,index]; response=observations[:,index]
        full=helper.fit_stats(response,predictor,np.ones(years.shape,dtype=bool)); first=helper.fit_stats(response,predictor,early); second=helper.fit_stats(response,predictor,late); blocked=helper.blocked_stats(response,predictor,years)
        rows.append({
            "network_index_1based":index+1,"formal_identity":record.formal_identity,"season_code":record.season_code,"measurement":record.measurement,
            "full_n":full["n"],"full_slope":full["slope"],"full_correlation":full["correlation"],"full_abs_correlation":abs(full["correlation"]),
            "full_insample_rmse_improvement":full["improvement"],"blocked_n":blocked["n"],"blocked_rmse_improvement":blocked["improvement"],"blocked_prediction_correlation":blocked["correlation"],
            "early_slope":first["slope"],"late_slope":second["slope"],"early_late_slope_sign_agreement":helper.sign_agrees(first["slope"],second["slope"]),
            "full_positive_direction_agreement":helper.sign_agrees(full["slope"],1.0),"finite_fold_slope_count":int(np.isfinite(blocked["slopes"]).sum()),
        })
    write_csv(output/"record_new_temperature_validation.csv",rows);frame=pd.DataFrame(rows)
    group_rows=[]
    for dimension in ("season_code","measurement"):
        for value,group in frame.groupby(dimension):
            if len(group)<10: continue
            group_rows.append({"group_dimension":dimension,"group_value":value,"identity_count":int(len(group)),"median_abs_full_correlation":float(group.full_abs_correlation.median()),"median_insample_improvement":float(group.full_insample_rmse_improvement.median()),"median_blocked_improvement":float(group.blocked_rmse_improvement.median()),"positive_blocked_fraction":float((group.blocked_rmse_improvement>0).mean()),"early_late_sign_agreement":float(group.early_late_slope_sign_agreement.mean()),"positive_direction_agreement":float(group.full_positive_direction_agreement.mean())})
    write_csv(output/"group_new_temperature_validation.csv",group_rows)
    blocked=frame.blocked_rmse_improvement.to_numpy(float); insample=frame.full_insample_rmse_improvement.to_numpy(float); abs_corr=frame.full_abs_correlation.to_numpy(float)
    blocked_ci=helper.bootstrap_median(blocked,1301); median_blocked=float(np.nanmedian(blocked)); positive_fraction=float(np.mean(blocked>0)); early_late=float(frame.early_late_slope_sign_agreement.mean()); direction=float(frame.full_positive_direction_agreement.mean())
    old=json.loads(paths["round09_summary"].read_text(encoding="utf-8"))
    gates={"median_blocked_improvement_at_least_1pct":median_blocked>=0.01,"blocked_bootstrap_lower_positive":blocked_ci[0]>0,"positive_blocked_identity_fraction_over_50pct":positive_fraction>0.50,"early_late_sign_agreement_at_least_70pct":early_late>=0.70,"positive_direction_agreement_at_least_70pct":direction>=0.70}
    passes=bool(all(gates.values()))
    summary={"status":"complete","locked_1921_1939_accessed":False,"identity_count":214,"unique_site_count":int(manifest[["latitude","longitude"]].drop_duplicates().shape[0]),"median_absolute_full_period_correlation":float(np.nanmedian(abs_corr)),"median_full_period_insample_rmse_improvement":float(np.nanmedian(insample)),"median_blocked_rmse_improvement":median_blocked,"median_blocked_rmse_improvement_bootstrap_95":list(blocked_ci),"positive_blocked_identity_fraction":positive_fraction,"early_late_slope_sign_agreement_fraction":early_late,"full_slope_positive_direction_agreement_fraction":direction,"round09_existing_network_comparison":{"median_blocked_improvement":old["median_blocked_rmse_improvement"],"early_late_sign_agreement":old["early_late_slope_sign_agreement_fraction"]},"gates":gates,"passes_new_temperature_network":passes,"decision":"permit_separate_temperature_operator_test" if passes else "close_new_lmr_linear_temperature_psm_promotion"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True)
    axes[0].scatter(insample*100,blocked*100,s=18,alpha=.65,color="#005f73");axes[0].axhline(0,color="#9b2226");axes[0].axvline(1,color="#9b2226",linestyle="--");axes[0].set(title="Calibration versus blocked skill",xlabel="In-sample improvement (%)",ylabel="Blocked improvement (%)")
    axes[1].bar(["New LMR","Existing proxy9"],[early_late,old["early_late_slope_sign_agreement_fraction"]],color=["#ee9b00","#94d2bd"]);axes[1].axhline(.70,color="#9b2226",linestyle="--");axes[1].set_ylim(0,1);axes[1].set(title="Early/late slope stability",ylabel="Agreement fraction")
    fig.suptitle("New LMR temperature network validation",fontsize=14);fig.savefig(output/"new_lmr_temperature_network_validation.png",dpi=180);plt.close(fig)
    (output/"FINAL_RESULT.md").write_text("# New LMR Temperature Network Validation\n\n"+f"- Identities/sites: `214/{summary['unique_site_count']}`.\n- Median |r| / in-sample improvement: `{summary['median_absolute_full_period_correlation']:.3f}` / `{summary['median_full_period_insample_rmse_improvement']:.3%}`.\n- Median blocked improvement: `{median_blocked:.3%}` (95% `{blocked_ci[0]:.3%}` to `{blocked_ci[1]:.3%}`); positive identities `{positive_fraction:.3%}`.\n- Early/late sign agreement: `{early_late:.3%}`; formal positive-direction agreement: `{direction:.3%}`.\n- All gates pass: `{str(passes).lower()}`.\n\nDecision: `{summary['decision']}`.\n",encoding="utf-8")
    after={name:state(path) for name,path in paths.items()};(output/"source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after,"bootstrap_draws":5000},indent=2),encoding="utf-8")
    if before!=after: raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
