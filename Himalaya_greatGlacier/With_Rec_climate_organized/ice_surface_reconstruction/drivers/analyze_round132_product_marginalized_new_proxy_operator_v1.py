"""Analyze Round 132 product-marginalized new-proxy operator."""
from __future__ import annotations
import importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    s=importlib.util.spec_from_file_location("r132base",path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def sm(a): return pd.Series(a).rolling(31,center=True,min_periods=15).mean().to_numpy()
def corr(a,b):
    v=np.isfinite(a)&np.isfinite(b); return float(np.corrcoef(a[v],b[v])[0,1])
def main():
    driver=Path(__file__).resolve().parent; root=driver.parents[3]; base=load(driver/"run_round129_new_d18o_seasonal_prior_coupling_v2.py")
    loop=root/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; r130=loop/"round_130_xinpu_keylong_frozen_enkf_addition_v1"; out=loop/"round_132_product_marginalized_new_proxy_operator_v1"
    paths={"facts":root/"LIA_PROJECT_FACTS.md","contract":driver/"round132_product_marginalized_new_proxy_operator_v1_contract.md","runner":driver/"run_round132_product_marginalized_new_proxy_operator_v1.m","analysis":Path(__file__).resolve(),"round130":r130/"enkf_branches.mat","marginalized":out/"marginalized_branch.mat","prior":loop/"round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat"}
    before={k:base.fingerprint(v) for k,v in paths.items()}
    with h5py.File(paths["prior"],"r") as f: lat=np.asarray(f["lat"]).ravel(); lon=np.asarray(f["lon"]).ravel()
    with h5py.File(paths["round130"],"r") as f: std=np.asarray(f["posterior_mean"]).transpose(2,1,0); years=np.asarray(f["years"]).ravel().astype(int)
    with h5py.File(paths["marginalized"],"r") as f:
        mar=np.asarray(f["posterior_mean"]).T; z=np.asarray(f["innovation_z"]).T; rnew=np.asarray(f["rnew"]).ravel(); rmean=np.asarray(f["rmean"]).ravel(); opvar=np.asarray(f["operator_product_variance"]).ravel()
    w=np.tile(np.cos(np.deg2rad(lat)),len(lon)); baseline=np.average(std[:,:,0],axis=0,weights=w); bea=np.average(std[:,:,5],axis=0,weights=w)-baseline; cru=np.average(std[:,:,6],axis=0,weights=w)-baseline; marg=np.average(mar,axis=0,weights=w)-baseline
    active=years>=1352; bstd=float(np.std(baseline,ddof=1)); m31=sm(marg); b31=sm(bea); c31=sm(cru)
    metrics={"max_abs_increment_mm_day":float(np.max(np.abs(marg[active]))),"max_abs_over_baseline_std":float(np.max(np.abs(marg[active]))/bstd),"annual_correlation_with_bea":corr(marg[active],bea[active]),"annual_correlation_with_cru":corr(marg[active],cru[active]),"smoothed_correlation_with_bea":corr(m31[active],b31[active]),"smoothed_correlation_with_cru":corr(m31[active],c31[active])}
    uncertainty=pd.DataFrame({"site":["xinpu","keylong"],"mean_oof_r":rmean,"operator_product_variance":opvar,"total_r":rnew,"r_inflation_factor":rnew/rmean})
    innovation=[]
    for j,site in enumerate(("xinpu","keylong")):
        x=z[j]; x=x[np.isfinite(x)]; innovation.append({"site":site,"count":len(x),"median_abs_z":float(np.median(np.abs(x))),"p95_abs_z":float(np.percentile(np.abs(x),95)),"maximum_abs_z":float(np.max(np.abs(x)))})
    innovations=pd.DataFrame(innovation); pd.DataFrame([metrics]).to_csv(out/"marginalized_impact_metrics.csv",index=False); uncertainty.to_csv(out/"product_uncertainty_inflation.csv",index=False); innovations.to_csv(out/"marginalized_innovations.csv",index=False)
    pd.DataFrame({"year":years,"standard_bea_increment":bea,"standard_cru_increment":cru,"marginalized_increment":marg,"standard_bea_31yr":b31,"standard_cru_31yr":c31,"marginalized_31yr":m31}).to_csv(out/"marginalized_increment_curves.csv",index=False)
    gates={"max_increment_below_half_baseline_std":metrics["max_abs_over_baseline_std"]<=.5,"smoothed_correlation_with_both_products_ge_0_80":min(metrics["smoothed_correlation_with_bea"],metrics["smoothed_correlation_with_cru"])>=.8,"innovation_p95_below_4":bool((innovations.p95_abs_z<=4).all())}; passed=all(gates.values()); decision="advance_product_marginalized_branch_to_independent_validation" if passed else "close_product_marginalized_branch_as_insufficient"
    fig,ax=plt.subplots(figsize=(12.5,4.8),constrained_layout=True); ax.plot(years,b31,color="#0072B2",lw=1,label="standard BEA+GPCC"); ax.plot(years,c31,color="#D55E00",lw=1,label="standard CRU"); ax.plot(years,m31,color="#111111",lw=1.6,label="product-marginalized"); ax.axhline(0,color="#555",lw=.7); ax.set(xlabel="year",ylabel="HMA increment (mm/day)",title="New-proxy increment, 31-year mean"); ax.legend(frameon=False); ax.grid(axis="y",color="#ddd",lw=.5); ax.spines[["top","right"]].set_visible(False); fig.savefig(out/"round132_product_marginalized_increment_curves.png",dpi=220); plt.close(fig)
    after={k:base.fingerprint(v) for k,v in paths.items()}; unchanged=before==after
    summary={"experiment_id":"round132_product_marginalized_new_proxy_operator_v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","years":[1000,1920],"locked_1921_1939_accessed":False,"external_validation_reserve_opened":False,"metrics":metrics,"gates":gates,"passes_all_gates":passed,"decision":decision,"source_fingerprints_unchanged":unchanged}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); (out/"source_manifest.json").write_text(json.dumps({"before":before,"after":after,"unchanged":unchanged},indent=2),encoding="utf-8"); (out/"FINAL_RESULT.md").write_text(f"# Round 132 Product-Marginalized Operator\n\n- Maximum increment / baseline standard deviation: `{metrics['max_abs_over_baseline_std']:.3f}`.\n- Decision: `{decision}`.\n",encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
