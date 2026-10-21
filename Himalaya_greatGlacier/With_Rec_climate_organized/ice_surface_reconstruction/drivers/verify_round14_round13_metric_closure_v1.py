"""Close Round 14 Berkeley arrays and pooled OOF metrics against Round 13."""

from __future__ import annotations
import importlib.util, json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

SEASONS={"ANN":0,"JJA":1,"AMJ":2,"APR":3}


def root()->Path:
    for parent in Path(__file__).resolve().parents:
        if (parent/"LIA_PROJECT_FACTS.md").exists():return parent
    raise RuntimeError("project-root-not-found")


def main()->int:
    project=root();drivers=Path(__file__).resolve().parent;base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round13=base/"round_13_new_lmr_temperature_network_validation_v2";round14=base/"round_14_temperature_product_cross_validation_v1"
    manifest=pd.read_csv(round13/"network_manifest.csv");reference=pd.read_csv(round13/"record_new_temperature_validation.csv").sort_values("network_index_1based")
    spec=importlib.util.spec_from_file_location("round14_closure_helper",drivers/"run_climate_reconstruction_loop_v2_round09_temperature_proxy_association_stability_v1.py")
    if spec is None or spec.loader is None:raise RuntimeError("helper-load-failed")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with h5py.File(round13/"temperature_seasons_v1.mat","r") as handle:
        old={"ANN":np.asarray(handle["annual"],dtype=float).T,"JJA":np.asarray(handle["jja"],dtype=float).T,"AMJ":np.asarray(handle["amj"],dtype=float).T,"APR":np.asarray(handle["april"],dtype=float).T}
    with h5py.File(round14/"temperature_products_v1.mat","r") as handle:
        years=np.asarray(handle["years"],dtype=int).ravel();observations=np.asarray(handle["observations"],dtype=float).T;products=np.asarray(handle["temperature"],dtype=float).transpose(3,2,1,0)
    chosen=np.full((61,214),np.nan);old_chosen=np.full((61,214),np.nan)
    for record in range(214):
        code=str(manifest.iloc[record].season_code);chosen[:,record]=products[:,record,SEASONS[code],0];old_chosen[:,record]=old[code][:,record]
    finite_equal=bool(np.array_equal(np.isfinite(chosen),np.isfinite(old_chosen)));maximum=float(np.nanmax(np.abs(chosen-old_chosen)))
    pooled=[]
    for record in range(214):pooled.append(float(helper.blocked_stats(observations[:,record],chosen[:,record],years)["improvement"]))
    reference_values=reference.blocked_rmse_improvement.to_numpy(float);metric_finite_equal=bool(np.array_equal(np.isfinite(pooled),np.isfinite(reference_values)));metric_maximum=float(np.nanmax(np.abs(np.asarray(pooled)-reference_values)))
    result={"status":"complete","locked_1921_1939_accessed":False,"berkeley_selected_season_finite_mask_equal":finite_equal,"berkeley_selected_season_max_abs_difference":maximum,"round13_pooled_metric_finite_mask_equal":metric_finite_equal,"round13_pooled_metric_max_abs_difference":metric_maximum,"closure_passes":bool(finite_equal and maximum<=1e-12 and metric_finite_equal and metric_maximum<=1e-12),"explanation":"Round 14 summarizes median fold-level relative improvements; Round 13 uses one pooled OOF RMSE ratio. Both are valid but not numerically interchangeable."}
    (round14/"round13_metric_closure.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result["closure_passes"]:raise RuntimeError("round13-round14-closure-failed")
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
