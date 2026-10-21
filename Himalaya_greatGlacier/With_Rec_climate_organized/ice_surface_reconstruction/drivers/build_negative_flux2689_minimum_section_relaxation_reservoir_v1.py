"""Minimum-change section relaxation plus passive reservoir for system 2689."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .audit_negative_flux2342_dynamic_subnetwork_reservoir_v1 import reservoir_lp
from .audit_negative_flux2342_hypsometric_residual_reservoir_v1 import sha256


PROJECT=Path(__file__).resolve().parents[4]
OUTPUT=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2689_minimum_section_relaxation_reservoir_v1"
PAIRED=PROJECT/"validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2689_RGI60-15.07122_system_grid_v1.nc"
FORMAL=PROJECT/"w586inv2689v1/inversion_summary_formal.json"
CONFIG=PROJECT/"w586inv2689v1/config_snapshot.json"
FLOWLINES=PROJECT/"w586inv2689v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92689/model_flowlines.pkl"


def load_flowlines():
    with gzip.open(FLOWLINES,"rb") as stream:return pickle.load(stream)


def main():
    if OUTPUT.exists():raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal=json.loads(FORMAL.read_text(encoding="utf-8")); config=json.loads(CONFIG.read_text(encoding="utf-8")); original=load_flowlines()
    with xr.open_dataset(PAIRED) as ds:
        h=np.asarray(ds["thickness_m"].values[0],float); s=np.asarray(ds["surface_elevation_m"].values[0],float)
        valid=np.asarray(ds["valid_mask"].values[0],bool); lia=np.asarray(ds["lia_mask"].values,bool)
        gt=[float(v) for v in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    cell_area=abs(gt[1]*gt[5]); mask=valid&lia&np.isfinite(h)&np.isfinite(s)&(h>1e-6)
    ph=h[mask]; ps=s[mask]; paired_area=float(mask.sum()*cell_area); paired_volume=float(np.sum(ph)*cell_area)
    all_orig_surface=np.concatenate([np.asarray(fl.surface_h)[np.asarray(fl.thick)>1e-6] for fl in original])
    lo=100*np.floor(min(np.min(ps),np.min(all_orig_surface))/100); hi=100*np.ceil(max(np.max(ps),np.max(all_orig_surface))/100)
    edges=np.arange(lo,hi+100,100); n=len(edges)-1; pids=np.clip(np.digitize(ps,edges)-1,0,n-1)
    paired_bins=np.bincount(pids,weights=np.full(len(pids),cell_area),minlength=n)
    hmin=np.zeros(n);hmax=np.zeros(n)
    for j in range(n):
        vals=ph[pids==j]
        if vals.size:hmin[j]=float(vals.min());hmax[j]=float(vals.max())
    trials=[]; selected=None
    # Descending q implements the registered minimum-change rule.
    for q in np.round(np.arange(0.815,0.699,-0.005),3):
        fls=copy.deepcopy(original); within=0;junction=0; area_parts=[];volume_parts=[];surface_parts=[]
        for fl in fls:
            fl.section=np.asarray(fl.section,dtype=float)*float(q)
            active=np.asarray(fl.thick)>1e-6; ids=np.flatnonzero(active)
            area=np.asarray(fl.widths_m)*fl.dx_meter; volume=np.asarray(fl.section)*fl.dx_meter
            area_parts.append(area[active]);volume_parts.append(volume[active]);surface_parts.append(np.asarray(fl.surface_h)[active])
            if len(ids)>1:
                within+=int(np.sum((ids[1:]==ids[:-1]+1)&(np.diff(np.asarray(fl.surface_h)[ids])>1e-7)))
        identity={id(fl):i for i,fl in enumerate(fls)}
        for i,fl in enumerate(fls):
            parent=getattr(fl,"flows_to",None)
            if parent is None:continue
            ids=np.flatnonzero(np.asarray(fl.thick)>1e-6)
            if not len(ids):continue
            receiver=int(fl.flows_to_indice)
            if receiver<parent.nx and parent.thick[receiver]>1e-6 and parent.surface_h[receiver]>fl.surface_h[ids[-1]]+1e-7:junction+=1
        ca=np.concatenate(area_parts);cv=np.concatenate(volume_parts);cs=np.concatenate(surface_parts)
        core_area=float(np.sum(ca));core_volume=float(np.sum(cv))
        cids=np.clip(np.digitize(cs,edges)-1,0,n-1);core_bins=np.bincount(cids,weights=ca,minlength=n)
        result=reservoir_lp(core_bins,core_area,core_volume,paired_bins,paired_area,paired_volume,hmin,hmax)
        success=bool(result is not None and result.success); cdf=float(result.x[-1]) if success else None
        record={"q_section":float(q),"core_area_m2":core_area,"core_volume_m3":core_volume,"core_area_fraction":core_area/paired_area,"core_volume_fraction":core_volume/paired_volume,"within_reverse_count":within,"junction_reverse_count":junction,"lp_success":success,"cdf_error":cdf}
        trials.append(record)
        if success and cdf<=.02 and within==0 and junction==0 and core_area/paired_area>=.50 and core_volume/paired_volume>=.50:
            selected=(record,result,fls,core_bins);break
    pd.DataFrame(trials).to_csv(OUTPUT/"section_relaxation_search.csv",index=False)
    if selected:
        record,result,fls,core_bins=selected;ra=result.x[:n];rv=result.x[n:2*n]
        area_err=record["core_area_m2"]+float(np.sum(ra))-paired_area;volume_err=record["core_volume_m3"]+float(np.sum(rv))-paired_volume
        pd.DataFrame({"bin_lower_m":edges[:-1],"bin_upper_m":edges[1:],"paired_area_m2":paired_bins,"dynamic_core_area_m2":core_bins,"reservoir_area_m2":ra,"reservoir_volume_m3":rv,"reservoir_mean_thickness_m":np.divide(rv,ra,out=np.zeros_like(rv),where=ra>0)}).to_csv(OUTPUT/"reservoir_by_100m_bin.csv",index=False)
        with gzip.open(OUTPUT/"candidate_dynamic_core_flowlines.pkl.gz","wb") as stream:pickle.dump(fls,stream)
        gates={"selected_q_at_least_0p70":record["q_section"]>=.70,"core_area_at_least_50pct":record["core_area_fraction"]>=.50,"core_volume_at_least_50pct":record["core_volume_fraction"]>=.50,"no_active_reverse":record["within_reverse_count"]==0 and record["junction_reverse_count"]==0,"combined_cdf_within_2pct":record["cdf_error"]<=.02,"area_closure":abs(area_err)<=1e-5,"volume_closure":abs(volume_err)<=1e-3,"root_length_within_10pct":abs(float(formal["model_root_length_m"])-float(config["registered_lia_length_m"]))/float(config["registered_lia_length_m"])<=.10,"free_tail_at_least_5km":float(formal["model_free_tail_m"])>=5000,"root_terminal_zero":float(formal["root_terminal_thickness_m"])<=10}
        passed=all(gates.values());selected_payload={**record,"area_closure_error_m2":area_err,"volume_closure_error_m3":volume_err}
    else:
        gates={"feasible_q_in_registered_range_found":False};passed=False;selected_payload=None
    summary={"schema":"lia-191-negative-flux2689-minimum-section-relaxation-reservoir-v1","status":"static-pass" if passed else "static-fail","decision":"zero-mb-authorized" if passed else "mechanism-closed-at-static-gate","system_id":2689,"mechanism":"maximum-q uniform positive-section relaxation plus passive source-supported hypsometric reservoir","q_search_range":[.815,.70],"q_step":.005,"selected":selected_payload,"gate_results":gates,"climate_or_stage2_modified":False,"zero_thickness_nodes_filled":False,"bed_or_topology_modified":False,"oggm_run":False,"zero_mb_run":False,"steady_50y_run":False,"historical_run":False,"provenance":{"paired_sha256":sha256(PAIRED),"formal_sha256":sha256(FORMAL),"config_sha256":sha256(CONFIG),"flowlines_sha256":sha256(FLOWLINES)}}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=="__main__":main()
