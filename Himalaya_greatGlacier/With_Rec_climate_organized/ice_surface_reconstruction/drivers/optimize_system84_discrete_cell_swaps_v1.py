"""Deterministic pair-swap realization of the system-84 continuous target."""

from __future__ import annotations

import gzip, json, pickle
from pathlib import Path

import numpy as np

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m

P=Path(__file__).resolve().parents[4]
GD=P/"validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
ARR=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2/assignment_audit_arrays.npz"
TARGET=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_native_surface_lp_bounds_v2/continuous_target_v2.npz"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_discrete_cell_swaps_v1"

def main():
    a=np.load(ARR);owner=np.asarray(a["owner"],int).copy();cs=np.asarray(a["cell_surface_m"],float);ch=np.asarray(a["cell_thickness_m"],float);xy=np.asarray(a["cell_xy_m"],float);nxy=np.asarray(a["node_xy_m"],float);quota=np.asarray(a["node_quota"],int);native=np.asarray(a["native_surface_m"],float)
    target=np.asarray(np.load(TARGET)["target_surface_m"],float);n=len(quota);target_sum=target*quota
    cells=[set(np.flatnonzero(owner==i).tolist()) for i in range(n)];current_sum=np.bincount(owner,weights=cs,minlength=n);err=current_sum-target_sum
    with gzip.open(GD/"inversion_flowlines.pkl","rb") as f:fls=pickle.load(f)
    byid={id(fl):i for i,fl in enumerate(fls)};offs=[];c=0
    for fl in fls:offs.append(c);c+=fl.nx
    edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1):edges.append((offs[b]+j,offs[b]+j+1,"within"))
        if fl.flows_to is not None:r=byid[id(fl.flows_to)];edges.append((offs[b]+fl.nx-1,offs[r]+int(fl.flows_to_indice),"junction"))
    def metrics():
        mean=current_sum/quota;rev=[(u,v,k,float(mean[v]-mean[u])) for u,v,k in edges if mean[v]-mean[u]>1e-7];delta=mean-native
        return mean,rev,float(np.max(np.abs(delta))),float(np.average(np.abs(delta),weights=quota)),cdf_error_100m(cs,mean,quota)
    history=[];max_iter=5000
    for it in range(max_iter+1):
        mean,rev,mx,ma,cdf=metrics()
        if not rev and mx<=150 and ma<=20 and cdf<=.02:break
        mean_err=err/quota;pos_nodes=np.argsort(-mean_err)[:20];neg_nodes=np.argsort(mean_err)[:20]
        best=None
        for i in pos_nodes:
            if mean_err[i]<=1e-10:continue
            hi=sorted(cells[i],key=lambda x:cs[x],reverse=True)[:5]
            for j in neg_nodes:
                if i==j or mean_err[j]>=-1e-10:continue
                lo=sorted(cells[j],key=lambda x:cs[x])[:5]
                old=(err[i]/quota[i])**2+(err[j]/quota[j])**2
                for ci in hi:
                    for cj in lo:
                        d=cs[ci]-cs[cj]
                        if d<=0:continue
                        ni=err[i]-d;nj=err[j]+d;new=(ni/quota[i])**2+(nj/quota[j])**2;gain=old-new
                        if gain<=0:continue
                        move_before=np.linalg.norm(xy[ci]-nxy[i])+np.linalg.norm(xy[cj]-nxy[j]);move_after=np.linalg.norm(xy[ci]-nxy[j])+np.linalg.norm(xy[cj]-nxy[i]);key=(gain,-max(0.,move_after-move_before),-move_after,-i,-j,-ci,-cj)
                        if best is None or key>best[0]:best=(key,i,j,ci,cj,d)
        if best is None:break
        _,i,j,ci,cj,d=best;cells[i].remove(ci);cells[j].remove(cj);cells[i].add(cj);cells[j].add(ci);owner[ci]=j;owner[cj]=i;current_sum[i]-=d;current_sum[j]+=d;err[i]-=d;err[j]+=d
        if it%100==0:history.append({"iteration":it,"reverse_count":len(rev),"max_native_adjustment_m":mx,"mean_native_adjustment_m":ma,"target_rmse_m":float(np.sqrt(np.average((mean-target)**2,weights=quota)))})
    mean,rev,mx,ma,cdf=metrics();movement=np.linalg.norm(xy-nxy[owner],axis=1);source_volume=float(np.sum(ch)*280.5625);mapped_volume=float(np.sum(np.bincount(owner,weights=ch*280.5625,minlength=n)))
    checks={"reverse":len(rev)==0,"maximum_native_surface_adjustment":mx<=150,"mean_native_surface_adjustment":ma<=20,"hypsometry":cdf<=.02,"quota":bool(np.array_equal(np.bincount(owner,minlength=n),quota)),"volume":abs(mapped_volume-source_volume)<=max(1e-6,source_volume*1e-12)}
    summary={"schema":"lia-191-system84-discrete-cell-swaps-v1","system_id":84,"status":"discrete-static-candidate-not-materialized" if all(checks.values()) else "discrete-optimizer-did-not-reach-formal-gates","checks":checks,"iterations_executed":it,"final_reverse_count":len(rev),"maximum_native_surface_adjustment_m":mx,"mean_native_surface_adjustment_m":ma,"hypsometry_cdf_error_100m":cdf,"target_surface_rmse_m":float(np.sqrt(np.average((mean-target)**2,weights=quota))),"cell_movement_mean_m":float(np.mean(movement)),"cell_movement_p95_m":float(np.percentile(movement,95)),"cell_movement_max_m":float(np.max(movement)),"source_volume_m3":source_volume,"mapped_volume_m3":mapped_volume,"volume_error_fraction":(mapped_volume-source_volume)/source_volume,"history":history,"materialized_geometry":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");np.savez_compressed(OUT/"candidate_assignment_v1.npz",owner=owner,node_surface_m=mean,node_quota=quota,cell_surface_m=cs,cell_thickness_m=ch,cell_xy_m=xy,node_xy_m=nxy,native_surface_m=native,target_surface_m=target);print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
