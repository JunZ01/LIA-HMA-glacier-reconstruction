"""Surface-moment-conserving LP bounds for system-84 formal geometry gates."""

from __future__ import annotations

import gzip, json, pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

P=Path(__file__).resolve().parents[4]
GD=P/"validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
ARR=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2/assignment_audit_arrays.npz"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_native_surface_lp_bounds_v2"

def main():
    data=np.load(ARR); weight=np.asarray(data["node_quota"],float); cell_surface=np.asarray(data["cell_surface_m"],float)
    with gzip.open(GD/"inversion_flowlines.pkl","rb") as f: fls=pickle.load(f)
    byid={id(fl):i for i,fl in enumerate(fls)};offs=[];native=[];c=0
    for fl in fls:offs.append(c);native.extend(np.asarray(fl.surface_h,float));c+=fl.nx
    native=np.asarray(native);edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1):edges.append((offs[b]+j,offs[b]+j+1))
        if fl.flows_to is not None:r=byid[id(fl.flows_to)];edges.append((offs[b]+fl.nx-1,offs[r]+int(fl.flows_to_indice)))
    n=len(native);required_delta_sum=float(np.sum(cell_surface)-np.dot(weight,native));aeq_delta=weight.reshape(1,-1)
    # minimax
    rr=[];cc=[];dd=[];bb=[];row=0
    for u,v in edges:rr += [row,row];cc += [v,u];dd += [1.,-1.];bb.append(float(native[u]-native[v]));row+=1
    for i in range(n):
        rr += [row,row];cc += [i,n];dd += [1.,-1.];bb.append(0.);row+=1
        rr += [row,row];cc += [i,n];dd += [-1.,-1.];bb.append(0.);row+=1
    A=coo_matrix((dd,(rr,cc)),shape=(row,n+1)).tocsr();obj=np.zeros(n+1);obj[n]=1.;Aeq=np.zeros((1,n+1));Aeq[0,:n]=weight
    r1=linprog(obj,A_ub=A,b_ub=np.asarray(bb),A_eq=Aeq,b_eq=np.asarray([required_delta_sum]),bounds=[(None,None)]*n+[(0,None)],method="highs")
    # min weighted L1 with max150
    rr=[];cc=[];dd=[];bb=[];row=0
    for u,v in edges:rr += [row,row];cc += [v,u];dd += [1.,-1.];bb.append(float(native[u]-native[v]));row+=1
    for i in range(n):
        rr += [row,row];cc += [i,n+i];dd += [1.,-1.];bb.append(0.);row+=1
        rr += [row,row];cc += [i,n+i];dd += [-1.,-1.];bb.append(0.);row+=1
    A2=coo_matrix((dd,(rr,cc)),shape=(row,2*n)).tocsr();obj2=np.r_[np.zeros(n),weight/np.sum(weight)];Aeq2=np.zeros((1,2*n));Aeq2[0,:n]=weight
    r2=linprog(obj2,A_ub=A2,b_ub=np.asarray(bb),A_eq=Aeq2,b_eq=np.asarray([required_delta_sum]),bounds=[(-150.,150.)]*n+[(0.,None)]*n,method="highs")
    minimax=None if not r1.success else float(r1.x[n]);mean=None if not r2.success else float(r2.fun)
    target_mean=float(np.mean(cell_surface));native_weighted=float(np.dot(weight,native)/np.sum(weight))
    passed=r1.success and minimax<=150 and r2.success and mean<=20
    summary={"schema":"lia-191-system84-native-surface-lp-bounds-v2","system_id":84,"status":"surface-moment-conserving-continuous-bounds-pass" if passed else "surface-moment-conserving-continuous-bound-fail-mechanism-closed","source_cell_mean_surface_m":target_mean,"native_quota_weighted_mean_surface_m":native_weighted,"required_quota_weighted_mean_adjustment_m":target_mean-native_weighted,"surface_moment_equality_enforced":True,"minimax_lp_success":bool(r1.success),"minimum_possible_maximum_adjustment_m":minimax,"maximum_adjustment_gate_m":150.,"mean_l1_lp_success_under_max150":bool(r2.success),"minimum_possible_area_weighted_mean_absolute_adjustment_m":mean,"mean_adjustment_gate_m":20.,"v1_superseded_reason":"omitted exact source-cell surface-moment equality","discrete_assignment_authorized":bool(passed),"geometry_materialized":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    if r2.success:np.savez_compressed(OUT/"continuous_target_v2.npz",delta_m=r2.x[:n],target_surface_m=native+r2.x[:n],node_quota=weight)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
