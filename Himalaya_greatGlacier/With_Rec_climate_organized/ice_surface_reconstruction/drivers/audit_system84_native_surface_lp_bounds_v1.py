"""Strict continuous LP lower bounds for system-84 formal surface gates."""

from __future__ import annotations

import gzip, json, pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

P=Path(__file__).resolve().parents[4]
GD=P/"validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
ARR=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2/assignment_audit_arrays.npz"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_native_surface_lp_bounds_v1"

def edges_and_native():
    with gzip.open(GD/"inversion_flowlines.pkl","rb") as f: fls=pickle.load(f)
    byid={id(fl):i for i,fl in enumerate(fls)}; offs=[]; native=[]; c=0
    for fl in fls: offs.append(c); native.extend(np.asarray(fl.surface_h,float)); c+=fl.nx
    edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1): edges.append((offs[b]+j,offs[b]+j+1))
        if fl.flows_to is not None: r=byid[id(fl.flows_to)]; edges.append((offs[b]+fl.nx-1,offs[r]+int(fl.flows_to_indice)))
    return np.asarray(native),edges

def main():
    a=np.load(ARR); weight=np.asarray(a["node_quota"],float); native,edges=edges_and_native(); n=len(native)
    # LP1: min z, monotone native+d, |d|<=z.
    rr=[];cc=[];dd=[];bb=[];row=0
    for u,v in edges: rr += [row,row]; cc += [v,u]; dd += [1.,-1.]; bb.append(float(native[u]-native[v])); row+=1
    for i in range(n):
        rr += [row,row]; cc += [i,n]; dd += [1.,-1.]; bb.append(0.); row+=1
        rr += [row,row]; cc += [i,n]; dd += [-1.,-1.]; bb.append(0.); row+=1
    A=coo_matrix((dd,(rr,cc)),shape=(row,n+1)).tocsr(); obj=np.zeros(n+1);obj[n]=1
    r1=linprog(obj,A_ub=A,b_ub=np.asarray(bb),bounds=[(None,None)]*n+[(0,None)],method="highs")
    if not r1.success: raise RuntimeError("minimax-lp-failed:"+r1.message)
    # LP2: min weighted mean |d| with |d|<=150 and monotonicity.
    # variables d[0:n], t[0:n].
    rr=[];cc=[];dd=[];bb=[];row=0
    for u,v in edges: rr += [row,row];cc += [v,u];dd += [1.,-1.];bb.append(float(native[u]-native[v]));row+=1
    for i in range(n):
        rr += [row,row];cc += [i,n+i];dd += [1.,-1.];bb.append(0.);row+=1
        rr += [row,row];cc += [i,n+i];dd += [-1.,-1.];bb.append(0.);row+=1
    A2=coo_matrix((dd,(rr,cc)),shape=(row,2*n)).tocsr(); obj2=np.r_[np.zeros(n),weight/np.sum(weight)]
    r2=linprog(obj2,A_ub=A2,b_ub=np.asarray(bb),bounds=[(-150.,150.)]*n+[(0.,None)]*n,method="highs")
    mean_min=None if not r2.success else float(r2.fun)
    summary={"schema":"lia-191-system84-native-surface-lp-bounds-v1","system_id":84,"minimax_lp_success":bool(r1.success),"minimum_possible_maximum_adjustment_m":float(r1.x[n]),"maximum_adjustment_gate_m":150.,"mean_l1_lp_success_under_max150":bool(r2.success),"minimum_possible_area_weighted_mean_absolute_adjustment_m":mean_min,"mean_adjustment_gate_m":20.,"status":"continuous-formal-bounds-pass-discrete-optimizer-required" if r1.x[n]<=150 and r2.success and mean_min<=20 else "continuous-formal-bound-fail-mechanism-closed","continuous_only":True,"geometry_materialized":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
