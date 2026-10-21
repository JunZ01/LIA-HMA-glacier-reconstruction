"""Localized grouped-cell transport LP for the system-84 discrete route."""

from __future__ import annotations

import gzip, json, pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

P=Path(__file__).resolve().parents[4]
GD=P/"validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
ARR=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2/assignment_audit_arrays.npz"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_group_transport_lp_v1"
GROUPS_PER_NODE=5;K_NEAREST=16;MAX_DISTANCE_M=2500.

def main():
    a=np.load(ARR);owner=np.asarray(a["owner"],int);cs=np.asarray(a["cell_surface_m"],float);xy=np.asarray(a["cell_xy_m"],float);nxy=np.asarray(a["node_xy_m"],float);quota=np.asarray(a["node_quota"],float);native=np.asarray(a["native_surface_m"],float);n=len(quota)
    groups=[]
    for node in range(n):
        cells=np.flatnonzero(owner==node);cells=cells[np.argsort(cs[cells],kind="stable")]
        for chunk in np.array_split(cells,min(GROUPS_PER_NODE,len(cells))):
            if len(chunk):groups.append((node,float(len(chunk)),float(np.mean(cs[chunk])),np.mean(xy[chunk],axis=0)))
    tree=cKDTree(nxy);var_group=[];var_node=[];cost=[];group_vars=[[] for _ in groups];node_vars=[[] for _ in range(n)]
    for gi,(source,supply,smean,centroid) in enumerate(groups):
        dist,idx=tree.query(centroid,k=min(K_NEAREST,n));dist=np.atleast_1d(dist);idx=np.atleast_1d(idx).astype(int)
        candidates={int(source)}|{int(j) for d,j in zip(dist,idx) if float(d)<=MAX_DISTANCE_M}
        for dest in sorted(candidates):
            vi=len(var_group);var_group.append(gi);var_node.append(dest);cost.append(float(np.linalg.norm(centroid-nxy[dest])));group_vars[gi].append(vi);node_vars[dest].append(vi)
    m=len(var_group);total=m+n
    # Equalities: each group supply, each node quota.
    rr=[];cc=[];dd=[];beq=[];row=0
    for gi,(_,supply,_,_) in enumerate(groups):
        for vi in group_vars[gi]:rr.append(row);cc.append(vi);dd.append(1.)
        beq.append(supply);row+=1
    for node in range(n):
        for vi in node_vars[node]:rr.append(row);cc.append(vi);dd.append(1.)
        beq.append(quota[node]);row+=1
    Aeq=coo_matrix((dd,(rr,cc)),shape=(row,total)).tocsr()
    # Inequalities: network monotonic, native +/-150, |adjustment|<=t,
    # and quota-weighted mean t <=20.
    with gzip.open(GD/"inversion_flowlines.pkl","rb") as f:fls=pickle.load(f)
    byid={id(fl):i for i,fl in enumerate(fls)};offs=[];c=0
    for fl in fls:offs.append(c);c+=fl.nx
    edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1):edges.append((offs[b]+j,offs[b]+j+1))
        if fl.flows_to is not None:r=byid[id(fl.flows_to)];edges.append((offs[b]+fl.nx-1,offs[r]+int(fl.flows_to_indice)))
    rr=[];cc=[];dd=[];bub=[];row=0
    def add_mean(node,coef):
        for vi in node_vars[node]:rr.append(row);cc.append(vi);dd.append(coef*groups[var_group[vi]][2]/quota[node])
    for u,v in edges:
        add_mean(v,1.);add_mean(u,-1.);bub.append(0.);row+=1
    for node in range(n):
        add_mean(node,1.);bub.append(float(native[node]+150.));row+=1
        add_mean(node,-1.);bub.append(float(-native[node]+150.));row+=1
        add_mean(node,1.);rr.append(row);cc.append(m+node);dd.append(-1.);bub.append(float(native[node]));row+=1
        add_mean(node,-1.);rr.append(row);cc.append(m+node);dd.append(-1.);bub.append(float(-native[node]));row+=1
    for node in range(n):rr.append(row);cc.append(m+node);dd.append(float(quota[node]/np.sum(quota)))
    bub.append(20.);row+=1
    Aub=coo_matrix((dd,(rr,cc)),shape=(row,total)).tocsr();objective=np.r_[np.asarray(cost)/1000.,np.zeros(n)]
    result=linprog(objective,A_ub=Aub,b_ub=np.asarray(bub),A_eq=Aeq,b_eq=np.asarray(beq),bounds=[(0.,None)]*total,method="highs",options={"time_limit":600.})
    summary={"schema":"lia-191-system84-local-group-transport-lp-v1","system_id":84,"status":"localized-group-lp-pass-integerization-authorized" if result.success else "localized-group-lp-fail-no-integer-run","solver_status":int(result.status),"solver_message":result.message,"group_count":len(groups),"transport_variable_count":m,"node_auxiliary_count":n,"groups_per_node_max":GROUPS_PER_NODE,"candidate_nearest_nodes":K_NEAREST,"maximum_candidate_distance_m":MAX_DISTANCE_M,"source_and_node_quota_equalities":True,"network_monotonic_constraints":len(edges),"maximum_adjustment_gate_m":150.,"mean_adjustment_gate_m":20.,"objective_kilometer_cell_count":None if not result.success else float(result.fun),"integer_run_authorized":bool(result.success),"geometry_materialized":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    if result.success:np.savez_compressed(OUT/"group_transport_lp_solution_v1.npz",transport=result.x[:m],var_group=np.asarray(var_group),var_node=np.asarray(var_node),group_source=np.asarray([g[0] for g in groups]),group_supply=np.asarray([g[1] for g in groups]),group_surface_mean=np.asarray([g[2] for g in groups]),group_centroid=np.asarray([g[3] for g in groups]),node_aux=result.x[m:])
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
