"""Deterministic local-pool exact-cell reassignment audit for system 84.

Starts from the branchwise hypsometric assignment.  For each remaining
reversed network edge, progressively larger graph-neighbourhood pools are
topologically reordered while preserving each node's integer cell quota.
Only a candidate that strictly reduces the global reversal count (or the
maximum reverse gap at equal count) is accepted.  No geometry is written.
"""

from __future__ import annotations

import gzip
import heapq
import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
GDIR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v1"


def topological_order(pool, edges, priority):
    pool = set(pool); indegree = {n: 0 for n in pool}; outgoing = {n: [] for n in pool}
    for u, v, _ in edges:
        if u in pool and v in pool:
            outgoing[u].append(v); indegree[v] += 1
    heap = []
    for n, degree in indegree.items():
        if degree == 0: heapq.heappush(heap, (-float(priority[n]), int(n)))
    order = []
    while heap:
        _, n = heapq.heappop(heap); order.append(n)
        for v in outgoing[n]:
            indegree[v] -= 1
            if indegree[v] == 0: heapq.heappush(heap, (-float(priority[v]), int(v)))
    if len(order) != len(pool): raise RuntimeError("pool-not-dag")
    return order


def main() -> None:
    grid = json.loads((GDIR/"glacier_grid.json").read_text(encoding="utf-8")); x0,y0=map(float,grid["x0y0"]); dx,dy=map(float,grid["dxdy"])
    with xr.open_dataset(SOURCE) as ds:
        h=np.asarray(ds.thickness_m.isel(member=0),float); s=np.asarray(ds.surface_elevation_m.isel(member=0),float)
        positive=np.asarray(ds.valid_mask.isel(member=0),bool)&np.asarray(ds.lia_mask,bool)&np.isfinite(h)&np.isfinite(s)&(h>1e-6)
        yy,xx=np.meshgrid(np.asarray(ds.y,float),np.asarray(ds.x,float),indexing="ij"); cell_xy=np.column_stack([xx[positive],yy[positive]]); cell_s=s[positive]; cell_h=h[positive]
        cell_area=float(abs(np.diff(np.asarray(ds.x,float)).mean()*np.diff(np.asarray(ds.y,float)).mean()))
    with gzip.open(GDIR/"inversion_flowlines.pkl","rb") as stream: fls=pickle.load(stream)
    branch_by_object={id(fl):i for i,fl in enumerate(fls)}; offsets=[]; parts=[]; branch_of=[]; cursor=0
    for b,fl in enumerate(fls):
        offsets.append(cursor); c=np.asarray(fl.line.coords,float); parts.append(np.column_stack([x0+c[:,0]*dx,y0+c[:,1]*dy])); branch_of.extend([b]*fl.nx); cursor+=fl.nx
    node_xy=np.vstack(parts); branch_of=np.asarray(branch_of,int); nearest_distance,nearest=cKDTree(node_xy).query(cell_xy,k=1,workers=1)
    quota=np.bincount(nearest,minlength=len(node_xy)).astype(int); old_mean=np.bincount(nearest,weights=cell_s,minlength=len(node_xy))/quota
    cell_branch=branch_of[nearest]; owner=np.empty(len(cell_s),int)
    for b,fl in enumerate(fls):
        cells=np.flatnonzero(cell_branch==b); cells=cells[np.argsort(-cell_s[cells],kind="stable")]; p=0
        for j in range(fl.nx):
            n=offsets[b]+j; q=int(quota[n]); owner[cells[p:p+q]]=n; p+=q
        if p!=len(cells): raise RuntimeError("branch-quota")
    edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1): edges.append((offsets[b]+j,offsets[b]+j+1,"within"))
        if fl.flows_to is not None:
            r=branch_by_object[id(fl.flows_to)]; edges.append((offsets[b]+fl.nx-1,offsets[r]+int(fl.flows_to_indice),"junction"))
    neighbours=[set() for _ in range(len(node_xy))]
    for u,v,_ in edges: neighbours[u].add(v); neighbours[v].add(u)

    def surfaces(own): return np.bincount(own,weights=cell_s,minlength=len(node_xy))/quota
    def reversals(own):
        ns=surfaces(own); return [(u,v,k,float(ns[v]-ns[u])) for u,v,k in edges if ns[v]-ns[u]>1e-7]
    def score(own):
        rev=reversals(own); return (len(rev),max([r[3] for r in rev],default=0.0))
    def pool_nodes(seed,radius):
        seen=set(seed); queue=deque((n,0) for n in seed)
        while queue:
            n,d=queue.popleft()
            if d>=radius: continue
            for m in neighbours[n]:
                if m not in seen: seen.add(m); queue.append((m,d+1))
        return seen
    def candidate(own,pool):
        own2=own.copy(); ns=surfaces(own); order=topological_order(pool,edges,ns)
        mask=np.isin(own2,np.fromiter(pool,dtype=int)); cells=np.flatnonzero(mask); cells=cells[np.argsort(-cell_s[cells],kind="stable")]; p=0
        for n in order:
            q=int(quota[n]); own2[cells[p:p+q]]=n; p+=q
        if p!=len(cells): raise RuntimeError("pool-quota")
        return own2

    history=[]; radii=(1,2,4,8,12,20); cycles=0
    while cycles<6:
        cycles+=1; improved=False; current_rev=reversals(owner)
        if not current_rev: break
        for u,v,kind,gap in sorted(current_rev,key=lambda x:-x[3]):
            before=score(owner); accepted=False
            for radius in radii:
                pool=pool_nodes((u,v),radius); trial=candidate(owner,pool); after=score(trial)
                if after[0]<before[0] or (after[0]==before[0] and after[1]<before[1]-1e-9):
                    owner=trial; history.append({"cycle":cycles,"edge_u":u,"edge_v":v,"edge_kind":kind,"gap_before_m":gap,"radius":radius,"pool_node_count":len(pool),"reverse_count_before":before[0],"reverse_count_after":after[0],"maximum_gap_after_m":after[1]}); improved=True; accepted=True; break
            if not accepted: history.append({"cycle":cycles,"edge_u":u,"edge_v":v,"edge_kind":kind,"gap_before_m":gap,"radius":None,"pool_node_count":None,"reverse_count_before":before[0],"reverse_count_after":before[0],"maximum_gap_after_m":before[1]})
        if not improved: break
    ns=surfaces(owner); rev=reversals(owner); movement=np.linalg.norm(cell_xy-node_xy[owner],axis=1); adjustment=ns-old_mean
    cdf=cdf_error_100m(cell_s,ns,quota); source_volume=float(np.sum(cell_h)*cell_area)
    node_volume=np.bincount(owner,weights=cell_h*cell_area,minlength=len(node_xy)); mapped_volume=float(np.sum(node_volume))
    summary={
        "schema":"lia-191-system84-local-topological-pool-reassignment-v1","system_id":84,
        "status":"exact-cell-static-candidate-not-materialized" if not rev and np.max(np.abs(adjustment))<=150 and cdf<=0.02 else "deterministic-local-pool-fail",
        "cycles":cycles,"accepted_pool_updates":sum(x["radius"] is not None for x in history),"final_reverse_count":len(rev),
        "final_within_reverse_count":sum(x[2]=="within" for x in rev),"final_junction_reverse_count":sum(x[2]=="junction" for x in rev),"final_maximum_reverse_gap_m":max([x[3] for x in rev],default=0.0),
        "maximum_absolute_node_surface_adjustment_m":float(np.max(np.abs(adjustment))),"mean_absolute_node_surface_adjustment_m":float(np.average(np.abs(adjustment),weights=quota)),
        "hypsometry_cdf_error_100m":cdf,"cell_movement_mean_m":float(np.mean(movement)),"cell_movement_p95_m":float(np.percentile(movement,95)),"cell_movement_max_m":float(np.max(movement)),
        "source_area_m2":float(len(cell_s)*cell_area),"mapped_area_m2":float(np.sum(quota)*cell_area),"source_volume_m3":source_volume,"mapped_volume_m3":mapped_volume,"volume_error_fraction":(mapped_volume-source_volume)/source_volume,
        "integer_cell_unique":int(len(owner))==int(len(np.flatnonzero(owner>=0))),"per_node_integer_quota_preserved":bool(np.array_equal(np.bincount(owner,minlength=len(node_xy)),quota)),
        "materialized_geometry":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False,"update_history":history,
    }
    OUTPUT.mkdir(parents=True,exist_ok=False); (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
