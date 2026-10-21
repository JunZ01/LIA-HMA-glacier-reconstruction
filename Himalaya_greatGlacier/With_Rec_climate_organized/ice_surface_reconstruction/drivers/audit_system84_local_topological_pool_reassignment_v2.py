"""Formal-native-surface adjudication of system-84 local exact-cell pools."""

from __future__ import annotations

import csv, gzip, heapq, json, pickle
from collections import deque
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m
from .audit_system84_local_topological_pool_reassignment_v1 import topological_order

P=Path(__file__).resolve().parents[4]
SRC=P/"validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
GD=P/"validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2"

def main():
    grid=json.loads((GD/"glacier_grid.json").read_text(encoding="utf-8")); x0,y0=map(float,grid["x0y0"]); dx,dy=map(float,grid["dxdy"])
    with xr.open_dataset(SRC) as ds:
        h=np.asarray(ds.thickness_m.isel(member=0),float); s=np.asarray(ds.surface_elevation_m.isel(member=0),float)
        m=np.asarray(ds.valid_mask.isel(member=0),bool)&np.asarray(ds.lia_mask,bool)&np.isfinite(h)&np.isfinite(s)&(h>1e-6)
        yy,xx=np.meshgrid(np.asarray(ds.y,float),np.asarray(ds.x,float),indexing="ij"); xy=np.column_stack([xx[m],yy[m]]); cs=s[m]; ch=h[m]
        ca=float(abs(np.diff(np.asarray(ds.x,float)).mean()*np.diff(np.asarray(ds.y,float)).mean()))
    with gzip.open(GD/"inversion_flowlines.pkl","rb") as f: fls=pickle.load(f)
    byid={id(fl):i for i,fl in enumerate(fls)}; offs=[]; parts=[]; bof=[]; native=[]; cur=0
    for b,fl in enumerate(fls):
        offs.append(cur); c=np.asarray(fl.line.coords,float); parts.append(np.column_stack([x0+c[:,0]*dx,y0+c[:,1]*dy])); bof.extend([b]*fl.nx); native.extend(np.asarray(fl.surface_h,float)); cur+=fl.nx
    nxy=np.vstack(parts); bof=np.asarray(bof,int); native=np.asarray(native,float); ndist,nearest=cKDTree(nxy).query(xy,k=1,workers=1)
    quota=np.bincount(nearest,minlength=len(nxy)).astype(int); cell_branch=bof[nearest]; owner=np.empty(len(cs),int)
    for b,fl in enumerate(fls):
        cells=np.flatnonzero(cell_branch==b); cells=cells[np.argsort(-cs[cells],kind="stable")]; q0=0
        for j in range(fl.nx):
            node=offs[b]+j; q=int(quota[node]); owner[cells[q0:q0+q]]=node; q0+=q
    edges=[]
    for b,fl in enumerate(fls):
        for j in range(fl.nx-1): edges.append((offs[b]+j,offs[b]+j+1,"within"))
        if fl.flows_to is not None: r=byid[id(fl.flows_to)]; edges.append((offs[b]+fl.nx-1,offs[r]+int(fl.flows_to_indice),"junction"))
    adj=[set() for _ in range(len(nxy))]
    for u,v,_ in edges: adj[u].add(v); adj[v].add(u)
    def surf(o): return np.bincount(o,weights=cs,minlength=len(nxy))/quota
    def rev(o):
        ns=surf(o); return [(u,v,k,float(ns[v]-ns[u])) for u,v,k in edges if ns[v]-ns[u]>1e-7]
    def score(o):
        r=rev(o); return len(r),max([x[3] for x in r],default=0.)
    def pool(seed,radius):
        seen=set(seed); dq=deque((x,0) for x in seed)
        while dq:
            x,d=dq.popleft()
            if d>=radius: continue
            for y in adj[x]:
                if y not in seen: seen.add(y); dq.append((y,d+1))
        return seen
    def reorder(o,pn):
        o2=o.copy(); ns=surf(o); order=topological_order(pn,edges,ns); cells=np.flatnonzero(np.isin(o2,np.fromiter(pn,int))); cells=cells[np.argsort(-cs[cells],kind="stable")]; p0=0
        for node in order: q=int(quota[node]); o2[cells[p0:p0+q]]=node; p0+=q
        return o2
    history=[]
    for cycle in range(1,7):
        improved=False; current=rev(owner)
        if not current: break
        for u,v,k,gap in sorted(current,key=lambda x:-x[3]):
            before=score(owner)
            for radius in (1,2,4,8,12,20):
                pn=pool((u,v),radius); trial=reorder(owner,pn); after=score(trial)
                if after[0]<before[0] or (after[0]==before[0] and after[1]<before[1]-1e-9):
                    owner=trial; history.append((cycle,u,v,k,gap,radius,len(pn),before[0],after[0])); improved=True; break
        if not improved: break
    ns=surf(owner); r=rev(owner); native_delta=ns-native; cdf=cdf_error_100m(cs,ns,quota); movement=np.linalg.norm(xy-nxy[owner],axis=1)
    volume=float(np.sum(ch)*ca); mapped_volume=float(np.sum(np.bincount(owner,weights=ch*ca,minlength=len(nxy))))
    gates={"reverse":len(r)==0,"maximum_native_surface_adjustment":float(np.max(np.abs(native_delta)))<=150.,"mean_native_surface_adjustment":float(np.average(np.abs(native_delta),weights=quota))<=20.,"hypsometry":cdf<=.02,"area_volume":abs(mapped_volume-volume)<=max(1e-6,volume*1e-12),"integer_quota":bool(np.array_equal(np.bincount(owner,minlength=len(nxy)),quota))}
    summary={"schema":"lia-191-system84-local-topological-pool-reassignment-v2","system_id":84,"status":"formal-native-static-candidate-not-materialized" if all(gates.values()) else "formal-native-static-gate-fail","checks":gates,"final_reverse_count":len(r),"maximum_native_surface_adjustment_m":float(np.max(np.abs(native_delta))),"mean_native_surface_adjustment_m":float(np.average(np.abs(native_delta),weights=quota)),"hypsometry_cdf_error_100m":cdf,"cell_movement_mean_m":float(np.mean(movement)),"cell_movement_p95_m":float(np.percentile(movement,95)),"cell_movement_max_m":float(np.max(movement)),"source_area_m2":float(len(cs)*ca),"mapped_area_m2":float(np.sum(quota)*ca),"source_volume_m3":volume,"mapped_volume_m3":mapped_volume,"accepted_pool_update_count":len(history),"materialized_geometry":False,"oggm_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False); (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(OUT/"assignment_audit_arrays.npz",owner=owner,node_surface_m=ns,node_quota=quota,native_surface_m=native,cell_surface_m=cs,cell_thickness_m=ch,cell_xy_m=xy,node_xy_m=nxy)
    with (OUT/"node_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["node","branch","quota","native_surface_m","mapped_surface_m","adjustment_m"])
        for i in range(len(nxy)): w.writerow([i,int(bof[i]),int(quota[i]),float(native[i]),float(ns[i]),float(native_delta[i])])
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
