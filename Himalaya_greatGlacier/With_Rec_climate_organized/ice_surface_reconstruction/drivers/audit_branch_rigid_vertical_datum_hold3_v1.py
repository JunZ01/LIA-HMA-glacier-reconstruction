"""Read-only branch-rigid vertical-datum feasibility for three geometry holds."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import linprog


PROJECT = Path(__file__).resolve().parents[4]
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_13/remaining_137_current_route.csv"
GRID_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_branch_rigid_vertical_datum_hold3_v1"
SYSTEMS = {
    1886: PROJECT / "w401geom1886v1",
    2534: PROJECT / "w505geom2534v1",
    2559: PROJECT / "w265geom2559v1",
}
MAX_OFFSET_M = 100.0
MAX_CDF_ERROR = 0.02


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def cumulative_hypsometry_error(values_a, weights_a, values_b, weights_b):
    low = 100.0 * np.floor(min(np.min(values_a), np.min(values_b)) / 100.0)
    high = 100.0 * np.ceil(max(np.max(values_a), np.max(values_b)) / 100.0) + 100.0
    edges = np.arange(low, high + 1e-7, 100.0)
    hist_a, _ = np.histogram(values_a, bins=edges, weights=weights_a)
    hist_b, _ = np.histogram(values_b, bins=edges, weights=weights_b)
    cdf_a = np.cumsum(hist_a) / np.sum(hist_a); cdf_b = np.cumsum(hist_b) / np.sum(hist_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def solve_offsets(surfaces, receivers, branch_areas):
    count = len(surfaces)
    # Phase 1: minimize z=max|c| with exact area-weighted mean(c)=0.
    objective = np.r_[np.zeros(count), 1.0]
    aub = []; bub = []
    for branch, receiver, receiver_index in receivers:
        row = np.zeros(count + 1); row[receiver] = 1.0; row[branch] = -1.0
        aub.append(row); bub.append(float(surfaces[branch][-1] - surfaces[receiver][receiver_index]))
    for branch in range(count):
        row = np.zeros(count + 1); row[branch] = 1.0; row[-1] = -1.0; aub.append(row); bub.append(0.0)
        row = np.zeros(count + 1); row[branch] = -1.0; row[-1] = -1.0; aub.append(row); bub.append(0.0)
    aeq = np.zeros((1, count + 1)); aeq[0, :count] = branch_areas / np.sum(branch_areas)
    phase1 = linprog(objective, A_ub=np.asarray(aub), b_ub=np.asarray(bub), A_eq=aeq, b_eq=np.array([0.0]), bounds=[(None,None)]*count+[(0,None)], method="highs")
    if not phase1.success: return None, {"phase1_status": phase1.message}
    zstar = float(phase1.x[-1])
    # Phase 2: with z fixed, minimize area-weighted L1 displacement.
    objective2 = np.r_[np.zeros(count), branch_areas / np.sum(branch_areas)]
    aub2 = []; bub2 = []
    for branch, receiver, receiver_index in receivers:
        row = np.zeros(2*count); row[receiver] = 1.0; row[branch] = -1.0
        aub2.append(row); bub2.append(float(surfaces[branch][-1] - surfaces[receiver][receiver_index]))
    for branch in range(count):
        row=np.zeros(2*count); row[branch]=1.0; aub2.append(row); bub2.append(zstar+1e-8)
        row=np.zeros(2*count); row[branch]=-1.0; aub2.append(row); bub2.append(zstar+1e-8)
        row=np.zeros(2*count); row[branch]=1.0; row[count+branch]=-1.0; aub2.append(row); bub2.append(0.0)
        row=np.zeros(2*count); row[branch]=-1.0; row[count+branch]=-1.0; aub2.append(row); bub2.append(0.0)
    aeq2=np.zeros((1,2*count)); aeq2[0,:count]=branch_areas/np.sum(branch_areas)
    phase2=linprog(objective2,A_ub=np.asarray(aub2),b_ub=np.asarray(bub2),A_eq=aeq2,b_eq=np.array([0.0]),bounds=[(None,None)]*count+[(0,None)]*count,method="highs")
    if not phase2.success: return None,{"phase1_z_m":zstar,"phase2_status":phase2.message}
    return phase2.x[:count],{"phase1_z_m":zstar,"phase2_weighted_l1_m":float(np.dot(branch_areas/np.sum(branch_areas),np.abs(phase2.x[:count])))}


def main():
    route={int(row["lia_unique_id"]):row for row in csv.DictReader(ROUTE.open("r",encoding="utf-8-sig"))}
    system_rows=[]; offset_rows=[]
    for system_id,root in SYSTEMS.items():
        geometry=json.loads((root/"geometry_summary.json").read_text(encoding="utf-8")); gdir=Path(geometry["synthetic_gdir"])
        flow_path=gdir/"inversion_flowlines.pkl"; grid_path=GRID_ROOT/f"lia_{system_id}_{route[system_id]['rgi_id']}_system_grid_v1.nc"
        before={str(flow_path.relative_to(PROJECT)):sha256(flow_path),str(grid_path.relative_to(PROJECT)):sha256(grid_path)}
        with gzip.open(flow_path,"rb") as stream: flowlines=pickle.load(stream)
        index={id(flowline):i for i,flowline in enumerate(flowlines)}
        surfaces=[np.asarray(flowline.surface_h,dtype=float) for flowline in flowlines]
        weights=[np.asarray(flowline.widths_m,dtype=float)*float(flowline.dx_meter) for flowline in flowlines]
        branch_areas=np.array([float(np.sum(item)) for item in weights])
        receivers=[]
        for branch,flowline in enumerate(flowlines):
            if flowline.flows_to is not None: receivers.append((branch,index[id(flowline.flows_to)],int(flowline.flows_to_indice)))
        offsets,solver=solve_offsets(surfaces,receivers,branch_areas)
        if offsets is None:
            system_rows.append({"system_id":system_id,"branch_count":len(flowlines),"junction_count":len(receivers),"lp_feasible":False,"static_feasible":False,**solver}); continue
        shifted=[surface+offsets[i] for i,surface in enumerate(surfaces)]
        remaining=[]
        for branch,receiver,receiver_index in receivers:
            gap=float(shifted[receiver][receiver_index]-shifted[branch][-1]);
            if gap>1e-7: remaining.append(gap)
        node_values=np.concatenate(surfaces); shifted_values=np.concatenate(shifted); node_weights=np.concatenate(weights)
        with xr.open_dataset(grid_path) as ds:
            h=np.asarray(ds.thickness_m.isel(member=0),dtype=float); s=np.asarray(ds.surface_elevation_m.isel(member=0),dtype=float)
            positive=np.asarray(ds.valid_mask.isel(member=0),dtype=bool)&np.asarray(ds.lia_mask,dtype=bool)&np.isfinite(h)&np.isfinite(s)&(h>1e-6)
            source_values=s[positive]; source_weights=np.ones_like(source_values)
        original_cdf=cumulative_hypsometry_error(node_values,node_weights,source_values,source_weights)
        shifted_cdf=cumulative_hypsometry_error(shifted_values,node_weights,source_values,source_weights)
        weighted_mean=float(np.dot(branch_areas,offsets)/np.sum(branch_areas)); max_offset=float(np.max(np.abs(offsets)))
        after={str(flow_path.relative_to(PROJECT)):sha256(flow_path),str(grid_path.relative_to(PROJECT)):sha256(grid_path)}
        feasible=bool(max_offset<=MAX_OFFSET_M and abs(weighted_mean)<=1e-8 and not remaining and shifted_cdf<=MAX_CDF_ERROR and before==after)
        system_rows.append({
            "system_id":system_id,"branch_count":len(flowlines),"junction_count":len(receivers),"lp_feasible":True,
            "minimum_max_offset_m":solver["phase1_z_m"],"weighted_l1_offset_m":solver["phase2_weighted_l1_m"],
            "maximum_absolute_offset_m":max_offset,"area_weighted_mean_offset_m":weighted_mean,
            "remaining_junction_reverse_count":len(remaining),"maximum_remaining_junction_reverse_m":max(remaining) if remaining else 0.0,
            "original_hypsometry_cdf_error_100m":original_cdf,"shifted_hypsometry_cdf_error_100m":shifted_cdf,
            "source_immutable":before==after,"static_feasible":feasible,
        })
        for branch,offset in enumerate(offsets):
            offset_rows.append({"system_id":system_id,"branch_index":branch,"branch_area_weight_m2":branch_areas[branch],"offset_m":float(offset)})
    eligible=[row for row in system_rows if row.get("static_feasible")]
    representative=min(eligible,key=lambda row:(row["minimum_max_offset_m"],row["branch_count"]))["system_id"] if eligible else None
    OUTPUT.mkdir(parents=True,exist_ok=False)
    for name,data in (("systems.csv",system_rows),("branch_offsets.csv",offset_rows)):
        with (OUTPUT/name).open("w",newline="",encoding="utf-8-sig") as stream:
            fields=sorted({key for row in data for key in row}); writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(data)
    summary={
        "schema":"lia-191-branch-rigid-vertical-datum-hold3-v1","status":"representative-selected-static-materialization-not-run" if representative is not None else "class-mathematically-closed",
        "system_count":3,"eligible_count":len(eligible),"eligible_system_ids":[row["system_id"] for row in eligible],"selected_representative":representative,
        "mechanism":"per-branch rigid surface vertical offset; exact area-weighted mean-zero; lexicographic min max-absolute then weighted-L1",
        "gates":{"maximum_absolute_offset_m":MAX_OFFSET_M,"area_weighted_mean_absolute_m":1e-8,"junction_reverse_count":0,"hypsometry_cdf_error_100m":MAX_CDF_ERROR},
        "bed_available_pre_inversion":False,"thickness_or_volume_modified":False,"geometry_materialized":False,"inversion_run":False,"zero_mb_run":False,"historical_run":False,
    }
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary))


if __name__=="__main__": main()
