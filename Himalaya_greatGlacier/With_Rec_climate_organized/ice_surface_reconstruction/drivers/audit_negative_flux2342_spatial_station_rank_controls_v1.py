"""Read-only spatial-station-rank control-volume audit for system 2342.

Cells remain on their nearest stored LIA branch.  Within each branch they are
ordered by nearest network station and split contiguously into the exact number
of 50 m controls implied by that branch's spatial length (the main branch uses
the registered LIA length).  No elevation sorting or surface regularization is
allowed.  The audit tests whether this raw spatial ordering is already a valid
monotonic, conservative state.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .build_negative_flux2342_decoupled_branch_elevation_bands_v1 import (
    CONFIG, FORMAL, NETWORK, PAIRED, PROJECT, sha256, cdf_error,
)


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_spatial_station_rank_controls_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    branches = network["branches"]
    main_index = int(network["main_branch_index"])
    with xr.open_dataset(PAIRED) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    dx_cell = abs(transform[1]); dy_cell = abs(transform[5]); cell_area = dx_cell * dy_cell
    positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
    rr, cc = np.nonzero(positive)
    xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])
    all_nodes=[]; all_branch=[]; all_station=[]
    branch_lengths={}
    for branch in branches:
        idx=int(branch["branch_index"]); coords=np.asarray(branch["coords_projected_m"],float)
        station=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(coords,axis=0),axis=1))])
        branch_lengths[idx]=float(station[-1]); all_nodes.append(coords)
        all_branch.extend([idx]*len(coords)); all_station.extend(station.tolist())
    all_nodes=np.vstack(all_nodes); all_branch=np.asarray(all_branch,int); all_station=np.asarray(all_station,float)
    distance, nearest=cKDTree(all_nodes).query(xy,k=1)
    assigned_branch=all_branch[nearest]; nearest_station=all_station[nearest]

    source_surface=surface[positive]; source_thick=thick[positive]
    rows=[]; node_rows=[]; model_surface=[]; model_area=[]; reverse_total=0
    total_area=0.0; total_volume=0.0; all_movement=[]; failures=[]
    for branch in branches:
        idx=int(branch["branch_index"]); select=np.flatnonzero(assigned_branch==idx)
        target=float(config["registered_lia_length_m"]) if idx==main_index else branch_lengths[idx]
        n=max(2,int(round(target/50.0))); exact_dx=target/n
        if len(select)<n:
            failures.append({"branch_index":idx,"reason":"fewer-cells-than-controls","cells":len(select),"controls":n})
            continue
        order=select[np.argsort(nearest_station[select],kind="stable")]
        groups=np.array_split(order,n)
        branch_surfaces=[]; branch_areas=[]; branch_volumes=[]; movements=[]
        for j,group in enumerate(groups):
            area=float(len(group)*cell_area); volume=float(np.sum(source_thick[group])*cell_area)
            s=float(np.mean(source_surface[group])); h=volume/area
            original_scaled=nearest_station[group]/branch_lengths[idx]*target if branch_lengths[idx]>0 else np.zeros(len(group))
            destination=(j+0.5)*exact_dx
            movements.extend(np.abs(original_scaled-destination).tolist())
            branch_surfaces.append(s); branch_areas.append(area); branch_volumes.append(volume)
            node_rows.append({"branch_index":idx,"node_index":j,"distance_m":destination,"surface_m":s,"bed_m":s-h,"thickness_m":h,"width_m":area/exact_dx,"area_m2":area,"volume_m3":volume,"source_cell_count":len(group)})
        branch_surfaces=np.asarray(branch_surfaces); branch_areas=np.asarray(branch_areas); branch_volumes=np.asarray(branch_volumes)
        reverses=int(np.sum(np.diff(branch_surfaces)>1e-7)); reverse_total+=reverses
        total_area+=float(np.sum(branch_areas)); total_volume+=float(np.sum(branch_volumes))
        model_surface.extend(branch_surfaces.tolist()); model_area.extend(branch_areas.tolist()); all_movement.extend(movements)
        rows.append({"branch_index":idx,"is_main":idx==main_index,"source_cell_count":len(select),"control_count":n,"target_length_m":target,"dx_m":exact_dx,"reverse_count":reverses,"maximum_reverse_m":float(max(0.0,np.max(np.diff(branch_surfaces)))) if n>1 else 0.0,"movement_p95_m":float(np.quantile(movements,.95)),"movement_max_m":float(np.max(movements)),"area_m2":float(np.sum(branch_areas)),"volume_m3":float(np.sum(branch_volumes))})
    pd.DataFrame(rows).to_csv(OUTPUT/"branch_audit.csv",index=False)
    pd.DataFrame(node_rows).to_csv(OUTPUT/"control_nodes.csv",index=False)
    if failures: pd.DataFrame(failures).to_csv(OUTPUT/"failures.csv",index=False)
    source_area=float(len(source_surface)*cell_area); source_volume=float(np.sum(source_thick)*cell_area)
    cdf=cdf_error(source_surface,np.full(len(source_surface),cell_area),np.asarray(model_surface),np.asarray(model_area)) if model_surface else float("inf")
    main_row=next((row for row in rows if row["is_main"]),None)
    main_length_error=abs(main_row["target_length_m"]-float(config["registered_lia_length_m"]))/float(config["registered_lia_length_m"]) if main_row else float("inf")
    gates={
        "all_cells_assigned_once":int(sum(row["source_cell_count"] for row in rows))==len(source_surface),
        "all_branches_retained":len(rows)==len(branches) and not failures,
        "area_conserved":abs(total_area-source_area)<=1e-5,
        "volume_conserved":abs(total_volume-source_volume)<=1e-3,
        "hypsometry_within_2pct":cdf<=.02,
        "main_length_within_10pct":main_length_error<=.10,
        "raw_spatial_controls_monotonic":reverse_total==0,
        "movement_p95_within_1km":bool(all_movement and np.quantile(all_movement,.95)<=1000.0),
        "movement_max_within_2p5km":bool(all_movement and np.max(all_movement)<=2500.0),
        "paired_matches_formal_area":abs(source_area-float(formal["paired_area_m2"]))<=1e-6,
        "paired_matches_formal_volume":abs(source_volume-float(formal["paired_volume_m3"]))<=1e-3,
    }
    passed=all(gates.values())
    summary={
        "schema":"lia-191-negative-flux2342-spatial-station-rank-controls-v1",
        "status":"static-feasibility-pass" if passed else "static-feasibility-fail",
        "decision":"materialization-authorized" if passed else "mechanism-closed-at-raw-spatial-gate",
        "system_id":2342,"branch_count":len(branches),"main_branch_index":main_index,
        "source_cell_count":len(source_surface),"source_area_m2":source_area,"candidate_area_m2":total_area,
        "source_volume_m3":source_volume,"candidate_volume_m3":total_volume,"hypsometry_cdf_error_100m":cdf,
        "total_reverse_count":reverse_total,"movement_p95_m":None if not all_movement else float(np.quantile(all_movement,.95)),"movement_max_m":None if not all_movement else float(np.max(all_movement)),
        "gate_results":gates,"failures":failures,"surface_or_bed_regularization_used":False,
        "elevation_sorting_used":False,"oggm_run":False,"zero_mb_run":False,"steady_50y_run":False,"historical_run":False,
        "provenance":{"paired_sha256":sha256(PAIRED),"network_sha256":sha256(NETWORK),"formal_sha256":sha256(FORMAL),"config_sha256":sha256(CONFIG)},
    }
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=="__main__": main()
