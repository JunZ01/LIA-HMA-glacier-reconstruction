"""Read-only downstream receiver-index feasibility for repaired system 334."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT=Path(__file__).resolve().parents[4]
GEOMETRY=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_input_geometry_plus1_pilot_v2/representative_334/topology_repair/system_334/geometry/geometry_summary.json"
OUTPUT=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline334_receiver_reindex_v1"
MAX_MOVE_M=250.0


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main():
    geometry=json.loads(GEOMETRY.read_text(encoding="utf-8")); gdir=Path(geometry["synthetic_gdir"]); path=gdir/"inversion_flowlines.pkl"
    before=sha256(path)
    with gzip.open(path,"rb") as stream: flowlines=pickle.load(stream)
    index={id(flowline):i for i,flowline in enumerate(flowlines)}; rows=[]
    for branch,flowline in enumerate(flowlines):
        if flowline.flows_to is None: continue
        receiver=index[id(flowline.flows_to)]; current=int(flowline.flows_to_indice)
        terminal_surface=float(flowline.surface_h[-1]); receiver_surface=np.asarray(flowline.flows_to.surface_h,dtype=float)
        original_gap=float(receiver_surface[current]-terminal_surface)
        candidates=np.flatnonzero(receiver_surface[current:]<=terminal_surface+1e-7)+current
        selected=int(candidates[0]) if candidates.size else None
        if selected is None:
            move=np.inf; spatial=np.inf; final_gap=np.inf
        else:
            move=float((selected-current)*flowline.flows_to.dx_meter)
            terminal_xy=np.asarray(flowline.line.coords,dtype=float)[-1]*float(flowline.map_dx)
            receiver_xy=np.asarray(flowline.flows_to.line.coords,dtype=float)[selected]*float(flowline.flows_to.map_dx)
            spatial=float(np.linalg.norm(receiver_xy-terminal_xy)); final_gap=float(receiver_surface[selected]-terminal_surface)
        rows.append({
            "branch_index":branch,"receiver_index":receiver,"current_receiver_node":current,
            "selected_receiver_node":selected,"original_junction_gap_m":original_gap,
            "downstream_index_move_m":move,"terminal_to_selected_spatial_distance_m":spatial,
            "final_junction_gap_m":final_gap,
            "candidate_exists":selected is not None,
            "move_gate_pass":bool(selected is not None and move<=MAX_MOVE_M and spatial<=MAX_MOVE_M),
            "was_originally_reverse":original_gap>1e-7,
        })
    reverse=[row for row in rows if row["was_originally_reverse"]]
    pass_all=bool(reverse and all(row["move_gate_pass"] and row["final_junction_gap_m"]<=1e-7 for row in reverse))
    after=sha256(path)
    summary={
        "schema":"lia-191-invalid-outline334-receiver-reindex-v1","status":"receiver-reindex-static-candidate-pass-materialization-not-run" if pass_all else "receiver-reindex-mechanism-closed",
        "system_id":334,"branch_count":len(flowlines),"junction_count":len(rows),"original_reverse_junction_count":len(reverse),
        "resolved_reverse_junction_count":sum(row["move_gate_pass"] and row["final_junction_gap_m"]<=1e-7 for row in reverse),
        "unresolved_reverse_junction_count":sum(not(row["move_gate_pass"] and row["final_junction_gap_m"]<=1e-7) for row in reverse),
        "maximum_allowed_downstream_or_spatial_move_m":MAX_MOVE_M,
        "maximum_required_downstream_move_m":max((row["downstream_index_move_m"] for row in reverse),default=0.0),
        "maximum_required_spatial_distance_m":max((row["terminal_to_selected_spatial_distance_m"] for row in reverse),default=0.0),
        "source_immutable":before==after,"surface_or_outline_modified":False,"receiver_indices_materialized":False,
        "inversion_run":False,"zero_mb_run":False,"historical_run":False,
        "next_action":"materialize receiver indices and rerun unchanged static geometry gates" if pass_all else "close downstream receiver-reindex mechanism for invalid-outline class",
    }
    OUTPUT.mkdir(parents=True,exist_ok=False)
    with (OUTPUT/"junctions.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8");print(json.dumps(summary,allow_nan=False))


if __name__=="__main__":main()
