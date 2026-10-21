"""Refresh 2689 after minimum-section-relaxation static failure."""

from __future__ import annotations
import json, shutil
from pathlib import Path
import pandas as pd

PROJECT=Path(__file__).resolve().parents[4]
SOURCE=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_18"
OUTPUT=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_19"

def main():
    if OUTPUT.exists():raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for p in SOURCE.glob("*.csv"):shutil.copy2(p,OUTPUT/p.name)
    path=OUTPUT/"lia_191_current_route.csv";route=pd.read_csv(path);mask=route["lia_unique_id"].astype(int).eq(2689)
    route.loc[mask,"current_gate"]="minimum-section-relaxation-reservoir-static-fail"
    route.loc[mask,"latest_evidence_stage"]="static-section-relaxation-feasibility"
    route.loc[mask,"latest_evidence_status"]="static-fail"
    route.loc[mask,"latest_flags"]="q=0.815..0.700 all fail: optimized CDF remains >2pct and scaling creates 109-147 within plus4 junction reversals"
    route.loc[mask,"latest_evidence"]="validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2689_minimum_section_relaxation_reservoir_v1/summary.json"
    route.loc[mask,"next_legal_action"]="do not lower q below0.70 or regularize standard core; require a fully rebuilt current-paired state/dynamics family distinct from closed EB, branch-ensemble and paired-cell graph"
    route.loc[mask,"update_source"]="negative-flux2689-section-relaxation-v1"
    route.to_csv(path,index=False);remaining=route[~route["accepted"].astype(str).str.lower().eq("true")];remaining.to_csv(OUTPUT/"remaining_137_current_route.csv",index=False)
    accepted=int(route["accepted"].astype(str).str.lower().eq("true").sum())
    summary={"status":"current-route-snapshot-complete","population":len(route),"accepted":accepted,"remaining":len(route)-accepted,"controller_state_counts":route["controller_state"].value_counts().to_dict(),"mutually_exclusive_class_counts":route["mutually_exclusive_class"].value_counts().to_dict(),"active_systems":[],"active_process_count":0,"queue_rows":0,"actionable_now":0,"launchable_now":0,"supersedes":"lia_191_current_route_v10_18","new_decisions":{"2689":"minimum uniform positive-section relaxation plus passive reservoir closed at static gate; no q<0.70 or regularizer retry"},"driver":str(Path(__file__).resolve())}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
