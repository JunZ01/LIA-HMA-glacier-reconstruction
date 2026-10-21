"""Route refresh after system-84 solver pause and system-1014 static closure."""

from __future__ import annotations

import csv,json
from collections import Counter
from pathlib import Path

P=Path(__file__).resolve().parents[4]
SRC=P/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_3/lia_191_current_route.csv"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_4"
DEC=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_support_geometry_decision_v1"

def main():
    rows=list(csv.DictReader(SRC.open("r",encoding="utf-8-sig")))
    for r in rows:
        sid=int(r["lia_unique_id"])
        if sid==84:
            r.update({"current_gate":"continuous-feasible-local-group-lp-technical-unresolved","latest_evidence_status":"continuous-pass-discrete-and-group-solvers-unresolved","latest_flags":"formal-continuous-pass;greedy-discrete-fail;group-lp-timeout;reduced-lp-highs-error","latest_evidence":"validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_group_transport_lp_v2/summary.json","next_legal_action":"pause this optimizer family; require a different exact min-cost/MILP implementation without further group shrinking","update_source":"lia_191_system84_local_group_transport_lp_v2"})
        elif sid==1014:
            r.update({"current_route":"input-geometry-mechanism-hold","current_gate":"raster-support-outline-static-junction-fail","latest_evidence_stage":"geometry","latest_evidence_status":"geometry-fail","latest_flags":"outline-pass;area-error4.663pct;root-error1.740pct;free7.897km;within0;junction5","latest_evidence":"Himalaya_greatGlacier/With_Rec_climate_organized/w1014rasteroutlinev1/geometry/geometry_summary.json","controller_state":"hold","mechanism_class":"raster-support-outline-native-multiflowline-closed","next_legal_action":"await a new confluence-capable geometry representation; do not promote raster-support outline to the remaining invalid-polygon class","update_source":"lia_191_system1014_raster_support_geometry_decision_v1","mutually_exclusive_class":"hold-new-mechanism-required"})
    OUT.mkdir(parents=True,exist_ok=False);fields=list(rows[0])
    for name,data in (("lia_191_current_route.csv",rows),("remaining_137_current_route.csv",[r for r in rows if r["accepted"].lower()!="true"])):
        with (OUT/name).open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    counts=Counter(r["controller_state"] for r in rows);classes=Counter(r["mutually_exclusive_class"] for r in rows)
    summary={"status":"current-route-snapshot-complete","population":191,"accepted":54,"remaining":137,"controller_state_counts":dict(counts),"mutually_exclusive_class_counts":dict(classes),"active_systems":[],"active_process_count":0,"queue_rows":0,"actionable_now":0,"launchable_now":0,"supersedes":"lia_191_current_route_v10_3","new_decisions":{"84":"continuous formal bounds pass, but both current discrete and localized LP implementations are unresolved/failed; pause optimizer family","1014":"raster-support outline passes preflight but native static fails five junction reverse slopes; no inversion or promotion"},"driver":str(Path(__file__).resolve())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    DEC.mkdir(parents=True,exist_ok=False);decision={"schema":"lia-191-system1014-raster-support-geometry-decision-v1","system_id":1014,"status":"mechanism-closed-at-static-junction-gate","outline_preflight":"pass","derived_outline_area_m2":63262620.0,"source_derived_iou":0.9626587780428482,"geometry_status":"geometry-fail","outline_area_relative_error":0.04662911526585525,"root_length_relative_error":0.017403681076979417,"downstream_free_length_m":7897.193600171105,"within_branch_reverse_slope_count":0,"junction_reverse_slope_count":5,"inversion_run":False,"zero_mb_run":False,"historical_run":False,"promotion":False,"evidence":{"outline":"validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_support_outline_v1/summary.json","geometry":"Himalaya_greatGlacier/With_Rec_climate_organized/w1014rasteroutlinev1/geometry/geometry_summary.json"}}
    (DEC/"summary.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
