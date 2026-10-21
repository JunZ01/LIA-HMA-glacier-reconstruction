"""Route refresh after system-2005 closure and system-84 feasibility bounds."""

from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path

P=Path(__file__).resolve().parents[4]
SOURCE=P/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_2/lia_191_current_route.csv"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_3"
DECISION=P/"validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_decision_v1"

def main():
    rows=list(csv.DictReader(SOURCE.open("r",encoding="utf-8-sig")))
    for row in rows:
        sid=int(row["lia_unique_id"])
        if sid==2005:
            row.update({"current_route":"steady-mechanism-closed","current_gate":"connected-front-50y-internal-gap-fail","latest_evidence_stage":"steady-50y","latest_evidence_status":"steady-fail","latest_flags":"daily-1y-pass;zero-mb-5y-pass;long-run-branch10-internal-gap;historical-not-run","latest_evidence":"validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_decision_v1/summary.json","controller_state":"closed-current-branch","mechanism_class":"network-capacity-connected-front-exact-zero-v2-closed","next_legal_action":"requires a genuinely new internal-deglaciation/network-state representation; do not repeat timestep, zero-MB, steady, or historical","update_source":"lia_191_network_connected_front_system2005_decision_v1","active_process_count":"0","active_process_ids":"","mutually_exclusive_class":"closed-current-branch"})
        elif sid==84:
            row.update({"current_route":"geometry-optimization-hold","current_gate":"surface-moment-conserving-continuous-bounds-pass-discrete-optimizer-required","latest_evidence_stage":"geometry-discrete-feasibility","latest_evidence_status":"continuous-pass-current-greedy-fail","latest_flags":"continuous-max87.278m;continuous-mean11.414m;greedy-max162.906m;greedy-mean26.461m","latest_evidence":"validation_data/results/paleo_surface_validation_v0/lia_191_system84_native_surface_lp_bounds_v2/summary.json","controller_state":"hold","mechanism_class":"network-constrained-min-cost-integer-cell-assignment-required","next_legal_action":"implement one graph-constrained integer/min-cost assignment against the v2 continuous target; static gates only; do not repeat greedy pair swaps","update_source":"lia_191_system84_native_surface_lp_bounds_v2","active_process_count":"0","active_process_ids":"","mutually_exclusive_class":"hold-new-mechanism-required"})
    if len(rows)!=191 or len({r["lia_unique_id"] for r in rows})!=191:raise RuntimeError("population")
    OUT.mkdir(parents=True,exist_ok=False)
    fields=list(rows[0]);
    with (OUT/"lia_191_current_route.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    remaining=[r for r in rows if r["accepted"].lower()!="true"]
    with (OUT/"remaining_137_current_route.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(remaining)
    counts=Counter(r["controller_state"] for r in rows);classes=Counter(r["mutually_exclusive_class"] for r in rows)
    summary={"status":"current-route-snapshot-complete","population":191,"accepted":sum(r["accepted"].lower()=="true" for r in rows),"remaining":len(remaining),"controller_state_counts":dict(counts),"mutually_exclusive_class_counts":dict(classes),"active_systems":[],"active_process_count":0,"queue_rows":0,"actionable_now":1,"launchable_now":0,"supersedes":"lia_191_current_route_v10_2","new_decisions":{"2005":"connected-front exact-zero passed 5y zero-MB and 1y daily climate but failed 50y steady at a branch-10 internal gap","84":"surface-moment-conserving continuous formal bounds pass; current greedy integer realization fails, so one constrained integer optimizer remains legal"},"driver":str(Path(__file__).resolve())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    DECISION.mkdir(parents=True,exist_ok=False)
    decision={"schema":"lia-191-network-connected-front-system2005-decision-v1","system_id":2005,"status":"mechanism-closed-at-50y-steady-gate","mechanism":"network-capacity-connected-front-exact-zero-v2","zero_mb_0p1y":"pass","zero_mb_5y":"pass","zero_mb_5y_volume_drift_fraction":-7.873944443431242e-14,"daily_climate_1y":"pass","daily_climate_1y_volume_change_fraction":-0.0034368397999864266,"steady_50y":"fail-incomplete","steady_failure":"preexisting-internal-gap:branch10","failure_year":"not persisted by atomic-on-success v2 runner; decisive failure does not depend on exact year","historical_run":False,"promotion":False,"do_not_repeat":["timestep refinement","same connected-active MB mask","zero-MB","historical"],"evidence":{"zero_5y":"validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_zero_5y_dt24h_v2/summary.json","climate_1y":"validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_climate_1y_daily_v2/summary.json","steady_50y":"validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_steady_50y_daily_v2/summary.json"}}
    (DECISION/"summary.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
