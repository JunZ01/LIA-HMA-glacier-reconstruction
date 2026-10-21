"""Route refresh after branch-rigid vertical-datum closure for three holds."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT=Path(__file__).resolve().parents[4]
SOURCE=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_13/lia_191_current_route.csv"
OUTPUT=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_14"
IDS={1886,2534,2559}
EVIDENCE="validation_data/results/paleo_surface_validation_v0/lia_191_branch_rigid_vertical_datum_hold3_v1/summary.json"


def main():
    rows=list(csv.DictReader(SOURCE.open("r",encoding="utf-8-sig")))
    flags={1886:"min-max-offset115.354m>100m;shifted-CDF5.443pct",2534:"shifted-CDF3.703pct>2pct",2559:"shifted-CDF3.069pct>2pct"}
    for row in rows:
        system_id=int(row["lia_unique_id"])
        if system_id not in IDS: continue
        row.update({
            "current_route":"branch-rigid-vertical-datum-mechanism-closed",
            "current_gate":"rigid-offset-hypsometry-or-adjustment-fail",
            "latest_evidence_stage":"geometry-static-mathematical-audit",
            "latest_evidence_status":"mathematical-fail",
            "latest_flags":flags[system_id],"latest_evidence":EVIDENCE,
            "controller_state":"closed-current-branch","mechanism_class":"branch-rigid-vertical-datum-closed",
            "next_legal_action":"requires a topology/control-volume representation that resolves confluences without branch-wide elevation shifts; do not retry rigid offsets or node smoothing",
            "update_source":"branch-rigid-vertical-datum-hold3-v1","mutually_exclusive_class":"closed-current-branch",
        })
    OUTPUT.mkdir(parents=True,exist_ok=False); fields=list(rows[0]); remaining=[row for row in rows if row["accepted"].lower()!="true"]
    for name,data in (("lia_191_current_route.csv",rows),("remaining_137_current_route.csv",remaining)):
        with (OUTPUT/name).open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(data)
    summary={"status":"current-route-snapshot-complete","population":191,"accepted":54,"remaining":137,
        "controller_state_counts":dict(Counter(row["controller_state"] for row in rows)),
        "mutually_exclusive_class_counts":dict(Counter(row["mutually_exclusive_class"] for row in rows)),
        "active_systems":[],"active_process_count":0,"queue_rows":0,"actionable_now":0,"launchable_now":0,
        "supersedes":"lia_191_current_route_v10_13",
        "new_decisions":{"1886/2534/2559":"branch-rigid mean-zero offsets remove junction reversals but all fail frozen max-adjustment or 100m hypsometry gates; no materialization"},
        "driver":str(Path(__file__).resolve())}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (OUTPUT/"pending_fact_section.md").write_text(
        "# Pending LIA facts section: branch-rigid vertical-datum closure\n\n"
        "- Systems 1886, 2534, and 2559 were audited with one constant surface-elevation offset per native branch, exact area-weighted mean offset zero, and lexicographic minimum maximum-absolute then weighted-L1 displacement. Within-branch slopes, horizontal topology, widths, and source files remained unchanged; no geometry was materialized.\n"
        "- All three LPs can remove every junction reversal, but none passes the frozen static gates. System 1886 requires 115.354 m maximum offset and yields 5.443% 100 m hypsometry CDF error. Systems 2534 and 2559 require only 45.111/38.793 m but yield 3.703%/3.069% CDF error.\n"
        "- Decision: close branch-rigid vertical-datum reconciliation for this three-system hold class. No inversion, zero-MB, steady, or historical run occurred.\n",
        encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
