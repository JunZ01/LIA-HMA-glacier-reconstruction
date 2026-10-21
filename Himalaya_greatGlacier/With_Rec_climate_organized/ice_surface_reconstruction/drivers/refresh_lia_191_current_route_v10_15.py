"""Route refresh after downstream receiver-reindex non-promotion for invalid outlines."""

from __future__ import annotations

import csv,json
from collections import Counter
from pathlib import Path


PROJECT=Path(__file__).resolve().parents[4]
SOURCE=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_14/lia_191_current_route.csv"
OUTPUT=PROJECT/"validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_15"
IDS={1120,1133,1499,1639,1653,2535,2692,3075,5014,5231,5508,5598}
EVIDENCE="validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline334_receiver_reindex_v1/summary.json"


def main():
    rows=list(csv.DictReader(SOURCE.open("r",encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) not in IDS:continue
        row.update({
            "current_gate":"representative-334-receiver-reindex-failed-no-promotion",
            "latest_evidence_stage":"geometry-topology-readonly-audit","latest_evidence_status":"not-run-after-representative-mathematical-failure",
            "latest_flags":"334 resolved2/6 reverse junctions; remaining moves require up to1550m and1373.7m spatial jump >250m",
            "latest_evidence":EVIDENCE,"next_legal_action":"do not retry downstream receiver reindex; requires a network-coupled confluence solver or a genuinely different control-volume topology",
            "update_source":"invalid-outline334-receiver-reindex-no-promotion-v1",
        })
    OUTPUT.mkdir(parents=True,exist_ok=False);fields=list(rows[0]);remaining=[row for row in rows if row["accepted"].lower()!="true"]
    for name,data in (("lia_191_current_route.csv",rows),("remaining_137_current_route.csv",remaining)):
        with (OUTPUT/name).open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    summary={"status":"current-route-snapshot-complete","population":191,"accepted":54,"remaining":137,
        "controller_state_counts":dict(Counter(row["controller_state"] for row in rows)),"mutually_exclusive_class_counts":dict(Counter(row["mutually_exclusive_class"] for row in rows)),
        "active_systems":[],"active_process_count":0,"queue_rows":0,"actionable_now":0,"launchable_now":0,"supersedes":"lia_191_current_route_v10_14",
        "new_decisions":{"invalid-outline-plus1-hole-class":"334 downstream receiver reindex resolves only2/6 and needs up to1.55km movement; no promotion to12 holds"},"driver":str(Path(__file__).resolve())}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":main()
