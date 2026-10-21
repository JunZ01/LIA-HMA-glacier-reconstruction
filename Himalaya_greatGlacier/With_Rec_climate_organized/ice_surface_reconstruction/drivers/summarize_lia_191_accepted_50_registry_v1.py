"""Create accepted-50 registry by adding independently reproduced system 5536."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VAL = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VAL / "lia_191_accepted_49_registry_v1/accepted_49_curve_metrics.csv"
HISTORY = ROOT / "w704hist_linear_5536_endp100_from_refine5_v1/historical_summary.json"
CHECK = VAL / "lia_191_reproduction_checks_v1/system_5536_linear_reproduction_v1.json"
OUT = VAL / "lia_191_accepted_50_registry_v1"

def main() -> None:
    h=json.loads(HISTORY.read_text(encoding="utf-8")); c=json.loads(CHECK.read_text(encoding="utf-8"))
    if h["status"]!="historical-pass" or c["status"]!="reproduction-pass": raise RuntimeError("5536-not-reproduced")
    with SOURCE.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f)); fields=list(rows[0])
    if any(int(r["system_id"])==5536 for r in rows): raise RuntimeError("5536-already-registered")
    lia=11666.956426940536; rgi=9182.62221768731; tol=.2*abs(lia-rgi)/1000
    rows.append({"system_id":"5536","rgi_id":"RGI60-14.17284","route":"formal-historical-reproduction-pass","lia_km":str(lia/1000),"rgi_km":str(rgi/1000),"lia_tol_km":str(tol),"rgi_tol_km":str(tol),"e_lia":str(h["e_lia_separation"]),"e_rgi":str(h["e_rgi_separation"]),"history_root":HISTORY.parent.name,"reproduction_root":"w704hist_linear_5536_endp100_from_refine5_v1_independent_reproduction_v1"})
    rows.sort(key=lambda r:int(r["system_id"])); OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/"accepted_50_curve_metrics.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    s={"status":"accepted-registry-complete","accepted_system_count":50,"remaining_system_count":141,"new_systems":[5536],"source_registry":str(SOURCE),"reproduction_check":str(CHECK)}
    (OUT/"summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8");print(json.dumps(s,indent=2))

if __name__=="__main__": main()
