"""Promote independently reproduced systems 2328, 2547, and 5168."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VAL = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VAL / "lia_191_accepted_50_registry_v1/accepted_50_curve_metrics.csv"
REFS = VAL / "lia_191_controller_v1/core191_length_references.csv"
OUT = VAL / "lia_191_accepted_53_registry_v1"
CASES = {
    2328: ("w730hist_power_2328_endm0p55_g0p25_from_w726steady_v1", "w730hist_power_2328_endm0p55_g0p25_from_w726steady_v1_independent_reproduction_v1", VAL / "lia_191_reproduction_checks_v1/system_2328_power_reproduction_v1.json"),
    2547: ("w734hist_linear_2547_endp0p625_from_w733steady_v1", "w738hist_linear_2547_endp0p625_from_w733steady_reproduction_v1", VAL / "lia_191_2547_historical_reproduction_v1/comparison.json"),
    5168: ("w723repro5168_linear_e1p0_v1", "w723repro5168_linear_e1p0_v1_independent_reproduction_v1", VAL / "lia_191_reproduction_checks_v1/system_5168_linear_reproduction_v1.json"),
}

def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows=list(csv.DictReader(f)); fields=list(rows[0])
    with REFS.open(encoding="utf-8-sig", newline="") as f:
        refs={int(r["lia_unique_id"]):r for r in csv.DictReader(f)}
    existing={int(r["system_id"]) for r in rows}
    evidence=[]
    for sid,(history_root,repro_root,check_path) in CASES.items():
        if sid in existing: raise RuntimeError(f"already-registered:{sid}")
        h=json.loads((ROOT/history_root/"historical_summary.json").read_text(encoding="utf-8"))
        c=json.loads(check_path.read_text(encoding="utf-8"))
        if h["status"]!="historical-pass" or c["status"]!="reproduction-pass": raise RuntimeError(f"not-reproduced:{sid}")
        ref=refs[sid]; lia=float(ref["lia_registered_length_m"]); rgi=float(ref["rgi_registered_length_m"]); tol=.2*abs(lia-rgi)/1000
        rows.append({"system_id":str(sid),"rgi_id":ref["rgi_id"],"route":"formal-historical-reproduction-pass","lia_km":str(lia/1000),"rgi_km":str(rgi/1000),"lia_tol_km":str(tol),"rgi_tol_km":str(tol),"e_lia":str(h["e_lia_separation"]),"e_rgi":str(h["e_rgi_separation"]),"history_root":history_root,"reproduction_root":repro_root})
        evidence.append(str(check_path))
    rows.sort(key=lambda r:int(r["system_id"])); OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/"accepted_53_curve_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={"status":"accepted-registry-complete","accepted_system_count":53,"remaining_system_count":138,"new_systems":sorted(CASES),"source_registry":str(SOURCE),"reproduction_checks":evidence}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
