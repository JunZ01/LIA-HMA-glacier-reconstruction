"""V2: build immutable zero-MB rescore overlays and four steady configs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

CASES={2328:("w228zero2328v1","w227inv2328v1"),2547:("w274zero2547mindt30v1","w264inv2547v1"),2580:("w273zero2580v1","w261inv2580v1"),5443:("w274zero5443mindt30v1","w270inv5443v1")}
RES="validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_legacy_four_formal_rescore_v1"
CFG="Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
    if not x: raise RuntimeError(m)
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f: json.dump(x,f,indent=2); f.write("\n")

def main():
    root=Path(__file__).resolve().parents[4]; records=[]
    manifest=root/"w725zero_rescore4_steady_manifest_v2.json"
    need(not manifest.exists(),f"collision:{manifest}")
    for sid,(zname,iname) in CASES.items():
        zroot=root/zname; iroot=root/iname
        rsp=root/RES/f"system_{sid}_formal_rescore.json"
        zsp=zroot/"zero_mb_summary.json"; zcp=zroot/"config_snapshot.json"
        isp=iroot/"inversion_summary_formal.json"; icp=iroot/"config_snapshot.json"
        for p in (rsp,zsp,zcp,isp,icp): need(p.is_file(),f"missing:{p}")
        r,zs,zc,isum,ic=map(load,(rsp,zsp,zcp,isp,icp))
        need(r["formal_status"]=="zero-mb-pass" and not r["formal_review_flags"],f"rescore:{sid}")
        need(r["source_summary_sha256"]==digest(zsp),f"hash:{sid}")
        need(isum["status"]=="inversion-pass",f"inversion:{sid}")
        need(list(ic["spinup_window"])==[1441,1470],f"window:{sid}")
        need(ic["stage2"]["prcp_anomaly_mode"]=="off",f"prcp:{sid}")
        oname=f"w725zero_rescore_overlay_{sid}_v1"; sname=f"w725steady_parent_rescore4_{sid}_v1"
        oroot=root/oname; sroot=root/sname; cp=root/CFG/f"steady_parent_rescore4_{sid}_v1.json"
        for p in (oroot,sroot,cp): need(not p.exists(),f"collision:{p}")
        fixed=dict(zs); fixed["status"]="zero-mb-pass"; fixed["review_flags"]=[]
        fixed["formal_rescore"]={"version":"legacy-four-per-step-closure-v1","source_root":zname,"source_summary_sha256":r["source_summary_sha256"],"criterion":r["closure_gate"],"maximum_absolute_step_closure_residual_m3":r["maximum_absolute_step_closure_residual_m3"],"per_step_closure_limit_m3":r["per_step_closure_limit_m3"],"evidence":str(rsp.relative_to(root))}
        put(oroot/"zero_mb_summary_formal.json",fixed); put(oroot/"config_snapshot.json",zc)
        cfg={"experiment":"conservative-one-way-tributary-fluxbased-v1","stage":"remaining-route-zero-rescore4-parent-steady-50y-v1","system_id":sid,"synthetic_rgi_id":ic["synthetic_rgi_id"],"inversion_root":{"anchor":"project","sub":iname},"zero_mb_root":{"anchor":"project","sub":oname},"output_root":{"anchor":"project","sub":sname},"climate_filesuffix":"_batch1_1441_1470_prcp_off","spinup_window":[1441,1470],"registered_lia_length_m":float(ic["registered_lia_length_m"]),"stage2":ic["stage2"],"gates":{"volume_change_fraction_max":.02,"lia_length_relative_error_max":.10,"minimum_free_downstream_m":5000.0},"authorization":{"historical":False}}
        put(cp,cfg)
        records.append({"system_id":sid,"source_zero_root":zname,"overlay_root":oname,"inversion_root":iname,"steady_config":str(cp.relative_to(root)),"steady_output_root":sname,"delta_t_c":float(ic["stage2"]["delta_t_c"]),"rescore_sha256":digest(rsp),"overlay_summary_sha256":digest(oroot/"zero_mb_summary_formal.json")})
    out={"status":"configs-built-not-run","count":4,"historical_authorized":False,"oggm_started":False,"records":records}
    put(manifest,out); print(json.dumps(out,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
