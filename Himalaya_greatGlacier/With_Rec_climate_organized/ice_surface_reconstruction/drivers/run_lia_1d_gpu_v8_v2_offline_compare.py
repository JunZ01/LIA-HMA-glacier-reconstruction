"""CPU-only, post-GPU comparison of frozen v5 oracle and v8-v2 results."""
from __future__ import annotations
import hashlib, json, tempfile, time
from pathlib import Path
from lia_1d_gpu_v8_v2_offline_compare_guard import PROJECT_ROOT, ROOT, OUTPUT, GPU_SUMMARY, ORACLE_SUMMARY, ORACLE_STATES, validate

def sha256(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def write_new(p: Path, x: dict) -> None:
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=p.parent,delete=False) as f:
        json.dump(x,f,indent=2,ensure_ascii=False); f.write("\n"); q=Path(f.name)
    q.replace(p)
def rel(a: float,b: float) -> float: return abs(a-b)/max(abs(a),1e-300)

def main() -> int:
    gate=validate(); OUTPUT.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    import numpy as np
    cpu=json.loads(ORACLE_SUMMARY.read_text(encoding="utf-8"))
    gpu=json.loads(GPU_SUMMARY.read_text(encoding="utf-8"))
    tol=gate["transaction"]["tolerances"]
    stages=("after_step","after_month","after_year"); records=[]; all_pass=True
    with np.load(ORACLE_STATES) as oz:
        for stage in stages:
            gp=ROOT/f"lia_1d_gpu_only_v8_v2_fixed24_stage_states/states/{stage}.npz"
            gd=ROOT/f"lia_1d_gpu_only_v8_v2_fixed24_stage_states/diagnostics/{stage}.json"
            with np.load(gp) as gz: gt=gz["thick"]
            ct=oz[f"{stage}_thick"]
            delta=gt-ct; max_abs=float(np.max(np.abs(delta))); l2=float(np.linalg.norm(delta)/max(np.linalg.norm(ct),1e-300))
            di=json.loads(gd.read_text(encoding="utf-8")); lanes=[]
            for i,(cr,gr) in enumerate(zip(cpu["records"],di["lanes"])):
                cm=cr[stage]; initial=cr["initial_volume_m3"]
                checks={
                    "length":abs(gr["length_m"]-cm["length_m"]) <= tol["length_absolute_m"],
                    "area":rel(cm["area_m2"],gr["area_m2"]) <= tol["area_relative"],
                    "volume":rel(cm["volume_m3"],gr["volume_m3"]) <= tol["volume_relative"],
                    "climatic_mass_balance":rel(cm["cumulative_climatic_mass_balance_m3"],gr["cumulative_climatic_mass_balance_m3"]) <= tol["climatic_mass_balance_relative"],
                    "mass_budget_residual":abs(gr["cumulative_mass_budget_residual_m3"]-cm["cumulative_mass_budget_residual_m3"]) <= max(1e-3,1e-10*initial),
                    "accepted_steps":gr["accepted_steps"] == cm["accepted_steps"],
                    "rejected_steps":gr["rejected_steps"] == cm["rejected_steps"] == 0,
                    "minimum_timestep":abs(gr["minimum_actual_timestep_seconds"]-cm["minimum_actual_timestep_seconds"]) <= tol["minimum_actual_timestep_seconds_absolute"]
                }
                passed=all(checks.values()); all_pass &= passed
                lanes.append({"lane":i,"cell":cr["cell"],"pass":passed,"checks":checks,"relative_area":rel(cm["area_m2"],gr["area_m2"]),"relative_volume":rel(cm["volume_m3"],gr["volume_m3"]),"relative_climatic_mass_balance":rel(cm["cumulative_climatic_mass_balance_m3"],gr["cumulative_climatic_mass_balance_m3"]),"mass_budget_residual_delta_m3":abs(gr["cumulative_mass_budget_residual_m3"]-cm["cumulative_mass_budget_residual_m3"])})
            thick_pass=max_abs <= tol["thickness_max_absolute_m"] and l2 <= tol["thickness_relative_l2"]
            all_pass &= thick_pass
            records.append({"stage":stage,"pass":thick_pass and all(x["pass"] for x in lanes),"thickness_max_absolute_m":max_abs,"thickness_relative_l2":l2,"lanes":lanes})
    result={"schema":"lia-1d-gpu-v8-v2-offline-comparison-result-v1","status":"equivalence-pass" if all_pass else "equivalence-fail","gpu_process_id":31188,"gpu_process_confirmed_exited_before_start":True,"process_type":"cpu-only","torch_or_cuda_imported":False,"cell_count":24,"stages":records,"tolerances":tol,"inputs":{"gpu_summary_sha256":sha256(GPU_SUMMARY),"oracle_summary_sha256":sha256(ORACLE_SUMMARY),"oracle_states_sha256":sha256(ORACLE_STATES)},"checkpoint_run":False,"throughput_run":False,"full_191_gpu_run":False,"planform_2d_or_igm_run":False,"cpu12_or_cpu8_modified_or_controlled":False,"wall_seconds":time.perf_counter()-started}
    write_new(OUTPUT/"summary.json",result); print(json.dumps({"status":result["status"],"output":str(OUTPUT)},indent=2)); return 0 if all_pass else 2
if __name__ == "__main__": raise SystemExit(main())
