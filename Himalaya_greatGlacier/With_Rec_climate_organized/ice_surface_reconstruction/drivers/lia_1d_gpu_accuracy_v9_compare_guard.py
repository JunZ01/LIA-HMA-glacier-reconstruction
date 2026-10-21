from __future__ import annotations
import ctypes,hashlib,json
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[4]; R=PROJECT_ROOT/"validation_data/results/paleo_surface_validation_v0"; TRANSACTION=R/"lia_1d_gpu_accuracy_v9_compare_transaction_v1/transaction.json"; OUTPUT=R/"lia_1d_gpu_accuracy_v9_comparison"; GPU_SUMMARY=R/"lia_1d_gpu_accuracy_v9_fixed24_states/summary.json"; ORACLE_SUMMARY=R/"lia_1d_gpu_batch_preflight_v5/cpu/summary.json"; ORACLE_STATES=R/"lia_1d_gpu_batch_preflight_v5/cpu/oracle_states.npz"; GPU_PID=67424
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def alive(pid):
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not h:return False
    c=ctypes.c_ulong(); ok=ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(c)); ctypes.windll.kernel32.CloseHandle(h); return bool(ok and c.value==259)
def validate(output_absent=True):
    t=load(TRANSACTION); s=load(PROJECT_ROOT/"LIA_CURRENT_STATE.json"); bad=[]
    if alive(GPU_PID):bad.append("GPU alive")
    for p,k in ((GPU_SUMMARY,"gpu_summary_sha256"),(ORACLE_SUMMARY,"oracle_summary_sha256"),(ORACLE_STATES,"oracle_states_sha256")):
        if sha256(p)!=t["inputs"][k]:bad.append(k)
    b=s.get("research_branches",{}).get("lia_1d_gpu_accuracy_v9_comparison",{})
    if b.get("launch_authorized") is not True:bad.append("state")
    if output_absent and OUTPUT.exists():bad.append("output")
    if bad:raise RuntimeError("v9 compare guard failed: "+", ".join(bad))
    return {"transaction":t,"state":s}
