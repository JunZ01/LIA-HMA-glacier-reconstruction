"""Fail-closed guard for the non-overlapping CPU-only v8-v2 comparison."""
from __future__ import annotations
import ctypes, hashlib, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION = ROOT / "lia_1d_gpu_v8_v2_offline_comparison_transaction_v1/transaction.json"
OUTPUT = ROOT / "lia_1d_gpu_v8_v2_offline_comparison_v1"
GPU_SUMMARY = ROOT / "lia_1d_gpu_only_v8_v2_fixed24_stage_states/summary.json"
ORACLE_SUMMARY = ROOT / "lia_1d_gpu_batch_preflight_v5/cpu/summary.json"
ORACLE_STATES = ROOT / "lia_1d_gpu_batch_preflight_v5/cpu/oracle_states.npz"
GPU_PID = 31188

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest().upper()

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def pid_alive(pid: int) -> bool:
    handle=ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle: return False
    code=ctypes.c_ulong()
    ok=ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok and code.value == 259)

def validate(output_absent: bool=True) -> dict:
    t=load(TRANSACTION); s=load(PROJECT_ROOT/"LIA_CURRENT_STATE.json"); failures=[]
    if pid_alive(GPU_PID): failures.append("GPU child still alive")
    if load(GPU_SUMMARY).get("status") != "gpu-only-stage-states-complete": failures.append("GPU phase not complete")
    for p,k in ((GPU_SUMMARY,"gpu_summary_sha256"),(ORACLE_SUMMARY,"oracle_summary_sha256"),(ORACLE_STATES,"oracle_states_sha256")):
        if sha256(p) != t.get("inputs",{}).get(k): failures.append(f"input changed: {p.name}")
    b=s.get("research_branches",{}).get("lia_1d_gpu_v8_v2_offline_comparison",{})
    if b.get("launch_authorized") is not True: failures.append("comparison not authorized in state")
    if b.get("gpu_process_overlap_allowed") is not False: failures.append("non-overlap rule missing")
    if output_absent and OUTPUT.exists(): failures.append("comparison output exists")
    if failures: raise RuntimeError("offline comparison guard failed: "+"; ".join(failures))
    return {"transaction":t,"state":s}
