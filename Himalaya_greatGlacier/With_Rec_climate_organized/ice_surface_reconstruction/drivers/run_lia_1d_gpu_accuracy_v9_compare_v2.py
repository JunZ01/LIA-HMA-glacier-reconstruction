from __future__ import annotations
import hashlib
from pathlib import Path
SOURCE=Path(__file__).with_name("run_lia_1d_gpu_v8_v2_offline_compare.py")
EXPECTED="2582BC34245C36C72BE9CC552DE9A01424D1F8C61C7F62DC6BF9369D55976163"
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest().upper()!=EXPECTED:raise RuntimeError("frozen comparator changed")
code=raw.decode("utf-8").replace("lia_1d_gpu_v8_v2_offline_compare_guard","lia_1d_gpu_accuracy_v9_compare_guard_v2").replace("lia_1d_gpu_only_v8_v2_fixed24_stage_states","lia_1d_gpu_accuracy_v9_fixed24_states").replace("lia_1d_gpu_v8_v2_offline_comparison_v1","lia_1d_gpu_accuracy_v9_comparison")
code=code.replace('"gpu_process_id":31188','"gpu_process_id":67424')
exec(compile(code,str(SOURCE)+"#v9-compare-v2","exec"),{"__name__":"__main__","__file__":str(SOURCE)})
