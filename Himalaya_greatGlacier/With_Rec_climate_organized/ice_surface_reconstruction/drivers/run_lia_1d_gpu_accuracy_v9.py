"""Fixed-source v9 GPU accuracy runner with annual MB cache."""
from __future__ import annotations
import hashlib
from pathlib import Path

SOURCE=Path(__file__).with_name("run_lia_1d_gpu_only_v8.py")
EXPECTED="F9C5032EE9C1FEC322917EF2AADE1C6228D57A8873EF4CFFA4D033B14B5A2601"
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest().upper()!=EXPECTED: raise RuntimeError("frozen source changed")
code=raw.decode("utf-8")
def one(old,new):
    global code
    if code.count(old)!=1: raise RuntimeError(f"replacement count != 1: {old}")
    code=code.replace(old,new,1)
one("from lia_1d_gpu_only_v8_guard import (","from lia_1d_gpu_accuracy_v9_guard import (")
one("from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState","from torch_flowline_1d_batched_semiimplicit_annual_mb_v2 import BatchedFlowlineStateAnnualMB as BatchedFlowlineState")
one('stage_record = {\n                "stage": stage,','stage_record = {\n                "result_stage": stage,')
code=code.replace("LIA_1D_GPU_ONLY_V8_CHILD","LIA_1D_GPU_ACCURACY_V9_CHILD")
code=code.replace("lia-1d-gpu-only-v8-fixed24-stage-states","lia-1d-gpu-accuracy-v9-fixed24")
exec(compile(code,str(SOURCE)+"#v9-annual-mb","exec"),{"__name__":"__main__","__file__":str(SOURCE)})
