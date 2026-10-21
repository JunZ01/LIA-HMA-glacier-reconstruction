"""Fixed-source v10 GPU accuracy runner with OGGM calendar landings."""
from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(__file__).with_name("run_lia_1d_gpu_only_v8.py")
EXPECTED = "F9C5032EE9C1FEC322917EF2AADE1C6228D57A8873EF4CFFA4D033B14B5A2601"
raw = SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest().upper() != EXPECTED:
    raise RuntimeError("frozen source changed")
code = raw.decode("utf-8")


def replace_once(old: str, new: str) -> None:
    global code
    if code.count(old) != 1:
        raise RuntimeError(f"replacement count != 1: {old}")
    code = code.replace(old, new, 1)


replace_once(
    "from lia_1d_gpu_only_v8_guard import (",
    "from lia_1d_gpu_accuracy_v10_guard import (",
)
replace_once(
    "from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState",
    "from torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3 import "
    "BatchedFlowlineStateOGGMCalendar as BatchedFlowlineState",
)
replace_once(
    'stage_record = {\n                "stage": stage,',
    'stage_record = {\n                "result_stage": stage,',
)
code = code.replace("LIA_1D_GPU_ONLY_V8_CHILD", "LIA_1D_GPU_ACCURACY_V10_CHILD")
code = code.replace(
    "lia-1d-gpu-only-v8-fixed24-stage-states",
    "lia-1d-gpu-accuracy-v10-fixed24",
)
exec(
    compile(code, str(SOURCE) + "#v10-oggm-calendar", "exec"),
    {"__name__": "__main__", "__file__": str(SOURCE)},
)
