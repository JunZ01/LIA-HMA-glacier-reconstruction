"""Immutable-source hotfix runner for the fixed-24 GPU-only v8 trajectory."""
from __future__ import annotations
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name("run_lia_1d_gpu_only_v8.py")
EXPECTED_SOURCE_SHA256 = "F9C5032EE9C1FEC322917EF2AADE1C6228D57A8873EF4CFFA4D033B14B5A2601"

def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"v8-v2 {label} replacement count is not one")
    return source.replace(old, new, 1)

raw = SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest().upper() != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("frozen v8-v1 source hash mismatch")
code = raw.decode("utf-8")
code = replace_once(code, "from lia_1d_gpu_only_v8_guard import (", "from lia_1d_gpu_only_v8_v2_guard import (", "guard import")
code = code.replace("LIA_1D_GPU_ONLY_V8_CHILD", "LIA_1D_GPU_ONLY_V8_V2_CHILD")
code = code.replace("lia-1d-gpu-only-v8-fixed24-stage-states", "lia-1d-gpu-only-v8-v2-fixed24-stage-states")
code = replace_once(code, 'stage_record = {\n                "stage": stage,', 'stage_record = {\n                "result_stage": stage,', "duplicate event stage keyword hotfix")
exec(compile(code, str(SOURCE) + "#v8-v2-hotfix", "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
