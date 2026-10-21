from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


# The imported v1 file defines a dataclass while loaded from a file spec. Python
# 3.12 requires that spec name to exist during annotation resolution.
sys.modules.setdefault("imd_leaf_audit_v1", types.ModuleType("imd_leaf_audit_v1"))

runpy.run_path(
    Path(__file__).with_name("audit_imd_annual_temperature_table_leaves_v2.py"),
    run_name="__main__",
)
