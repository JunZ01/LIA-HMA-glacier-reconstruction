from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVER_DIR = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
V1_DRIVER = DRIVER_DIR / "run_round230_yang2010_project_reference_locked_safe_comparison_v1.py"
V2_CONTRACT = DRIVER_DIR / "round230_yang2010_project_reference_locked_safe_comparison_v2_contract.md"
V2_OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_230_yang2010_project_reference_locked_safe_comparison_v2"
)


def forbidden_workbook_call(*args, **kwargs):
    raise RuntimeError("Round 230 v2 must not call any workbook reader")


def main() -> None:
    placeholder = types.ModuleType("openpyxl")
    placeholder.load_workbook = forbidden_workbook_call
    sys.modules["openpyxl"] = placeholder

    spec = importlib.util.spec_from_file_location("round230_v1_for_v2", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 230 v1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = V2_OUT
    module.CONTRACT = V2_CONTRACT
    module.DRIVER = Path(__file__).resolve()
    module.main()


if __name__ == "__main__":
    main()
