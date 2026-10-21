from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
V1_DRIVER = DRIVERS / "run_round206_setp_winter_temperature_locked_safe_comparison_v1.py"
V2_OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_206_setp_winter_temperature_locked_safe_comparison_v2"
)


def load_v1():
    spec = importlib.util.spec_from_file_location("round206_v1", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 206 v1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def centered_modern_z(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1940) & (years <= 2000) & np.isfinite(values)
    center = float(np.mean(values[baseline]))
    scale = float(np.std(values[baseline], ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid modern standard deviation")
    return (values - center) / scale


def main() -> None:
    round206 = load_v1()
    round206.OUT = V2_OUT
    round206.CONTRACT = V2_OUT / "ROUND_CONTRACT.md"
    round206.DRIVER = Path(__file__).resolve()
    round206.modern_z = centered_modern_z
    round206.main()


if __name__ == "__main__":
    main()
