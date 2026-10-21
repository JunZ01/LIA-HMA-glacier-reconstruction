"""Analyze the archived-axis v4 calibration-window replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    driver = Path(__file__).with_name("prepare_and_analyze.py")
    spec = importlib.util.spec_from_file_location("calibration_window_analysis", driver)
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis-module-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module.load_v73

    def load_v4(_path):
        return original_loader(
            module.project_root()
            / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_calibration_window_stability_v1/calibration_windows_v4.mat"
        )

    module.load_v73 = load_v4
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
