"""Dependency-isolated launcher for the 1861-1900 western attribution."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


MODULE = "run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1"
stub = types.ModuleType(MODULE)
stub.project_root = project_root
stub.correlation = correlation
sys.modules[MODULE] = stub

import attribute_temperature_west_increment_1861_1900_v1 as experiment


experiment.OUTPUT = experiment.RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2"


if __name__ == "__main__":
    experiment.main()
