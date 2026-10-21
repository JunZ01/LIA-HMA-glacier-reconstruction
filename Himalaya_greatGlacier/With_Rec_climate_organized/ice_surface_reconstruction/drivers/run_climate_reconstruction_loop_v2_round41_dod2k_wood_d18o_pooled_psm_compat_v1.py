"""Run Round 41 with the local Matplotlib boxplot keyword compatibility shim."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from matplotlib.axes import Axes


def main() -> int:
    original_boxplot = Axes.boxplot

    def compatible_boxplot(self, *args, **kwargs):
        if "tick_labels" in kwargs and "labels" not in kwargs:
            kwargs["labels"] = kwargs.pop("tick_labels")
        return original_boxplot(self, *args, **kwargs)

    Axes.boxplot = compatible_boxplot
    target = Path(__file__).with_name("run_climate_reconstruction_loop_v2_round41_dod2k_wood_d18o_pooled_psm_v1.py")
    spec = importlib.util.spec_from_file_location("round41_scientific_driver", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("round41-driver-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
