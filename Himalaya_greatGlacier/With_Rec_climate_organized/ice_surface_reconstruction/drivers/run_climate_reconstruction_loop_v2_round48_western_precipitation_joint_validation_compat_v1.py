"""Run Round 48 with the Matplotlib boxplot keyword used by this environment."""

from __future__ import annotations

import runpy
from pathlib import Path

from matplotlib.axes import Axes


_boxplot = Axes.boxplot


def _boxplot_compat(self, *args, **kwargs):
    if "tick_labels" in kwargs and "labels" not in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return _boxplot(self, *args, **kwargs)


Axes.boxplot = _boxplot_compat
target = Path(__file__).with_name(
    "run_climate_reconstruction_loop_v2_round48_western_precipitation_joint_validation_v1.py"
)
runpy.run_path(str(target), run_name="__main__")
