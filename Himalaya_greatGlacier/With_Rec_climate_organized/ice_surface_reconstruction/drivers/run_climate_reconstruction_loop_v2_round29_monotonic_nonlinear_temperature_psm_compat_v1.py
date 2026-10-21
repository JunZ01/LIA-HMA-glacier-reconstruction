"""Run Round 29 with the Matplotlib 3.8 boxplot keyword compatibility shim."""

from __future__ import annotations

from matplotlib.axes import Axes

from run_climate_reconstruction_loop_v2_round29_monotonic_nonlinear_temperature_psm_v1 import main


_boxplot = Axes.boxplot


def _compatible_boxplot(self, *args, **kwargs):
    if "tick_labels" in kwargs and "labels" not in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return _boxplot(self, *args, **kwargs)


if __name__ == "__main__":
    Axes.boxplot = _compatible_boxplot
    raise SystemExit(main())
