"""Run the residual-missingness analysis with two non-numerical fixes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_analysis_module():
    path = Path(__file__).with_name("analyze_repair.py")
    spec = importlib.util.spec_from_file_location("residual_missingness_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis-module-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrected_plot_variance_diagnostic(frame, output: Path) -> None:
    ratio = frame.corrected_error_variance / frame.archived_error_variance
    size = 18 + 90 * np.minimum(
        np.abs(frame.missingness_repaired_R_hma_trend_contribution_c_per_century),
        1.0,
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    scatter = axis.scatter(
        frame.missing_calibration_count,
        ratio,
        c=frame.repaired_minus_archived_hma_trend_contribution_c_per_century,
        s=size,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        alpha=0.75,
        edgecolors="none",
    )
    axis.axhline(1.0, color="#555555", lw=0.8)
    axis.set(
        xlabel="Missing calibration years (1940-2000)",
        ylabel="Corrected / archived EnKF error variance",
        title="Residual missingness changes proxy error weights",
    )
    axis.grid(alpha=0.18)
    figure.colorbar(
        scatter,
        ax=axis,
        label="Change in 1901-1950 HMA trend contribution (degC/century)",
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    module = load_analysis_module()
    module.plot_variance_diagnostic = corrected_plot_variance_diagnostic
    original_merge = pd.DataFrame.merge

    def corrected_merge(self, right, *args, **kwargs):
        if kwargs.get("how") == "validate":
            kwargs["how"] = "inner"
            kwargs["validate"] = "one_to_one"
        return original_merge(self, right, *args, **kwargs)

    pd.DataFrame.merge = corrected_merge
    try:
        return int(module.main())
    finally:
        pd.DataFrame.merge = original_merge


if __name__ == "__main__":
    raise SystemExit(main())
