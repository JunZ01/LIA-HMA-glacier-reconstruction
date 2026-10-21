from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import benchmark_temperature_reconstructions_with_cru_target_v1 as core


core.OUT = (
    Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_reconstruction_cru_target_benchmark_v2"
)


def plot_metric_grid_v2(table, path: Path, title: str) -> None:
    labels = table["product"].tolist()
    colors = ["#1f4e79", "#c55a11", "#548235", "#8064a2", "#bf9000", "#666666"][:len(labels)]
    if "decadal_rmse_degC" in table.columns:
        panels = [
            ("hma_mean_rmse_degC", "Annual RMSE", "degC"),
            ("hma_mean_mae_degC", "Annual MAE", "degC"),
            ("hma_mean_correlation_r", "Annual correlation", "r"),
            ("hma_mean_bias_degC", "Annual bias", "degC"),
            ("hma_mean_variance_ratio", "Annual variance ratio", "ratio"),
            ("hma_mean_ce", "Annual CE", "score"),
            ("hma_mean_abs_trend_error_degC_per_century", "Median trend error", "degC/century"),
            ("decadal_rmse_degC", "Decadal RMSE", "degC"),
            ("decadal_correlation_r", "Decadal correlation", "r"),
        ]
    else:
        panels = [
            ("hma_mean_rmse_degC", "HMA mean RMSE", "degC"),
            ("hma_mean_mae_degC", "HMA mean MAE", "degC"),
            ("hma_mean_correlation_r", "HMA mean correlation", "r"),
            ("hma_mean_bias_degC", "HMA mean bias", "degC"),
            ("hma_mean_variance_ratio", "HMA mean variance ratio", "ratio"),
            ("hma_mean_ce", "HMA mean CE", "score"),
            ("hma_mean_abs_trend_error_degC_per_century", "Trend error", "degC/century"),
            ("median_cell_temporal_correlation_r", "Median cell temporal r", "r"),
            ("median_annual_spatial_correlation_r", "Median annual spatial r", "r"),
        ]
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    x = np.arange(len(labels))
    for axis, (column, panel_title, ylabel) in zip(axes.ravel(), panels):
        axis.bar(x, table[column], color=colors)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        if "correlation" in column or column.endswith("_ce"):
            axis.axhline(0, color="#222222", linewidth=0.8)
        if "variance_ratio" in column:
            axis.axhline(1, color="#222222", linewidth=0.8)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=220)
    plt.close(figure)


core.plot_metric_grid = plot_metric_grid_v2


if __name__ == "__main__":
    core.main()
