from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
V1_DRIVER = DRIVERS / "run_round206_setp_winter_temperature_locked_safe_comparison_v1.py"
V3_OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_206_setp_winter_temperature_locked_safe_comparison_v3"
)


def load_v1():
    spec = importlib.util.spec_from_file_location("round206_v1_for_v3", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 206 v1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_display(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    del years
    return values.copy()


def replace_plot_and_drop_misleading_columns() -> None:
    annual_path = V3_OUT / "annual_allowed_curves.csv"
    windows_path = V3_OUT / "temperature_50year_windows_1500_1899.csv"
    annual = pd.read_csv(annual_path)
    windows = pd.read_csv(windows_path)
    misleading = [column for column in annual if column.endswith("_smooth31_z1940_2000")]
    annual = annual.drop(columns=misleading)
    annual.to_csv(annual_path, index=False)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.16},
    )
    year = annual.year.to_numpy(int)
    axis = axes[0]
    axis.plot(
        year,
        annual.external_winter_tmin_anomaly_degC_smooth31,
        color="#222222",
        linewidth=2.4,
        label="Huang 2019 SETP winter Tmin",
    )
    axis.plot(
        year,
        annual.project_setp_annual_anomaly_degC_smooth31,
        color="#B33A3A",
        linewidth=2.0,
        label="Project six-member SETP annual",
    )
    axis.plot(
        year,
        annual.reference_setp_annual_anomaly_degC_smooth31,
        color="#147D78",
        linewidth=2.0,
        label="Four-family SETP annual",
    )
    axis.plot(
        year,
        annual.project_hma_annual_anomaly_degC_smooth31,
        color="#C89021",
        linewidth=1.2,
        alpha=0.85,
        label="Project HMA annual context",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_ylabel("31-year smooth anomaly (degC; 1961-1990)")
    axis.set_title("Southeastern Tibetan winter temperature versus frozen project curves")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    axis = axes[1]
    centers = windows.center_year.to_numpy(float)
    axis.plot(
        centers,
        windows.external_winter_tmin_anomaly_degC,
        color="#222222",
        marker="o",
        linewidth=2.0,
        label="Winter Tmin",
    )
    axis.plot(
        centers,
        windows.project_setp_annual_anomaly_degC,
        color="#B33A3A",
        marker="s",
        linewidth=1.8,
        label="Project SETP annual",
    )
    axis.plot(
        centers,
        windows.reference_setp_annual_anomaly_degC,
        color="#147D78",
        marker="^",
        linewidth=1.8,
        label="Four-family SETP annual",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlim(1490, 2005)
    axis.set_xlabel("Year")
    axis.set_ylabel("50-year anomaly (degC)\nrelative to 1961-1990")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.text(
        0.99,
        0.01,
        "Gray gap: sealed 1921-1939; no smoothing or interpolation crosses it. Seasonal targets differ.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.1)
    figure.savefig(V3_OUT / "setp_winter_project_curves_1500_2000.png", dpi=220)
    plt.close(figure)

    summary_path = V3_OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["authoritative_version"] = "round_206_v3"
    summary["display"] = {
        "top_panel": "31-year smooth anomalies relative to 1961-1990",
        "bottom_panel": "non-overlapping 50-year anomalies relative to 1961-1990",
        "z_score_display_used": False,
        "v1_v2_display_superseded": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    final_path = V3_OUT / "FINAL_RESULT.md"
    final_path.write_text(
        final_path.read_text(encoding="utf-8")
        + "\n## Display correction\n\n"
        + "Round 206 v3 is authoritative. The figure shows raw 31-year smooth "
        + "anomalies relative to 1961-1990. The v1/v2 smoothed-SD display is "
        + "superseded and is not used for interpretation.\n",
        encoding="utf-8",
    )


def main() -> None:
    round206 = load_v1()
    round206.OUT = V3_OUT
    round206.CONTRACT = V3_OUT / "ROUND_CONTRACT.md"
    round206.DRIVER = Path(__file__).resolve()
    round206.modern_z = identity_display
    round206.main()
    replace_plot_and_drop_misleading_columns()


if __name__ == "__main__":
    main()
