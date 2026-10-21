"""Plot separate spin-up and forward length panels for formal dual-pass curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


INVENTORY = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_analog_v1"
)
FROZEN_991 = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_frozen_curve_registry_v1"
    / "system_991"
    / "timeseries.csv"
)
SYSTEMS = (991, 1077, 1616)


def _candidate_timeseries(row: pd.Series) -> Path:
    if int(row["lia_unique_id"]) == 991:
        return FROZEN_991
    return Path(str(row["output_dir"])) / "timeseries.csv"


def main() -> None:
    best = pd.read_csv(INVENTORY / "best_formal_dual_pass_per_system.csv")
    best["lia_unique_id"] = pd.to_numeric(best["lia_unique_id"]).astype(int)
    best = best[best["lia_unique_id"].isin(SYSTEMS)].copy()
    if set(best["lia_unique_id"]) != set(SYSTEMS):
        raise ValueError("expected formal dual-pass rows for 991, 1077, and 1616")

    plot_dir = INVENTORY / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for row in best.sort_values("lia_unique_id").itertuples(index=False):
        system_id = int(row.lia_unique_id)
        source = FROZEN_991 if system_id == 991 else Path(str(row.output_dir)) / "timeseries.csv"
        series = pd.read_csv(source)
        spin = series[series["phase"].eq("spinup")].copy()
        forward = series[series["phase"].eq("forward")].copy()
        lia = float(row.lia_target_m)
        rgi = float(row.rgi_target_m)
        separation = abs(lia - rgi)
        lower = min(lia, rgi) - 0.20 * separation
        upper = max(lia, rgi) + 0.20 * separation
        all_length = pd.to_numeric(series["length_m"], errors="coerce") / 1000.0
        y_min = min(float(all_length.min()), lower / 1000.0) - 0.25
        y_max = max(float(all_length.max()), upper / 1000.0) + 0.25

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 5.0),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.55]},
        )
        for axis in axes:
            axis.axhspan(lower / 1000.0, upper / 1000.0, color="#dceeff", alpha=0.55)
            axis.axhline(lia / 1000.0, color="#c73e1d", linewidth=1.5, linestyle="--", label="LIA target")
            axis.axhline(rgi / 1000.0, color="#2b6cb0", linewidth=1.5, linestyle="--", label="RGI target")
            axis.grid(alpha=0.22, linewidth=0.7)
            axis.set_ylim(y_min, y_max)
        axes[0].plot(spin["year"], spin["length_m"] / 1000.0, color="#444444", marker="o", linewidth=2.0)
        axes[0].set_xlabel("Spin-up model year")
        axes[0].set_ylabel("Length (km)")
        axes[0].set_title("Adaptive spin-up")

        axes[1].plot(forward["year"], forward["length_m"] / 1000.0, color="#168aad", marker="o", markersize=3.0, linewidth=2.0)
        peak_index = forward["length_m"].idxmax()
        peak = forward.loc[peak_index]
        end = forward.iloc[-1]
        axes[1].scatter([peak["year"]], [peak["length_m"] / 1000.0], color="#c73e1d", zorder=4)
        axes[1].scatter([end["year"]], [end["length_m"] / 1000.0], color="#2b6cb0", zorder=4)
        axes[1].annotate(
            f"peak {peak['length_m']/1000.0:.2f} km ({int(peak['year'])})",
            (peak["year"], peak["length_m"] / 1000.0),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )
        axes[1].annotate(
            f"2000 {end['length_m']/1000.0:.2f} km",
            (end["year"], end["length_m"] / 1000.0),
            xytext=(-8, -18),
            ha="right",
            textcoords="offset points",
            fontsize=9,
        )
        axes[1].set_xlabel("Calendar year")
        axes[1].set_title("Forward reconstruction")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(
            f"LIA system {system_id} | ΔT={float(row.delta_t):+.4g} K | "
            f"Esep(LIA/RGI)={float(row.e_lia_sep_formal):.3f}/{float(row.e_rgi_sep_formal):.3f}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        output = plot_dir / f"system_{system_id}_spinup_forward_length.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        index_rows.append(
            {
                "lia_unique_id": system_id,
                "task_id": row.task_id,
                "timeseries": str(source),
                "plot": str(output),
                "delta_t": float(row.delta_t),
                "forward_peak_m": float(row.forward_peak_m),
                "forward_2000_m": float(row.forward_2000_m),
                "e_lia_sep": float(row.e_lia_sep_formal),
                "e_rgi_sep": float(row.e_rgi_sep_formal),
            }
        )
    pd.DataFrame(index_rows).to_csv(plot_dir / "plot_index.csv", index=False)
    print(plot_dir)


if __name__ == "__main__":
    main()
