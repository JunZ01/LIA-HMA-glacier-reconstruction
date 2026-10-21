"""Plot all trajectories passing the LIA-RGI envelope screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis


def _plot(root: Path, system_id: int, accepted: pd.DataFrame, screen: pd.DataFrame) -> None:
    system_screen = screen[screen["system_id"].eq(system_id)]
    band = system_screen.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if accepted.empty:
        ax.text(0.5, 0.5, "0 trajectories pass the new envelope constraint",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        norm = Normalize(vmin=float(accepted["e_total"].min()), vmax=float(accepted["e_total"].max()))
        for row in accepted.itertuples(index=False):
            summary_path = root / f"system_{system_id}" / f"point_{int(row.point):04d}" / "summary.json"
            ts = pd.read_csv(summary_path.parent / "timeseries.csv")
            ts = ts[ts["phase"].eq("forward")].copy()
            ax.plot(ts["year"], ts["length_m"] / 1000.0, color=viridis(norm(row.e_total)),
                    linewidth=2.6 if row.e_total == accepted["e_total"].min() else 1.1,
                    alpha=0.95 if row.e_total == accepted["e_total"].min() else 0.65)
        sm = ScalarMappable(norm=norm, cmap=viridis)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="E_total")
    ax.axhspan(band.band_lower_m / 1000.0, band.band_upper_m / 1000.0,
               color="tab:green", alpha=0.12, label="LIA-RGI ±20% envelope")
    ax.axhline(band.band_lower_m / 1000.0, color="tab:green", linewidth=1.0)
    ax.axhline(band.band_upper_m / 1000.0, color="tab:green", linewidth=1.0)
    # All three selected systems have LIA length greater than RGI length.
    lia = band.band_upper_m - 0.2 * band.band_amplitude_m
    rgi = lia - band.band_amplitude_m
    ax.axhline(lia / 1000.0, color="black", linewidth=1.1, label=f"LIA ({lia/1000:.3f} km)")
    ax.axhline(rgi / 1000.0, color="gray", linewidth=1.1, label=f"RGI ({rgi/1000:.3f} km)")
    ax.set_title(f"LIA system {system_id}: trajectories passing envelope screen ({len(accepted)})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Glacier length (km)")
    ax.set_xlim(1836, 2000)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(root / f"accepted_trajectories_system_{system_id}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    screen = pd.read_csv(root / "trajectory_bound_screen.csv")
    for system_id in sorted(screen["system_id"].unique()):
        accepted = screen[(screen["system_id"].eq(system_id)) & screen["new_accept"].astype(bool)]
        _plot(root, int(system_id), accepted, screen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
