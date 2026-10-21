"""Plot length, area, and volume response for envelope-accepted trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    screen = pd.read_csv(root / "trajectory_bound_screen.csv")
    for system_id in sorted(screen["system_id"].unique()):
        accepted = screen[(screen.system_id == system_id) & screen.new_accept.astype(bool)].copy()
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        if accepted.empty:
            axes[1].text(0.5, 0.5, "0 trajectories pass the envelope constraint",
                         ha="center", va="center", transform=axes[1].transAxes, fontsize=14)
        else:
            norm = Normalize(vmin=float(accepted.e_total.min()), vmax=float(accepted.e_total.max()))
            for row in accepted.itertuples(index=False):
                ts = pd.read_csv(root / f"system_{system_id}" / f"point_{int(row.point):04d}" / "timeseries.csv")
                ts = ts[ts.phase.eq("forward")].sort_values("year")
                color = viridis(norm(row.e_total))
                rel_area = (ts.area_m2 / ts.area_m2.iloc[0] - 1.0) * 100.0
                rel_volume = (ts.volume_m3 / ts.volume_m3.iloc[0] - 1.0) * 100.0
                width = 2.7 if row.e_total == accepted.e_total.min() else 1.0
                alpha = 0.95 if width > 2 else 0.65
                axes[0].plot(ts.year, ts.length_m / 1000.0, color=color, lw=width, alpha=alpha)
                axes[1].plot(ts.year, rel_area, color=color, lw=width, alpha=alpha)
                axes[2].plot(ts.year, rel_volume, color=color, lw=width, alpha=alpha)
            sm = ScalarMappable(norm=norm, cmap=viridis)
            sm.set_array([])
            fig.colorbar(sm, ax=axes, label="E_total")
        row = screen[screen.system_id.eq(system_id)].iloc[0]
        lia = row.band_upper_m - 0.2 * row.band_amplitude_m
        rgi = lia - row.band_amplitude_m
        axes[0].axhspan(row.band_lower_m / 1000, row.band_upper_m / 1000, color="tab:green", alpha=0.12)
        axes[0].axhline(lia / 1000, color="black", lw=1.1, label="LIA")
        axes[0].axhline(rgi / 1000, color="gray", lw=1.1, label="RGI")
        axes[0].set_ylabel("Length (km)")
        axes[1].set_ylabel("Area change (%)")
        axes[2].set_ylabel("Volume change (%)")
        axes[2].set_xlabel("Year")
        axes[0].set_title(f"LIA system {system_id}: accepted trajectories ({len(accepted)})")
        axes[0].legend(frameon=False)
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.set_xlim(1836, 2000)
        fig.savefig(root / f"accepted_multimetric_system_{system_id}.png", dpi=180)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
