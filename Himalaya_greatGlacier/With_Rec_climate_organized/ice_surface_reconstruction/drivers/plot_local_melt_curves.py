"""Plot saved local melt-factor chain trajectories without rerunning OGGM."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "d_layer_lhs_D_v1"
    / "modern_mb_probe_v1"
    / "local_melt_search_v1"
)

SYSTEMS = {
    "1044": {"lia": 15.456, "rgi": 13.248},
    "3584": {"lia": 24.130, "rgi": 21.470},
}
COLORS = {"6.5": "#0072B2", "7.5": "#D55E00"}


def load_series(data_root: Path, system_id: str, melt: str) -> pd.DataFrame:
    path = data_root / f"system_{system_id}_melt_{melt}" / "timeseries.csv"
    df = pd.read_csv(path)
    # Spin-up elapsed time 0..500 is displayed as 1336..1836 so the
    # repeated 1807-1836 climate window is visible on the historical axis.
    spin = df[df["phase"].eq("spinup")].copy()
    max_spinup = int(spin["year"].max())
    spin["display_year"] = 1836 - max_spinup + spin["year"]
    forward = df[df["phase"].eq("forward")].copy()
    forward["display_year"] = forward["year"]
    return pd.concat([spin, forward], ignore_index=True).sort_values("display_year")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or (data_root / "local_melt_curves.png")).resolve()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    specs = [
        ("length_m", "Length (km)", 1000.0),
        ("area_m2", "Area (km$^2$)", 1e6),
        ("volume_m3", "Volume (km$^3$)", 1e9),
    ]

    for row, system_id in enumerate(SYSTEMS):
        target = SYSTEMS[system_id]
        for col, (field, ylabel, scale) in enumerate(specs):
            ax = axes[row, col]
            for melt in ("6.5", "7.5"):
                ts = load_series(data_root, system_id, melt)
                ax.plot(
                    ts["display_year"],
                    ts[field] / scale,
                    color=COLORS[melt],
                    lw=2,
                    marker="o",
                    ms=2.5,
                    label=fr"$melt_f$={melt}",
                )
            if field == "length_m":
                ax.axhline(target["lia"], color="#444444", ls="--", lw=1.2, label="LIA target")
                ax.axhline(target["rgi"], color="#666666", ls=":", lw=1.5, label="RGI target")
            ax.axvline(1836, color="#888888", ls="-.", lw=1)
            ax.set_title(f"System {system_id}  |  {ylabel}")
            ax.set_xlabel("Display year")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            spin_max = int(
                load_series(data_root, system_id, "6.5")
                .query("phase == 'spinup'")["year"]
                .max()
            )
            ax.set_xlim(1836 - spin_max, 2000)
            if col == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        "Saved local melt-factor chains: extended spin-up + 1836–2000 forward",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
