"""Plot saved targeted LIA chain curves without starting OGGM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _display_year(frame: pd.DataFrame, forward_start: int = 1836) -> pd.Series:
    """Map the artificial spin-up clock to a display axis ending at the forward start."""
    values = frame["year"].astype(float).copy()
    spin = frame["phase"].eq("spinup")
    if spin.any():
        spin_end = float(frame.loc[spin, "year"].max())
        values.loc[spin] = forward_start - spin_end + values.loc[spin]
    return values


def _load_targeted(chain_dir: Path, modern_json: Path | None) -> tuple[pd.DataFrame, dict, dict | None]:
    series = pd.read_csv(chain_dir / "timeseries.csv")
    summary = json.loads((chain_dir / "summary.json").read_text(encoding="utf-8"))
    modern = None
    if modern_json is not None and modern_json.exists():
        modern = json.loads(modern_json.read_text(encoding="utf-8"))
    return series, summary, modern


def plot_chain(chain_dir: Path, modern_json: Path | None, output_dir: Path) -> Path:
    series, summary, modern = _load_targeted(chain_dir, modern_json)
    uid = int(summary["lia_unique_id"])
    lia = summary["targets"]["lia_length_m"]
    rgi = summary["targets"]["rgi_length_m"]
    window = summary.get("spinup_window", ["unknown", "unknown"])
    window_label = f"{window[0]}_{window[1]}"
    forward_start = int(summary.get("start_year", 1836))
    display_year = _display_year(series, forward_start=forward_start)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    specs = [
        ("length_m", 1000.0, "Length (km)"),
        ("area_m2", 1e6, "Area (km$^2$)"),
        ("volume_m3", 1e9, "Volume (km$^3$)"),
    ]
    phase_colors = {"spinup": "#3568a8", "forward": "#b64a3a"}
    phase_labels = {"spinup": "spin-up", "forward": f"{forward_start}-2000 forward"}
    for axis, (column, scale, ylabel) in zip(axes, specs):
        for phase in ("spinup", "forward"):
            part = series[series["phase"].eq(phase)]
            if part.empty:
                continue
            x = display_year.loc[part.index]
            axis.plot(x, part[column] / scale, color=phase_colors[phase], lw=1.8,
                      marker="o", ms=2.3, label=phase_labels[phase])
        if modern and modern.get("initial_2000") and modern.get("final_2020"):
            a = modern["initial_2000"]
            b = modern["final_2020"]
            axis.plot([a["year"], b["year"]], [a[column] / scale, b[column] / scale],
                      color="#6a3d9a", lw=1.8, ls="--", marker="o", ms=3,
                      label="2000-2020 modern probe")
        if column == "length_m":
            axis.axhline(lia / scale, color="#222222", lw=1.1, ls=":",
                         label=f"LIA target ({lia / scale:.2f} km)")
            axis.axhline(rgi / scale, color="#555555", lw=1.1, ls="-.",
                         label=f"RGI target ({rgi / scale:.2f} km)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8, loc="best")

    modern_note = "modern probe included" if modern else "modern probe unavailable"
    metrics = summary["metrics"]
    fig.suptitle(
        f"LIA system {uid} ({summary['rgi_id']}) | spin-up + forward | {modern_note}\n"
        f"spin-up window={window[0]}-{window[1]} | "
        f"delta_T={summary['parameters']['delta_t_c']:+.3g} K | "
        f"E_LIA={metrics['e_lia']:.1%} | E_2000={metrics['e_2000']:.1%}",
        fontsize=13,
    )
    axes[-1].set_xlabel(
        f"Display year (spin-up clock mapped to {forward_start}; forward years are actual)"
    )
    path = output_dir / f"system_{uid}_window_{window_label}_chain_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", action="append", nargs=2, metavar=("CHAIN_DIR", "MODERN_JSON"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for chain, modern in args.chain:
        modern_path = None if modern in ("", "none", "None") else Path(modern)
        print(plot_chain(Path(chain), modern_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
