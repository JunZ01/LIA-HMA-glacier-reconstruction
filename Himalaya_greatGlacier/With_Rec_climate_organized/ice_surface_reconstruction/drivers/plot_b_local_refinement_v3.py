"""Plot every B-window local-refinement parameter group with V3 status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE_GLEN_3584 = 1.045220095930216e-23
FORWARD_START = 1470


def _display_year(series: pd.DataFrame) -> pd.Series:
    result = series["year"].astype(float).copy()
    spin = series["phase"].astype(str).eq("spinup")
    if spin.any():
        spin_end = float(series.loc[spin, "year"].max())
        result.loc[spin] = FORWARD_START - spin_end + result.loc[spin]
    return result


def _group_name(row: pd.Series) -> tuple[str, str] | None:
    source = str(row["source_dir"])
    for branch in ("1044_low_melt", "1044_mid_melt", "1044_high_melt"):
        if branch in source:
            return branch, branch.replace("_", " ")
    if "3584_endpoint_shape" in source:
        return "3584_melt_delta", "3584 melt_f - delta_T"
    if "B_3584_glen_shape_refine_v1" in source:
        multiplier = float(row["glen_a_pa_3_s_1"]) / BASE_GLEN_3584
        tag = f"3584_glen_x{multiplier:g}".replace(".", "p")
        return tag, f"3584 glen_a x {multiplier:g}"
    return None


def _plot_group(
    group: pd.DataFrame,
    title: str,
    output_path: Path,
    include_spinup: bool = True,
) -> dict:
    group = group.sort_values(["delta_t_c", "melt_f", "glen_a_pa_3_s_1"])
    lia = float(group.iloc[0]["lia_target_m"])
    rgi = float(group.iloc[0]["rgi_target_m"])
    gap = abs(lia - rgi)
    corridor_low = min(lia, rgi) - 0.20 * gap
    corridor_high = max(lia, rgi) + 0.20 * gap

    figure, axis = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, max(len(group), 2)))
    handles: list[Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = [corridor_low / 1000.0, corridor_high / 1000.0]

    for color, (_, row) in zip(colors, group.iterrows()):
        source_dir = Path(str(row["source_dir"]))
        series = pd.read_csv(source_dir / "timeseries.csv")
        x = _display_year(series)
        y = series["length_m"].astype(float) / 1000.0
        plotted = series["phase"].astype(str).eq("forward")
        if include_spinup:
            plotted |= series["phase"].astype(str).eq("spinup")
        all_x.extend(x.loc[plotted].tolist())
        all_y.extend(y.loc[plotted].tolist())
        accepted = bool(row["accepted_curve_v3"])
        emphasized = bool(row.get("plot_emphasis", accepted))
        if include_spinup:
            phases = (("spinup", "--"), ("forward", "-"))
        else:
            phases = (("forward", "-"),)
        for phase, style in phases:
            mask = series["phase"].astype(str).eq(phase)
            if mask.any():
                axis.plot(
                    x.loc[mask],
                    y.loc[mask],
                    color=color,
                    linestyle=style,
                    linewidth=2.2 if emphasized and phase == "forward" else 1.25,
                    alpha=1.0 if emphasized else 0.72,
                )
        delta_t = float(row["delta_t_c"])
        melt_f = float(row["melt_f"])
        glen_multiplier = float(row["glen_a_pa_3_s_1"]) / BASE_GLEN_3584
        if pd.notna(row.get("plot_parameter_label")):
            parameter_label = str(row["plot_parameter_label"])
        elif int(row["system_id"]) == 1044 or "melt_delta" in output_path.stem:
            parameter_label = f"mf={melt_f:.4g}, dT={delta_t:+.4g}"
        else:
            parameter_label = f"glen x{glen_multiplier:g}, dT={delta_t:+.2f}"
        if pd.notna(row.get("plot_status")):
            status = str(row["plot_status"])
        else:
            status = "pass" if accepted else str(row["failure_reasons_v3"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.2 if emphasized else 1.25,
                label=f"{parameter_label} | {status}",
            )
        )

    axis.axhspan(
        corridor_low / 1000.0,
        corridor_high / 1000.0,
        color="0.75",
        alpha=0.18,
        label="20% corridor",
    )
    axis.axhline(lia / 1000.0, color="black", linestyle="--", linewidth=1.5)
    axis.axhline(rgi / 1000.0, color="0.25", linestyle="-.", linewidth=1.4)
    axis.axvline(FORWARD_START, color="0.35", linestyle=":", linewidth=1.2)
    y_span = max(max(all_y) - min(all_y), 0.5)
    axis.text(min(all_x), lia / 1000.0 + 0.025 * y_span, f"LIA {lia / 1000.0:.2f} km")
    axis.text(min(all_x), rgi / 1000.0 - 0.035 * y_span, f"RGI {rgi / 1000.0:.2f} km")
    period = (
        "adaptive spin-up and 1470-2000 length"
        if include_spinup
        else "1470-2000 length"
    )
    axis.set_title(f"{title}: {period}")
    axis.set_xlabel("Display year")
    axis.set_ylabel("Length (km)")
    axis.grid(True, alpha=0.20)
    axis.legend(handles=handles, loc="best", frameon=False, fontsize=8, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "group": output_path.stem,
        "title": title,
        "chain_count": len(group),
        "accepted_count": int(group["accepted_curve_v3"].sum()),
        "output_png": str(output_path.resolve()),
    }


def _plot_best(frame: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for system_id, group in frame.groupby("system_id"):
        rows.append(group.sort_values("objective_v3").iloc[0])
    figure, axes = plt.subplots(len(rows), 1, figsize=(13.5, 9.0), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for axis, row in zip(axes, rows):
        series = pd.read_csv(Path(str(row["source_dir"])) / "timeseries.csv")
        x = _display_year(series)
        y = series["length_m"].astype(float) / 1000.0
        lia = float(row["lia_target_m"]) / 1000.0
        rgi = float(row["rgi_target_m"]) / 1000.0
        gap = abs(lia - rgi)
        axis.axhspan(min(lia, rgi) - 0.2 * gap, max(lia, rgi) + 0.2 * gap, color="0.75", alpha=0.18)
        for phase, style in (("spinup", "--"), ("forward", "-")):
            mask = series["phase"].astype(str).eq(phase)
            axis.plot(x.loc[mask], y.loc[mask], linestyle=style, linewidth=1.5 if phase == "spinup" else 2.2)
        axis.axhline(lia, color="black", linestyle="--", linewidth=1.4)
        axis.axhline(rgi, color="0.25", linestyle="-.", linewidth=1.3)
        axis.axvline(FORWARD_START, color="0.35", linestyle=":", linewidth=1.1)
        status = "V3 pass" if bool(row["accepted_curve_v3"]) else f"fails: {row['failure_reasons_v3']}"
        axis.set_title(
            f"System {int(row['system_id'])} | mf={float(row['melt_f']):.4g}, "
            f"dT={float(row['delta_t_c']):+.4g} | objective={float(row['objective_v3']):.3f} | {status}"
        )
        axis.set_ylabel("Length (km)")
        axis.grid(True, alpha=0.20)
    axes[-1].set_xlabel("Display year")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(scores_path: Path, output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(scores_path)
    memberships = frame.apply(_group_name, axis=1)
    frame = frame[memberships.notna()].copy()
    frame["plot_group"] = [value[0] for value in memberships[memberships.notna()]]
    frame["plot_title"] = [value[1] for value in memberships[memberships.notna()]]

    records: list[dict] = []
    for group_name, group in frame.groupby("plot_group"):
        records.append(
            _plot_group(
                group,
                str(group.iloc[0]["plot_title"]),
                output_dir / f"{group_name}.png",
            )
        )
    _plot_best(frame, output_dir / "best_by_system_v3.png")
    manifest = pd.DataFrame(records).sort_values("group")
    manifest.to_csv(output_dir / "plot_manifest.csv", index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = run(args.scores, args.output_dir)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
