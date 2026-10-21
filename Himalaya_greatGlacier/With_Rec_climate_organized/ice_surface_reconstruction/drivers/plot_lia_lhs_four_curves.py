"""Plot four representative length trajectories for each LHS glacier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_records(root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(root.glob("system_*/point_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        rows.append({
            "system_id": int(summary["lia_unique_id"]),
            "point": int(summary_path.parent.name.split("_")[1]),
            "status": summary.get("status", "crash"),
            "summary_path": str(summary_path),
            "timeseries_path": str(summary_path.parent / "timeseries.csv"),
            "e_lia": metrics.get("e_lia"),
            "e_2000": metrics.get("e_2000"),
            "e_total": metrics.get("e_total"),
            "l_2000_m": metrics.get("l_2000_m"),
            "ratio": metrics.get("forward_to_initial_length_ratio"),
            "lia_target_m": summary["targets"]["lia_length_m"],
            "rgi_target_m": summary["targets"]["rgi_length_m"],
        })
    return pd.DataFrame(rows)


def _select(group: pd.DataFrame) -> pd.DataFrame:
    selected = []
    valid = group[group["status"].eq("ok")].dropna(subset=["e_total"]).sort_values("e_total")
    if valid.empty:
        raise ValueError(f"no-valid-trial:{group.name}")
    selected.append(("best", valid.iloc[0]))
    selected.append(("median", valid.iloc[len(valid) // 2]))

    hard_short = group[(group["status"] == "out_of_range") & (group["ratio"] < 0.05)]
    hard_long = group[(group["status"] == "out_of_range") & (group["ratio"] > 2.0)]
    used = {int(row["point"]) for _, row in selected}
    hard_short = hard_short[~hard_short["point"].isin(used)]
    hard_long = hard_long[~hard_long["point"].isin(used)]
    if not hard_short.empty:
        selected.append(("short_out_of_range", hard_short.sort_values("l_2000_m").iloc[0]))
    else:
        candidates = group[~group["point"].isin(used)]
        selected.append(("short_endpoint_no_hard_short", candidates.sort_values("l_2000_m").iloc[0]))
    used = {int(row["point"]) for _, row in selected}
    hard_long = hard_long[~hard_long["point"].isin(used)]
    if not hard_long.empty:
        selected.append(("long_out_of_range", hard_long.sort_values("l_2000_m", ascending=False).iloc[0]))
    else:
        candidates = group[~group["point"].isin(used)]
        selected.append(("long_endpoint_no_hard_long", candidates.sort_values("l_2000_m", ascending=False).iloc[0]))
    out = pd.DataFrame([{"category": category, **row.to_dict()} for category, row in selected])
    return out


def _plot_system(root: Path, selected: pd.DataFrame, system_id: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = {"best": "tab:green", "median": "tab:blue", "short_out_of_range": "tab:red",
              "short_endpoint_no_hard_short": "tab:orange", "long_out_of_range": "tab:purple",
              "long_endpoint_no_hard_long": "tab:brown"}
    styles = {"best": "-", "median": "--", "short_out_of_range": ":",
              "short_endpoint_no_hard_short": ":", "long_out_of_range": "-.",
              "long_endpoint_no_hard_long": "-."}
    for row in selected.itertuples(index=False):
        ts = pd.read_csv(row.timeseries_path)
        ts["plot_year"] = ts["year"]
        spin = ts["phase"].eq("spinup")
        ts.loc[spin, "plot_year"] = ts.loc[spin, "year"] + 1836 - 500
        label = f"{row.category} (p{int(row.point)}, E={row.e_total:.3f})"
        ax.plot(ts["plot_year"], ts["length_m"] / 1000.0, color=colors[row.category],
                linestyle=styles[row.category], linewidth=2.0, label=label)
    target = selected.iloc[0]
    ax.axhline(target.lia_target_m / 1000.0, color="black", linewidth=1.2, alpha=0.7,
               label=f"LIA target ({target.lia_target_m/1000:.2f} km)")
    ax.axhline(target.rgi_target_m / 1000.0, color="gray", linewidth=1.2, alpha=0.8,
               label=f"RGI target ({target.rgi_target_m/1000:.2f} km)")
    note = "short class is endpoint-side fallback" if "short_out_of_range" not in set(selected.category) else ""
    ax.set_title(f"LIA system {system_id}: representative D-layer length trajectories")
    ax.set_xlabel("Model year (500-year spin-up mapped to 1336–1836)")
    ax.set_ylabel("Glacier length (km)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    if note:
        ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8, color="tab:orange")
    fig.savefig(root / f"four_curves_system_{system_id}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    records = _read_records(root)
    selections = []
    for system_id, group in records.groupby("system_id"):
        chosen = _select(group)
        selections.append(chosen)
        _plot_system(root, chosen, int(system_id))
    selected = pd.concat(selections, ignore_index=True)
    selected.to_csv(root / "four_curve_selected_points.csv", index=False)
    print(selected[["system_id", "category", "point", "status", "ratio", "e_lia", "e_2000", "e_total", "l_2000_m"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
