"""Summarize and plot a completed D-layer LHS batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    results = pd.read_csv(root / "results.csv")
    results["status"] = results["status"].astype(str)
    valid = results[results["status"].eq("ok")].copy()
    valid = valid.dropna(subset=["e_total"])
    top = (
        valid.sort_values(["system_id", "e_total"])
        .groupby("system_id", as_index=False, group_keys=False)
        .head(10)
    )
    top.to_csv(root / "top10_by_system.csv", index=False)
    diagnostics = (
        results.groupby("system_id", as_index=False)
        .agg(
            n_total=("status", "size"),
            n_ok=("status", lambda s: int((s == "ok").sum())),
            n_out_of_range=("status", lambda s: int((s == "out_of_range").sum())),
            n_crash=("status", lambda s: int((s == "crash").sum())),
            failure_rate=("status", lambda s: float((s != "ok").mean())),
            min_e_lia=("e_lia", "min"),
            min_e_2000=("e_2000", "min"),
            min_e_total=("e_total", "min"),
            mean_runtime_s=("runtime_s", "mean"),
        )
    )
    diagnostics.to_csv(root / "diagnostics_enhanced.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (system_id, group) in zip(axes, sorted(results.groupby("system_id"))):
        group = group[group["status"].eq("ok")].dropna(subset=["e_lia", "e_2000"])
        ax.scatter(group["e_lia"] * 100, group["e_2000"] * 100, s=14, alpha=0.65)
        best = top[top["system_id"].eq(system_id)]
        ax.scatter(best["e_lia"] * 100, best["e_2000"] * 100, s=24, c="tab:red", label="top 10")
        ax.set_title(f"LIA system {system_id}")
        ax.set_xlabel("E_LIA (%)")
        ax.set_ylabel("E_2000 (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(root / "e_lia_vs_e_2000.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for system_id, group in sorted(top.groupby("system_id")):
        ax.plot(range(1, len(group) + 1), group["e_total"] * 100, marker="o", label=str(system_id))
    ax.set_xlabel("Rank among valid trials")
    ax.set_ylabel("E_total (%)")
    ax.set_title("Top-10 valid parameter trials")
    ax.grid(alpha=0.25)
    ax.legend(title="LIA system", frameon=False)
    fig.savefig(root / "top10_e_total.png", dpi=180)
    plt.close(fig)
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
