"""Plot exact failure-state geometry for the closed 2547/5443 one-shot pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASES = {
    2547: {"flowline": 8, "node": 101, "source": "system_2547"},
    5443: {"flowline": 7, "node": 73, "source": "system_5443"},
}


def run(diagnosis_root: Path, pilot_root: Path) -> None:
    for system_id, case in CASES.items():
        data = pd.read_csv(diagnosis_root / case["source"] / "local_state_nodes.csv")
        data = data[data.flowline == case["flowline"]].copy()
        summary = json.loads((pilot_root.parent / f"lia_191_climate_steady_cfl_local_continuity_{system_id}_v1" / "summary.json").read_text(encoding="utf-8"))
        stages = list(dict.fromkeys(data.stage.tolist()))
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True, constrained_layout=True)
        initial = data[data.stage == "initial"].sort_values("node")
        axes[0].plot(initial.node, initial.bed_m, color="black", linewidth=1.6, label="bed (fixed)")
        palette = plt.cm.viridis_r
        for i, stage in enumerate(stages):
            if stage == "initial" or stage.startswith("year_") or stage == "failure_state":
                block = data[data.stage == stage].sort_values("node")
                label = "failure state" if stage == "failure_state" else stage.replace("_", " ")
                axes[0].plot(block.node, block.surface_m, linewidth=2.0 if stage == "failure_state" else 1.2,
                             color=palette(i / max(len(stages) - 1, 1)), label=label)
                axes[1].plot(block.node, block.thickness_m, linewidth=2.0 if stage == "failure_state" else 1.2,
                             color=palette(i / max(len(stages) - 1, 1)), label=label)
        for ax in axes:
            ax.axvline(case["node"], color="crimson", linestyle="--", linewidth=1.2,
                       label=f"CFL node {case['node']}" if ax is axes[0] else None)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Elevation (m)")
        axes[1].set_ylabel("Ice thickness (m)")
        axes[1].set_xlabel("Flowline node")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].legend(ncol=3, fontsize=8)
        axes[0].set_title(
            f"System {system_id}: climate-steady CFL develops at tributary terminal\n"
            f"one-shot conservative continuity closed at static gate "
            f"(reverse step {summary['maximum_reverse_surface_step_m']:.2f} m)"
        )
        target = pilot_root.parent / f"lia_191_climate_steady_cfl_local_continuity_{system_id}_v1"
        fig.savefig(target / "local_cfl_diagnosis.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.diagnosis_root.resolve(), args.pilot_root.resolve())


if __name__ == "__main__":
    main()
