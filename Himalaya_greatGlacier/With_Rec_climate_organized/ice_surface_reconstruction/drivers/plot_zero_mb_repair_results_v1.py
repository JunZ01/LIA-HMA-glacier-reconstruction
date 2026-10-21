"""Versioned plots for the formal rescore and bounded CFL repair pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rescore = pd.read_csv(project / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_legacy_four_formal_rescore_v1/formal_rescore_table.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.bar(rescore.system_id.astype(str), rescore.closure_ratio, color="#4477AA")
    ax.axhline(1.0, color="#BB3333", linestyle="--", linewidth=1.2, label="formal limit")
    for i, value in enumerate(rescore.closure_ratio):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("max absolute per-step residual / formal limit")
    ax.set_xlabel("LIA system")
    ax.set_title("Formal rescore of four legacy cumulative-closure labels")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "legacy_four_formal_rescore.png", dpi=180)
    plt.close(fig)

    pilot = project / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_cfl_local_continuity_pilot_v1"
    diagnosis = pd.read_csv(project / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_cfl_local_structure_v1/cfl_local_nodes.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for ax, system_id in zip(axes, (2729, 2177, 2611)):
        repaired_path = pilot / f"system_{system_id}" / "local_repair_nodes.csv"
        if repaired_path.exists():
            data = pd.read_csv(repaired_path)
            ax.plot(data.node, data.old_surface_m, color="#777777", label="original surface")
            ax.plot(data.node, data.new_surface_m, color="#228833", label="local repair")
            ax.plot(data.node, data.bed_m, color="#AA7744", label="bed")
            result = json.loads((pilot / f"system_{system_id}" / "summary.json").read_text(encoding="utf-8"))
            subtitle = "5-y pass at 60 s" if result["status"] == "zero-mb-pass" else f"closed: {result.get('required_timestep_seconds_if_failed')} s"
        else:
            data = diagnosis.loc[diagnosis.system_id == system_id]
            ax.plot(data.node, data.surface_m, color="#777777", label="original surface")
            ax.plot(data.node, data.bed_m, color="#AA7744", label="bed")
            subtitle = "repair rejected by monotonic-surface gate"
        ax.set_title(f"System {system_id}\n{subtitle}")
        ax.set_xlabel("flowline node")
        ax.set_ylabel("elevation (m)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "three_cfl_local_repairs.png", dpi=180)
    plt.close(fig)

    classes = pd.read_csv(project / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_cfl_27_structural_classification_v1/cfl_27_structural_classification.csv")
    classes = classes.loc[classes.failure_type == "tributary_terminal_connector"].set_index("system_id")
    promotion = json.loads((project / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_terminal_connector_promotion_v1/promotion_summary.json").read_text(encoding="utf-8"))
    after = {2729: 1.0}
    status = {2729: "pass"}
    for item in promotion["systems"]:
        sid = int(item["system_id"])
        required = item.get("required_timestep_seconds_if_failed")
        after[sid] = (float(required) / 60.0) if required is not None else np.nan
        status[sid] = item["status"]
    ids = [2729, 2255, 553, 584, 2538, 2619]
    x = np.arange(len(ids))
    before = [float(classes.loc[sid, "required_ratio"]) for sid in ids]
    after_values = [after.get(sid, np.nan) for sid in ids]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.scatter(x, before, color="#4477AA", s=55, label="before repair")
    ax.scatter(x, after_values, color="#228833", marker="s", s=55, label="after one-shot repair")
    ax.axhline(1.0, color="#BB3333", linestyle="--", linewidth=1.2, label="60 s pass threshold")
    for i, sid in enumerate(ids):
        label = "excluded by newer geometry gate" if sid == 584 else status.get(sid, "not run")
        ax.text(i, 0.03, label, rotation=90, va="bottom", ha="center", fontsize=7)
    ax.set_xticks(x, [str(sid) for sid in ids])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("LIA system")
    ax.set_ylabel("required timestep / 60 s")
    ax.set_title("Terminal-connector repair is system-specific, not class-general")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "terminal_connector_class_outcomes.png", dpi=180)
    plt.close(fig)

    manifest = {"plots": ["legacy_four_formal_rescore.png", "three_cfl_local_repairs.png", "terminal_connector_class_outcomes.png"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
