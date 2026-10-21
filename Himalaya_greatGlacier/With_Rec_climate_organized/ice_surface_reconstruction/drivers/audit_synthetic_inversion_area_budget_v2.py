"""Read-only staged area budget for saved synthetic LIA inversions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def audit(result_root: Path) -> tuple[dict, list[dict]]:
    from oggm import cfg, utils

    config = json.loads((result_root / "config_snapshot.json").read_text(encoding="utf-8"))
    formal = json.loads(
        (result_root / "inversion_summary_formal.json").read_text(encoding="utf-8")
    )
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=result_root / "synthetic_gdir",
        reset=False,
    )
    inversion = gdir.read_pickle("inversion_flowlines")
    model = gdir.read_pickle("model_flowlines")
    catchments = gdir.read_pickle("geometries").get("catchment_indices", [])
    if not (len(inversion) == len(model) == len(catchments)):
        raise ValueError(
            f"count-mismatch:{len(inversion)}/{len(model)}/{len(catchments)}"
        )
    cell_area_m2 = float(abs(gdir.grid.dx * gdir.grid.dy))
    rows = []
    for branch, (inv_fl, model_fl, cells) in enumerate(
        zip(inversion, model, catchments)
    ):
        inv_width = np.asarray(inv_fl.widths_m, dtype=float)
        model_width = np.asarray(model_fl.widths_m, dtype=float)
        thick = np.asarray(model_fl.thick, dtype=float)
        if inv_width.shape != model_width.shape:
            raise ValueError(f"branch-{branch}-node-shape-changed")
        inv_node_area = inv_width * float(inv_fl.dx_meter)
        model_node_area = model_width * float(model_fl.dx_meter)
        active = thick > 0
        rows.append(
            {
                "branch_index": branch,
                "node_count": int(inv_width.size),
                "catchment_cell_count": int(len(np.asarray(cells))),
                "catchment_area_m2": float(len(np.asarray(cells)) * cell_area_m2),
                "inversion_width_area_all_nodes_m2": float(inv_node_area.sum()),
                "model_width_area_all_nodes_m2": float(model_node_area.sum()),
                "model_active_area_m2": float(model_node_area[active].sum()),
                "model_inactive_width_area_m2": float(model_node_area[~active].sum()),
                "active_node_count": int(active.sum()),
                "inactive_node_count": int((~active).sum()),
                "maximum_absolute_width_change_m": float(
                    np.max(np.abs(model_width - inv_width))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    outline = float(formal["outline_area_m2"])
    catchment = float(frame.catchment_area_m2.sum())
    inv_width = float(frame.inversion_width_area_all_nodes_m2.sum())
    model_width = float(frame.model_width_area_all_nodes_m2.sum())
    active = float(frame.model_active_area_m2.sum())
    summary = {
        "system_id": int(config["system_id"]),
        "source_result_root": str(result_root),
        "outline_area_m2": outline,
        "catchment_area_m2": catchment,
        "inversion_width_area_all_nodes_m2": inv_width,
        "model_width_area_all_nodes_m2": model_width,
        "model_active_area_m2": active,
        "catchment_vs_outline_relative_difference": (catchment - outline) / outline,
        "inversion_width_vs_catchment_relative_difference": (inv_width - catchment) / catchment,
        "model_all_width_vs_inversion_width_relative_difference": (model_width - inv_width) / inv_width,
        "active_vs_model_all_width_relative_difference": (active - model_width) / model_width,
        "active_vs_outline_relative_difference": (active - outline) / outline,
        "inactive_width_area_m2": float(frame.model_inactive_width_area_m2.sum()),
        "inactive_node_count": int(frame.inactive_node_count.sum()),
        "negative_apparent_flux_branches": formal.get("negative_apparent_flux_branches", []),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    branch_rows = []
    for root in args.root:
        summary, rows = audit(root.resolve())
        summaries.append(summary)
        system_id = int(summary["system_id"])
        branch_rows.extend({"system_id": system_id, **row} for row in rows)
        write_json(output / f"system_{system_id}_area_budget.json", summary)
        pd.DataFrame(rows).to_csv(
            output / f"system_{system_id}_branch_area_budget.csv", index=False
        )
    pd.DataFrame(summaries).to_csv(output / "representative_area_budget.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(
        output / "representative_branch_area_budget.csv", index=False
    )
    write_json(
        output / "manifest.json",
        {
            "experiment": "synthetic-inversion-area-budget-audit-v2",
            "read_only": True,
            "systems": [int(row["system_id"]) for row in summaries],
            "source_roots": [str(root.resolve()) for root in args.root],
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
