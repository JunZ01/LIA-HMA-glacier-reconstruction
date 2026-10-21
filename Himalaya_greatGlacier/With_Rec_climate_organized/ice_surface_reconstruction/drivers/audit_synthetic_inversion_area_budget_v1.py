"""Read-only area-budget audit for saved synthetic LIA inversions.

The audit separates the two-dimensional outline/catchment area from the
one-dimensional inversion-flowline width integral and the initialized
active-ice model area.  It never edits a glacier directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _branch_budget(branch: int, inv_fl, model_fl, catchment_cells: int, cell_area_m2: float) -> dict:
    inv_widths = np.asarray(inv_fl.widths_m, dtype=float)
    model_widths = np.asarray(model_fl.widths_m, dtype=float)
    model_thick = np.asarray(model_fl.thick, dtype=float)
    if inv_widths.shape != model_widths.shape:
        raise ValueError(
            f"branch-{branch}-node-count-changed:{inv_widths.size}!={model_widths.size}"
        )
    active = model_thick > 0
    inv_node_area = inv_widths * float(inv_fl.dx_meter)
    model_node_area = model_widths * float(model_fl.dx_meter)
    catchment_area = float(catchment_cells * cell_area_m2)
    return {
        "branch_index": int(branch),
        "node_count": int(inv_widths.size),
        "catchment_cell_count": int(catchment_cells),
        "catchment_area_m2": catchment_area,
        "inversion_width_area_all_nodes_m2": float(np.sum(inv_node_area)),
        "model_width_area_all_nodes_m2": float(np.sum(model_node_area)),
        "model_active_area_m2": float(np.sum(model_node_area[active])),
        "model_inactive_width_area_m2": float(np.sum(model_node_area[~active])),
        "active_node_count": int(np.count_nonzero(active)),
        "inactive_node_count": int(np.count_nonzero(~active)),
        "width_integral_minus_catchment_m2": float(np.sum(inv_node_area) - catchment_area),
        "active_area_minus_width_integral_m2": float(
            np.sum(model_node_area[active]) - np.sum(inv_node_area)
        ),
        "maximum_absolute_width_change_m": float(
            np.max(np.abs(model_widths - inv_widths))
        ),
    }


def audit_root(result_root: Path) -> tuple[dict, list[dict]]:
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
    inv_flowlines = gdir.read_pickle("inversion_flowlines")
    model_flowlines = gdir.read_pickle("model_flowlines")
    if len(inv_flowlines) != len(model_flowlines):
        raise ValueError(
            f"flowline-count-changed:{len(inv_flowlines)}!={len(model_flowlines)}"
        )
    geometries = gdir.read_pickle("geometries")
    catchments = geometries.get("catchment_indices", [])
    if len(catchments) != len(inv_flowlines):
        raise ValueError(
            f"catchment-count:{len(catchments)}!=flowline-count:{len(inv_flowlines)}"
        )
    with gdir.get_filepath("gridded_data") as _:
        pass
    cell_area_m2 = float(abs(gdir.grid.dx * gdir.grid.dy))
    branch_rows = [
        _branch_budget(
            index,
            inv_fl,
            model_fl,
            int(len(np.asarray(cells))),
            cell_area_m2,
        )
        for index, (inv_fl, model_fl, cells) in enumerate(
            zip(inv_flowlines, model_flowlines, catchments)
        )
    ]
    frame = pd.DataFrame(branch_rows)
    catchment_area = float(frame["catchment_area_m2"].sum())
    inv_area = float(frame["inversion_width_area_all_nodes_m2"].sum())
    model_all_area = float(frame["model_width_area_all_nodes_m2"].sum())
    active_area = float(frame["model_active_area_m2"].sum())
    outline_area = float(formal["outline_area_m2"])
    summary = {
        "system_id": int(config["system_id"]),
        "source_result_root": str(result_root),
        "outline_area_m2": outline_area,
        "catchment_area_m2": catchment_area,
        "inversion_width_area_all_nodes_m2": inv_area,
        "model_width_area_all_nodes_m2": model_all_area,
        "model_active_area_m2": active_area,
        "catchment_vs_outline_relative_difference": (catchment_area - outline_area) / outline_area,
        "inversion_width_vs_catchment_relative_difference": (inv_area - catchment_area) / catchment_area,
        "model_all_width_vs_inversion_width_relative_difference": (model_all_area - inv_area) / inv_area,
        "active_vs_model_all_width_relative_difference": (active_area - model_all_area) / model_all_area,
        "active_vs_outline_relative_difference": (active_area - outline_area) / outline_area,
        "inactive_width_area_m2": float(frame["model_inactive_width_area_m2"].sum()),
        "inactive_node_count": int(frame["inactive_node_count"].sum()),
        "negative_apparent_flux_branches": formal.get("negative_apparent_flux_branches", []),
    }
    return summary, branch_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    all_rows = []
    for root in args.root:
        summary, rows = audit_root(root.resolve())
        system_id = int(summary["system_id"])
        summaries.append(summary)
        for row in rows:
            all_rows.append({"system_id": system_id, **row})
        _write_json(output / f"system_{system_id}_area_budget.json", summary)
        pd.DataFrame(rows).to_csv(
            output / f"system_{system_id}_branch_area_budget.csv", index=False
        )
    pd.DataFrame(summaries).to_csv(output / "representative_area_budget.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output / "representative_branch_area_budget.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "experiment": "synthetic-inversion-area-budget-audit-v1",
            "read_only": True,
            "representative_system_ids": [int(item["system_id"]) for item in summaries],
            "source_roots": [str(path.resolve()) for path in args.root],
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
