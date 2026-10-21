"""Read-only area budget that separates the appended root safety tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _audit(root: Path) -> tuple[dict, list[dict]]:
    from oggm import cfg, utils

    config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    formal = json.loads((root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]), base_dir=root / "synthetic_gdir", reset=False
    )
    inversion = gdir.read_pickle("inversion_flowlines")
    model = gdir.read_pickle("model_flowlines")
    catchments = gdir.read_pickle("geometries").get("catchment_indices", [])
    if not (len(inversion) == len(model) == len(catchments)):
        raise ValueError("flowline/catchment count mismatch")
    cell_area = float(abs(gdir.grid.dx * gdir.grid.dy))
    rows = []
    for branch, (inv, mod, cells) in enumerate(zip(inversion, model, catchments)):
        iw = np.asarray(inv.widths_m, dtype=float)
        mw = np.asarray(mod.widths_m, dtype=float)
        thick = np.asarray(mod.thick, dtype=float)
        if mw.size < iw.size:
            raise ValueError(f"branch-{branch}-model-shorter-than-inversion")
        source_mw = mw[: iw.size]
        source_thick = thick[: iw.size]
        tail_mw = mw[iw.size :]
        tail_thick = thick[iw.size :]
        inv_area = iw * float(inv.dx_meter)
        source_model_area = source_mw * float(mod.dx_meter)
        tail_area = tail_mw * float(mod.dx_meter)
        active_source = source_thick > 0
        active_tail = tail_thick > 0
        rows.append(
            {
                "branch_index": branch,
                "inversion_node_count": int(iw.size),
                "appended_tail_node_count": int(tail_mw.size),
                "catchment_cell_count": int(len(np.asarray(cells))),
                "catchment_area_m2": float(len(np.asarray(cells)) * cell_area),
                "inversion_width_area_m2": float(inv_area.sum()),
                "model_source_domain_width_area_m2": float(source_model_area.sum()),
                "model_active_source_area_m2": float(source_model_area[active_source].sum()),
                "model_inactive_source_width_area_m2": float(source_model_area[~active_source].sum()),
                "model_tail_width_area_m2": float(tail_area.sum()),
                "model_active_tail_area_m2": float(tail_area[active_tail].sum()),
                "active_source_node_count": int(active_source.sum()),
                "inactive_source_node_count": int((~active_source).sum()),
                "active_tail_node_count": int(active_tail.sum()),
                "maximum_absolute_source_width_change_m": float(
                    np.max(np.abs(source_mw - iw))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    outline = float(formal["outline_area_m2"])
    catchment = float(frame.catchment_area_m2.sum())
    inv_area = float(frame.inversion_width_area_m2.sum())
    model_source = float(frame.model_source_domain_width_area_m2.sum())
    active_source = float(frame.model_active_source_area_m2.sum())
    active_tail = float(frame.model_active_tail_area_m2.sum())
    summary = {
        "system_id": int(config["system_id"]),
        "source_result_root": str(root),
        "outline_area_m2": outline,
        "catchment_area_m2": catchment,
        "inversion_width_area_m2": inv_area,
        "model_source_domain_width_area_m2": model_source,
        "model_active_source_area_m2": active_source,
        "model_active_tail_area_m2": active_tail,
        "catchment_vs_outline_relative_difference": (catchment - outline) / outline,
        "inversion_width_vs_catchment_relative_difference": (inv_area - catchment) / catchment,
        "model_source_vs_inversion_width_relative_difference": (model_source - inv_area) / inv_area,
        "active_source_vs_model_source_relative_difference": (active_source - model_source) / model_source,
        "active_total_vs_outline_relative_difference": (active_source + active_tail - outline) / outline,
        "inactive_source_width_area_m2": float(frame.model_inactive_source_width_area_m2.sum()),
        "inactive_source_node_count": int(frame.inactive_source_node_count.sum()),
        "negative_apparent_flux_branches": formal.get("negative_apparent_flux_branches", []),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    branches = []
    for root in args.root:
        summary, rows = _audit(root.resolve())
        summaries.append(summary)
        system_id = int(summary["system_id"])
        branches.extend({"system_id": system_id, **row} for row in rows)
        _dump(output / f"system_{system_id}_area_budget.json", summary)
        pd.DataFrame(rows).to_csv(output / f"system_{system_id}_branch_area_budget.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "representative_area_budget.csv", index=False)
    pd.DataFrame(branches).to_csv(output / "representative_branch_area_budget.csv", index=False)
    _dump(
        output / "manifest.json",
        {
            "experiment": "synthetic-inversion-area-budget-audit-v3",
            "read_only": True,
            "systems": [int(row["system_id"]) for row in summaries],
            "source_roots": [str(root.resolve()) for root in args.root],
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
