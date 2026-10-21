"""Read-only node audit for apparent-flux clipping and active-area loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _dump(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _audit(root: Path) -> tuple[dict, list[dict]]:
    from oggm import cfg, utils

    config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]), base_dir=root / "synthetic_gdir", reset=False
    )
    inversion_flowlines = gdir.read_pickle("inversion_flowlines")
    model_flowlines = gdir.read_pickle("model_flowlines")
    inversion_output = gdir.read_pickle("inversion_output")
    rows = []
    for branch, (inv_fl, model_fl, inv_out) in enumerate(
        zip(inversion_flowlines, model_flowlines, inversion_output)
    ):
        node_count = int(inv_fl.nx)
        flux = np.asarray(inv_fl.flux, dtype=float)
        model_thick = np.asarray(model_fl.thick, dtype=float)[:node_count]
        model_width = np.asarray(model_fl.widths_m, dtype=float)[:node_count]
        surface = np.asarray(model_fl.surface_h, dtype=float)[:node_count]
        inversion_thick = np.asarray(inv_out.get("thick", np.zeros(node_count)), dtype=float)
        if inversion_thick.size != node_count:
            raise ValueError(f"branch-{branch}-inversion-output-size")
        for node in range(node_count):
            rows.append(
                {
                    "branch_index": branch,
                    "node_index": node,
                    "distance_m": float(node * model_fl.dx_meter),
                    "surface_h_m": float(surface[node]),
                    "apparent_flux_native": float(flux[node]),
                    "negative_apparent_flux": bool(flux[node] < 0),
                    "nonpositive_apparent_flux": bool(flux[node] <= 0),
                    "inversion_output_thickness_m": float(inversion_thick[node]),
                    "model_thickness_m": float(model_thick[node]),
                    "active_ice": bool(model_thick[node] > 0),
                    "node_plan_area_m2": float(model_width[node] * model_fl.dx_meter),
                }
            )
    frame = pd.DataFrame(rows)
    negative = frame.negative_apparent_flux
    inactive = ~frame.active_ice
    summary = {
        "system_id": int(config["system_id"]),
        "source_result_root": str(root),
        "source_node_count": int(len(frame)),
        "negative_flux_node_count": int(negative.sum()),
        "inactive_node_count": int(inactive.sum()),
        "negative_and_inactive_node_count": int((negative & inactive).sum()),
        "negative_but_active_node_count": int((negative & ~inactive).sum()),
        "nonnegative_but_inactive_node_count": int((~negative & inactive).sum()),
        "negative_flux_node_area_m2": float(frame.loc[negative, "node_plan_area_m2"].sum()),
        "inactive_node_area_m2": float(frame.loc[inactive, "node_plan_area_m2"].sum()),
        "negative_inactive_area_overlap_m2": float(
            frame.loc[negative & inactive, "node_plan_area_m2"].sum()
        ),
        "minimum_negative_flux_native": float(frame.apparent_flux_native.min()),
        "maximum_inversion_model_thickness_difference_m": float(
            np.max(
                np.abs(
                    frame.inversion_output_thickness_m.to_numpy()
                    - frame.model_thickness_m.to_numpy()
                )
            )
        ),
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
    all_rows = []
    for root in args.root:
        summary, rows = _audit(root.resolve())
        summaries.append(summary)
        system_id = int(summary["system_id"])
        all_rows.extend({"system_id": system_id, **row} for row in rows)
        _dump(output / f"system_{system_id}_active_loss_summary.json", summary)
        pd.DataFrame(rows).to_csv(output / f"system_{system_id}_node_active_loss.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "representative_active_loss_summary.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output / "representative_node_active_loss.csv", index=False)
    _dump(
        output / "manifest.json",
        {
            "experiment": "synthetic-inversion-active-loss-audit-v1",
            "read_only": True,
            "systems": [int(row["system_id"]) for row in summaries],
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
