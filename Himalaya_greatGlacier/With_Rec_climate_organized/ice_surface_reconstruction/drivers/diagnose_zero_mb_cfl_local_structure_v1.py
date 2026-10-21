"""Read-only local-structure audit at three registered zero-MB CFL nodes."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


CASES = {
    2729: {"root": "w301inv2729v1", "flowline": 1, "bin": 52, "required_s": 9.9, "configured_s": 60.0},
    2177: {"root": "w621inv2177v2", "flowline": 9, "bin": 147, "required_s": 51.7, "configured_s": 60.0},
    2611: {"root": "w582inv2611v1", "flowline": 18, "bin": 180, "required_s": 59.7, "configured_s": 60.0},
}


def _load_flowlines(root: Path):
    paths = list(root.rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl below {root}, found {len(paths)}")
    with gzip.open(paths[0], "rb") as handle:
        return pickle.load(handle), paths[0]


def _safe_array(flowline, name: str, count: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(flowline, name), dtype=float)
    except Exception:
        return np.full(count, np.nan)
    if values.ndim == 0:
        return np.full(count, float(values))
    if len(values) != count:
        return np.full(count, np.nan)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    all_rows: list[dict] = []
    case_rows: list[dict] = []
    for system_id, case in CASES.items():
        flowlines, source_path = _load_flowlines(project_root / case["root"])
        flowline = flowlines[case["flowline"]]
        n = int(flowline.nx)
        cfl_bin = int(np.clip(case["bin"], 0, n - 1))
        surface = np.asarray(flowline.surface_h, dtype=float)
        bed = np.asarray(flowline.bed_h, dtype=float)
        thick = np.asarray(flowline.thick, dtype=float)
        section = np.asarray(flowline.section, dtype=float)
        width = _safe_array(flowline, "widths_m", n)
        dx = float(flowline.dx_meter)
        slope = np.full(n, np.nan)
        slope[:-1] = (surface[:-1] - surface[1:]) / dx
        section_ratio = np.full(n, np.nan)
        section_ratio[1:] = section[1:] / np.where(section[:-1] != 0, section[:-1], np.nan)
        width_ratio = np.full(n, np.nan)
        width_ratio[1:] = width[1:] / np.where(width[:-1] != 0, width[:-1], np.nan)
        driving_proxy = thick * np.maximum(slope, 0.0)

        target_index = None
        junction_index = None
        if flowline.flows_to is not None:
            target_index = next(i for i, candidate in enumerate(flowlines) if candidate is flowline.flows_to)
            junction_index = int(flowline.flows_to_indice)
        incoming = []
        for index, candidate in enumerate(flowlines):
            if candidate.flows_to is flowline:
                incoming.append({"flowline": index, "junction_bin": int(candidate.flows_to_indice)})

        start = max(0, cfl_bin - 8)
        end = min(n, cfl_bin + 9)
        for node in range(start, end):
            all_rows.append(
                {
                    "system_id": system_id,
                    "flowline": case["flowline"],
                    "node": node,
                    "offset_from_cfl": node - cfl_bin,
                    "distance_m": node * dx,
                    "surface_m": surface[node],
                    "bed_m": bed[node],
                    "thickness_m": thick[node],
                    "width_m": width[node],
                    "section_m2": section[node],
                    "downslope_fraction": slope[node],
                    "section_ratio_to_previous": section_ratio[node],
                    "width_ratio_to_previous": width_ratio[node],
                    "driving_proxy_m": driving_proxy[node],
                    "active_ice": bool(thick[node] > 0),
                }
            )

        case_rows.append(
            {
                "system_id": system_id,
                "source_model_flowlines": str(source_path),
                "flowline_count": len(flowlines),
                "cfl_flowline": case["flowline"],
                "reported_bin": case["bin"],
                "audited_bin": cfl_bin,
                "node_count": n,
                "distance_to_branch_end_nodes": n - 1 - cfl_bin,
                "is_root": flowline.flows_to is None,
                "flows_to_flowline": target_index,
                "flows_to_bin": junction_index,
                "incoming_tributaries": incoming,
                "required_timestep_s": case["required_s"],
                "configured_timestep_s": case["configured_s"],
                "surface_m": float(surface[cfl_bin]),
                "bed_m": float(bed[cfl_bin]),
                "thickness_m": float(thick[cfl_bin]),
                "width_m": float(width[cfl_bin]),
                "section_m2": float(section[cfl_bin]),
                "downslope_fraction": float(slope[cfl_bin]),
                "section_ratio_to_previous": float(section_ratio[cfl_bin]),
                "width_ratio_to_previous": float(width_ratio[cfl_bin]),
                "driving_proxy_m": float(driving_proxy[cfl_bin]),
            }
        )

    pd.DataFrame(all_rows).to_csv(output_root / "cfl_local_nodes.csv", index=False)
    pd.DataFrame(
        [{**row, "incoming_tributaries": json.dumps(row["incoming_tributaries"])} for row in case_rows]
    ).to_csv(output_root / "cfl_case_summary.csv", index=False)
    (output_root / "cfl_case_summary.json").write_text(
        json.dumps(case_rows, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(case_rows, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
