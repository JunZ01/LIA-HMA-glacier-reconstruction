"""Read-only structural classification of the 27 registered CFL cases."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


PATTERN = re.compile(
    r"required time step smaller than the minimum allowed:\s*(?P<required>[0-9.]+)s\s*vs\s*(?P<configured>[0-9.]+)s.*?"
    r"fl_id\s*(?P<fl>[0-9]+),\s*bin_id\s*(?P<bin>[0-9]+)", re.I | re.S
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.input_csv.resolve())
    frame = frame.loc[frame.failure_class == "CFL"].copy()
    rows = []
    for row in frame.itertuples(index=False):
        zero_root = project_root / str(row.zero_mb_evidence_path).replace("\\", "/").rsplit("/", 1)[0]
        inversion_root = project_root / str(row.inversion_evidence_path).replace("\\", "/").rsplit("/", 1)[0]
        formal = zero_root / "zero_mb_summary_formal.json"
        summary_path = formal if formal.exists() else zero_root / "zero_mb_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        match = PATTERN.search(summary.get("runtime_error") or "")
        if not match:
            raise RuntimeError(f"cannot parse CFL for system {row.lia_unique_id}")
        parsed = {k: float(v) for k, v in match.groupdict().items()}
        fl_index, bin_index = int(parsed["fl"]), int(parsed["bin"])
        pickle_paths = list(inversion_root.rglob("model_flowlines.pkl"))
        if len(pickle_paths) != 1:
            raise RuntimeError(f"model_flowlines count {len(pickle_paths)} for {row.lia_unique_id}")
        with gzip.open(pickle_paths[0], "rb") as handle:
            flowlines = pickle.load(handle)
        flowline = flowlines[fl_index]
        n = int(flowline.nx)
        audited_bin = int(np.clip(bin_index, 0, n - 1))
        is_root = flowline.flows_to is None
        terminal_distance = n - 1 - audited_bin
        incoming = [
            int(candidate.flows_to_indice)
            for candidate in flowlines if candidate.flows_to is flowline
        ]
        junction_distance = min([abs(audited_bin - value) for value in incoming], default=999999)
        bed = np.asarray(flowline.bed_h, float)
        thick = np.asarray(flowline.thick, float)
        lo, hi = max(0, audited_bin - 4), min(n - 1, audited_bin + 4)
        local_bed_step = float(np.max(np.abs(np.diff(bed[lo:hi + 1])))) if hi > lo else 0.0
        local_thickness_step = float(np.max(np.abs(np.diff(thick[lo:hi + 1])))) if hi > lo else 0.0
        if not is_root and (bin_index >= n or terminal_distance <= 1):
            failure_type = "tributary_terminal_connector"
        elif is_root and junction_distance <= 4 and max(local_bed_step, local_thickness_step) >= 50.0:
            failure_type = "root_near_junction_bed_thickness_step"
        elif is_root and max(local_bed_step, local_thickness_step) >= 50.0:
            failure_type = "root_interior_bed_thickness_step"
        elif is_root and junction_distance <= 4:
            failure_type = "root_near_junction_other"
        else:
            failure_type = "other_local_stiffness"
        rows.append({
            "system_id": int(row.lia_unique_id), "failure_type": failure_type,
            "required_timestep_s": parsed["required"], "configured_timestep_s": parsed["configured"],
            "required_ratio": parsed["required"] / parsed["configured"],
            "flowline": fl_index, "bin": bin_index, "audited_bin": audited_bin,
            "node_count": n, "is_root": is_root,
            "distance_to_branch_end_nodes": terminal_distance,
            "nearest_incoming_junction_distance_nodes": None if junction_distance == 999999 else junction_distance,
            "maximum_local_bed_step_m": local_bed_step,
            "maximum_local_thickness_step_m": local_thickness_step,
            "zero_summary_path": str(summary_path), "model_flowlines_path": str(pickle_paths[0]),
        })
    result = pd.DataFrame(rows).sort_values(["failure_type", "required_ratio", "system_id"])
    result.to_csv(output_root / "cfl_27_structural_classification.csv", index=False)
    counts = result.groupby("failure_type").size().rename("count").reset_index()
    counts.to_csv(output_root / "class_counts.csv", index=False)
    manifest = {"system_count": int(len(result)), "class_counts": counts.to_dict(orient="records"),
                "oggm_advanced": False, "source_files_modified": False}
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
