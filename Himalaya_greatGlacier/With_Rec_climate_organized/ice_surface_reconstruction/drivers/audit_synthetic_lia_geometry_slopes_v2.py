"""Read-only slope and junction audit for gzip-compressed OGGM flowlines."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    tolerance = float(config.get("tolerance_m", 1e-7))
    for case in config["cases"]:
        system_id = int(case["system_id"])
        pickle_path = Path(case["gdir_root"]) / "inversion_flowlines.pkl"
        with gzip.open(pickle_path, "rb") as stream:
            flowlines = pickle.load(stream)
        by_identity = {id(fl): i for i, fl in enumerate(flowlines)}
        within_positive, junction_positive = [], []
        for index, flowline in enumerate(flowlines):
            surface = np.asarray(flowline.surface_h, dtype=float)
            for edge, difference in enumerate(np.diff(surface)):
                if difference > tolerance:
                    within_positive.append(float(difference))
                    rows.append({
                        "system_id": system_id, "kind": "within_branch",
                        "branch_index": index, "receiver_branch_index": np.nan,
                        "node_index": edge, "receiver_node_index": edge + 1,
                        "reverse_step_m": float(difference),
                    })
            if flowline.flows_to is not None:
                receiver_index = int(flowline.flows_to_indice)
                difference = float(flowline.flows_to.surface_h[receiver_index] - surface[-1])
                if difference > tolerance:
                    junction_positive.append(difference)
                    rows.append({
                        "system_id": system_id, "kind": "junction",
                        "branch_index": index,
                        "receiver_branch_index": by_identity[id(flowline.flows_to)],
                        "node_index": len(surface) - 1,
                        "receiver_node_index": receiver_index,
                        "reverse_step_m": difference,
                    })
        summaries.append({
            "system_id": system_id, "branch_count": len(flowlines),
            "within_reverse_count": len(within_positive),
            "within_reverse_max_m": max(within_positive, default=0.0),
            "junction_reverse_count": len(junction_positive),
            "junction_reverse_max_m": max(junction_positive, default=0.0),
            "junction_reverse_median_m": float(np.median(junction_positive)) if junction_positive else 0.0,
        })
    pd.DataFrame(rows).to_csv(output_root / "reverse_slope_details.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_root / "system_summary.csv", index=False)
    payload = {"status": "complete", "tolerance_m": tolerance, "systems": summaries}
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
