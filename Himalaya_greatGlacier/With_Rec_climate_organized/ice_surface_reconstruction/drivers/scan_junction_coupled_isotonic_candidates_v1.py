"""Read-only scan of saved 50-year states for the frozen three-node mechanism."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _system_id(root: Path) -> int | None:
    for name in ("steady_50y_summary.json", "config_snapshot.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig")).get("system_id")
            if value is not None:
                return int(value)
        except Exception:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for state in project_root.glob("w*/lia_equilibrium_state_50y.pkl.gz"):
        try:
            with gzip.open(state, "rb") as stream:
                flowlines = pickle.load(stream)
        except Exception:
            continue
        system_id = _system_id(state.parent)
        incoming: dict[int, list[int]] = {}
        for branch_id, branch in enumerate(flowlines):
            if branch.flows_to is None:
                continue
            receiver_id = next((i for i, candidate in enumerate(flowlines) if candidate is branch.flows_to), None)
            if receiver_id is not None:
                incoming.setdefault(receiver_id, []).append(int(branch.flows_to_indice))
        reversals = []
        for flowline_id, flowline in enumerate(flowlines):
            active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
            if len(active) < 2:
                continue
            last = int(active[-1])
            surface = np.asarray(flowline.surface_h, float)
            for node in np.flatnonzero(np.diff(surface[: last + 1]) > 1e-8):
                node = int(node)
                if flowline.flows_to is not None:
                    distance = last - node
                    junction_kind = "tributary-terminal"
                    junction_node = last
                elif incoming.get(flowline_id):
                    junction_node = min(incoming[flowline_id], key=lambda value: abs(value - node))
                    distance = abs(junction_node - node)
                    junction_kind = "receiver-root"
                else:
                    junction_node = None
                    distance = 10**9
                    junction_kind = "none"
                reversals.append({
                    "flowline": flowline_id, "node": node,
                    "reverse_step_m": float(surface[node + 1] - surface[node]),
                    "junction_kind": junction_kind, "junction_node": junction_node,
                    "distance_to_junction_nodes": distance,
                })
        near = [item for item in reversals if item["distance_to_junction_nodes"] <= 2]
        exact = len(reversals) == 1 and len(near) == 1
        rows.append({
            "system_id": system_id, "steady_root": str(state.parent), "flowline_count": len(flowlines),
            "active_reverse_count": len(reversals), "near_junction_reverse_count": len(near),
            "frozen_three_node_candidate": exact,
            "reverse_records": json.dumps(reversals, separators=(",", ":")),
        })
    frame = pd.DataFrame(rows).sort_values(["frozen_three_node_candidate", "near_junction_reverse_count"], ascending=[False, False])
    frame.to_csv(output_root / "candidate_scan.csv", index=False)
    candidates = frame.loc[frame["frozen_three_node_candidate"] & frame["system_id"].ne(5443)]
    manifest = {
        "experiment": "junction-coupled-isotonic-candidate-scan-v1",
        "saved_state_count": int(len(frame)),
        "candidate_count_excluding_5443": int(len(candidates)),
        "candidate_systems": sorted(set(int(value) for value in candidates.system_id.dropna())),
        "read_only": True,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

