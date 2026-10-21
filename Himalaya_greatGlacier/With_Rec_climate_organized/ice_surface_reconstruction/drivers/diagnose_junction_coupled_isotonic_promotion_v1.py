from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
SOURCES = {
    2570: PROJECT / "w256steady2570dtm04375mindt30retryv1",
    3244: PROJECT / "w648steady_conservative_batch35_39_round2_3244_v1",
    3906: PROJECT / "w648steady_conservative_batch35_39_round2_3906_v1",
}
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_junction_coupled_isotonic_promotion_preflight_v1"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    records = []
    for system_id, root in SOURCES.items():
        with gzip.open(root / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
            flowlines = pickle.load(stream)
        reverse = []
        for flowline_id, flowline in enumerate(flowlines):
            active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
            if len(active) < 2:
                continue
            last = int(active[-1])
            surface = np.asarray(flowline.surface_h, float)
            for node in np.flatnonzero(np.diff(surface[: last + 1]) > 1e-8):
                reverse.append((flowline_id, int(node), last))
        if len(reverse) != 1:
            raise RuntimeError(f"system {system_id}: expected one reversal, got {reverse}")
        flowline_id, node, last = reverse[0]
        branch = flowlines[flowline_id]
        if branch.flows_to is None:
            raise RuntimeError(f"system {system_id}: reversal is not on a tributary")
        receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is branch.flows_to)
        receiver_node = int(branch.flows_to_indice)
        path = [(flowline_id, n) for n in range(node, last + 1)] + [(receiver_id, receiver_node)]
        values = []
        for path_index, (fid, n) in enumerate(path):
            flowline = flowlines[fid]
            values.append(
                {
                    "path_index": path_index,
                    "flowline": fid,
                    "node": n,
                    "surface_m": float(flowline.surface_h[n]),
                    "bed_m": float(flowline.bed_h[n]),
                    "thickness_m": float(flowline.thick[n]),
                    "width_m": float(flowline.widths_m[n]),
                    "dx_m": float(flowline.dx_meter),
                    "rectangular": bool(np.asarray(flowline.is_rectangular)[n]),
                }
            )
        records.append(
            {
                "system_id": system_id,
                "source_steady_root": root.name,
                "reverse_flowline": flowline_id,
                "reverse_node": node,
                "tributary_last_active_node": last,
                "receiver_flowline": receiver_id,
                "receiver_node": receiver_node,
                "full_junction_path_node_count": len(path),
                "path": values,
            }
        )
    payload = {"experiment": "junction-coupled-isotonic-promotion-preflight-v1", "systems": records}
    (OUTPUT / "preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
