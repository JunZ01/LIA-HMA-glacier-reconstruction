"""Read-only MixedBed setter conditioning audit for the 3660 coarse root."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
CANDIDATE = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v4/candidate_model_flowlines.pkl"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_conditioning_v1"
)
ROOT_ID = 32


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with gzip.open(CANDIDATE, "rb") as stream:
        flowlines = pickle.load(stream)
    root = flowlines[ROOT_ID]
    active = np.asarray(root.thick, dtype=float) > 0.0
    lambdas = np.asarray(root._lambdas, dtype=float)
    section0 = np.asarray(root.section, dtype=float).copy()
    volume0 = float(root.volume_m3)
    test = copy.deepcopy(root)
    test.section = section0.copy()
    section1 = np.asarray(test.section, dtype=float)
    difference = section1 - section0
    worst = np.argsort(np.abs(difference))[::-1][:20]
    positive = lambdas[active & (lambdas > 0.0)]
    payload = {
        "experiment": "zero-mb-3660-root-coarsen2x-mixedbed-conditioning-v1",
        "system_id": 3660,
        "read_only": True,
        "candidate": str(CANDIDATE),
        "active_node_count": int(np.count_nonzero(active)),
        "positive_lambda_count": int(len(positive)),
        "positive_lambda_below_1e-12_count": int(np.count_nonzero(positive < 1e-12)),
        "positive_lambda_below_1e-10_count": int(np.count_nonzero(positive < 1e-10)),
        "minimum_positive_lambda": float(np.min(positive)) if len(positive) else None,
        "maximum_positive_lambda": float(np.max(positive)) if len(positive) else None,
        "setter_roundtrip": {
            "volume_before_m3": volume0,
            "volume_after_m3": float(test.volume_m3),
            "volume_difference_m3": float(test.volume_m3 - volume0),
            "maximum_absolute_section_difference_m2": float(np.max(np.abs(difference))),
            "material_node_count_over_1e-6_m2": int(np.count_nonzero(np.abs(difference) > 1e-6)),
            "worst_nodes": [
                {
                    "node": int(node),
                    "lambda": float(lambdas[node]),
                    "section_before_m2": float(section0[node]),
                    "section_after_m2": float(section1[node]),
                    "difference_m2": float(difference[node]),
                }
                for node in worst
            ],
        },
        "finding": (
            "If the setter roundtrip loses the same order of volume as the first "
            "zero-MB step and affected nodes have near-zero positive lambda, the "
            "pilot is an ill-conditioned representation failure rather than a valid "
            "coarse-grid dynamics test."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
