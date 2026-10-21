"""Corrected 30-second zero-MB pilot for the conditioned 2x root."""

from __future__ import annotations

import json
from pathlib import Path

from . import run_zero_mb_3660_mixedbed_shape_mindt30_v1 as runner


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PREFLIGHT = RESULTS / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v5/summary.json"
CANDIDATE = RESULTS / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v5/candidate_model_flowlines.pkl"
OUTPUT = RESULTS / "lia_191_zero_mb_3660_root_coarsen2x_mindt30_v2"


def main() -> None:
    runner.PREFLIGHT = PREFLIGHT
    runner.CANDIDATE = CANDIDATE
    runner.OUTPUT = OUTPUT
    runner.main()
    summary_path = OUTPUT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    payload["experiment"] = "zero-mb-3660-uniform-root-coarsen2x-conditioned-mindt30-v2"
    payload["mechanism"] = "uniform-root-2x-width-weighted-conditioned-mixedbed-control-volumes"
    payload["supersedes_invalid_pilot"] = str(
        RESULTS / "lia_191_zero_mb_3660_root_coarsen2x_mindt30_v1/summary.json"
    )
    payload["supersession_reason"] = (
        "v1 used 61 ill-conditioned near-zero positive lambdas and failed the "
        "new mandatory MixedBed setter-roundtrip conservation gate"
    )
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
