"""Run Round 32 with the archived Round 31 observation-column compatibility shim."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_climate_reconstruction_loop_v2_round32_precipitation_nested_amplitude_transfer_v1 import main


_read_csv = pd.read_csv


def _compatible_read_csv(path, *args, **kwargs):
    frame = _read_csv(path, *args, **kwargs)
    if Path(path).name == "fixed_candidate_block_scores.csv" and "observation" in frame.columns:
        frame = frame.rename(columns={"observation": "eval_product"})
    return frame


if __name__ == "__main__":
    pd.read_csv = _compatible_read_csv
    raise SystemExit(main())
