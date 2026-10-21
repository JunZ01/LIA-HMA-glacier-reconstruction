"""Zero-MB wrapper adding the formal initial-LIA observation-domain gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from . import synthetic_lia_system_zero_mb_v1 as v1
from .lia_observation_domain import observation_domain_metrics


_state_v1 = v1._state
_run_v1 = v1.run
_domain_node_count = None


def _state(model, initial_volume: float, year: float) -> dict:
    global _domain_node_count
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"zero-mb-root-count:{len(roots)}")
    if _domain_node_count is None:
        _domain_node_count = int(np.count_nonzero(np.asarray(roots[0].thick) > 0))
    return {
        **_state_v1(model, initial_volume, year),
        **observation_domain_metrics(
            roots[0].thick,
            roots[0].dx_meter,
            _domain_node_count,
        ),
    }


def run(config_path: Path) -> dict:
    global _domain_node_count
    _domain_node_count = None
    result = _run_v1(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    checkpoints = pd.read_csv(output_root / "zero_mb_checkpoints.csv")
    overflow = checkpoints["observation_domain_overflow"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    flags = list(result.get("review_flags", []))
    if bool(overflow.any()):
        flags.append("observation-domain-overflow")
    formal = {
        **result,
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "review_flags": list(dict.fromkeys(flags)),
        "observation_domain_node_count": int(_domain_node_count),
        "observation_domain_length_m": float(
            checkpoints.iloc[0]["observation_domain_length_m"]
        ),
        "observation_domain_overflow": bool(overflow.any()),
        "first_overflow_year": (
            None
            if not overflow.any()
            else float(checkpoints.loc[overflow, "year"].iloc[0])
        ),
        "maximum_overflow_thickness_m": float(
            checkpoints["maximum_observation_domain_overflow_thickness_m"].max()
        ),
    }
    (output_root / "zero_mb_summary_formal.json").write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return formal


v1._state = _state
v1.run = run


if __name__ == "__main__":
    v1.main()
