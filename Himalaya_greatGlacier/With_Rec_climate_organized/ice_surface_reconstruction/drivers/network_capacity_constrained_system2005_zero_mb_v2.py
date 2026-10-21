"""V2 launcher adding the explicit node residual required by the old shell."""

import json
import sys
from pathlib import Path

import pandas as pd

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_capacity_constrained_system2005_step_v2 import capacity_one_step_v2


PROJECT = Path(__file__).resolve().parents[4]
implementation.one_step = capacity_one_step_v2
from . import network_coupled_semiimplicit_system2005_zero_mb_v3 as adaptive


def main() -> None:
    exit_code = 0
    try:
        adaptive.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    if not output.is_absolute():
        output = PROJECT / output
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoints = output / "checkpoints.csv"
    maximum_gaps = maximum_limited = None
    if checkpoints.exists():
        frame = pd.read_csv(checkpoints)
        maximum_gaps = int(frame["internal_active_gap_count"].max())
        maximum_limited = int(frame["limited_edge_count"].max())
    summary["mechanism"] = "network-dag-capacity-constrained-least-change-flux-v1"
    summary["maximum_internal_active_gap_count"] = maximum_gaps
    summary["maximum_limited_edge_count_at_recorded_checkpoints"] = maximum_limited
    summary["posthoc_section_clip_used"] = False
    if maximum_gaps is not None and maximum_gaps > int(config["gates"]["internal_active_gap_count_max"]):
        summary["status"] = "zero-mb-fail-mechanism-closed"
        summary.setdefault("review_flags", []).append("internal-active-gap")
        exit_code = 1
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
