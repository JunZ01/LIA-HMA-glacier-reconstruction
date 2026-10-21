"""Tracked 5y zero-MB gate for the capacity-constrained network solver."""

import gzip
import json
import pickle
import sys
from pathlib import Path

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_capacity_constrained_system2005_step_v2 import capacity_one_step_v2


PROJECT = Path(__file__).resolve().parents[4]
MAX_GAPS = 0
MAX_LIMITED = 0
FINAL_FLOWLINES = None
ORIGINAL_LOAD = implementation.load_flowlines


def tracked_load(path):
    global FINAL_FLOWLINES
    FINAL_FLOWLINES = ORIGINAL_LOAD(path)
    return FINAL_FLOWLINES


def tracked_step(flowlines, coefficients, dt_seconds, closure_limit_m3):
    global MAX_GAPS, MAX_LIMITED
    result = capacity_one_step_v2(flowlines, coefficients, dt_seconds, closure_limit_m3)
    MAX_GAPS = max(MAX_GAPS, int(result["internal_active_gap_count"]))
    MAX_LIMITED = max(MAX_LIMITED, int(result["limited_edge_count"]))
    return result


implementation.load_flowlines = tracked_load
implementation.one_step = tracked_step
from . import network_coupled_semiimplicit_system2005_zero_mb_v3 as adaptive


def main():
    exit_code = 0
    try:
        adaptive.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adjudication = Path(config["timestep_adjudication"])
    if not adjudication.is_absolute():
        adjudication = PROJECT / adjudication
    adjudication_data = json.loads(adjudication.read_text(encoding="utf-8"))
    if adjudication_data["selected_for_5y"] != "24h":
        raise RuntimeError("24h-timestep-not-authorized")
    output = Path(config["output_root"])
    if not output.is_absolute():
        output = PROJECT / output
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root = next(fl for fl in FINAL_FLOWLINES if fl.flows_to is None)
    active = [i for i, value in enumerate(root.thick) if value > implementation.ACTIVE_TOLERANCE_M]
    root_length = float((active[-1] + 1) * root.dx_meter) if active else 0.0
    free_tail = float((root.nx - active[-1] - 1) * root.dx_meter) if active else float(root.nx * root.dx_meter)
    length_error = abs(root_length - float(config["registered_lia_length_m"])) / float(config["registered_lia_length_m"])
    gates = config["gates"]
    extra_checks = {
        "no_internal_active_gaps": MAX_GAPS <= int(gates["internal_active_gap_count_max"]),
        "root_length": length_error <= float(gates["root_length_relative_error_max"]),
        "free_downstream": free_tail >= float(gates["minimum_free_downstream_m"]),
    }
    summary["mechanism"] = "network-dag-capacity-constrained-least-change-flux-v1"
    summary["maximum_internal_active_gap_count"] = MAX_GAPS
    summary["maximum_limited_edge_count"] = MAX_LIMITED
    summary["root_length_m"] = root_length
    summary["root_length_relative_error"] = length_error
    summary["free_downstream_m"] = free_tail
    summary["extra_checks"] = extra_checks
    summary["posthoc_section_clip_used"] = False
    if summary.get("status") != "zero-mb-pass" or not all(extra_checks.values()):
        summary["status"] = "zero-mb-fail-mechanism-closed"
        exit_code = 1
    if FINAL_FLOWLINES is not None:
        with gzip.open(output / "final_model_flowlines.pkl.gz", "wb") as stream:
            pickle.dump(FINAL_FLOWLINES, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
