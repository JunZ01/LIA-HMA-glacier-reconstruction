"""Real 30-second connected-front gate for system 2005."""

import json
import traceback
from pathlib import Path

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_front_step_v1 import connected_front_step


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005/model_flowlines.pkl"
FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_fixture_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_step_v1"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        if json.loads(FIXTURE.read_text(encoding="utf-8"))["status"] != "fixture-pass-real-system-not-authorized":
            raise RuntimeError("fixture-not-pass")
        from oggm import cfg
        cfg.initialize(logging_level="WARNING")
        flowlines = base.load_flowlines(SOURCE)
        v0 = base.total_volume(flowlines)
        limit = max(1e-6, v0 * 1e-12)
        step = connected_front_step(flowlines, base.Coefficients(4.355770498109968e-23, 0.0), 30.0, limit)
        summary = {"status": "real-single-step-pass", "system_id": 2005, "mechanism": "connected-ice-mb-mask-plus-one-dry-front-capacity-transport-v1", "dt_seconds": 30.0, "initial_volume_m3": v0, "final_volume_m3": base.total_volume(flowlines), "closure_limit_m3": limit, "step": step, "source_model_flowlines_sha256": base.sha256(SOURCE), "source_modified": False, "zero_mb_run": False, "steady_run": False, "historical_run": False}
    except Exception as exc:
        summary = {"status": "real-single-step-fail-mechanism-closed", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "source_modified": False, "zero_mb_run": False, "steady_run": False, "historical_run": False}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "real-single-step-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
