"""Tracked direct-timestep runner for the exact-zero connected-front step."""

from __future__ import annotations

import json
from pathlib import Path

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation
from .network_capacity_connected_front_step_v2 import connected_front_step_v2


PROJECT = Path(__file__).resolve().parents[4]
_MAX_GAPS = 0
_MAX_LIMITED = 0
_MAX_PROJECTION_CONSERVATION = 0.0
_MAX_ZERO_CANONICALIZATION = 0.0


def tracked_step(flowlines, coefficients, dt_seconds, closure_limit_m3):
    global _MAX_GAPS, _MAX_LIMITED, _MAX_PROJECTION_CONSERVATION, _MAX_ZERO_CANONICALIZATION
    result = connected_front_step_v2(flowlines, coefficients, dt_seconds, closure_limit_m3)
    _MAX_GAPS = max(_MAX_GAPS, int(result["internal_active_gap_count"]))
    _MAX_LIMITED = max(_MAX_LIMITED, int(result["limited_edge_count"]))
    _MAX_PROJECTION_CONSERVATION = max(
        _MAX_PROJECTION_CONSERVATION, abs(float(result["projection_global_conservation_m3"]))
    )
    _MAX_ZERO_CANONICALIZATION = max(
        _MAX_ZERO_CANONICALIZATION,
        abs(float(result["numerical_zero_canonicalization_volume_m3"])),
    )
    return result


def main() -> None:
    implementation.one_step = tracked_step
    import sys

    try:
        implementation.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    else:
        exit_code = 0
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    if not output.is_absolute():
        output = PROJECT / output
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update({
            "solver": "network-capacity-connected-front-exact-zero-v2",
            "maximum_internal_active_gap_count": _MAX_GAPS,
            "maximum_limited_edge_count": _MAX_LIMITED,
            "maximum_absolute_projection_conservation_m3": _MAX_PROJECTION_CONSERVATION,
            "maximum_numerical_zero_canonicalization_volume_m3": _MAX_ZERO_CANONICALIZATION,
            "posthoc_section_clipping": False,
            "connected_front_local_mb_required_for_climate": True,
        })
        if summary.get("status") == "zero-mb-pass" and (
            _MAX_GAPS != 0
            or _MAX_ZERO_CANONICALIZATION != 0.0
            or _MAX_PROJECTION_CONSERVATION > float(summary["closure_limit_m3"])
        ):
            summary["status"] = "zero-mb-fail-mechanism-closed"
            summary.setdefault("checks", {})["tracked_intermediate_state"] = False
            exit_code = 1
        else:
            summary.setdefault("checks", {})["tracked_intermediate_state"] = True
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

