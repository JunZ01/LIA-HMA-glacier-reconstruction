"""Finite steady-state delta-T bracket for system 3660 root-coarsen2x.

Run the single authorised cold endpoint (delta-T=-0.25 K).  Only when its
50-year volume-change sign opposes the already completed +0.25 K endpoint and
both endpoints are numerically/domain clean, run the single authorised
midpoint (delta-T=0 K).  All other registered inputs and gates remain frozen.
No historical continuation is performed here.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

from oggm import utils

import synthetic_lia_conservative_steady_50y_registered_floor_v1 as steady


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").is_file():
            return parent
    raise RuntimeError("project root containing LIA_PROJECT_FACTS.md not found")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _clean_for_bracket(summary: dict) -> bool:
    flags = set(summary.get("review_flags", []))
    final = summary.get("final", {})
    return (
        summary.get("runtime_error") is None
        and flags.issubset({"50y-volume-change-over-2pct"})
        and int(final.get("year", -1)) == 50
        and bool(final.get("finite_geometry", False))
        and int(final.get("negative_section_count", -1)) == 0
        and float(final.get("remaining_free_downstream_m", -1.0)) >= 5000.0
        and float(final.get("length_relative_error_lia", 1.0)) <= 0.10
    )


def main() -> None:
    project = _project_root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    inversion = project / "w382inv3660v1"
    preflight = base / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v5"
    zero = base / "lia_191_zero_mb_3660_root_coarsen2x_mindt30_v2"
    warm_output = base / "lia_191_zero_mb_3660_root_coarsen2x_steady50_v5"
    adapter = base / "lia_191_zero_mb_3660_root_coarsen2x_zero_formal_adapter_bracket_v1"
    decision_root = base / "lia_191_zero_mb_3660_root_coarsen2x_steady_bracket_v1"
    candidate = preflight / "candidate_model_flowlines.pkl"
    zero_summary_path = zero / "summary.json"
    warm_summary_path = warm_output / "steady_50y_summary.json"

    zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    warm_summary = json.loads(warm_summary_path.read_text(encoding="utf-8"))
    if zero_summary.get("status") != "zero-mb-pass":
        raise RuntimeError("formal zero-MB prerequisite is not pass")
    if float(zero_summary.get("minimum_timestep_seconds", -1.0)) != 30.0:
        raise RuntimeError("zero-MB timestep floor is not 30 s")
    expected_sha = "eaa00d2088bd70a7409fd13922861aedead66e5d45808d95a46cd2a770fab189"
    if _sha256(candidate) != expected_sha:
        raise RuntimeError("v5 candidate geometry hash conflict")

    adapter.mkdir(parents=True, exist_ok=False)
    decision_root.mkdir(parents=True, exist_ok=False)
    formal_zero = copy.deepcopy(zero_summary)
    formal_zero.update(
        status="zero-mb-pass",
        review_flags=[],
        minimum_timestep_seconds=30.0,
        source_summary=str(zero_summary_path),
        candidate_model_flowlines_sha256=expected_sha,
    )
    _write_json(adapter / "zero_mb_summary_formal.json", formal_zero)

    base_stage2 = {
        "melt_f": 3.3728089223844293,
        "prcp_fac": 3.9019813553910025,
        "glen_a": 1.4759277199273846e-23,
        "fs": 0.0,
        "temp_melt_c": -1.0,
        "base_temp_bias_c": -4.121092849873314,
        "precipitation_anomaly": "off",
    }

    def run_point(delta_t_c: float, label: str) -> dict:
        output = base / f"lia_191_zero_mb_3660_root_coarsen2x_steady50_{label}_v1"
        stage2 = dict(base_stage2)
        stage2["delta_t_c"] = float(delta_t_c)
        stage2["total_temp_bias_c"] = float(base_stage2["base_temp_bias_c"] + delta_t_c)
        config = {
            "experiment": f"lia-191-3660-root-coarsen2x-steady50-{label}-v1",
            "system_id": 3660,
            "synthetic_rgi_id": "RGI60-14.93660",
            "inversion_root": {"anchor": "project", "sub": str(inversion.relative_to(project))},
            "zero_mb_root": {"anchor": "project", "sub": str(adapter.relative_to(project))},
            "output_root": {"anchor": "project", "sub": str(output.relative_to(project))},
            "climate_filesuffix": "_batch1_1441_1470_prcp_off",
            "spinup_window": [1441, 1470],
            "minimum_timestep_seconds": 30.0,
            "registered_lia_length_m": 27508.72902948928,
            "stage2": stage2,
            "gates": {
                "volume_change_fraction_max": 0.02,
                "lia_length_relative_error_max": 0.10,
                "minimum_free_downstream_m": 5000.0,
            },
        }
        config_path = adapter / f"steady50_{label}_config.json"
        _write_json(config_path, config)
        return steady.run(config_path)

    with gzip.open(candidate, "rb") as stream:
        candidate_flowlines = pickle.load(stream)
    original_read_pickle = utils.GlacierDirectory.read_pickle

    def read_pickle_with_candidate(self, filename, *args, **kwargs):
        if filename == "model_flowlines":
            return copy.deepcopy(candidate_flowlines)
        return original_read_pickle(self, filename, *args, **kwargs)

    utils.GlacierDirectory.read_pickle = read_pickle_with_candidate
    cold_summary = midpoint_summary = None
    try:
        cold_summary = run_point(-0.25, "dt_m025")
        warm_clean = _clean_for_bracket(warm_summary)
        cold_clean = _clean_for_bracket(cold_summary)
        warm_dv = float(warm_summary["volume_change_fraction_50y"])
        cold_dv = float(cold_summary["volume_change_fraction_50y"])
        opposite_sign = warm_dv * cold_dv < 0.0
        if warm_clean and cold_clean and opposite_sign:
            midpoint_summary = run_point(0.0, "dt_0")
    finally:
        utils.GlacierDirectory.read_pickle = original_read_pickle

    warm_clean = _clean_for_bracket(warm_summary)
    cold_clean = _clean_for_bracket(cold_summary)
    warm_dv = float(warm_summary["volume_change_fraction_50y"])
    cold_dv = float(cold_summary["volume_change_fraction_50y"])
    opposite_sign = warm_dv * cold_dv < 0.0
    if not warm_clean or not cold_clean:
        decision = "close-numerical-or-domain-gate"
    elif not opposite_sign:
        decision = "close-no-volume-change-sign-bracket"
    elif midpoint_summary is None:
        decision = "internal-error-midpoint-not-run"
    elif midpoint_summary.get("status") == "steady-50y-pass":
        decision = "midpoint-steady-pass-historical-not-run"
    else:
        decision = "midpoint-complete-steady-not-yet-pass"
    result = {
        "status": decision,
        "system_id": 3660,
        "candidate_model_flowlines_sha256": expected_sha,
        "frozen_inputs": {
            "window": [1441, 1470],
            "minimum_timestep_seconds": 30.0,
            "base_temp_bias_c": base_stage2["base_temp_bias_c"],
            "melt_f": base_stage2["melt_f"],
            "prcp_fac": base_stage2["prcp_fac"],
            "glen_a": base_stage2["glen_a"],
            "fs": base_stage2["fs"],
            "temp_melt_c": base_stage2["temp_melt_c"],
        },
        "warm_delta_t_c": 0.25,
        "warm_clean": warm_clean,
        "warm_volume_change_fraction_50y": warm_dv,
        "cold_delta_t_c": -0.25,
        "cold_clean": cold_clean,
        "cold_volume_change_fraction_50y": cold_dv,
        "opposite_sign": opposite_sign,
        "midpoint_ran": midpoint_summary is not None,
        "midpoint_summary": midpoint_summary,
        "historical_run": False,
    }
    _write_json(decision_root / "steady_bracket_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
