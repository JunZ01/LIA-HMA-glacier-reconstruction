"""Run the unique registered 50-y B-window steady test for system 3660.

This is deliberately a thin adapter around the registered steady runner.  It
changes only the initial flowline pickle: the formally screened, conditioned
root-2x geometry is supplied in place of ``model_flowlines``.  Climate,
Stage-2 parameters, solver, timestep floor, and acceptance gates stay frozen.
"""

from __future__ import annotations

import copy
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


def main() -> None:
    project = _project_root()
    base = project / "validation_data" / "results" / "paleo_surface_validation_v0"
    inversion = base / "w382inv3660v1"
    preflight = base / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v5"
    zero = base / "lia_191_zero_mb_3660_root_coarsen2x_mindt30_v2"
    adapter = base / "lia_191_zero_mb_3660_root_coarsen2x_zero_formal_adapter_v1"
    output = base / "lia_191_zero_mb_3660_root_coarsen2x_steady50_v1"
    candidate = preflight / "candidate_model_flowlines.pkl"
    zero_summary_path = zero / "summary.json"

    zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    if zero_summary.get("status") != "zero-mb-pass":
        raise RuntimeError(f"formal zero-MB prerequisite failed: {zero_summary.get('status')}")
    if float(zero_summary.get("minimum_timestep_seconds", 30.0)) != 30.0:
        raise RuntimeError("zero-MB timestep floor is not the registered 30 s")
    expected_candidate_sha = "eaa00d2088bd70a7409fd13922861aedead66e5d45808d95a46cd2a770fab189"
    if _sha256(candidate) != expected_candidate_sha:
        raise RuntimeError("v5 candidate geometry hash conflict")

    adapter.mkdir(parents=True, exist_ok=True)
    formal_zero = copy.deepcopy(zero_summary)
    formal_zero["status"] = "zero-mb-pass"
    formal_zero["review_flags"] = []
    formal_zero["minimum_timestep_seconds"] = 30.0
    formal_zero["source_summary"] = str(zero_summary_path)
    formal_zero["candidate_model_flowlines_sha256"] = expected_candidate_sha
    (adapter / "zero_mb_summary_formal.json").write_text(
        json.dumps(formal_zero, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = {
        "experiment": "lia-191-3660-root-coarsen2x-registered-parent-steady50-v1",
        "system_id": 3660,
        "synthetic_rgi_id": "RGI60-14.93660",
        "inversion_root": {"anchor": "project", "sub": str(inversion.relative_to(project))},
        "zero_mb_root": {"anchor": "project", "sub": str(adapter.relative_to(project))},
        "output_root": {"anchor": "project", "sub": str(output.relative_to(project))},
        "climate_filesuffix": "_batch1_1441_1470_prcp_off",
        "spinup_window": [1441, 1470],
        "minimum_timestep_seconds": 30.0,
        "registered_lia_length_m": 27508.72902948928,
        "stage2": {
            "melt_f": 3.3728089223844293,
            "prcp_fac": 3.9019813553910025,
            "glen_a": 1.4759277199273846e-23,
            "fs": 0.0,
            "temp_melt": -1.0,
            "base_temp_bias": -4.121092849873314,
            "delta_t": 0.25,
            "total_temp_bias": -3.871092849873314,
            "precipitation_anomaly": "off",
        },
        "gates": {
            "volume_change_fraction_max": 0.02,
            "lia_length_relative_error_max": 0.10,
            "minimum_free_downstream_m": 5000.0,
        },
    }
    config_path = adapter / "steady50_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with candidate.open("rb") as stream:
        candidate_flowlines = pickle.load(stream)
    original_read_pickle = utils.GlacierDirectory.read_pickle

    def read_pickle_with_candidate(self, filename, *args, **kwargs):
        if filename == "model_flowlines":
            return copy.deepcopy(candidate_flowlines)
        return original_read_pickle(self, filename, *args, **kwargs)

    utils.GlacierDirectory.read_pickle = read_pickle_with_candidate
    try:
        steady.run(config_path)
    finally:
        utils.GlacierDirectory.read_pickle = original_read_pickle


if __name__ == "__main__":
    main()
