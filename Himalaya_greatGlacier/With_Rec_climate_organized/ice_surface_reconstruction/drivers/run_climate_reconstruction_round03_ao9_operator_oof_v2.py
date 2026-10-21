"""Round 03 v2 runner: omit fold-specific PSM fits that fail the training gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


BASE_PATH = Path(__file__).with_name("run_climate_reconstruction_round03_ao9_operator_oof_v1.py")
SPEC = importlib.util.spec_from_file_location("round03_oof_v1", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("round03-v1-load-failed")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def run_branch(prior, sample, observations, hxb, residual_variance, branch_by_year):
    background = prior[:, sample]
    analysis = np.full((prior.shape[0], len(base.YEARS)), np.nan)
    selected = np.full((len(base.YEARS), 100), -1, dtype=int)
    maximum_closure = 0.0
    for year_index, branch in enumerate(branch_by_year):
        operator_valid = (
            np.isfinite(residual_variance[:, branch])
            & (residual_variance[:, branch] > 0)
            & np.all(np.isfinite(hxb[:, branch][:, sample]), axis=1)
        )
        active = np.flatnonzero(np.isfinite(observations[:, year_index]) & operator_valid)
        if len(active) == 0:
            analysis[:, year_index] = np.mean(background, axis=1)
            continue
        branch_hxb = hxb[active, branch][:, sample]
        local = base.select_analogues(observations[active, year_index], branch_hxb)
        selected[year_index] = local
        result, closure = base.analysis_mean(
            background[:, local],
            branch_hxb[:, local],
            observations[active, year_index],
            residual_variance[active, branch],
        )
        analysis[:, year_index] = result
        maximum_closure = max(maximum_closure, closure)
    return analysis, selected, maximum_closure


def main() -> int:
    base.run_branch = run_branch
    result = base.main()
    output = base.root() / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    manifest_path = output / "operator_oof_source_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runner_v2_sha256"] = base.digest(Path(__file__).resolve())
    manifest["runner_v2_change"] = "Fold-specific proxies without a valid training fit are excluded from that fold."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
