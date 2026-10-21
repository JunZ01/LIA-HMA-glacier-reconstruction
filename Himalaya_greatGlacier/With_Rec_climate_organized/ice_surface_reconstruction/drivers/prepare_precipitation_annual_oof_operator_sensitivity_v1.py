"""Prepare annual-scale OOF proxy groups for a frozen operator sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_operator_sensitivity_v1"


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    project = root()
    result_root = project / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "nested_scores": result_root / "climate_reconstruction_precipitation_nested_seasonal_tree_psm_cv_v1/nested_seasonal_proxy_scores.csv",
        "replay": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "baseline": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/attribution_output.mat",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "matlab_driver": Path(__file__).with_name("run_precipitation_annual_oof_operator_sensitivity_v1.m"),
        "analysis_driver": Path(__file__).with_name("analyze_precipitation_annual_oof_operator_sensitivity_v1.py"),
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: digest(path) for name, path in sources.items()}
    scores = pd.read_csv(sources["nested_scores"]).sort_values("proxy_index")
    if len(scores) != 201 or not np.array_equal(scores["proxy_index"].to_numpy(int), np.arange(1, 202)):
        raise RuntimeError("tree-index-contract-failed")
    admissible_mask = (
        (scores["nested_nrmse_bea_annual"] < 1)
        & (scores["nested_nrmse_cru_annual"] < 1)
        & (scores["nested_nrmse_bea_to_cru_annual"] < 1)
        & (scores["nested_nrmse_cru_to_bea_annual"] < 1)
        & (scores["nested_correlation_bea_annual"] > 0)
        & (scores["nested_correlation_cru_annual"] > 0)
    )
    admissible = scores.loc[admissible_mask, "proxy_index"].to_numpy(int)
    failing = scores.loc[~admissible_mask, "proxy_index"].to_numpy(int)
    if len(admissible) != 15 or len(failing) != 186:
        raise RuntimeError(f"annual-group-count-conflict:{len(admissible)}:{len(failing)}")
    output.mkdir(parents=True)
    scores.assign(annual_oof_admissible=admissible_mask).to_csv(output / "annual_oof_proxy_classification.csv", index=False)
    group_path = output / "annual_operator_groups.mat"
    scipy.io.savemat(group_path, {
        "tree_indices": np.arange(1, 202, dtype=float).reshape(-1, 1),
        "annual_admissible_indices": admissible.astype(float).reshape(-1, 1),
        "annual_failing_indices": failing.astype(float).reshape(-1, 1),
        "glacier_ice_index": np.array([[202.0]]),
    })
    after = {name: digest(path) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared",
        "locked_1921_1939_accessed": False,
        "rule": "Annual PSM NRMSE<1 on BEA+GPCC, CRU, BEA-to-CRU, and CRU-to-BEA nested predictions, with positive BEA and CRU self correlations.",
        "tree_count": 201,
        "annual_oof_admissible_count": int(len(admissible)),
        "annual_oof_failing_count": int(len(failing)),
        "annual_oof_admissible_indices": admissible.tolist(),
        "source_hashes_unchanged": before == after,
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Annual OOF Operator Sensitivity\n\nFreeze prior members, observations, RegEM errors, operator means, and the glacier-ice operator. Retain annual operator anomalies for either the 15 annual four-path OOF-admissible tree proxies or the complementary 186 records. Do not access 1921-1939, modify sources, or run OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
