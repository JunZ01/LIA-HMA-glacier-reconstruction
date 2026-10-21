"""Prepare frozen proxy groups for the OOF-admissibility operator sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_precipitation_oof_admissibility_operator_sensitivity_v1"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "admissibility": result_root / "climate_reconstruction_precipitation_oof_psm_admissibility_v2/proxy_oof_admissibility_and_hma_exp1_contribution.csv",
        "admissibility_summary": result_root / "climate_reconstruction_precipitation_oof_psm_admissibility_v2/summary.json",
        "replay_input": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "baseline": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/attribution_output.mat",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "matlab_driver": Path(__file__).with_name("run_precipitation_oof_admissibility_operator_sensitivity_v1.m"),
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: sha256(path) for name, path in sources.items()}
    table = pd.read_csv(sources["admissibility"]).sort_values("proxy_index")
    if len(table) != 201 or not np.array_equal(table["proxy_index"].to_numpy(int), np.arange(1, 202)):
        raise RuntimeError("tree-index-contract-failed")
    admissible = table.loc[table["selected_skill_all_four_paths"], "proxy_index"].to_numpy(int)
    not_admissible = table.loc[~table["selected_skill_all_four_paths"], "proxy_index"].to_numpy(int)
    significant = table.loc[table["archived_significant_p_le_0_05"], "proxy_index"].to_numpy(int)
    if not (len(admissible) == 27 and len(not_admissible) == 174 and len(significant) == 100):
        raise RuntimeError(f"group-count-contract-failed:{len(admissible)}:{len(not_admissible)}:{len(significant)}")
    if len(np.intersect1d(admissible, not_admissible)) or not np.array_equal(np.sort(np.concatenate([admissible, not_admissible])), np.arange(1, 202)):
        raise RuntimeError("admissibility-partition-failed")
    output.mkdir(parents=True)
    group_path = output / "operator_groups.mat"
    scipy.io.savemat(group_path, {
        "tree_indices": np.arange(1, 202, dtype=float).reshape(-1, 1),
        "admissible_indices": admissible.astype(float).reshape(-1, 1),
        "not_admissible_indices": not_admissible.astype(float).reshape(-1, 1),
        "archived_significant_indices": significant.astype(float).reshape(-1, 1),
        "glacier_ice_index": np.array([[202.0]]),
    }, do_compression=False)
    after = {name: sha256(path) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared",
        "locked_1921_1939_accessed": False,
        "tree_count": 201,
        "oof_admissible_count": int(len(admissible)),
        "oof_not_admissible_count": int(len(not_admissible)),
        "archived_significant_count": int(len(significant)),
        "glacier_ice_operator_retained_in_all_branches": True,
        "source_hashes_unchanged": before == after,
        "group_input_sha256": sha256(group_path),
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# OOF-Admissibility Operator Sensitivity Contract\n\n"
        "Freeze prior members, observations, diagonal RegEM errors, proxy-operator means, glacier-ice operator, and all group definitions. Compare baseline, OOF-admissible-only, OOF-not-admissible-only, archived-significant-only, and zero-tree-anomaly branches. This is a mechanism sensitivity, not record deletion or a production rerun. Do not access 1921-1939 or run OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
