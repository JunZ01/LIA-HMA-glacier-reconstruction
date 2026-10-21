"""Prepare a fixed positive-b2 proxy group for a frozen EnKF ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scipy.io
import pandas as pd


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    project = root()
    components = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1/operator_components_input.mat"
    audit = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_trend_table.csv"
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_group_ablation_delta_anchor_v1"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    before = {"components": digest(components), "audit": digest(audit)}
    selected = scipy.io.loadmat(components)["selected_proxy_indices"].reshape(-1).astype(int)
    table = pd.read_csv(audit).set_index("proxy_index")
    positive = table.loc[selected]
    positive = positive.loc[(positive["psm_significance"] == "p<=0.05") & (positive["coefficient_b2"] >= 0)].index.to_numpy(dtype=int)
    if len(selected) != 65 or len(positive) != 60:
        raise RuntimeError(f"selection-contract-failed:selected={len(selected)}:positive={len(positive)}")
    output.mkdir(parents=True)
    scipy.io.savemat(output / "positive_b2_group_input.mat", {"positive_b2_proxy_indices": positive.reshape(-1, 1).astype(float)})
    after = {"components": digest(components), "audit": digest(audit)}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-preparation")
    (output / "prepare_summary.json").write_text(json.dumps({"period": [1901, 1920], "locked_final_period_accessed": False, "selected_significant_proxy_count": 65, "positive_b2_proxy_count": 60, "positive_b2_proxy_indices": positive.tolist()}, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    print(json.dumps({"positive_b2_proxy_count": int(len(positive)), "source_hashes_unchanged": before == after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
