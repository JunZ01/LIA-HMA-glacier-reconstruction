from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import validate_temperature_candidate_b2_allowed_input_v2 as v2


OUT = v2.v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_preflight_v3"


class DeltaAnchorAlias:
    def __init__(self, data: Any):
        self.data = data
        self.files = list(data.files)

    def __getitem__(self, key: str):
        if key.startswith("proxy") and key.endswith("_cesm_archived_operator"):
            key = key.replace("_cesm_archived_operator", "_cesm_delta_anchored_operator")
        return self.data[key]

    def close(self) -> None:
        self.data.close()


def main(input_path: Path) -> None:
    original_load = v2.np.load
    resolved_input = input_path.resolve()

    def aliased_load(path: Any, *args: Any, **kwargs: Any):
        loaded = original_load(path, *args, **kwargs)
        if Path(path).resolve() == resolved_input:
            return DeltaAnchorAlias(loaded)
        return loaded

    v2.OUT = OUT
    v2.np.load = aliased_load
    try:
        v2.main(input_path)
    finally:
        v2.np.load = original_load
    table_path = OUT / "operator_stage_b_closure.csv"
    table = pd.read_csv(table_path)
    table.loc[table.model == "cesm", "operator_source"] = (
        "archived_HXb_through_1848_then_annual_delta_anchor"
    )
    table.to_csv(table_path, index=False)
    old_prior = OUT / "candidate_b2_stage_c_prior_input_v2.npz"
    new_prior = OUT / "candidate_b2_stage_c_prior_input_v3.npz"
    if old_prior.exists():
        old_prior.replace(new_prior)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["cesm_operator_branch"] = (
        "archived HXb through 1848; annual-operator delta anchored at 1848 afterward"
    )
    summary["extension_contract"] = str(
        v2.v1.LOOP
        / "temperature_balanced_pooled_candidate_b2_cesm_operator_extension_preregistration_v1/CONTRACT.md"
    )
    summary["stage_c_prior_input"] = str(new_prior) if new_prior.exists() else None
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    main(args.input)
