"""Freeze the formal split-forward-delta confirmation pass for system 2827."""

from __future__ import annotations

import json

import pandas as pd

from . import freeze_lia_191_split_forward_delta_passes_v1 as freeze
from .build_lia_191_iterative_loop import project_root


def main() -> None:
    root = project_root()
    experiment = "lia_191_split_forward_delta_confirmation_v1"
    freeze.EXPERIMENT = experiment
    experiment_root = root / "validation_data/results/paleo_surface_validation_v0" / experiment
    scores = pd.read_csv(experiment_root / "scores.csv")
    registry_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1"
    new_row = freeze._freeze_one(
        root,
        registry_root,
        scores,
        2827,
        {"forward_delta_t_c": 1.370, "run_tag": "forward_dt_p1p370"},
    )

    registry_path = registry_root / "frozen_candidates.csv"
    registry = pd.read_csv(registry_path)
    registry = pd.concat([registry, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    registry["lia_unique_id"] = pd.to_numeric(registry["lia_unique_id"]).astype(int)
    registry = registry.sort_values("lia_unique_id").drop_duplicates("lia_unique_id", keep="last")
    expected = {
        991, 1077, 1546, 1616, 2266, 2337, 2569, 2622, 2815, 2827,
        2902, 2928, 2939, 4319,
    }
    if set(registry["lia_unique_id"]) != expected:
        raise ValueError(f"frozen-registry-membership-changed:{set(registry['lia_unique_id'])}")
    registry.to_csv(registry_path, index=False)
    summary = {
        "schema_version": 1,
        "scope": (
            "reviewed or auto-approved provisional curves passing both the 10% "
            "total-length and 20% separation endpoint gates"
        ),
        "candidate_count": len(expected),
        "systems": sorted(expected),
        "launched_oggm": False,
    }
    (registry_root / "registry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    mismatches = []
    for system_id in sorted(expected):
        system_root = registry_root / f"system_{system_id}"
        hashes = json.loads((system_root / "file_hashes.json").read_text(encoding="utf-8"))
        for name, metadata in hashes.items():
            path = system_root / name
            if (
                not path.exists()
                or path.stat().st_size != int(metadata["bytes"])
                or freeze._sha256(path) != str(metadata["sha256"])
            ):
                mismatches.append(f"{system_id}:{name}")
    if mismatches:
        raise ValueError(f"registry-hash-validation-failed:{mismatches}")
    print(json.dumps({**summary, "hash_mismatches": 0}, indent=2))


if __name__ == "__main__":
    main()
