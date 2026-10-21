"""Prepare fixed longitude blocks for the other 33 positive-b2 proxies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_precipitation_other_positive_b2_longitude_block_ablation_v1"


def project_root() -> Path:
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


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest(path)}


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "source_blocks": result_root / "climate_reconstruction_precipitation_positive_b2_source_block_ablation_v1/source_block_groups.mat",
        "source_trends": result_root / "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_trend_table.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    other = scipy.io.loadmat(
        sources["source_blocks"], variable_names=["other_positive_b2_proxy_indices"]
    )["other_positive_b2_proxy_indices"].reshape(-1).astype(int)
    trends = pd.read_csv(sources["source_trends"]).set_index("proxy_index").loc[other].copy()
    band_definition = {
        "west_lt80": "<80E",
        "central_80_90": "80-90E",
        "east_90_100": "90-100E",
        "far_east_ge100": ">=100E",
    }
    groups = {
        name: sorted(trends.index[trends["longitude_band"].eq(label)].astype(int).tolist())
        for name, label in band_definition.items()
    }
    expected_counts = {"west_lt80": 10, "central_80_90": 6, "east_90_100": 14, "far_east_ge100": 3}
    if len(other) != 33 or {name: len(values) for name, values in groups.items()} != expected_counts:
        raise RuntimeError(f"longitude-group-count-contract-failed:{len(other)}:{groups}")
    group_sets = [set(values) for values in groups.values()]
    if any(group_sets[i] & group_sets[j] for i in range(len(group_sets)) for j in range(i + 1, len(group_sets))):
        raise RuntimeError("longitude-group-overlap")
    if set().union(*group_sets) != set(other.tolist()):
        raise RuntimeError("longitude-group-union-contract-failed")

    output.mkdir(parents=True)
    scipy.io.savemat(
        output / "longitude_block_groups.mat",
        {
            "west_lt80_proxy_indices": pd.Series(groups["west_lt80"], dtype=float).to_numpy().reshape(-1, 1),
            "central_80_90_proxy_indices": pd.Series(groups["central_80_90"], dtype=float).to_numpy().reshape(-1, 1),
            "east_90_100_proxy_indices": pd.Series(groups["east_90_100"], dtype=float).to_numpy().reshape(-1, 1),
            "far_east_ge100_proxy_indices": pd.Series(groups["far_east_ge100"], dtype=float).to_numpy().reshape(-1, 1),
            "other_positive_b2_proxy_indices": other.astype(float).reshape(-1, 1),
        },
    )
    membership = trends.reset_index().copy()
    reverse = {proxy_index: group for group, values in groups.items() for proxy_index in values}
    membership["longitude_block"] = membership["proxy_index"].map(reverse)
    membership.sort_values(["longitude", "latitude", "proxy_index"]).to_csv(
        output / "longitude_block_membership.csv", index=False
    )

    after = {name: source_state(path) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "other_positive_b2_proxy_count": int(len(other)),
        "longitude_band_definition": band_definition,
        "group_counts": {name: len(values) for name, values in groups.items()},
        "group_proxy_indices": groups,
        "groups_pairwise_disjoint": True,
        "group_union_equals_other_positive_b2": True,
        "all_declared_sources_unchanged": before == after,
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("source-changed-during-longitude-block-preparation")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
