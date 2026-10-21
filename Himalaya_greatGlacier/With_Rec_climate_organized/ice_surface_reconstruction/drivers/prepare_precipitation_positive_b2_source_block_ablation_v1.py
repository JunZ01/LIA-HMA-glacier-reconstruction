"""Prepare mutually exclusive positive-b2 source blocks for fixed EnKF ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_precipitation_positive_b2_source_block_ablation_v1"


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
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    sources = {
        "components": result_root / "climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1/operator_components_input.mat",
        "source_trends": result_root / "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_trend_table.csv",
        "metadata": result_root / "climate_reconstruction_pages2k_fixed_network_study_batch_v1/fixed_network_study_metadata.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    selected = scipy.io.loadmat(
        sources["components"], variable_names=["selected_proxy_indices"]
    )["selected_proxy_indices"].reshape(-1).astype(int)
    trends = pd.read_csv(sources["source_trends"]).set_index("proxy_index")
    selected_trends = trends.loc[selected]
    positive = selected_trends.loc[
        selected_trends["psm_significance"].eq("p<=0.05")
        & selected_trends["coefficient_b2"].ge(0)
    ].index.to_numpy(dtype=int)

    metadata = pd.read_csv(sources["metadata"])
    pakistan = metadata.loc[
        metadata["data_author"].eq("Cook, E.R.") & metadata["latitude"].ge(35.0),
        "proxy_index",
    ].to_numpy(dtype=int)
    shao = metadata.loc[
        metadata["data_author"].eq("Shao, X.M."), "proxy_index"
    ].to_numpy(dtype=int)

    positive_set = set(positive.tolist())
    pakistan_set = set(pakistan.tolist())
    shao_set = set(shao.tolist())
    if len(selected) != 65 or len(positive_set) != 60:
        raise RuntimeError(f"positive-network-contract-failed:{len(selected)}:{len(positive_set)}")
    if len(pakistan_set) != 14 or len(shao_set) != 13:
        raise RuntimeError(f"named-group-count-contract-failed:{len(pakistan_set)}:{len(shao_set)}")
    if pakistan_set & shao_set:
        raise RuntimeError("pakistan-shao-overlap")
    if not pakistan_set <= positive_set or not shao_set <= positive_set:
        raise RuntimeError("named-group-outside-positive-b2-network")
    other_set = positive_set - pakistan_set - shao_set
    if len(other_set) != 33 or (pakistan_set | shao_set | other_set) != positive_set:
        raise RuntimeError(f"group-union-contract-failed:{len(other_set)}")

    groups = {
        "pakistan": sorted(pakistan_set),
        "shao": sorted(shao_set),
        "other_positive_b2": sorted(other_set),
        "all_positive_b2": sorted(positive_set),
    }
    output.mkdir(parents=True)
    scipy.io.savemat(
        output / "source_block_groups.mat",
        {
            "pakistan_proxy_indices": pd.Series(groups["pakistan"], dtype=float).to_numpy().reshape(-1, 1),
            "shao_proxy_indices": pd.Series(groups["shao"], dtype=float).to_numpy().reshape(-1, 1),
            "other_positive_b2_proxy_indices": pd.Series(groups["other_positive_b2"], dtype=float).to_numpy().reshape(-1, 1),
            "positive_b2_proxy_indices": pd.Series(groups["all_positive_b2"], dtype=float).to_numpy().reshape(-1, 1),
        },
    )
    membership = pd.concat(
        [pd.DataFrame({"proxy_index": values, "source_block": name}) for name, values in groups.items() if name != "all_positive_b2"],
        ignore_index=True,
    ).sort_values("proxy_index")
    membership.merge(metadata, on="proxy_index", how="left", validate="one_to_one").to_csv(
        output / "source_block_membership.csv", index=False
    )

    after = {name: source_state(path) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "selected_significant_proxy_count": int(len(selected)),
        "positive_b2_proxy_count": int(len(positive_set)),
        "group_counts": {name: len(values) for name, values in groups.items()},
        "group_proxy_indices": groups,
        "groups_pairwise_disjoint": True,
        "subgroup_union_equals_positive_b2": True,
        "all_declared_sources_unchanged": before == after,
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "prepare_source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("source-changed-during-source-block-preparation")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
