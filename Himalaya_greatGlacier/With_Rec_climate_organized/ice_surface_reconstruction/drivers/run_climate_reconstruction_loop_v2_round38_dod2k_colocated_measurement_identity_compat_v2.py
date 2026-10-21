"""Correct the Round 38 Tree Rings/Wood archive-label compatibility case."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def archive_group(value: object) -> str:
    token = normalize(value)
    if token == "wood" or token.startswith("tree"):
        return "wood"
    if "speleothem" in token or "cave" in token:
        return "speleothem"
    if "glacier" in token or "ice" in token:
        return "glacier_ice"
    return token


def measurement_group(value: object) -> str:
    token = normalize(value)
    if "d18o" in token or "delta18o" in token:
        return "d18o"
    if token == "trw" or "ringwidth" in token or "trsgi" in token or "breit" in token:
        return "ring_width"
    if "dust" in token:
        return "dust"
    return token


def main() -> int:
    project = root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "round_38_dod2k_colocated_measurement_identity_v1"
    output = loop / "round_38_dod2k_colocated_measurement_identity_v2"
    paths = {
        "v1_result": source / "colocated_measurement_identity.csv",
        "v1_nearby": source / "nearby_round11_measurement_matches.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    result = pd.read_csv(paths["v1_result"])
    nearby = pd.read_csv(paths["v1_nearby"])
    if len(result) != 7 or set(result[result.route == "identity_unresolved"].datasetId) != {"iso2k_1344"}:
        raise RuntimeError("v1-compatibility-contract-failed")

    for index, item in result.iterrows():
        matches = nearby[nearby.datasetId == item.datasetId]
        candidate_archive = archive_group(item.archiveType)
        candidate_measurement = measurement_group(item.paleoData_proxy)
        same_archive = matches.round11_archive.map(archive_group) == candidate_archive
        same_measurement = matches.round11_measurement.map(measurement_group) == candidate_measurement
        result.loc[index, "same_archive_same_measurement_count"] = int((same_archive & same_measurement).sum())
        result.loc[index, "same_archive_different_measurement_count"] = int((same_archive & ~same_measurement).sum())
        if item.route == "identity_unresolved" and bool(item.annual_metadata_chronology_gates) and int((same_archive & ~same_measurement).sum()) > 0:
            result.loc[index, "route"] = "colocated_distinct_annual_measurement_candidate"

    result = result.sort_values(["route", "datasetId"])
    annual = result[result.route == "colocated_distinct_annual_measurement_candidate"]
    if set(annual.datasetId) != {"iso2k_1086", "iso2k_1344", "iso2k_503", "iso2k_556"}:
        raise RuntimeError("corrected-annual-candidate-contract-failed")
    result.to_csv(output / "colocated_measurement_identity.csv", index=False)
    nearby.to_csv(output / "nearby_round11_measurement_matches.csv", index=False)
    annual.to_csv(output / "colocated_distinct_annual_candidates.csv", index=False)
    result[result.route == "source_linked_distinct_low_frequency_only"].to_csv(output / "source_linked_low_frequency_candidates.csv", index=False)

    route_counts = {str(key): int(value) for key, value in result.route.value_counts().items()}
    summary = {
        "status": "complete",
        "compatibility_correction": "Tree Rings and Wood normalized to one archive group",
        "locked_1921_1939_numeric_values_accessed": False,
        "route_counts": route_counts,
        "colocated_distinct_annual_candidate_count": int(len(annual)),
        "colocated_distinct_annual_candidate_ids": annual.datasetId.tolist(),
        "decision": "advance_colocated_distinct_annual_candidates_to_nested_validation_with_site_cluster_sensitivity",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# DoD2k Co-Located Measurement Identity v2\n\n"
        f"- Distinct co-located annual measurements: `{len(annual)}`: `{', '.join(annual.datasetId)}`.\n"
        "- v1 remains unchanged; v2 only normalizes `Tree Rings` and `Wood` archive labels.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
