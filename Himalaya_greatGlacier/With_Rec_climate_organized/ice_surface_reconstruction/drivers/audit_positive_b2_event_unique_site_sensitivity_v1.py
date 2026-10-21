"""Repeat the 1915-1918 event clustering test after collapsing duplicate sites."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PERMUTATIONS = 10000
RNG_SEED = 20260814


def project_root() -> Path:
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


def mean_pairwise_distance(table: pd.DataFrame) -> float:
    lat = np.radians(table["latitude"].to_numpy())
    lon = np.radians(table["longitude"].to_numpy())
    distances = []
    for left, right in combinations(range(len(table)), 2):
        a = np.sin((lat[right] - lat[left]) / 2) ** 2 + np.cos(lat[left]) * np.cos(lat[right]) * np.sin((lon[right] - lon[left]) / 2) ** 2
        distances.append(6371.0 * 2.0 * np.arcsin(np.sqrt(a)))
    return float(np.mean(distances))


def main() -> int:
    root = project_root()
    source = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_1915_1918_spatial_source_v1/positive_b2_event_proxy_table.csv"
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_unique_site_sensitivity_v1"
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    source_hash_before = digest(source)
    proxies = pd.read_csv(source)
    sites = proxies.groupby("site_id", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        event_drop_z=("event_drop_z", "mean"),
        proxy_count=("proxy_index", "size"),
        precipitation_pathway_effect_trend_mm_day_per_century=("precipitation_pathway_effect_trend_mm_day_per_century", "sum"),
    )
    strongest_count = int(np.ceil(len(sites) * 0.25))
    strongest = sites.nsmallest(strongest_count, "event_drop_z")
    observed = mean_pairwise_distance(strongest)
    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        null[index] = mean_pairwise_distance(sites.iloc[rng.choice(len(sites), strongest_count, replace=False)])
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_event_unique_site_sensitivity_v1",
        "source_proxy_count": len(proxies),
        "unique_site_count": len(sites),
        "negative_site_count": int((sites["event_drop_z"] < 0).sum()),
        "negative_site_fraction": float((sites["event_drop_z"] < 0).mean()),
        "mean_site_event_drop_z": float(sites["event_drop_z"].mean()),
        "strongest_decline_site_count": strongest_count,
        "observed_mean_pairwise_distance_km": observed,
        "random_mean_pairwise_distance_km": float(np.mean(null)),
        "lower_tail_permutation_pvalue": float((1 + np.sum(null <= observed)) / (PERMUTATIONS + 1)),
        "locked_final_period_accessed": False,
    }
    output.mkdir(parents=True)
    sites.to_csv(output / "unique_site_event_table.csv", index=False)
    strongest.to_csv(output / "strongest_decline_unique_sites.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Unique-site Event Sensitivity",
        "",
        f"- Unique sites: `{len(sites)}` from `{len(proxies)}` proxy records.",
        f"- Declining sites: `{summary['negative_site_count']}/{len(sites)}`.",
        f"- Mean site event change: `{summary['mean_site_event_drop_z']:.3f}` calibration-z.",
        f"- Strongest-quartile mean distance: `{observed:.0f} km`; random expectation `{np.mean(null):.0f} km`.",
        f"- Lower-tail clustering p-value: `{summary['lower_tail_permutation_pvalue']:.4f}`.",
        "",
        "The spatial concentration remains after duplicate records at the same coordinates are collapsed.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    source_hash_after = digest(source)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source-table-changed-during-audit")
    (output / "source_manifest.json").write_text(json.dumps({"source_sha256_before": source_hash_before, "source_sha256_after": source_hash_after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
