"""Audit ModE-RA HMA proxy-source overlap at frozen LIA window midpoints."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANCHOR_YEARS = tuple(range(1475, 1900, 50))
SEASONS = ("00", "05")
BBOX = (55.0, 116.0, 18.0, 51.0)
BASE_URL = (
    "https://mode-ra.unibe.ch/data/ModE-RA/outdata/"
    "lowres_20mem_Set_1420-3_1850-1/observation_feedback_archive/"
    "inout_form_cyc_1"
)
USECOLS = (
    "PRAOBSID",
    "PRARECID",
    "NAME",
    "LON",
    "LAT",
    "YEAR_ASSIM",
    "YEAR_OBS",
    "MONTH",
    "SEASON",
    "VARIABLE",
    "TYPE",
    "CYCLE",
    "ACTIVE",
    "REF_DATA",
    "ASSIM",
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def identity_token(name: str) -> str:
    value = str(name)
    if value.startswith("P2k_"):
        value = value[4:]
    elif value.startswith("ISO2k_"):
        value = value[6:]
    value = re.sub(r"_(boreal)?summer$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.(rwl|txt|csv)$", "", value, flags=re.IGNORECASE)
    return normalize(value)


def source_family(name: str) -> str:
    value = str(name)
    if value.startswith("P2k_"):
        return "PAGES2k_named"
    if value.startswith("ISO2k_"):
        return "Iso2k_named"
    if value.startswith("B14_"):
        return "B14_tree_network"
    if value.startswith("NTREND_"):
        return "NTREND"
    if value.lower().startswith("schweingruber_"):
        return "Schweingruber_gridded"
    return "other"


def haversine_km(lat: float, lon: float, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(latitudes)
    dlat = lat2 - lat1
    dlon = np.deg2rad(longitudes - lon)
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_73_mode_ra_hma_source_overlap_v1"
    round72 = loop / "round_72_mode_ra_five_family_temperature_reference_v1"
    paths = {
        "round72_summary": round72 / "summary.json",
        "round11_inventory": loop
        / "round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round72_summary"].read_text(encoding="utf-8"))[
        "passes_five_family_promotion"
    ]:
        raise RuntimeError("round72-five-family-candidate-did-not-pass")

    inventory = pd.read_csv(paths["round11_inventory"])
    inventory = inventory[np.isfinite(inventory.latitude) & np.isfinite(inventory.longitude)].copy()
    inventory_tokens = inventory.formal_identity.fillna("").map(normalize).to_numpy(str)
    inventory_lat = inventory.latitude.to_numpy(float)
    inventory_lon = inventory.longitude.to_numpy(float)
    session = requests.Session()
    session.headers.update({"User-Agent": "LIA-HMA-ModE-RA-source-overlap-audit/1.0"})
    manifests: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    try:
        for year in ANCHOR_YEARS:
            if 1921 <= year <= 1939:
                raise RuntimeError(f"locked-anchor-year-blocked:{year}")
            for season in SEASONS:
                url = f"{BASE_URL}/inoutform_{year}_{season}.tsv"
                response = session.get(url, timeout=240)
                response.raise_for_status()
                payload = response.content
                declared = int(response.headers.get("Content-Length", len(payload)))
                if len(payload) != declared:
                    raise RuntimeError(f"feedback-size-mismatch:{year}:{season}:{len(payload)}:{declared}")
                digest = hashlib.sha256(payload).hexdigest()
                frame = pd.read_csv(io.BytesIO(payload), sep="\t", usecols=list(USECOLS), low_memory=False)
                west, east, south, north = BBOX
                hma = frame[
                    (frame.LON > west)
                    & (frame.LON < east)
                    & (frame.LAT > south)
                    & (frame.LAT < north)
                ].copy()
                hma["anchor_year"] = year
                hma["anchor_season"] = season
                frames.append(hma)
                manifests.append(
                    {
                        "year": year,
                        "season": season,
                        "url": url,
                        "http_status": response.status_code,
                        "size_bytes": len(payload),
                        "sha256": digest,
                        "hma_row_count": len(hma),
                        "locked_period_overlap": False,
                    }
                )
    finally:
        session.close()
    snapshots = pd.concat(frames, ignore_index=True)
    proxy_like = snapshots[
        snapshots.TYPE.fillna("").str.contains("proxy", case=False)
        | snapshots.TYPE.fillna("").str.contains("docu", case=False)
    ].copy()
    if proxy_like.empty:
        raise RuntimeError("no-hma-proxy-or-documentary-sources-in-anchor-snapshots")

    unique_rows: list[dict[str, Any]] = []
    for record_id, group in proxy_like.groupby("PRARECID", dropna=False):
        first = group.iloc[0]
        name = str(first.NAME)
        token = identity_token(name)
        distances = haversine_km(
            float(first.LAT), float(first.LON), inventory_lat, inventory_lon
        )
        nearest_position = int(np.argmin(distances))
        nearest = inventory.iloc[nearest_position]
        token_positions = [
            index
            for index, candidate in enumerate(inventory_tokens)
            if len(token) >= 6 and (token in candidate or candidate in token)
        ]
        family = source_family(name)
        named_shared_pool = family in ("PAGES2k_named", "Iso2k_named")
        site_match = float(distances[nearest_position]) <= 10.0
        token_match = len(token_positions) > 0
        likely_overlap = named_shared_pool or site_match or token_match
        unique_rows.append(
            {
                "PRARECID": record_id,
                "NAME": name,
                "longitude": float(first.LON),
                "latitude": float(first.LAT),
                "TYPE": first.TYPE,
                "VARIABLE": first.VARIABLE,
                "source_family": family,
                "snapshot_occurrence_count": int(len(group)),
                "first_anchor_year": int(group.anchor_year.min()),
                "last_anchor_year": int(group.anchor_year.max()),
                "named_shared_pool": named_shared_pool,
                "identity_token": token,
                "identity_token_match": token_match,
                "identity_token_matches": " || ".join(
                    inventory.iloc[token_positions].formal_identity.astype(str).tolist()
                ),
                "nearest_existing_identity": str(nearest.formal_identity),
                "nearest_existing_source_database": str(nearest.source_database),
                "nearest_existing_distance_km": float(distances[nearest_position]),
                "existing_site_within_10km": site_match,
                "likely_existing_evidence_overlap": likely_overlap,
            }
        )
    unique = pd.DataFrame(unique_rows).sort_values(
        ["likely_existing_evidence_overlap", "source_family", "PRARECID"],
        ascending=[False, True, True],
    )
    snapshot_rows = (
        proxy_like.groupby(["anchor_year", "anchor_season", "TYPE"], dropna=False)
        .agg(row_count=("PRAOBSID", "size"), unique_record_count=("PRARECID", "nunique"))
        .reset_index()
    )
    pd.DataFrame(manifests).to_csv(output / "feedback_anchor_source_manifest.csv", index=False)
    proxy_like[list(USECOLS) + ["anchor_year", "anchor_season"]].to_csv(
        output / "hma_proxy_documentary_snapshot_rows.csv", index=False
    )
    unique.to_csv(output / "hma_unique_source_overlap.csv", index=False)
    snapshot_rows.to_csv(output / "hma_anchor_source_counts.csv", index=False)

    unique_count = len(unique)
    named_count = int(unique.named_shared_pool.sum())
    token_count = int(unique.identity_token_match.sum())
    site_count = int(unique.existing_site_within_10km.sum())
    overlap_count = int(unique.likely_existing_evidence_overlap.sum())
    overlap_fraction = overlap_count / unique_count
    named_fraction = named_count / unique_count
    site_fraction = site_count / unique_count
    independence_gates = {
        "named_pages2k_or_iso2k_fraction_below_10pct": named_fraction < 0.10,
        "existing_site_within_10km_fraction_below_25pct": site_fraction < 0.25,
        "combined_likely_overlap_fraction_below_25pct": overlap_fraction < 0.25,
        "no_exact_normalized_identity_token_match": token_count == 0,
    }
    independent = bool(all(independence_gates.values()))
    summary = {
        "status": "complete",
        "audit_design": "cycle1 two-season midpoint snapshots for nine complete LIA windows",
        "anchor_years": list(ANCHOR_YEARS),
        "feedback_file_count": len(manifests),
        "feedback_bytes_read": int(sum(item["size_bytes"] for item in manifests)),
        "locked_1921_1939_accessed": False,
        "hma_snapshot_proxy_documentary_row_count": len(proxy_like),
        "hma_unique_proxy_documentary_record_count": unique_count,
        "named_pages2k_or_iso2k_record_count": named_count,
        "named_pages2k_or_iso2k_fraction": named_fraction,
        "exact_normalized_identity_token_match_count": token_count,
        "existing_site_within_10km_count": site_count,
        "existing_site_within_10km_fraction": site_fraction,
        "combined_likely_existing_evidence_overlap_count": overlap_count,
        "combined_likely_existing_evidence_overlap_fraction": overlap_fraction,
        "independence_gates": independence_gates,
        "eligible_as_proxy_evidence_independent_family": independent,
        "decision": (
            "mode_ra_may_be_treated_as_proxy_evidence_independent"
            if independent
            else "label_mode_ra_as_method_independent_but_proxy_evidence_overlapping"
        ),
        "five_family_modern_skill_decision_unchanged": True,
        "five_family_spread_must_not_be_interpreted_as_five_independent_proxy_evidence_sources": not independent,
        "audit_is_presence_proof_not_exhaustive_full_period_inventory": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    colors = np.where(unique.likely_existing_evidence_overlap, "#D55E00", "#0072B2")
    axes[0].scatter(
        unique.longitude,
        unique.latitude,
        c=colors,
        s=25 + 7 * unique.snapshot_occurrence_count,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
    )
    axes[0].set(
        title="ModE-RA HMA proxy/documentary anchor sources",
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=(BBOX[0], BBOX[1]),
        ylim=(BBOX[2], BBOX[3]),
    )
    family_counts = unique.source_family.value_counts().sort_values()
    axes[1].barh(family_counts.index, family_counts.values, color="#009E73")
    axes[1].set(title="Unique source identities by family", xlabel="Record count")
    figure.savefig(output / "mode_ra_hma_source_overlap.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-RA HMA Source Overlap",
                "",
                f"- Anchor files/bytes: `{len(manifests)}` / `{sum(item['size_bytes'] for item in manifests)}`.",
                f"- Unique HMA proxy/documentary records: `{unique_count}`.",
                f"- Named PAGES2k/Iso2k: `{named_count}` (`{named_fraction:.1%}`).",
                f"- Existing site within 10 km: `{site_count}` (`{site_fraction:.1%}`).",
                f"- Combined likely evidence overlap: `{overlap_count}` (`{overlap_fraction:.1%}`).",
                f"- Eligible as proxy-evidence-independent family: `{str(independent).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "remote_feedback_files": manifests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("local-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
