from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
UPSTREAM = LOOP / "round_224_central_himalaya_mxd2026_public_recovery_v5"
OUT = LOOP / "round_224_central_himalaya_mxd2026_public_recovery_v6"
RAW = OUT / "official_sources_unmodified"
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
TITLE = "Central Himalayan tree-ring density network reveals warm-season temperature variability back to 1775 CE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    upstream_summary_path = UPSTREAM / "summary.json"
    upstream_raw = UPSTREAM / "official_sources_unmodified"
    source_files = sorted(path for path in upstream_raw.iterdir() if path.is_file())
    tracked = {"facts": ROOT / "LIA_PROJECT_FACTS.md", "contract": CONTRACT, "driver": DRIVER, "upstream_summary": upstream_summary_path}
    tracked.update({f"upstream_{path.name}": path for path in source_files})
    before = {name: state(path) for name, path in tracked.items()}
    upstream = json.loads(upstream_summary_path.read_text(encoding="utf-8"))
    if upstream.get("version") != 5 or upstream.get("zenodo_exact_title_hits") != 555951:
        raise RuntimeError("Unexpected V5 correction state")
    for path in source_files:
        target = RAW / path.name
        shutil.copy2(path, target)
        if sha256(path) != sha256(target):
            raise RuntimeError(f"Official response copy mismatch: {path.name}")

    zenodo = json.loads((RAW / "zenodo_search.json").read_text(encoding="utf-8"))
    rows = []
    expected = normalize(TITLE)
    for item in zenodo.get("hits", {}).get("hits", []):
        title = str(item.get("metadata", {}).get("title", ""))
        rows.append({"record_id": item.get("id"), "title": title, "normalized_exact_match": normalize(title) == expected})
    title_audit = pd.DataFrame(rows)
    title_audit.to_csv(OUT / "zenodo_title_match_audit.csv", index=False)
    exact_hits = int(title_audit["normalized_exact_match"].sum()) if len(title_audit) else 0
    relation_count = len(upstream["crossref_relation_dois"])
    datacite_hits = int(upstream["datacite_exact_title_hits"])
    embedded_count = sum(int(item["embedded_file_count"]) for item in upstream["pdf_containers"].values())
    public_numeric = bool(exact_hits or relation_count or datacite_hits or embedded_count)
    if public_numeric:
        raise RuntimeError("V6 expected no machine-readable public numeric series")

    decision = "central_hma_post1775_diagnostic_admitted_numeric_series_not_public"
    summary = dict(upstream)
    summary.update(
        {
            "version": 6,
            "decision": decision,
            "public_machine_readable_numeric_series_found": False,
            "zenodo_broad_query_total": int(zenodo.get("hits", {}).get("total", 0)),
            "zenodo_returned_record_count": len(rows),
            "zenodo_exact_title_hits": exact_hits,
            "v5_correction": "broad_free_text_total_was_not_an_exact_title_match_count",
            "source_states_unchanged": True,
        }
    )
    after = {name: state(path) for name, path in tracked.items()}
    if before != after:
        raise RuntimeError("A tracked source changed")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(UPSTREAM / "site_lineage_inventory.json", OUT / "site_lineage_inventory.json")
    shutil.copy2(UPSTREAM / "pdf_container_inventory.json", OUT / "pdf_container_inventory.json")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 224 V6 Result\n\n"
        f"`{decision}`\n\n"
        "The official article and supplement are valid PDFs with no embedded files. Crossref has no related data DOI, DataCite has no exact-title record, "
        f"and Unicode-normalized comparison finds `{exact_hits}` exact title matches among the `{len(rows)}` Zenodo records returned by the broad query. "
        "The annual chronology and reconstruction are therefore not publicly machine-readable.\n\n"
        "The new 2023 Rara and Cholangpati material is retained as a source-independent Central-Himalaya diagnostic lead, but the published reconstruction begins in 1775 and cannot pass the begin-by-1500 production gate. No figure was digitized and no annual value was opened.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
