from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
PANEL = LOOP / "imd_ghcn_overlap_pages_v2/station_year_page_panel.csv"
METADATA = LOOP / "temperature_historical_imd_direct_station_table_recovery_v1/official_fulltext_search_json"
OUT = LOOP / "imd_ia_annual_temperature_row_query_v1"
DRIVER = Path(__file__).resolve()
QUERIES = ("year", "annual", "average", "december")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str, attempts: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    raw = OUT / "raw_api"
    raw.mkdir()
    with PANEL.open("r", encoding="utf-8-sig", newline="") as stream:
        panel = list(csv.DictReader(stream))
    targets: dict[str, set[int]] = {}
    for row in panel:
        targets.setdefault(row["item_id"], set()).add(int(row["leaf"]))

    fragments = []
    sources = []
    for item_id, target_leaves in sorted(targets.items()):
        base = item_id.rsplit(".", 1)[-1]
        meta_path = METADATA / f"{base}_metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        server = str(metadata.get("server") or metadata.get("d2") or metadata["d1"])
        item_path = str(metadata["dir"])
        for query in QUERIES:
            params = urllib.parse.urlencode({
                "item_id": item_id, "doc": base, "path": item_path, "q": query,
                "pre_tag": "{{{", "post_tag": "}}}",
            })
            url = f"https://{server}/fulltext/inside.php?{params}"
            payload = request(url)
            path = raw / f"{base}_{query}.json"
            path.write_bytes(payload)
            sources.append({"item_id": item_id, "query": query, "url": url,
                            "path": str(path), "bytes": len(payload), "sha256": sha256(path)})
            parsed = json.loads(payload.decode("utf-8", errors="replace"))
            for match_index, match in enumerate(parsed.get("matches", []), start=1):
                pages = sorted({int(p["page"]) for p in (match.get("par") or []) if "page" in p})
                for target_leaf in sorted(target_leaves):
                    distances = [abs(page - target_leaf) for page in pages]
                    distance = min(distances) if distances else 9999
                    if distance <= 1:
                        fragments.append({
                            "item_id": item_id, "query": query, "target_leaf": target_leaf,
                            "match_index": match_index, "match_pages": ";".join(map(str, pages)),
                            "leaf_distance": distance,
                            "text": " ".join(str(match.get("text", "")).split()),
                        })

    fields = ["item_id", "query", "target_leaf", "match_index", "match_pages", "leaf_distance", "text"]
    with (OUT / "target_page_fragments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(fragments)
    summary = {
        "experiment": OUT.name, "items": len(targets), "queries_per_item": len(QUERIES),
        "target_pages": sum(len(v) for v in targets.values()), "near_page_fragments": len(fragments),
        "annual_values_promoted": False, "locked_period_accessed": False,
        "source_files_modified": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)}, "sources": sources,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
