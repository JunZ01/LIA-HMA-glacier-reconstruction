from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "imd_1886_temperature_mean_method_v1"
ITEM_ID = "india.history.resource.109237"
QUERIES = (
    "except Darjeeling",
    "Darjeeling Pedong Calcutta",
    "10h 16h observations",
    "mean temperature",
    "Sambhar Akyab",
)
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str, attempts: int = 4) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    raw = OUT / "official_fulltext_queries"
    raw.mkdir()

    metadata_url = f"https://archive.org/metadata/{ITEM_ID}"
    metadata_bytes = request(metadata_url)
    metadata_path = raw / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    metadata = json.loads(metadata_bytes.decode("utf-8", errors="replace"))
    server = str(metadata.get("server") or metadata.get("d2") or metadata["d1"])
    item_path = str(metadata["dir"])
    base = "109237"

    method_matches: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for index, query in enumerate(QUERIES, start=1):
        params = urllib.parse.urlencode(
            {
                "item_id": ITEM_ID,
                "doc": base,
                "path": item_path,
                "q": query,
                "pre_tag": "{{{",
                "post_tag": "}}}",
            }
        )
        url = f"https://{server}/fulltext/inside.php?{params}"
        payload = request(url)
        path = raw / f"query_{index:02d}.json"
        path.write_bytes(payload)
        manifest.append(
            {
                "kind": "fulltext_query",
                "query": query,
                "url": url,
                "path": str(path),
                "bytes": len(payload),
                "sha256": sha256(path),
            }
        )
        decoded = json.loads(payload.decode("utf-8", errors="replace"))
        for match in decoded.get("matches", []):
            leaves = sorted(
                {
                    int(par["page"])
                    for par in (match.get("par") or [])
                    if "page" in par
                }
            )
            if 56 in leaves:
                method_matches.append(
                    {"query": query, "leaves": leaves, "text": match.get("text", "")}
                )

    image_url = f"https://archive.org/download/{ITEM_ID}/page/leaf56.jpg"
    image_path = OUT / "official_leaf_0056.jpg"
    image_path.write_bytes(request(image_url))
    manifest.append(
        {
            "kind": "official_page_image",
            "query": "",
            "url": image_url,
            "path": str(image_path),
            "bytes": image_path.stat().st_size,
            "sha256": sha256(image_path),
        }
    )

    (OUT / "method_page_query_matches.json").write_text(
        json.dumps(method_matches, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "imd_1886_temperature_mean_method_v1",
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "sources": manifest,
                "locked_period_accessed": False,
                "source_files_modified": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": "imd_1886_temperature_mean_method_v1",
        "queries": len(QUERIES),
        "method_page_matches": len(method_matches),
        "method_page_bytes": image_path.stat().st_size,
        "locked_period_accessed": False,
        "decision": "ocr_method_page_and_recover_station_specific_mean_rules",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
