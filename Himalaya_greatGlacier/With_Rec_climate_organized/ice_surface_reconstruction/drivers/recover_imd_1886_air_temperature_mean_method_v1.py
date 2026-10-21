from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "imd_1886_air_temperature_mean_method_v1"
)
ITEM = "india.history.resource.109237"
LEAVES = (31, 32, 33)
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = []
    for leaf in LEAVES:
        url = f"https://archive.org/download/{ITEM}/page/leaf{leaf}.jpg"
        path = OUT / f"official_leaf_{leaf:04d}.jpg"
        path.write_bytes(download(url))
        sources.append(
            {
                "leaf": leaf,
                "url": url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    summary = {
        "experiment": "imd_1886_air_temperature_mean_method_v1",
        "pages": len(sources),
        "bytes": sum(int(source["bytes"]) for source in sources),
        "locked_period_accessed": False,
        "source_files_modified": False,
        "decision": "ocr_air_temperature_method_pages",
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "sources": sources,
                **summary,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
