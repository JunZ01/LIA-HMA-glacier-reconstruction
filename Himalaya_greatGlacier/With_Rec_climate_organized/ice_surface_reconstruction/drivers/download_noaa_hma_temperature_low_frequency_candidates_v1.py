"""Download public NOAA HMA temperature reconstruction candidates."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = PROJECT / "validation_data/sources/noaa_hma_temperature_low_frequency_candidates_v1"
URLS = {
    "tibet2019temp.txt": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/china/tibet2019temp.txt",
    "zhang2018easia-temp.txt": "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/zhang2018/zhang2018easia-temp.txt",
    "zhang2018easia-temp-grid.txt": "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/zhang2018/zhang2018easia-temp-grid.txt",
    "kathmandu_temp-noaa.txt": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/nepal/kathmandu_temp-noaa.txt",
    "kathmandu_temp-supp.txt": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/nepal/kathmandu_temp-supp.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows = []
    for filename, url in URLS.items():
        destination = OUTPUT / filename
        request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-climate-audit/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            status = int(response.status)
            content_type = response.headers.get("Content-Type", "")
        if status != 200 or len(payload) < 100:
            raise RuntimeError(f"Download failed: {filename}: status={status}, bytes={len(payload)}")
        destination.write_bytes(payload)
        rows.append(
            {
                "filename": filename,
                "url": url,
                "http_status": status,
                "content_type": content_type,
                "size_bytes": len(payload),
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "status": "complete",
        "source_policy": "public_noaa_ncei_direct_download",
        "files": rows,
        "original_project_files_modified": False,
    }
    (OUTPUT / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# NOAA HMA Temperature Low-Frequency Candidates\n\n"
        "Public candidate reconstructions downloaded without modifying any existing source. "
        "They remain candidate evidence until identity overlap and frozen low-frequency validation are complete.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
