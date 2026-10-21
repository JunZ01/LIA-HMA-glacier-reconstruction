from __future__ import annotations

import concurrent.futures
import json
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_167_chin074_qamdo2_identity_recovery_v1"

BASES = [
    "https://www.climatology.uni-mainz.de/files/2016/03",
    "https://www.climatology.uni-mainz.de/files/2016/06",
    "https://www.blogs.uni-mainz.de/fb09climatology/files/2016/03",
    "https://www.blogs.uni-mainz.de/fb09climatology/files/2016/06",
]

NAMES = [
    "Fig1_Qamdo.jpg", "Fig1_Qamdo.jpeg", "Fig1_Qamdo.png",
    "Qamdo_recon.jpg", "Qamdo_recon.jpeg", "Qamdo_recon.png",
    "Qamdo.txt", "qamdo.txt", "Qamdo_data.txt", "Qamdo_recon.txt",
    "Qamdo_RCS.txt", "Qamdo.csv", "Qamdo.xls", "Qamdo.xlsx", "Qamdo.zip",
    "Qamdo_data.csv", "Qamdo_data.xls", "Qamdo_data.xlsx", "Qamdo_data.zip",
]


def probe(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "LIA-HMA-source-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            return {
                "url": url,
                "status": int(getattr(response, "status", 200)),
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "location": response.headers.get("Location"),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "content_type": None, "content_length": None, "location": exc.headers.get("Location"), "error": None}
    except Exception as exc:
        return {"url": url, "status": None, "content_type": None, "content_length": None, "location": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    urls = [f"{base}/{name}" for base in BASES for name in NAMES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(probe, urls))
    hits = [row for row in results if row["status"] is not None and row["status"] < 400]
    payload = {
        "experiment": "round_167_chin074_qamdo2_identity_recovery_v1",
        "method": "HEAD_only_no_payload_download",
        "candidate_count": len(results),
        "hits": hits,
        "all_results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mainz_qamdo_attachment_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(results), "hit_count": len(hits), "hits": hits}, indent=2))


if __name__ == "__main__":
    main()
