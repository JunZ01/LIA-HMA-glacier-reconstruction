"""Download the two public PANGAEA tree-ring d18O candidates for Round 123."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "mimei_tree_ring_d18o.txt": "https://doi.pangaea.de/10.1594/PANGAEA.869839?format=textfile",
    "xinpu_tree_ring_d18o.txt": "https://doi.pangaea.de/10.1594/PANGAEA.869840?format=textfile",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    )
    raw = output / "raw_unopened"
    if output.exists():
        raise FileExistsError(output)
    raw.mkdir(parents=True)
    rows = []
    for name, url in SOURCES.items():
        target = raw / name
        request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if len(payload) < 1000:
            raise RuntimeError(f"download-too-small:{name}:{len(payload)}")
        target.write_bytes(payload)
        rows.append(
            {
                "name": name,
                "url": url,
                "size_bytes": len(payload),
                "sha256": sha256(target),
            }
        )
    manifest = {
        "experiment_id": "round123_southeast_tibet_d18o_proxy_qualification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "download_complete_values_unopened",
        "locked_1921_1939_values_decoded": 0,
        "sources": rows,
    }
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
