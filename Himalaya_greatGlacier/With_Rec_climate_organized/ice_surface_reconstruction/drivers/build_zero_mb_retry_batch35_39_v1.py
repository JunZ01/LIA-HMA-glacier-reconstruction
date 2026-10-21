"""Versioned retry configs for zero-MB runs that failed before writing summaries."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_IDS = [551, 902, 1326, 1645, 1880, 2141, 2727, 3010, 3185, 3808, 3906, 3909, 3912, 4464]


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []
    for system_id in SYSTEM_IDS:
        source = config_root / f"zero_mb_batch35_39_{system_id}_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["stage"] = "batch-35-39-formal-zero-mb-technical-retry"
        payload["output_root"] = {"anchor": "project", "sub": f"w628zero_retry_batch35_39_{system_id}_v1"}
        target = config_root / f"zero_mb_retry_batch35_39_{system_id}_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(target))
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
