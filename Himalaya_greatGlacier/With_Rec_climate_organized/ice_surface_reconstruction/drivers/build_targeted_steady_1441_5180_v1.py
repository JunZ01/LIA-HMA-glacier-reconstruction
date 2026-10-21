"""Build two near-threshold steady refinements from registered snapshots."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    cases = [
        (1441, "w479steady1441parentv1", -2.4, "w665steady1441_dtm2p4_v1"),
        (5180, "w492steady5180dtm0125v1", -0.25, "w665steady5180_dtm0p25_v1"),
    ]
    written = []
    for sid, source_root, delta, output_root in cases:
        cfg = json.loads((root / source_root / "config_snapshot.json").read_text(encoding="utf-8"))
        cfg["stage"] = "targeted-near-threshold-steady-refinement"
        cfg["stage2"]["delta_t_c"] = delta
        cfg["output_root"] = {"anchor": "project", "sub": output_root}
        target = cfg_root / f"s_w665_{sid}.json"
        target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({"system_id": sid, "delta_t_c": delta, "config": str(target), "output_root": output_root})
    out = root / "w665_targeted_steady_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
