"""Build two evidence-backed refinements after the historical bracket audit."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"

    hist = json.loads((root / "w653hist_conservative_batch35_39_round3_3495_end0p125_v1/config_snapshot.json").read_text(encoding="utf-8"))
    hist["stage"] = "targeted-linear-endpoint-bisection"
    hist["forward_delta_t_end_c"] = 0.25
    hist["output_root"] = {"anchor": "project", "sub": "w658hist3495_end0p25_v1"}
    hist_cfg = cfg_root / "h_w658_3495_0p25.json"
    hist_cfg.write_text(json.dumps(hist, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    steady = json.loads((root / "w645steady_conservative_batch35_39_3808_cold05_v1/config_snapshot.json").read_text(encoding="utf-8"))
    steady["stage"] = "targeted-steady-grid-crossing-refinement"
    steady["stage2"]["delta_t_c"] = 0.45
    steady["output_root"] = {"anchor": "project", "sub": "w658steady3808_delta0p45_v1"}
    steady_cfg = cfg_root / "s_w658_3808_0p45.json"
    steady_cfg.write_text(json.dumps(steady, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "historical": {"system_id": 3495, "config": str(hist_cfg), "output_root": "w658hist3495_end0p25_v1"},
        "steady": {"system_id": 3808, "config": str(steady_cfg), "output_root": "w658steady3808_delta0p45_v1"},
    }
    (root / "w658_targeted_refine_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
