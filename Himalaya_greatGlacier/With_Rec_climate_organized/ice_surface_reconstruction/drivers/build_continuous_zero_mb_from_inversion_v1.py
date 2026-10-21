from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "synthetic_lia_system_standard_inversion_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inversion-prefix", default="w673inv")
    parser.add_argument("--zero-prefix", default="w674zero")
    parser.add_argument("--systems", nargs="+", required=True, type=int)
    args = parser.parse_args()
    outputs = []
    for sid in args.systems:
        inv_root = ROOT / f"{args.inversion_prefix}{sid}_continuous_v1"
        summary = json.loads((inv_root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
        if summary.get("status") != "inversion-pass":
            continue
        cfg = {
            "experiment": "conservative-one-way-tributary-fluxbased-v1",
            "stage": "continuous-remaining-formal-conservative-zero-mb",
            "system_id": sid,
            "synthetic_rgi_id": summary["synthetic_rgi_id"] if "synthetic_rgi_id" in summary else f"RGI60-00.9{sid:04d}",
            "inversion_root": {"anchor": "project", "sub": inv_root.name},
            "output_root": {"anchor": "project", "sub": f"{args.zero_prefix}{sid}_continuous_v1"},
            "glen_a": float(summary["stage2"]["glen_a"]),
            "fs": float(summary["stage2"]["fs"]),
            "gates": {"volume_drift_fraction_max": 0.001, "root_terminal_thickness_m_max": 10.0, "mass_creation_relative_fraction": 1e-12, "mass_creation_absolute_m3": 1e-6},
        }
        if "synthetic_rgi_id" not in summary:
            inv_cfg = json.loads((CONFIG_ROOT / f"inversion_{sid}_continuous_v1.json").read_text(encoding="utf-8"))
            cfg["synthetic_rgi_id"] = inv_cfg["synthetic_rgi_id"]
        out = CONFIG_ROOT / f"zero_mb_conservative_{sid}_continuous_v1.json"
        out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        outputs.append(str(out.relative_to(ROOT)).replace("\\", "/"))
    print(json.dumps({"configs": outputs}, indent=2))


if __name__ == "__main__":
    main()
