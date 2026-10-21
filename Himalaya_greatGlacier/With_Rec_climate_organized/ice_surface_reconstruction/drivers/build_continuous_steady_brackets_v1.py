from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "synthetic_lia_system_standard_inversion_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", required=True, type=int)
    parser.add_argument("--offsets", nargs="+", default=[0.0, -0.5], type=float)
    parser.add_argument("--inversion-prefix", default="w673inv")
    parser.add_argument("--zero-prefix", default="w674zero")
    parser.add_argument("--steady-prefix", default="w675steady")
    args = parser.parse_args()
    outputs = []
    for sid in args.systems:
        inv_cfg = json.loads((CONFIG_ROOT / f"inversion_{sid}_continuous_v1.json").read_text(encoding="utf-8"))
        for offset in args.offsets:
            delta = float(inv_cfg["stage2"]["delta_t_c"]) + offset
            token = f"{offset:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
            out_root = f"{args.steady_prefix}{sid}_{token}_continuous_v1"
            cfg = {
                "experiment": "conservative-one-way-tributary-fluxbased-v1",
                "stage": "continuous-remaining-formal-conservative-steady-50y-bracket",
                "system_id": sid,
                "synthetic_rgi_id": inv_cfg["synthetic_rgi_id"],
                "inversion_root": {"anchor": "project", "sub": f"{args.inversion_prefix}{sid}_continuous_v1"},
                "zero_mb_root": {"anchor": "project", "sub": f"{args.zero_prefix}{sid}_continuous_v1"},
                "output_root": {"anchor": "project", "sub": out_root},
                "climate_filesuffix": "_batch1_1441_1470_prcp_off",
                "spinup_window": [1441, 1470],
                "registered_lia_length_m": inv_cfg["registered_lia_length_m"],
                "stage2": dict(inv_cfg["stage2"], delta_t_c=delta),
                "gates": {"volume_change_fraction_max": 0.02, "lia_length_relative_error_max": 0.1, "minimum_free_downstream_m": 5000.0},
                "authorization": {"historical": False},
            }
            path = CONFIG_ROOT / f"steady_conservative_{sid}_{token}_continuous_v1.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            outputs.append({"system_id": sid, "offset": offset, "delta_t_c": delta, "config": str(path.relative_to(ROOT)).replace("\\", "/"), "output_root": out_root})
    print(json.dumps({"configs": outputs}, indent=2))


if __name__ == "__main__":
    main()
