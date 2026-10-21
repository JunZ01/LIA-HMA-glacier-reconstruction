"""Run schema-compatible inversion v2 with delta-T and output overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synthetic_lia_system_standard_inversion_v2 as v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["stage"] = "independent-selected-delta-standard-inversion"
    config["stage2"]["delta_t_c"] = float(args.delta_t)
    config["output_root"] = {"anchor": "project", "sub": output_root.name}
    overlay = output_root.parent / f"{output_root.name}_input_config_overlay.json"
    overlay.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(v2.v1.run(overlay), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
