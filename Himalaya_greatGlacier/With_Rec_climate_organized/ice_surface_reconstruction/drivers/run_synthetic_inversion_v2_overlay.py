"""Run schema-compatible inversion v2 with an independent output-root overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synthetic_lia_system_standard_inversion_v2 as v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config["output_root"] = {"anchor": "project", "sub": output_root.name}
    config["stage"] = "three-representative-standard-inversion-v2-schema-compatible"
    overlay = output_root / "input_config_overlay.json"
    overlay.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(v2.v1.run(overlay), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
