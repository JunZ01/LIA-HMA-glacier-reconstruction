"""UTF-8-BOM compatibility wrapper for the v1 scientific adjudicator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import adjudicate_system2005_terminal_retreat_gap_v1 as adjudicator


def run(config_path: Path):
    original_open = Path.open

    def compatible_open(self, *args, **kwargs):
        if self.name == "event_trace.csv" and kwargs.get("encoding") == "utf-8":
            kwargs["encoding"] = "utf-8-sig"
        return original_open(self, *args, **kwargs)

    Path.open = compatible_open
    try:
        return adjudicator.run(config_path)
    finally:
        Path.open = original_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), allow_nan=False))


if __name__ == "__main__":
    main()
