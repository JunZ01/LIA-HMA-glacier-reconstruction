"""Schema-compatibility wrapper for standard inversion v1.

The paired anchor-reroute SourceState already encodes the LIA domain in
``valid_mask`` and has no separate ``lia_mask`` attribute.  This wrapper adds
that read-only alias in memory and otherwise delegates the full calculation,
physics, and gates unchanged to v1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synthetic_lia_system_standard_inversion_v1 as v1


_ORIGINAL_LOAD = v1.load_source_state


def _load_compatible(*args, **kwargs):
    state = _ORIGINAL_LOAD(*args, **kwargs)
    state.lia_mask = state.valid_mask
    return state


def run(config_path: Path) -> dict:
    v1.load_source_state = _load_compatible
    return v1.run(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
