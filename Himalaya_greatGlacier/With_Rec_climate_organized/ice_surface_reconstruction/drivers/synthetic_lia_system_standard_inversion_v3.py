"""Frozen-SourceState schema proxy for standard inversion v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from . import synthetic_lia_system_standard_inversion_v1 as v1


_ORIGINAL_LOAD = v1.load_source_state


def _load_compatible(*args, **kwargs):
    state = _ORIGINAL_LOAD(*args, **kwargs)
    values = {name: getattr(state, name) for name in state.__dataclass_fields__}
    values.update(
        lia_mask=state.valid_mask,
        cell_area_m2=state.cell_area_m2,
        shape=state.shape,
    )
    return SimpleNamespace(**values)


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
