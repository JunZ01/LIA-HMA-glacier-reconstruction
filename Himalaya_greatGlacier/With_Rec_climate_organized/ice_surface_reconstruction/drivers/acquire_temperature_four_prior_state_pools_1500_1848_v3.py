"""Executable repair of text-attribute decoding in the v2 acquisition."""

from __future__ import annotations

from pathlib import Path

import acquire_temperature_four_prior_state_pools_1500_1848_v2 as v2


def text_attr(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


v2.v1.range_access.text_attr = text_attr
v2.v1.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(v2.v1.main())
