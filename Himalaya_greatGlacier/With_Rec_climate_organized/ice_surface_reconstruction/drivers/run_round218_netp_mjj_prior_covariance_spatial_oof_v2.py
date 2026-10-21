from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(__file__).with_name(
    "run_round218_netp_mjj_prior_covariance_spatial_oof_v1.py"
)
EXPECTED_SOURCE_SHA256 = (
    "acd5b24a5b7c68eb445b8e59b7f71de7f9576997888b9904192f1954da6ede9d"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("Round 218 v1 driver changed")

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'round_218_netp_mjj_prior_covariance_spatial_oof_v1"',
    'round_218_netp_mjj_prior_covariance_spatial_oof_v2"',
    1,
)
source = source.replace(
    "cells_frame.product == product",
    'cells_frame["product"] == product',
    1,
)
if "cells_frame.product == product" in source:
    raise RuntimeError("Round 218 compatibility replacement incomplete")

namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace, namespace)
