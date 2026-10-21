from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


DRIVER = Path(__file__).with_name(
    "run_round270_xinzhu_current_network_identity_v1.py"
)


def load_metrics():
    spec = importlib.util.spec_from_file_location("round270_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics


def main() -> None:
    metrics = load_metrics()
    years = np.arange(61, dtype=float)
    candidate = np.sin(years / 4.0) + 0.015 * years

    exact = metrics(candidate, 4.5 - 2.25 * candidate)
    assert exact["exact_affine_alias"]
    assert exact["near_alias"]
    assert exact["affine_rmse_fraction"] <= 1e-12

    rng = np.random.default_rng(270)
    near_reference = candidate + rng.normal(0.0, 0.025, candidate.size)
    near = metrics(candidate, near_reference)
    assert not near["exact_affine_alias"]
    assert near["near_alias"]

    independent = metrics(candidate, rng.normal(size=candidate.size))
    assert not independent["exact_affine_alias"]
    assert not independent["near_alias"]

    short = metrics(candidate[:49], 3.0 * candidate[:49] + 2.0)
    assert not short["exact_affine_alias"]
    assert not short["near_alias"]

    print("ROUND270_SYNTHETIC_IDENTITY_TEST_OK")


if __name__ == "__main__":
    main()
