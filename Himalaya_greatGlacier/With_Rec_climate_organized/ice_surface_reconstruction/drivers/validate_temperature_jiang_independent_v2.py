"""Index-safe wrapper for Jiang Co blind validation v2."""

from __future__ import annotations

import numpy as np
import pandas as pd

import validate_temperature_external_low_frequency_v1 as external_base
import validate_temperature_jiang_independent_v1 as v1


v1.OUTPUT = (
    external_base.RESULTS
    / "climate_reconstruction_temperature_jiang_independent_blind_validation_v2"
)
original_reader = external_base.read_candidate


def read_candidate_with_locked_nan(path):
    allowed = original_reader(path).set_index("year")
    full = allowed.reindex(np.arange(1000, 2001))
    full.index.name = "year"
    return full.reset_index()


if __name__ == "__main__":
    v1.base.read_candidate = read_candidate_with_locked_nan
    v1.main()
