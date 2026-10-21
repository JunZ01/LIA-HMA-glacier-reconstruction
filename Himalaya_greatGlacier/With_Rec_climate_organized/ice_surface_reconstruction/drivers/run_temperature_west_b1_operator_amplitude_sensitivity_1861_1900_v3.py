"""Serialization-corrected launcher for the b1 sensitivity experiment."""

from __future__ import annotations

import json

import numpy as np

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v2 as experiment


OUTPUT = experiment.v1.RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v3"
ORIGINAL_DUMPS = json.dumps


def compatible_dumps(value, *args, **kwargs):
    def convert(item):
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    kwargs.setdefault("default", convert)
    return ORIGINAL_DUMPS(value, *args, **kwargs)


experiment.OUTPUT = OUTPUT
experiment.json.dumps = compatible_dumps


if __name__ == "__main__":
    experiment.main()
