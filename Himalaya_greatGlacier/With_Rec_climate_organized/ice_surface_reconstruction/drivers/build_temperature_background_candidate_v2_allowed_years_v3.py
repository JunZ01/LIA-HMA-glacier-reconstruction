"""Retry Candidate v2 with a netCDF multidimensional-mask compatibility adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

import build_temperature_background_candidate_v2_allowed_years as base


class VariableAdapter:
    def __init__(self, variable):
        self._variable = variable

    def __getitem__(self, key):
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[1], np.ndarray)
            and key[1].ndim == 2
            and key[1].dtype == bool
        ):
            values = np.asarray(self._variable[key[0], :, :])
            return values[:, key[1]] if values.ndim == 3 else values[key[1]]
        return self._variable[key]

    def __getattr__(self, name):
        return getattr(self._variable, name)


class DatasetAdapter:
    def __init__(self, *args, **kwargs):
        self._dataset = NetCDFDataset(*args, **kwargs)

    @property
    def variables(self):
        return {name: VariableAdapter(variable) for name, variable in self._dataset.variables.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._dataset.close()

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __setattr__(self, name, value):
        if name == "_dataset":
            object.__setattr__(self, name, value)
        else:
            setattr(self._dataset, name, value)


base.Dataset = DatasetAdapter
base.OUTPUT = base.LOOP / "temperature_background_candidate_v2_allowed_years_v3"
base.DRIVER = Path(base.__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(base.main())
