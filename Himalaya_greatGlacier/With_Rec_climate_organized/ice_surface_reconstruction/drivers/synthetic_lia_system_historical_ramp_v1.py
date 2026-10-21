"""Historical-chain wrapper with a continuous 1470-2000 temperature-bias ramp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import synthetic_lia_system_historical_v1 as v1


_write_case_climate_v1 = v1._write_case_climate
_run_v1 = v1.run
_active_config = None


def _tag(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _write_case_climate(
    gdir,
    reconstruction_path,
    pixel,
    prcp_mode,
    start_year=1470,
    end_year=2000,
):
    suffix, audit = _write_case_climate_v1(
        gdir,
        reconstruction_path,
        pixel,
        prcp_mode,
        start_year=start_year,
        end_year=end_year,
    )
    config = _active_config
    start_bias = float(config["stage2"]["delta_t_c"])
    end_bias = float(config["forward_delta_t_end_c"])
    ramp_suffix = f"{suffix}_bias_ramp_{_tag(start_bias)}_to_{_tag(end_bias)}"
    source_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    target_path = Path(gdir.get_filepath("gcm_data", filesuffix=ramp_suffix))
    if not target_path.exists():
        with xr.open_dataset(source_path, use_cftime=True) as source:
            dataset = source.load()
        dates = list(dataset["time"].values)
        decimal_year = np.asarray(
            [date.year + (date.month - 1) / 12.0 for date in dates], dtype=float
        )
        fraction = np.clip(
            (decimal_year - float(start_year)) / float(end_year - start_year),
            0.0,
            1.0,
        )
        added = (end_bias - start_bias) * fraction
        attrs = dict(dataset["temp"].attrs)
        dataset["temp"] = dataset["temp"] + xr.DataArray(
            added,
            dims=("time",),
            coords={"time": dataset["time"]},
        )
        dataset["temp"].attrs = attrs
        dataset.attrs = dict(dataset.attrs)
        dataset.attrs["forward_temperature_bias_policy"] = (
            f"linear effective bias from {start_bias:+.6f} K at {start_year} "
            f"to {end_bias:+.6f} K at {end_year}"
        )
        dataset.to_netcdf(target_path)
        dataset.close()
    ramp_audit = {
        **audit,
        "base_climate_file": str(source_path),
        "climate_file": str(target_path),
        "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "temperature_bias_policy": "linear-ramp-in-forward-climate-file",
        "effective_delta_t_start_c": start_bias,
        "effective_delta_t_end_c": end_bias,
        "additional_warming_2000_c": end_bias - start_bias,
    }
    return ramp_suffix, ramp_audit


def run(config_path: Path):
    global _active_config
    _active_config = json.loads(config_path.read_text(encoding="utf-8"))
    return _run_v1(config_path)


v1._write_case_climate = _write_case_climate
v1.run = run


if __name__ == "__main__":
    v1.main()
