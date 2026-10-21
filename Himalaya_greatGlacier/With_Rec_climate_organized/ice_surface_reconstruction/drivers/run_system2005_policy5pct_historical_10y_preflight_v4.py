"""Ten-year preflight rebuilding a copied ramp cache in the isolated clone."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import numpy as np
import xarray as xr

from . import run_system2005_policy5pct_historical_10y_preflight_v1 as preflight_v1
from . import run_system2005_topology_adaptive_historical_v1 as historical
from . import synthetic_lia_conservative_junction_historical_ramp_v1 as ramp


def main() -> None:
    original = historical._ramp_climate

    def isolated_clone_rebuilt_ramp(gdir, config, pixel):
        start_year = int(config["start_year"])
        end_year = int(config["end_year"])
        suffix, audit = ramp._write_case_climate(
            gdir,
            Path(config["reconstruction_climate"]),
            pixel,
            "off",
            start_year=start_year,
            end_year=end_year,
        )
        start_bias = float(config["stage2"]["delta_t_c"])
        end_bias = float(config["forward_delta_t_end_c"])
        ramp_suffix = f"{suffix}_bias_ramp_{ramp._tag(start_bias)}_to_{ramp._tag(end_bias)}"
        source_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
        target_path = Path(gdir.get_filepath("gcm_data", filesuffix=ramp_suffix))
        if target_path.exists():
            target_path.chmod(target_path.stat().st_mode | stat.S_IWRITE | stat.S_IREAD)
            target_path.unlink()
        with xr.open_dataset(source_path, use_cftime=True) as source:
            dataset = source.load()
        decimal_year = np.asarray(
            [date.year + (date.month - 1) / 12.0 for date in dataset["time"].values],
            dtype=float,
        )
        fraction = np.clip(
            (decimal_year - float(start_year)) / float(end_year - start_year), 0.0, 1.0
        )
        attrs = dict(dataset["temp"].attrs)
        dataset["temp"] = dataset["temp"] + xr.DataArray(
            (end_bias - start_bias) * fraction,
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
        return ramp_suffix, {
            **audit,
            "base_climate_file": str(source_path),
            "climate_file": str(target_path),
            "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "temperature_bias_policy": "linear-ramp-in-forward-climate-file",
            "effective_delta_t_start_c": start_bias,
            "effective_delta_t_end_c": end_bias,
            "additional_warming_2000_c": end_bias - start_bias,
            "isolated_clone_existing_ramp_cache_rebuilt": True,
        }

    historical._ramp_climate = isolated_clone_rebuilt_ramp
    try:
        preflight_v1.main()
    finally:
        historical._ramp_climate = original


if __name__ == "__main__":
    main()
