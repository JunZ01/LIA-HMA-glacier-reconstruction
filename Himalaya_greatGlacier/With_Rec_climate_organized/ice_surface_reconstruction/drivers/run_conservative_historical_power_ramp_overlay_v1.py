"""Run a conservative historical chain with a bounded power-law ramp.

Only the timing shape of the forward temperature correction is changed:
fraction = ((year - start) / (end - start)) ** gamma.  All geometry,
physics, climate inputs and the two endpoint temperature biases are inherited
from the registered linear-ramp configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import xarray as xr

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    synthetic_lia_conservative_junction_historical_ramp_v1 as historical,
)


def _power_ramp_climate(gdir, config: dict, pixel):
    suffix, audit = historical._write_case_climate(
        gdir,
        Path(config["reconstruction_climate"]),
        pixel,
        "off",
        int(config["forward_start_year"]),
        int(config["forward_end_year"]),
    )
    source = Path(gdir.get_filepath("climate_historical", filesuffix=suffix))
    gamma = float(config["ramp_shape_exponent"])
    start_bias = float(config["forward_start_delta_t"])
    end_bias = float(config["forward_end_delta_t"])
    start_year = int(config["forward_start_year"])
    end_year = int(config["forward_end_year"])

    with xr.open_dataset(source) as ds0:
        ds = ds0.load()
    years = ds.time.dt.year.values.astype(float)
    fraction = np.clip((years - start_year) / (end_year - start_year), 0.0, 1.0)
    added_bias = (end_bias - start_bias) * np.power(fraction, gamma)
    ds["temp"] = ds["temp"] + xr.DataArray(added_bias, dims=("time",))
    ds.attrs["forward_temperature_ramp"] = "power"
    ds.attrs["forward_temperature_ramp_exponent"] = gamma
    ds.attrs["forward_start_delta_t_K"] = start_bias
    ds.attrs["forward_end_delta_t_K"] = end_bias

    gamma_tag = f"{gamma:.4f}".replace(".", "p")
    end_tag = f"{end_bias:+.5f}".replace("+", "p").replace("-", "m").replace(".", "p")
    target_suffix = f"{suffix}_power_g{gamma_tag}_end{end_tag}"
    target = Path(gdir.get_filepath("climate_historical", filesuffix=target_suffix))
    ds.to_netcdf(target)
    audit = dict(audit)
    audit.update(
        {
            "ramp_type": "power",
            "ramp_shape_exponent": gamma,
            "forward_start_delta_t_K": start_bias,
            "forward_end_delta_t_K": end_bias,
            "climate_path": str(target),
        }
    )
    return target_suffix, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--end-delta-t", required=True, type=float)
    parser.add_argument("--shape-exponent", required=True, type=float)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_path = Path(args.config).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    overlay = copy.deepcopy(config)
    overlay["forward_end_delta_t"] = float(args.end_delta_t)
    overlay["ramp_shape_exponent"] = float(args.shape_exponent)
    overlay["output"] = {"anchor": "project", "sub": str(args.output_root)}
    overlay["stage"] = "historical-forward-power-ramp"
    historical._ramp_climate = _power_ramp_climate
    historical.run(overlay)


if __name__ == "__main__":
    main()
