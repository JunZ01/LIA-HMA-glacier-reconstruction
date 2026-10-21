"""Run a conservative historical chain with one bounded power-law ramp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import synthetic_lia_conservative_junction_historical_ramp_v1 as historical


def _power_ramp_climate(gdir, config: dict, pixel):
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    suffix, audit = historical._write_case_climate(
        gdir,
        Path(config["reconstruction_climate"]),
        pixel,
        "off",
        start_year=start_year,
        end_year=end_year,
    )
    source = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    gamma = float(config["ramp_shape_exponent"])
    start_bias = float(config["stage2"]["delta_t_c"])
    end_bias = float(config["forward_delta_t_end_c"])
    with xr.open_dataset(source, use_cftime=True) as ds0:
        ds = ds0.load()
    decimal_year = np.asarray(
        [date.year + (date.month - 1) / 12.0 for date in ds["time"].values], dtype=float
    )
    fraction = np.clip(
        (decimal_year - float(start_year)) / float(end_year - start_year), 0.0, 1.0
    )
    attrs = dict(ds["temp"].attrs)
    ds["temp"] = ds["temp"] + xr.DataArray(
        (end_bias - start_bias) * np.power(fraction, gamma),
        dims=("time",),
        coords={"time": ds["time"]},
    )
    ds["temp"].attrs = attrs
    ds.attrs = dict(ds.attrs)
    ds.attrs["forward_temperature_bias_policy"] = (
        f"power effective bias gamma={gamma:.6f} from {start_bias:+.6f} K at "
        f"{start_year} to {end_bias:+.6f} K at {end_year}"
    )
    gamma_tag = f"{gamma:.4f}".replace(".", "p")
    end_tag = f"{end_bias:+.5f}".replace("+", "p").replace("-", "m").replace(".", "p")
    target_suffix = f"{suffix}_bias_power_g{gamma_tag}_end{end_tag}"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
    ds.to_netcdf(target)
    ds.close()
    return target_suffix, {
        **audit,
        "base_climate_file": str(source),
        "climate_file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "temperature_bias_policy": "power-ramp-in-forward-climate-file",
        "ramp_shape_exponent": gamma,
        "effective_delta_t_start_c": start_bias,
        "effective_delta_t_end_c": end_bias,
        "additional_warming_2000_c": end_bias - start_bias,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--end-delta-t", required=True, type=float)
    parser.add_argument("--shape-exponent", required=True, type=float)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["stage"] = "historical-forward-power-ramp-screening"
    config["forward_delta_t_end_c"] = float(args.end_delta_t)
    config["ramp_shape_exponent"] = float(args.shape_exponent)
    config["output_root"] = {"anchor": "project", "sub": output_root.name}
    overlay = output_root.parent / f"{output_root.name}_input_config_overlay.json"
    overlay.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    historical._ramp_climate = _power_ramp_climate
    print(json.dumps(historical.run(overlay), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
