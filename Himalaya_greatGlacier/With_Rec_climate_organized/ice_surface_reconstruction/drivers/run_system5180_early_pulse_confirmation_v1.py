"""Run the preregistered system-5180 early-warm-pulse mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import synthetic_lia_conservative_junction_historical_ramp_v1 as historical


def _early_pulse_climate(gdir, config: dict, pixel):
    start_year = int(config["start_year"])
    run_end_year = int(config["end_year"])
    forcing_end_year = int(config["forcing_end_year"])
    suffix, audit = historical._write_case_climate(
        gdir,
        Path(config["reconstruction_climate"]),
        pixel,
        "off",
        start_year=start_year,
        end_year=run_end_year,
    )
    source = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    gamma = float(config["ramp_shape_exponent"])
    start_bias = float(config["stage2"]["delta_t_c"])
    end_bias = float(config["forward_delta_t_end_c"])
    pulse = config["early_warm_pulse"]
    pulse_peak_year = float(pulse["peak_year"])
    pulse_end_year = float(pulse["end_year"])
    pulse_amplitude = float(pulse["amplitude_c"])
    if not (start_year < pulse_peak_year < pulse_end_year <= forcing_end_year):
        raise ValueError("invalid-early-warm-pulse-years")

    with xr.open_dataset(source, use_cftime=True) as ds0:
        ds = ds0.load()
    decimal_year = np.asarray(
        [date.year + (date.month - 1) / 12.0 for date in ds["time"].values],
        dtype=float,
    )
    forcing_fraction = np.clip(
        (decimal_year - float(start_year)) / float(forcing_end_year - start_year),
        0.0,
        1.0,
    )
    power_increment = (end_bias - start_bias) * np.power(forcing_fraction, gamma)
    pulse_increment = np.zeros_like(decimal_year)
    rising = (decimal_year >= start_year) & (decimal_year <= pulse_peak_year)
    falling = (decimal_year > pulse_peak_year) & (decimal_year <= pulse_end_year)
    pulse_increment[rising] = pulse_amplitude * (
        (decimal_year[rising] - start_year) / (pulse_peak_year - start_year)
    )
    pulse_increment[falling] = pulse_amplitude * (
        (pulse_end_year - decimal_year[falling]) / (pulse_end_year - pulse_peak_year)
    )
    attrs = dict(ds["temp"].attrs)
    ds["temp"] = ds["temp"] + xr.DataArray(
        power_increment + pulse_increment,
        dims=("time",),
        coords={"time": ds["time"]},
    )
    ds["temp"].attrs = attrs
    ds.attrs = dict(ds.attrs)
    ds.attrs["forward_temperature_bias_policy"] = (
        f"power gamma={gamma:.6f} from {start_bias:+.6f} K at {start_year} to "
        f"{end_bias:+.6f} K at {forcing_end_year}; triangular early pulse "
        f"0 K at {start_year}, {pulse_amplitude:+.6f} K at {pulse_peak_year:g}, "
        f"0 K at {pulse_end_year:g}"
    )
    target_suffix = f"{suffix}_system5180_early_pulse_v1"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
    ds.to_netcdf(target)
    ds.close()
    return target_suffix, {
        **audit,
        "base_climate_file": str(source),
        "climate_file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "temperature_bias_policy": "registered-power-ramp-plus-finite-early-warm-pulse",
        "ramp_shape_exponent": gamma,
        "effective_delta_t_start_c": start_bias,
        "effective_delta_t_end_c": end_bias,
        "forcing_end_year": forcing_end_year,
        "early_warm_pulse": {
            "start_year": start_year,
            "peak_year": pulse_peak_year,
            "end_year": pulse_end_year,
            "amplitude_c": pulse_amplitude,
        },
        "initial_thickness_modification": False,
        "prohibited_source_accessed": False,
    }


def _assess_preflight(summary: dict, config: dict) -> dict:
    lia = float(config["lia_registered_length_m"])
    rgi = float(config["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    peak = float(summary["peak"]["length_m"])
    lower = max(0.9 * lia, lia - 0.2 * separation)
    upper = min(1.1 * lia, lia + 0.2 * separation)
    numerical_flags = [
        flag for flag in summary["review_flags"]
        if flag in {
            "cfl-error", "boundary-error", "numerical-error",
            "historical-incomplete", "nonfinite-geometry",
            "negative-section", "root-boundary-ice",
        }
    ]
    passed = not numerical_flags and lower <= peak <= upper
    return {
        "schema": "lia-191-system5180-early-pulse-preflight-v1",
        "status": "preflight-pass" if passed else "preflight-fail",
        "system_id": 5180,
        "run_end_year": int(config["end_year"]),
        "full_historical_evaluated": False,
        "initial_thickness_modification": False,
        "prohibited_source_accessed": False,
        "peak_length_m": peak,
        "allowed_peak_length_m": {"lower": lower, "upper": upper},
        "numerical_flags": numerical_flags,
        "base_summary_status_is_not_a_preflight_decision": summary["status"],
        "full_historical_authorized_by_this_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["system_id"]) != 5180:
        raise ValueError("driver-is-registered-for-system5180-only")
    historical._ramp_climate = _early_pulse_climate
    summary = historical.run(config_path)
    result = {"historical_summary": summary}
    if config.get("run_role") == "scientific-preflight":
        assessment = _assess_preflight(summary, config)
        output_root = historical.resolve(config["output_root"])
        historical._write_json(output_root / "preflight_assessment.json", assessment)
        result["preflight_assessment"] = assessment
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
