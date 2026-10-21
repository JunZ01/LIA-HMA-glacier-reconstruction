"""Export a frozen glacier-system MonthlyTI calendar for PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .lia_first10_spinup_batch import _write_case_climate
from .generic_planform2d_adapter_contract_v1 import (
    climate_schema_names,
    selected_sample_positions,
    validate_hashed_paths,
)


PROJECT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clone_runtime_gdir(config: dict[str, object], output: Path) -> tuple[Path, str, Path]:
    """Clone either a legacy synthetic base or one real glacier directory."""
    if "source_gdir" not in config:
        source_base = resolve(str(config["inversion_root"])) / "synthetic_gdir"
        cloned_base = output / "synthetic_gdir"
        shutil.copytree(source_base, cloned_base)
        return cloned_base, str(config["synthetic_rgi_id"]), source_base

    source_gdir = resolve(str(config["source_gdir"]))
    runtime_rgi_id = str(config.get("runtime_rgi_id", config["main_rgi_id"]))
    expected_climate = source_gdir / "climate_historical.nc"
    baseline = resolve(str(config["baseline_climate"]))
    if not source_gdir.is_dir():
        raise FileNotFoundError(f"source-gdir-missing:{source_gdir}")
    if sha256(expected_climate) != sha256(baseline):
        raise RuntimeError("source-gdir-baseline-climate-mismatch")

    cloned_base = (
        resolve(str(config["runtime_gdir_base"]))
        if "runtime_gdir_base" in config
        else output / "runtime_gdir"
    )
    if cloned_base.exists():
        raise FileExistsError(f"runtime-gdir-base-exists:{cloned_base}")
    destination = cloned_base / runtime_rgi_id[:8] / runtime_rgi_id[:-3] / runtime_rgi_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    required = ["climate_historical.nc", "outlines.tar.gz"]
    optional = ["glacier_grid.json", "outlines.tar.gz.properties"]
    for name in required:
        source = source_gdir / name
        if not source.is_file():
            raise FileNotFoundError(f"source-gdir-runtime-file-missing:{source}")
        shutil.copy2(source, destination / name)
    for name in optional:
        source = source_gdir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return cloned_base, runtime_rgi_id, source_gdir


def portable_formula(model, heights: np.ndarray, source_index: int, sec_month: float, sec_year: float):
    temperature = float(model.temp[source_index]) + float(model.grad[source_index]) * (
        heights - float(model.ref_hgt)
    )
    melt_temperature = np.maximum(temperature - float(model.t_melt), 0.0)
    solid_fraction = np.clip(
        1.0 - (temperature - float(model.t_solid)) / (float(model.t_liq) - float(model.t_solid)),
        0.0,
        1.0,
    )
    monthly_mass = float(model.prcp[source_index]) * solid_fraction
    monthly_mass -= float(model.monthly_melt_f) * melt_temperature
    monthly_mass -= float(model.bias) * sec_month / sec_year
    return monthly_mass / sec_month / float(model.rho)


def run(config_path: Path) -> dict[str, object]:
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel

    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_schema, generic_mode = climate_schema_names(config)
    system_id = int(config["system_id"])
    if not generic_mode and system_id != 2005:
        raise RuntimeError("wrong-system")
    required_keys = ["reconstruction_climate", "climate_pixels", "baseline_climate", "geology"]
    if not generic_mode:
        required_keys.extend(["expected_route", "audit_evidence"])
    for key in required_keys:
        path = resolve(config[key])
        if sha256(path) != str(config[f"{key}_sha256"]).upper():
            raise RuntimeError(f"{key}-hash-mismatch")
    generic_inputs = validate_hashed_paths(PROJECT, config.get("adapter_inputs", {}))

    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "config_snapshot.json", config)

    baseline_hash_before = sha256(resolve(config["baseline_climate"]))
    cloned_gdir, runtime_rgi_id, source_gdir = clone_runtime_gdir(config, output)

    cfg.initialize(logging_level="WARNING")
    runtime_dir = cloned_gdir / runtime_rgi_id[:8] / runtime_rgi_id[:-3] / runtime_rgi_id
    cfg.PARAMS["use_tar_shapefiles"] = (runtime_dir / "outlines.tar.gz").is_file()
    stage2 = config["stage2"]
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
    gdir = utils.GlacierDirectory(
        runtime_rgi_id, base_dir=cloned_gdir, reset=False
    )
    pixels = pd.read_csv(resolve(config["climate_pixels"]))
    rows = pixels[
        pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(rows) != 1:
        raise RuntimeError(f"climate-pixel-row-count:{len(rows)}")
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    suffix, climate_audit = _write_case_climate(
        gdir,
        resolve(config["reconstruction_climate"]),
        rows.iloc[0],
        "off",
        start_year=start_year,
        end_year=end_year,
    )
    climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year,
        ye=end_year,
        repeat=False,
        check_calib_params=False,
    )

    sequence_indices: list[int] = []
    sequence_years: list[int] = []
    sequence_months: list[int] = []
    years = np.asarray(model.years, dtype=np.int64)
    months = np.asarray(model.months, dtype=np.int64)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            matches = np.flatnonzero((years == year) & (months == month))
            if matches.size != 1:
                raise RuntimeError(f"nonunique-calendar-row:{year}:{month}:{matches.size}")
            sequence_indices.append(int(matches[0]))
            sequence_years.append(year)
            sequence_months.append(month)

    with Dataset(resolve(config["geology"]), "r") as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
    active_heights = (bed + thickness)[thickness > 1e-6]
    validation_heights = np.linspace(
        float(np.min(active_heights)) - 500.0,
        float(np.max(active_heights)) + 500.0,
        257,
        dtype=np.float64,
    )
    selected_positions = selected_sample_positions(len(sequence_indices))
    references = np.empty((selected_positions.size, validation_heights.size), dtype=np.float64)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for row, position in enumerate(selected_positions):
        source_index = sequence_indices[int(position)]
        year = sequence_years[int(position)] + (sequence_months[int(position)] - 1) / 12.0
        authoritative = np.asarray(model.get_monthly_mb(validation_heights, year=year), dtype=np.float64)
        portable = portable_formula(model, validation_heights, source_index, SEC_IN_MONTH, SEC_IN_YEAR)
        difference = np.abs(portable - authoritative)
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(difference)))
        maximum_relative_error = max(
            maximum_relative_error,
            float(np.max(difference / np.maximum(np.abs(authoritative), 1e-30))),
        )
        references[row] = authoritative

    package_name = (
        "monthly_ti_historical_1470_2000.npz"
        if not generic_mode
        else f"monthly_ti_historical_{start_year}_{end_year}.npz"
    )
    package_path = output / package_name
    reference_path = output / "reference_samples.npz"
    np.savez_compressed(
        package_path,
        temp=np.asarray(model.temp, dtype=np.float64),
        prcp=np.asarray(model.prcp, dtype=np.float64),
        grad=np.asarray(model.grad, dtype=np.float64),
        years=years,
        months=months,
        sequence_indices=np.asarray(sequence_indices, dtype=np.int64),
        sequence_years=np.asarray(sequence_years, dtype=np.int64),
        sequence_months=np.asarray(sequence_months, dtype=np.int64),
        ref_hgt=np.asarray(model.ref_hgt, dtype=np.float64),
        t_melt=np.asarray(model.t_melt, dtype=np.float64),
        t_solid=np.asarray(model.t_solid, dtype=np.float64),
        t_liq=np.asarray(model.t_liq, dtype=np.float64),
        monthly_melt_f=np.asarray(model.monthly_melt_f, dtype=np.float64),
        bias=np.asarray(model.bias, dtype=np.float64),
        rho=np.asarray(model.rho, dtype=np.float64),
        sec_in_month=np.asarray(SEC_IN_MONTH, dtype=np.float64),
        sec_in_year=np.asarray(SEC_IN_YEAR, dtype=np.float64),
        active_thickness_tolerance_m=np.asarray(1e-6, dtype=np.float64),
    )
    np.savez_compressed(
        reference_path,
        validation_heights_m=validation_heights,
        selected_sequence_positions=selected_positions,
        reference_monthly_mb_m_ice_s=references,
    )
    expected_months = 12 * (end_year - start_year + 1)
    checks = {
        "calendar_month_count": len(sequence_indices) == expected_months,
        "calendar_starts_at_requested_month": sequence_years[0] == start_year and sequence_months[0] == 1,
        "calendar_ends_at_requested_month": sequence_years[-1] == end_year and sequence_months[-1] == 12,
        "calendar_rows_unique": len(set(sequence_indices)) == expected_months,
        "formula_absolute_error": maximum_absolute_error <= 1e-18,
        "formula_relative_error": maximum_relative_error <= 1e-10,
        "all_arrays_finite": all(
            np.isfinite(np.asarray(value)).all()
            for value in (model.temp, model.prcp, model.grad, references)
        ),
        "production_source_unchanged": sha256(resolve(config["reconstruction_climate"]))
        == str(config["reconstruction_climate_sha256"]).upper(),
        "baseline_source_unchanged": sha256(resolve(config["baseline_climate"]))
        == baseline_hash_before
        == str(config["baseline_climate_sha256"]).upper(),
    }
    summary = {
        "schema": summary_schema,
        "status": "historical-climate-bridge-pass" if all(checks.values()) else "historical-climate-bridge-fail",
        "system_id": system_id,
        "calendar": [start_year, end_year],
        "sequence_month_count": len(sequence_indices),
        "stage2": stage2,
        "climate_pixel": rows.iloc[0].to_dict(),
        "source_climate_file": str(climate_path),
        "source_climate_sha256": sha256(climate_path),
        "reconstruction_climate": str(resolve(config["reconstruction_climate"])),
        "reconstruction_climate_sha256": sha256(resolve(config["reconstruction_climate"])),
        "package_file": str(package_path),
        "package_sha256": sha256(package_path),
        "reference_file": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "maximum_formula_absolute_error_m_ice_s": maximum_absolute_error,
        "maximum_formula_relative_error": maximum_relative_error,
        "climate_audit": climate_audit,
        "runtime_rgi_id": runtime_rgi_id,
        "source_gdir": str(source_gdir),
        "checks": checks,
        "dynamics_run": False,
        "production_temperature_modified": False,
        "production_source_temperature_file_modified": False,
        "forcing_package_registered_delta_t_c": float(stage2["delta_t_c"]),
        "initial_thickness_modified": False,
    }
    if generic_mode:
        summary["adapter_inputs"] = generic_inputs
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run(resolve(args.config))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if summary["status"] != "historical-climate-bridge-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
