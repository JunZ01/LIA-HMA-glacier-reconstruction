"""Export the authoritative system-2005 MonthlyTIModel state for PyTorch.

The export is produced by the existing Windows OGGM environment.  It contains
only already bias/factor-corrected MonthlyTIModel arrays, scalar parameters,
the exact 50-year repeated-month index sequence, and deterministic OGGM
reference evaluations.  No glacier dynamics are run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT = Path(__file__).resolve().parents[4]
SYSTEM_ID = 2005


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_formula(model, heights: np.ndarray, source_index: int, sec_month: float, sec_year: float):
    itemp = float(model.temp[source_index])
    iprcp = float(model.prcp[source_index])
    igrad = float(model.grad[source_index])
    temperature = itemp + igrad * (heights - float(model.ref_hgt))
    temperature_for_melt = np.maximum(temperature - float(model.t_melt), 0.0)
    solid_fraction = np.clip(
        1.0
        - (temperature - float(model.t_solid))
        / (float(model.t_liq) - float(model.t_solid)),
        0.0,
        1.0,
    )
    solid_precipitation = iprcp * solid_fraction
    monthly_mass = solid_precipitation - float(model.monthly_melt_f) * temperature_for_melt
    monthly_mass -= float(model.bias) * sec_month / sec_year
    return monthly_mass / sec_month / float(model.rho)


def run(config_path: Path, geology_path: Path, output: Path) -> dict[str, object]:
    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH, SEC_IN_YEAR
    from oggm.core.massbalance import MonthlyTIModel
    from oggm.utils import floatyear_to_date

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["system_id"]) != SYSTEM_ID:
        raise RuntimeError("wrong-system-config")
    if output.exists():
        raise FileExistsError(output)

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    stage2 = config["stage2"]
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=resolve(config["inversion_root"]) / "synthetic_gdir",
        reset=False,
    )
    model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=config["climate_filesuffix"],
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=int(config["spinup_window"][0]),
        ye=int(config["spinup_window"][1]),
        repeat=True,
        check_calib_params=False,
    )

    sequence_indices: list[int] = []
    sequence_years: list[int] = []
    sequence_months: list[int] = []
    for month_index in range(50 * 12):
        model_year = month_index / 12.0
        year, month = floatyear_to_date(model_year)
        repeated_year = model.ys + (year - model.ys) % (model.ye - model.ys + 1)
        matches = np.flatnonzero((np.asarray(model.years) == repeated_year) & (np.asarray(model.months) == month))
        if matches.size != 1:
            raise RuntimeError(f"nonunique-climate-row:{month_index}:{repeated_year}:{month}")
        sequence_indices.append(int(matches[0]))
        sequence_years.append(int(repeated_year))
        sequence_months.append(int(month))

    with Dataset(geology_path, "r") as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
    surface = bed + thickness
    active_flat_indices = np.flatnonzero((thickness > 1e-6).ravel())
    active_heights = surface.ravel()[active_flat_indices]
    if active_heights.size == 0:
        raise RuntimeError("no-active-grid-heights")

    validation_heights = np.linspace(
        float(np.min(active_heights)) - 500.0,
        float(np.max(active_heights)) + 500.0,
        2049,
        dtype=np.float64,
    )
    reference = np.empty((len(sequence_indices), validation_heights.size), dtype=np.float64)
    maximum_formula_absolute_error = 0.0
    maximum_formula_relative_error = 0.0
    for month_index, source_index in enumerate(sequence_indices):
        model_year = month_index / 12.0
        authoritative = np.asarray(model.get_monthly_mb(validation_heights, year=model_year), dtype=np.float64)
        portable = portable_formula(
            model,
            validation_heights,
            source_index,
            float(SEC_IN_MONTH),
            float(SEC_IN_YEAR),
        )
        difference = np.abs(portable - authoritative)
        maximum_formula_absolute_error = max(maximum_formula_absolute_error, float(np.max(difference)))
        denominator = np.maximum(np.abs(authoritative), 1e-30)
        maximum_formula_relative_error = max(
            maximum_formula_relative_error,
            float(np.max(difference / denominator)),
        )
        reference[month_index] = authoritative

    selected_month_indices = np.asarray([0, 1, 5, 11, 12, 119, 359, 599], dtype=np.int64)
    active_reference = np.empty((selected_month_indices.size, active_heights.size), dtype=np.float64)
    for row, month_index in enumerate(selected_month_indices):
        active_reference[row] = model.get_monthly_mb(active_heights, year=float(month_index) / 12.0)

    climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=config["climate_filesuffix"]))
    output.mkdir(parents=True, exist_ok=False)
    package_path = output / "monthly_ti_package.npz"
    reference_path = output / "reference_samples.npz"
    np.savez_compressed(
        package_path,
        temp=np.asarray(model.temp, dtype=np.float64),
        prcp=np.asarray(model.prcp, dtype=np.float64),
        grad=np.asarray(model.grad, dtype=np.float64),
        years=np.asarray(model.years, dtype=np.int64),
        months=np.asarray(model.months, dtype=np.int64),
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
        reference_monthly_mb_m_ice_s=reference,
        active_flat_indices=active_flat_indices.astype(np.int64),
        active_surface_heights_m=active_heights,
        selected_month_indices=selected_month_indices,
        active_reference_monthly_mb_m_ice_s=active_reference,
    )

    checks = {
        "source_formula_absolute_error": maximum_formula_absolute_error <= 1e-18,
        "source_formula_relative_error": maximum_formula_relative_error <= 1e-10,
        "sequence_has_600_months": len(sequence_indices) == 600,
        "all_export_arrays_finite": all(
            np.isfinite(np.asarray(value)).all()
            for value in (model.temp, model.prcp, model.grad, reference, active_reference)
        ),
        "package_written": package_path.exists(),
        "reference_written": reference_path.exists(),
    }
    summary = {
        "schema": "lia-191-system2005-monthly-ti-pytorch-bridge-export-v1",
        "status": "climate-bridge-export-pass" if all(checks.values()) else "climate-bridge-export-fail",
        "system_id": SYSTEM_ID,
        "oggm_version": __import__("oggm").__version__,
        "spinup_window": config["spinup_window"],
        "climate_filesuffix": config["climate_filesuffix"],
        "stage2": stage2,
        "effective_temp_bias_c": float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        "climate_source_file": str(climate_path),
        "climate_source_sha256": sha256(climate_path),
        "geology_source_file": str(geology_path),
        "geology_source_sha256": sha256(geology_path),
        "package_file": str(package_path),
        "package_sha256": sha256(package_path),
        "reference_file": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "source_climate_row_count": int(np.asarray(model.temp).size),
        "sequence_month_count": len(sequence_indices),
        "validation_height_count": int(validation_heights.size),
        "active_grid_height_count": int(active_heights.size),
        "maximum_formula_absolute_error_m_ice_s": maximum_formula_absolute_error,
        "maximum_formula_relative_error": maximum_formula_relative_error,
        "checks": checks,
        "dynamics_run": False,
        "steady_run": False,
        "historical_run": False,
    }
    (output / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = run(resolve(arguments.config), resolve(arguments.geology), resolve(arguments.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "climate-bridge-export-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
