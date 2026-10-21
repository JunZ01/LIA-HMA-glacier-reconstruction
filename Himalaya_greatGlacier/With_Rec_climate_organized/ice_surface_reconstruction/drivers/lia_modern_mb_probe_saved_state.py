"""Continue a saved LIA candidate state from 2000 to 2020.

This is the generic B-window counterpart of ``lia_modern_mb_probe.py``.  It
starts from a candidate directory containing ``summary.json`` and
``state_2000.pkl.gz`` instead of looking up a D-layer ``system_id/point`` row.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path
from .lia_modern_mb_probe import _write_modern_climate


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_label(candidate_dir: Path, explicit: str | None) -> str:
    if explicit:
        label = explicit
    else:
        parts = candidate_dir.parts[-3:]
        label = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", label).strip("_")


def _parameter(summary: dict, *names: str) -> float:
    parameters = summary.get("parameters", {})
    for name in names:
        if name in parameters:
            return float(parameters[name])
    raise KeyError(f"missing-parameter:{'/'.join(names)}")


def _prcp_fac(summary: dict) -> float:
    parameters = summary.get("parameters", {})
    if "prcp_fac" in parameters:
        return float(parameters["prcp_fac"])
    return float(parameters["prcp_fac_initial"]) * float(parameters.get("prcp_fac_multiplier", 1.0))


def _resolve_existing_path(raw_path: str, candidate_dir: Path) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(candidate_dir / path)
        candidates.extend(parent / path for parent in candidate_dir.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = [str(candidate) for candidate in candidates[:8]]
    raise FileNotFoundError(f"could-not-resolve-path:{raw_path}; tried={tried}")


def _source_gdir(summary: dict, candidate_dir: Path, override: Path | None) -> Path:
    if override is not None:
        source = override
    else:
        climate = summary.get("climate", {})
        forward = climate.get("forward", {})
        spinup = climate.get("spinup", {})
        climate_file = forward.get("climate_file") or spinup.get("climate_file")
        if not climate_file:
            raise KeyError("missing-climate-file-in-summary")
        source = _resolve_existing_path(str(climate_file), candidate_dir).parent
    if not source.exists():
        raise FileNotFoundError(f"source-gdir-missing:{source}")
    if not (source / "climate_historical.nc").exists():
        raise FileNotFoundError(f"source-gdir-missing-climate_historical:{source}")
    return source


def _geometry(model) -> dict:
    return {
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
    }


def _load_state(candidate_dir: Path) -> list:
    state_path = candidate_dir / "state_2000.pkl.gz"
    if not state_path.exists():
        raise FileNotFoundError(f"missing-state-2000:{state_path}")
    with gzip.open(state_path, "rb") as stream:
        flowlines = pickle.load(stream)
    if not flowlines:
        raise RuntimeError("empty-2000-state")
    return flowlines


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    candidate_dir = args.candidate_dir.resolve()
    args.workdir = args.workdir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.source_gdir is not None:
        args.source_gdir = args.source_gdir.resolve()
    summary_path = candidate_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing-summary:{summary_path}")

    summary = _read_json(summary_path)
    label = _candidate_label(candidate_dir, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    version = _configure_oggm(args.workdir, args.output_dir)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    rgi_id = str(summary["rgi_id"])
    source = _source_gdir(summary, candidate_dir, args.source_gdir)
    target = _gdir_path(args.workdir / "per_glacier", rgi_id)
    _clone_gdir(source, target)
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    climate = _write_modern_climate(gdir)

    delta_t = _parameter(summary, "delta_t_c", "delta_t")
    modern_delta_t = delta_t if args.modern_delta_t is None else float(args.modern_delta_t)
    base_temp_bias = _parameter(summary, "base_temp_bias_c", "base_temp_bias")
    modern_temp_bias = base_temp_bias + modern_delta_t
    melt_f = _parameter(summary, "melt_f")
    prcp_fac = _prcp_fac(summary)
    glen_a = _parameter(summary, "glen_a_pa_3_s_1", "glen_a")
    candidate_temp_melt = _parameter(summary, "temp_melt_c", "temp_melt")
    modern_temp_melt = (
        candidate_temp_melt
        if args.modern_temp_melt is None
        else float(args.modern_temp_melt)
    )

    cfg.PARAMS["temp_melt"] = modern_temp_melt
    mb = MonthlyTIModel(
        gdir,
        ys=2000,
        ye=2020,
        repeat=False,
        filename="gcm_data",
        input_filesuffix="_modern_2000_2020",
        melt_f=melt_f,
        temp_bias=modern_temp_bias,
        prcp_fac=prcp_fac,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(fl) for fl in _load_state(candidate_dir)],
        mb_model=mb,
        y0=2000.0,
        glen_a=glen_a,
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )

    initial = {"year": 2000, **_geometry(model)}
    annual_records = []
    annual_smb = []
    for year in range(2000, 2020):
        before = _geometry(model)
        surface_mb = float(mb.get_specific_mb(fls=model.fls, year=float(year)))
        annual_smb.append(surface_mb)
        model.run_until(float(year + 1))
        after = _geometry(model)
        annual_records.append(
            {
                "year": year,
                "surface_mb_kg_m2_yr": surface_mb,
                "start_area_m2": before["area_m2"],
                "start_volume_m3": before["volume_m3"],
                "start_length_m": before["length_m"],
                "end_area_m2": after["area_m2"],
                "end_volume_m3": after["volume_m3"],
                "end_length_m": after["length_m"],
            }
        )

    final = {"year": 2020, **_geometry(model)}
    mean_area = 0.5 * (initial["area_m2"] + final["area_m2"])
    geodetic_mb = 900.0 * (final["volume_m3"] - initial["volume_m3"]) / (mean_area * 20.0)
    result = {
        "label": label,
        "candidate_dir": str(candidate_dir),
        "candidate_status": str(summary.get("status", "")),
        "lia_unique_id": int(summary["lia_unique_id"]),
        "rgi_id": rgi_id,
        "layer": summary.get("layer"),
        "spinup_window": summary.get("spinup_window"),
        "start_year": summary.get("start_year"),
        "spinup_control": summary.get("spinup_control"),
        "targets": summary.get("targets"),
        "candidate_metrics": summary.get("metrics"),
        "delta_t_c": delta_t,
        "modern_delta_t_c": modern_delta_t,
        "base_temp_bias_c": base_temp_bias,
        "modern_temp_bias_c": modern_temp_bias,
        "melt_f": melt_f,
        "prcp_fac": prcp_fac,
        "glen_a": glen_a,
        "temp_melt_c": candidate_temp_melt,
        "candidate_temp_melt_c": candidate_temp_melt,
        "modern_temp_melt_c": modern_temp_melt,
        "initial_2000": initial,
        "final_2020": final,
        "volume_change_m3": final["volume_m3"] - initial["volume_m3"],
        "area_change_m2": final["area_m2"] - initial["area_m2"],
        "length_change_m": final["length_m"] - initial["length_m"],
        "model_geodetic_mb_kg_m2_yr": float(geodetic_mb),
        "model_surface_mb_mean_kg_m2_yr": float(np.mean(annual_smb)),
        "model_surface_mb_min_kg_m2_yr": float(np.min(annual_smb)),
        "model_surface_mb_max_kg_m2_yr": float(np.max(annual_smb)),
        "climate": climate,
        "source_gdir": str(source),
        "work_gdir": str(target),
        "python_executable": sys.executable,
        "oggm_version": version,
        "runtime_seconds": time.perf_counter() - started,
        "timeseries_file": str(args.output_dir / f"{label}_annual_timeseries.csv"),
    }

    pd.DataFrame(annual_records).to_csv(args.output_dir / f"{label}_annual_timeseries.csv", index=False)
    (args.output_dir / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--source-gdir", type=Path, default=None)
    parser.add_argument("--modern-delta-t", type=float, default=None)
    parser.add_argument("--modern-temp-melt", type=float, default=None)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

