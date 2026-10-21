"""Exactly-50-year repeated-LIA-climate equilibrium gate for system 3850."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _snapshot(model, year: int, registered_length_m: float) -> dict:
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"steady-root-count:{len(roots)}")
    root = roots[0]
    thick = np.asarray(root.thick, dtype=float)
    active = np.flatnonzero(thick > 0)
    last_active = int(active[-1]) if active.size else -1
    free_tail_m = float((root.nx - last_active - 1) * root.dx_meter)
    finite = bool(
        np.isfinite(model.volume_m3)
        and np.isfinite(model.area_m2)
        and np.isfinite(model.length_m)
        and all(
            np.isfinite(flowline.section).all()
            and np.isfinite(flowline.thick).all()
            for flowline in model.fls
        )
    )
    length_m = float(model.length_m)
    return {
        "year": int(year),
        "volume_m3": float(model.volume_m3),
        "area_m2": float(model.area_m2),
        "length_m": length_m,
        "length_relative_error_lia": float(
            abs(length_m - registered_length_m) / registered_length_m
        ),
        "root_last_active_index": last_active,
        "root_last_active_thickness_m": (
            None if last_active < 0 else float(thick[last_active])
        ),
        "root_array_end_thickness_m": float(thick[-1]),
        "remaining_free_downstream_m": free_tail_m,
        "finite_geometry": finite,
        "negative_section_count": int(
            sum(
                np.count_nonzero(np.asarray(flowline.section) < -1e-10)
                for flowline in model.fls
            )
        ),
    }


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inversion_root = resolve(config["inversion_root"])
    zero_mb_root = resolve(config["zero_mb_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", config)
    inversion = json.loads(
        (inversion_root / "inversion_summary_formal.json").read_text(encoding="utf-8")
    )
    if inversion.get("status") != "inversion-pass":
        raise RuntimeError("registered-inversion-not-pass")
    zero_mb = json.loads(
        (zero_mb_root / "zero_mb_summary_formal.json").read_text(encoding="utf-8")
    )
    numerical_flags = set(zero_mb.get("review_flags", [])) - {
        "observation-domain-overflow"
    }
    if numerical_flags:
        raise RuntimeError(f"zero-mb-numerical-gates-not-pass:{sorted(numerical_flags)}")

    from oggm import cfg, utils
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=inversion_root / "synthetic_gdir",
        reset=False,
    )
    suffix = str(config["climate_filesuffix"])
    climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if not climate_path.exists():
        raise FileNotFoundError(f"steady-climate-missing:{climate_path}")
    start_year, end_year = [int(value) for value in config["spinup_window"]]
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year,
        ye=end_year,
        repeat=True,
        check_calib_params=False,
    )
    flowlines = gdir.read_pickle("model_flowlines")
    model = FluxBasedModel(
        copy.deepcopy(flowlines),
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(stage2["glen_a"]),
        fs=float(stage2["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    registered_length = float(config["registered_lia_length_m"])
    rows = [_snapshot(model, 0, registered_length)]
    runtime_error = None
    failure_traceback = None
    try:
        for year in range(1, 51):
            model.run_until(float(year))
            rows.append(_snapshot(model, year, registered_length))
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "steady_50y_timeseries.csv", index=False)
    initial = rows[0]
    final = rows[-1]
    volume_change = float(
        (final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"]
    )
    area_change = float(
        (final["area_m2"] - initial["area_m2"]) / initial["area_m2"]
    )
    length_change = float(final["length_m"] - initial["length_m"])
    late_start = rows[-11] if len(rows) >= 11 else rows[0]
    late_volume_change = float(
        (final["volume_m3"] - late_start["volume_m3"])
        / late_start["volume_m3"]
    )
    flags = []
    if runtime_error:
        lower = runtime_error.lower()
        flags.append(
            "cfl-error"
            if "cfl" in lower
            else "boundary-error"
            if "boundar" in lower
            else "numerical-error"
        )
    if int(final["year"]) != 50:
        flags.append("50y-incomplete")
    if abs(volume_change) > float(config["gates"]["volume_change_fraction_max"]):
        flags.append("50y-volume-change-over-2pct")
    if float(final["length_relative_error_lia"]) > float(
        config["gates"]["lia_length_relative_error_max"]
    ):
        flags.append("lia-length-error-over-10pct")
    if float(final["remaining_free_downstream_m"]) < float(
        config["gates"]["minimum_free_downstream_m"]
    ):
        flags.append("remaining-free-domain-under-5km")
    if not bool(final["finite_geometry"]):
        flags.append("nonfinite-geometry")
    if int(final["negative_section_count"]):
        flags.append("negative-section")
    summary = {
        "status": "steady-50y-pass" if not flags else "steady-50y-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "solver": "FluxBasedModel",
        "climate": "repeated registered B-window MonthlyTIModel",
        "spinup_window": [start_year, end_year],
        "initial": initial,
        "final": final,
        "volume_change_fraction_50y": volume_change,
        "area_change_fraction_50y": area_change,
        "length_change_m_50y": length_change,
        "volume_change_fraction_last_10y": late_volume_change,
        "review_flags": flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "historical_run": False,
    }
    if not flags:
        with gzip.open(output_root / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(output_root / "steady_50y_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve()),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
