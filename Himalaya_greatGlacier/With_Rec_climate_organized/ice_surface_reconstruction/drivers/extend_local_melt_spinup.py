"""Continue saved local-melt chains from spin-up year 500 to year 1000.

The script deliberately reuses the saved year-500 flowline state and the
already-written case climate. It then reruns the forward period from the
year-1000 state so the effect of the longer spin-up is isolated.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


def configure_oggm(workdir: Path, output_dir: Path) -> None:
    import oggm
    from oggm import cfg

    for directory in (
        workdir,
        output_dir / "oggm_cache",
        output_dir / "oggm_tmp",
        output_dir / "oggm_rgi",
        output_dir / "oggm_tests",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(workdir)
    cfg.PATHS["dl_cache_dir"] = str(output_dir / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_dir / "oggm_tmp")
    cfg.PATHS["rgi_dir"] = str(output_dir / "oggm_rgi")
    cfg.PATHS["test_dir"] = str(output_dir / "oggm_tests")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["border"] = 320
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False


def snapshot(model, phase: str, year: int) -> dict:
    return {
        "phase": phase,
        "year": int(year),
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
        "terminal_thickness_m": float(model.fls[-1].thick[-1]),
        "minimum_thickness_m": float(np.min(model.fls[-1].thick)),
    }


def run_one(source_dir: Path, output_dir: Path, workdir: Path, system_id: int) -> dict:
    started = time.perf_counter()
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    params = summary["parameters"]
    rgi_id = str(summary["rgi_id"])
    configure_oggm(workdir, output_dir)

    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(params["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    spin_suffix = "_batch1_1807_1836_prcp_off"
    forward_suffix = "_batch1_1836_2000_prcp_off"
    for suffix in (spin_suffix, forward_suffix):
        if not Path(gdir.get_filepath("gcm_data", filesuffix=suffix)).exists():
            raise FileNotFoundError(f"missing-climate:{gdir.get_filepath('gcm_data', filesuffix=suffix)}")

    with gzip.open(source_dir / "spinup_state.pkl.gz", "rb") as stream:
        saved_fls = pickle.load(stream)
    if not isinstance(saved_fls, list) or not saved_fls:
        raise ValueError("invalid-spinup-state")

    common = {
        "filename": "gcm_data",
        "melt_f": float(params["melt_f"]),
        "temp_bias": float(params["base_temp_bias_c"]) + float(params["delta_t_c"]),
        "prcp_fac": float(params["prcp_fac"]),
        "check_calib_params": False,
    }
    spin_mb = MonthlyTIModel(
        gdir,
        input_filesuffix=spin_suffix,
        ys=1807,
        ye=1836,
        repeat=True,
        **common,
    )
    spin_model = SemiImplicitModel(
        [copy.deepcopy(fl) for fl in saved_fls],
        mb_model=spin_mb,
        y0=500.0,
        glen_a=float(params["glen_a_pa_3_s_1"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    rows = []
    for year in range(600, 1001, 100):
        spin_model.run_until(float(year))
        rows.append(snapshot(spin_model, "spinup", year))

    forward_mb = MonthlyTIModel(
        gdir,
        input_filesuffix=forward_suffix,
        ys=1836,
        ye=2000,
        repeat=False,
        **common,
    )
    forward_model = SemiImplicitModel(
        [copy.deepcopy(fl) for fl in spin_model.fls],
        mb_model=forward_mb,
        y0=1836.0,
        glen_a=float(params["glen_a_pa_3_s_1"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    rows.append(snapshot(forward_model, "forward", 1836))
    for year in range(1846, 2001, 10):
        forward_model.run_until(float(year))
        rows.append(snapshot(forward_model, "forward", year))

    old_rows = pd.read_csv(source_dir / "timeseries.csv")
    old_spin = old_rows[old_rows["phase"].eq("spinup") & (old_rows["year"] <= 500)]
    new_rows = pd.DataFrame(rows)
    for frame in (old_spin, new_rows):
        for key, value in {
            "lia_unique_id": system_id,
            "rgi_id": rgi_id,
            "delta_t_c": params["delta_t_c"],
            "melt_f": params["melt_f"],
            "prcp_fac": params["prcp_fac"],
            "glen_a_pa_3_s_1": params["glen_a_pa_3_s_1"],
            "temp_melt_c": params["temp_melt_c"],
        }.items():
            if key not in frame:
                frame[key] = value
    combined = pd.concat([old_spin, new_rows], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "timeseries.csv", index=False)
    with gzip.open(output_dir / "spinup_state_1000.pkl.gz", "wb") as stream:
        pickle.dump(spin_model.fls, stream)
    with gzip.open(output_dir / "state_2000.pkl.gz", "wb") as stream:
        pickle.dump(forward_model.fls, stream)

    spin_final = rows[4]
    forward_final = rows[-1]
    lia_target = float(summary["targets"]["lia_length_m"])
    rgi_target = float(summary["targets"]["rgi_length_m"])
    last_200 = [row for row in rows if row["phase"] == "spinup" and row["year"] >= 800]
    first_last = last_200[0]
    last = last_200[-1]
    result = {
        "status": "ok",
        "system_id": system_id,
        "rgi_id": rgi_id,
        "spinup_years": 1000,
        "source_spinup_years": 500,
        "parameters": params,
        "targets": summary["targets"],
        "metrics": {
            "l_spinup_1000_m": spin_final["length_m"],
            "l_2000_m": forward_final["length_m"],
            "e_lia_1000": abs(spin_final["length_m"] - lia_target) / lia_target,
            "e_2000": abs(forward_final["length_m"] - rgi_target) / rgi_target,
            "volume_spinup_1000_m3": spin_final["volume_m3"],
            "volume_2000_m3": forward_final["volume_m3"],
            "area_spinup_1000_m2": spin_final["area_m2"],
            "area_2000_m2": forward_final["area_m2"],
            "last_200y_length_change_m": last["length_m"] - first_last["length_m"],
            "last_200y_volume_change_fraction": (
                last["volume_m3"] - first_last["volume_m3"]
            ) / max(first_last["volume_m3"], 1.0),
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    (output_dir / "summary_1000.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--system-id", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(run_one(args.source_dir, args.output_dir, args.workdir, args.system_id), indent=2))


if __name__ == "__main__":
    main()
