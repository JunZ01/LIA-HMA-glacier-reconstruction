"""Continue a saved stable LIA spin-up state with an independent forward delta-T.

This driver is deliberately narrow: it does not rerun or alter spin-up, geometry,
or physical parameters.  It clones the source glacier directory, reuses the saved
``spinup_state.pkl.gz``, changes only the constant temperature bias used during
1470-2000, and scores the resulting length curve with the formal controller gates.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path
from .lia_lhs_single_chain_legacy_20260813 import _snapshot
from .lia_observation_domain import observation_domain_metrics


def _suffix_from_climate_path(path: Path) -> str:
    name = path.name
    if not name.startswith("gcm_data") or path.suffix != ".nc":
        raise ValueError(f"unexpected-climate-file:{path}")
    return name[len("gcm_data") : -len(".nc")]


def _source_task_path(source: Path) -> Path:
    for name in ("task.json", "source_task.json"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing-source-task-metadata:{source}")


def _task_option(task: dict, name: str) -> str | None:
    command = [str(value) for value in task.get("command", [])]
    try:
        index = command.index(name)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError(f"task-option-without-value:{name}")
    return command[index + 1]


def _observation_domain(source_dir: Path, summary: dict, task: dict) -> dict:
    domain = summary.get("observation_domain") or {}
    node_count = domain.get("node_count")
    source = domain.get("source")
    if node_count is None:
        root = _task_option(task, "--elevation-band-root")
        if root is None:
            raise ValueError("missing-observation-domain-source")
        audit_path = Path(root) / str(summary["lia_unique_id"]) / "elevation_band_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        node_count = audit.get("observation_domain_node_count", audit.get("ice_node_count"))
        source = str(audit_path.resolve())
    node_count = int(node_count)
    if node_count <= 0:
        raise ValueError("invalid-observation-domain-node-count")
    return {
        "node_count": node_count,
        "source": source,
        "definition": (
            "declared mapped observation-domain nodes; any >10 m downstream ice "
            "invalidates the length observation"
        ),
    }


def _source_forward_series(source_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_dir / "timeseries.csv")
    frame = frame[frame["phase"].astype(str).eq("forward")].copy()
    if frame.empty:
        raise ValueError("source-forward-series-empty")
    if "observation_length_m" not in frame.columns:
        frame["observation_length_m"] = pd.to_numeric(frame["length_m"], errors="coerce")
    return frame


def _baseline_reproduction(source_dir: Path, candidate: pd.DataFrame) -> dict:
    source = _source_forward_series(source_dir)
    joined = source.merge(candidate, on="year", suffixes=("_source", "_new"))
    if len(joined) != len(source):
        return {
            "pass": False,
            "reason": "checkpoint-count-mismatch",
            "source_rows": int(len(source)),
            "matched_rows": int(len(joined)),
        }
    length_diff = np.abs(
        pd.to_numeric(joined["observation_length_m_source"], errors="coerce")
        - pd.to_numeric(joined["observation_length_m_new"], errors="coerce")
    )
    area_rel = np.abs(
        pd.to_numeric(joined["area_m2_new"], errors="coerce")
        / pd.to_numeric(joined["area_m2_source"], errors="coerce")
        - 1.0
    )
    volume_rel = np.abs(
        pd.to_numeric(joined["volume_m3_new"], errors="coerce")
        / pd.to_numeric(joined["volume_m3_source"], errors="coerce")
        - 1.0
    )
    result = {
        "pass": bool(
            np.isfinite(length_diff).all()
            and float(length_diff.max()) <= 50.0
            and float(area_rel.max()) <= 5e-4
            and float(volume_rel.max()) <= 5e-4
        ),
        "max_length_difference_m": float(length_diff.max()),
        "max_area_relative_difference": float(area_rel.max()),
        "max_volume_relative_difference": float(volume_rel.max()),
        "length_tolerance_m": 50.0,
        "area_volume_relative_tolerance": 5e-4,
    }
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    workdir = args.workdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    task = json.loads(_source_task_path(source_dir).read_text(encoding="utf-8"))
    if str(summary.get("status")) != "ok":
        raise ValueError(f"source-status:{summary.get('status')}")
    spin_control = summary.get("spinup_control") or {}
    if not bool(spin_control.get("converged")):
        raise ValueError("source-spinup-not-converged")
    if float(spin_control.get("final_stability_fraction", math.inf)) > 0.02:
        raise ValueError("source-spinup-stability-fails")
    if "prcp_off" not in str(summary["climate"]["forward"]["source"]):
        raise ValueError("source-precipitation-mode-not-off")

    with gzip.open(source_dir / "spinup_state.pkl.gz", "rb") as stream:
        spinup_state = pickle.load(stream)

    domain = _observation_domain(source_dir, summary, task)
    source_forward = Path(summary["climate"]["forward"]["climate_file"])
    if not source_forward.exists():
        raise FileNotFoundError(f"missing-source-forward-climate:{source_forward}")
    source_gdir = source_forward.parent
    rgi_id = str(summary["rgi_id"])
    target_gdir = _gdir_path(workdir / "per_glacier", rgi_id)
    _clone_gdir(source_gdir, target_gdir)
    _configure_oggm(workdir, output_dir)

    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    parameters = summary["parameters"]
    cfg.PARAMS["temp_melt"] = float(parameters["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    suffix = _suffix_from_climate_path(source_forward)
    cloned_climate = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if not cloned_climate.exists():
        raise FileNotFoundError(f"missing-cloned-forward-climate:{cloned_climate}")

    base_temp_bias = float(parameters["base_temp_bias_c"])
    forward_mb = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        ys=int(summary["start_year"]),
        ye=2000,
        repeat=False,
        melt_f=float(parameters["melt_f"]),
        temp_bias=base_temp_bias + float(args.forward_delta_t),
        prcp_fac=float(parameters["prcp_fac"]),
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline) for flowline in spinup_state],
        mb_model=forward_mb,
        y0=float(summary["start_year"]),
        glen_a=float(parameters["glen_a_pa_3_s_1"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )

    start_year = int(summary["start_year"])
    checkpoints = [start_year, *range(start_year + 10, 2001, 10)]
    if checkpoints[-1] != 2000:
        checkpoints.append(2000)
    rows: list[dict] = []
    for year in checkpoints:
        if year != start_year:
            model.run_until(float(year))
        row = _snapshot(model, "forward", year)
        row["raw_model_length_m"] = float(row["length_m"])
        row.update(
            observation_domain_metrics(
                model.fls[-1].thick,
                model.fls[-1].dx_meter,
                domain["node_count"],
            )
        )
        row.update(
            {
                "lia_unique_id": int(summary["lia_unique_id"]),
                "rgi_id": rgi_id,
                "spinup_delta_t_c": float(parameters["delta_t_c"]),
                "forward_delta_t_c": float(args.forward_delta_t),
                "melt_f": float(parameters["melt_f"]),
                "prcp_fac": float(parameters["prcp_fac"]),
                "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
                "temp_melt_c": float(parameters["temp_melt_c"]),
            }
        )
        rows.append(row)

    series = pd.DataFrame(rows)
    series.to_csv(output_dir / "timeseries.csv", index=False)
    with gzip.open(output_dir / "state_2000.pkl.gz", "wb") as stream:
        pickle.dump(model.fls, stream)

    valid = not bool(series["observation_domain_overflow"].astype(bool).any())
    finite = bool(
        np.isfinite(series[["area_m2", "volume_m3", "raw_model_length_m"]].to_numpy()).all()
    )
    observed = pd.to_numeric(series["observation_length_m"], errors="coerce")
    lia_target = float(summary["targets"]["lia_length_m"])
    rgi_target = float(summary["targets"]["rgi_length_m"])
    separation = abs(lia_target - rgi_target)
    if separation <= 0:
        raise ValueError("nonpositive-lia-rgi-separation")
    peak = float(observed.max()) if valid else math.nan
    endpoint = float(observed.iloc[-1]) if valid else math.nan
    e_lia_sep = abs(peak - lia_target) / separation if valid else math.nan
    e_rgi_sep = abs(endpoint - rgi_target) / separation if valid else math.nan
    e_lia_total = abs(peak - lia_target) / lia_target if valid else math.nan
    e_rgi_total = abs(endpoint - rgi_target) / rgi_target if valid else math.nan
    accepted = bool(
        finite
        and valid
        and e_lia_sep <= 0.2
        and e_rgi_sep <= 0.2
        and e_lia_total <= 0.1
        and e_rgi_total <= 0.1
    )

    source_delta = float(parameters["delta_t_c"])
    baseline = None
    if math.isclose(float(args.forward_delta_t), source_delta, abs_tol=1e-12):
        baseline = _baseline_reproduction(source_dir, series)
        if not baseline["pass"]:
            accepted = False

    output = {
        "status": "ok" if finite else "technical_failure",
        "experiment": "saved_spinup_independent_forward_delta_t",
        "python_executable": sys.executable,
        "oggm_version": str(__import__("oggm").__version__),
        "lia_unique_id": int(summary["lia_unique_id"]),
        "rgi_id": rgi_id,
        "source_task_id": task.get("task_id"),
        "source_dir": str(source_dir),
        "spinup_state_reused": str(source_dir / "spinup_state.pkl.gz"),
        "source_spinup_stability_fraction": float(spin_control["final_stability_fraction"]),
        "start_year": start_year,
        "end_year": 2000,
        "parameters": {
            "spinup_delta_t_c": source_delta,
            "forward_delta_t_c": float(args.forward_delta_t),
            "base_temp_bias_c": base_temp_bias,
            "melt_f": float(parameters["melt_f"]),
            "prcp_fac": float(parameters["prcp_fac"]),
            "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
            "temp_melt_c": float(parameters["temp_melt_c"]),
            "precipitation_anomalies": "off",
        },
        "observation_domain": {
            **domain,
            "assessed": True,
            "overflow": not valid,
        },
        "targets": {
            "lia_length_m": lia_target,
            "rgi_length_m": rgi_target,
            "separation_m": separation,
        },
        "metrics": {
            "forward_peak_m": peak,
            "forward_peak_year": (
                int(series.loc[observed.idxmax(), "year"]) if valid else None
            ),
            "forward_2000_m": endpoint,
            "e_lia_sep": e_lia_sep,
            "e_rgi_sep": e_rgi_sep,
            "objective_sep": e_lia_sep + e_rgi_sep if valid else math.nan,
            "e_lia_total": e_lia_total,
            "e_rgi_total": e_rgi_total,
            "total_length_pass": bool(valid and e_lia_total <= 0.1 and e_rgi_total <= 0.1),
            "formal_separation_pass": bool(valid and e_lia_sep <= 0.2 and e_rgi_sep <= 0.2),
            "formal_dual_pass": accepted,
        },
        "baseline_reproduction": baseline,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=True), encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--forward-delta-t", required=True, type=float)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "technical_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
