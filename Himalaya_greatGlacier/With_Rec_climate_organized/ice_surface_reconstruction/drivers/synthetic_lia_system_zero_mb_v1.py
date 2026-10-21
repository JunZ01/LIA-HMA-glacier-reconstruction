"""FluxBased 0.1/5-year zero-MB gate for synthetic LIA system 3850."""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .flux_mass_budget import _step_budget


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _state(model, initial_volume: float, year: float) -> dict:
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"zero-mb-root-count:{len(roots)}")
    finite = bool(
        np.isfinite(model.volume_m3)
        and np.isfinite(model.area_m2)
        and all(
            np.isfinite(flowline.section).all()
            and np.isfinite(flowline.thick).all()
            for flowline in model.fls
        )
    )
    return {
        "year": float(year),
        "volume_m3": float(model.volume_m3),
        "volume_drift_fraction": float(
            (model.volume_m3 - initial_volume) / initial_volume
        ),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "root_terminal_thickness_m": float(roots[0].thick[-1]),
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
    input_root = resolve(config["inversion_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", config)
    formal = json.loads(
        (input_root / "inversion_summary_formal.json").read_text(encoding="utf-8")
    )
    if formal.get("status") != "inversion-pass":
        raise RuntimeError("registered-inversion-not-pass")

    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=input_root / "synthetic_gdir",
        reset=False,
    )
    initial_flowlines = gdir.read_pickle("model_flowlines")
    model = FluxBasedModel(
        copy.deepcopy(initial_flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["glen_a"]),
        fs=float(config["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    checkpoints = [_state(model, initial_volume, 0.0)]
    step_rows = []
    runtime_error = None
    failure_traceback = None
    budget_keys = (
        "actual_volume_change_m3",
        "upstream_gate_input_m3",
        "root_domain_outflow_m3",
        "negative_tributary_creation_m3",
        "negative_section_clip_creation_m3",
        "predicted_volume_change_m3",
        "closure_residual_m3",
    )
    cumulative = {key: 0.0 for key in budget_keys}
    for target_year in (0.1, 5.0):
        target_time = target_year * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                row = _step_budget(model, target_time - float(model.t))
                payload = {
                    "step_index": len(step_rows),
                    "checkpoint_target_year": target_year,
                    **row.__dict__,
                }
                step_rows.append(payload)
                for key in budget_keys:
                    cumulative[key] += float(payload[key])
            state = {**_state(model, initial_volume, target_year), **cumulative}
            checkpoints.append(state)
            if target_year == 0.1:
                numerical_limit = max(
                    float(config["gates"]["mass_creation_absolute_m3"]),
                    initial_volume
                    * float(config["gates"]["mass_creation_relative_fraction"]),
                )
                if (
                    abs(state["volume_drift_fraction"])
                    > float(config["gates"]["volume_drift_fraction_max"])
                    or state["negative_tributary_creation_m3"] > numerical_limit
                    or state["negative_section_clip_creation_m3"] > numerical_limit
                ):
                    break
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break

    pd.DataFrame(step_rows).to_csv(output_root / "zero_mb_steps.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(
        output_root / "zero_mb_checkpoints.csv", index=False
    )
    final = checkpoints[-1]
    numerical_limit = max(
        float(config["gates"]["mass_creation_absolute_m3"]),
        initial_volume
        * float(config["gates"]["mass_creation_relative_fraction"]),
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
    if float(final["year"]) < 5.0:
        flags.append("checkpoint-incomplete")
    if abs(float(final["volume_drift_fraction"])) > float(
        config["gates"]["volume_drift_fraction_max"]
    ):
        flags.append("volume-drift-over-0.1pct")
    if float(final.get("negative_tributary_creation_m3", 0.0)) > numerical_limit:
        flags.append("negative-tributary-mass-creation")
    if float(final.get("negative_section_clip_creation_m3", 0.0)) > numerical_limit:
        flags.append("negative-section-clip-mass-creation")
    if float(final["root_terminal_thickness_m"]) > float(
        config["gates"]["root_terminal_thickness_m_max"]
    ):
        flags.append("root-terminal-thickness")
    if not bool(final["finite_geometry"]):
        flags.append("nonfinite-geometry")
    if int(final["negative_section_count"]):
        flags.append("negative-section")
    closure = abs(float(final.get("closure_residual_m3", 0.0)))
    if closure > numerical_limit:
        flags.append("budget-closure-residual")
    summary = {
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "solver": "FluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "flowline_count": len(initial_flowlines),
        "initial_volume_m3": initial_volume,
        "completed_checkpoint_year": float(final["year"]),
        "final": final,
        "numerical_mass_creation_limit_m3": numerical_limit,
        "review_flags": flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "steady_50y_run": False,
        "historical_run": False,
    }
    _write_json(output_root / "zero_mb_summary.json", summary)
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
