"""Zero-MB pilot with conservative one-way tributary junction fluxes."""

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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _state(model, initial_volume: float, year: float) -> dict:
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    finite = bool(
        len(roots) == 1 and np.isfinite(model.volume_m3) and np.isfinite(model.area_m2)
        and all(np.isfinite(fl.section).all() and np.isfinite(fl.thick).all() for fl in model.fls)
    )
    return {
        "year": float(year), "volume_m3": float(model.volume_m3),
        "volume_drift_fraction": float((model.volume_m3 - initial_volume) / initial_volume),
        "area_m2": float(model.area_m2), "length_m": float(model.length_m),
        "root_terminal_thickness_m": float(roots[0].thick[-1]) if roots else float("nan"),
        "finite_geometry": finite,
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section) < -1e-10) for fl in model.fls)),
    }


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_root = resolve(config["inversion_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", config)
    formal = json.loads((input_root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    if formal.get("status") != "inversion-pass":
        raise RuntimeError("registered-inversion-not-pass")

    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    class ConservativeOneWayFluxBasedModel(FluxBasedModel):
        def step(self, dt):
            used_dt = float(super().step(dt))
            prevented = 0.0
            for index, flowline in enumerate(self.fls):
                if self._tributary_indices[index][0] is None:
                    continue
                terminal_flux = float(np.asarray(self.flux_stag[index], dtype=float)[:-1][-1])
                creation = max(-terminal_flux, 0.0) * used_dt
                if creation <= 0.0:
                    continue
                section = np.asarray(flowline.section, dtype=float).copy()
                remaining = creation / float(flowline.dx_meter)
                for node in range(len(section) - 1, -1, -1):
                    take = min(float(section[node]), remaining)
                    section[node] -= take
                    remaining -= take
                    if remaining <= 1e-10:
                        break
                if remaining > 1e-6:
                    raise RuntimeError(f"junction-correction-insufficient-section:{index}:{remaining}")
                flowline.section = section
                prevented += creation
            self.last_prevented_negative_tributary_m3 = prevented
            self.cumulative_prevented_negative_tributary_m3 = (
                getattr(self, "cumulative_prevented_negative_tributary_m3", 0.0) + prevented
            )
            return used_dt

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=input_root / "synthetic_gdir", reset=False)
    initial_flowlines = gdir.read_pickle("model_flowlines")
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(initial_flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
        glen_a=float(config["glen_a"]), fs=float(config["fs"]),
        smooth_trib_influx=True, check_for_boundaries=True,
    )
    initial_volume = float(model.volume_m3)
    checkpoints = [_state(model, initial_volume, 0.0)]
    step_rows = []
    totals = {
        key: 0.0 for key in (
            "actual_volume_change_m3", "upstream_gate_input_m3", "root_domain_outflow_m3",
            "negative_tributary_creation_m3", "prevented_negative_tributary_creation_m3",
            "unresolved_negative_tributary_creation_m3", "negative_section_clip_creation_m3",
            "predicted_volume_change_m3", "closure_residual_m3",
        )
    }
    runtime_error = None
    failure_traceback = None
    for target_year in (0.1, 5.0):
        target_time = target_year * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                before_sections = [np.asarray(fl.section, dtype=float).copy() for fl in model.fls]
                before_volume = float(model.volume_m3)
                start_year = float(model.yr)
                used_dt = float(model.step(target_time - float(model.t)))
                upstream, root_outflow, negative_trib, clip_creation = 0.0, 0.0, 0.0, 0.0
                for index, (flowline, before) in enumerate(zip(model.fls, before_sections)):
                    tributary = model._tributary_indices[index][0] is not None
                    flux = np.asarray(model.flux_stag[index], dtype=float)
                    if tributary:
                        flux = flux[:-1]
                        negative_trib += max(-float(flux[-1]), 0.0) * used_dt
                    else:
                        root_outflow += float(flux[-1]) * used_dt
                    upstream += float(flux[0]) * used_dt
                    raw = before + (flux[:-1] - flux[1:]) * used_dt / float(flowline.dx_meter)
                    raw += np.asarray(model.trib_flux[index], dtype=float) * used_dt / float(flowline.dx_meter)
                    clip_creation += float(np.sum(np.maximum(-raw, 0.0)) * float(flowline.dx_meter))
                prevented = float(model.last_prevented_negative_tributary_m3)
                unresolved = max(negative_trib - prevented, 0.0)
                actual = float(model.volume_m3 - before_volume)
                predicted = upstream - root_outflow + negative_trib - prevented + clip_creation
                row = {
                    "step_index": len(step_rows), "checkpoint_target_year": target_year,
                    "start_year": start_year, "end_year": float(model.yr), "dt_seconds": used_dt,
                    "actual_volume_change_m3": actual, "upstream_gate_input_m3": upstream,
                    "root_domain_outflow_m3": root_outflow,
                    "negative_tributary_creation_m3": negative_trib,
                    "prevented_negative_tributary_creation_m3": prevented,
                    "unresolved_negative_tributary_creation_m3": unresolved,
                    "negative_section_clip_creation_m3": clip_creation,
                    "predicted_volume_change_m3": predicted,
                    "closure_residual_m3": actual - predicted,
                }
                step_rows.append(row)
                for key in totals:
                    totals[key] += float(row[key])
            checkpoints.append({**_state(model, initial_volume, target_year), **totals})
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break

    pd.DataFrame(step_rows).to_csv(output_root / "zero_mb_steps.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    final = checkpoints[-1]
    limit = max(float(config["gates"]["mass_creation_absolute_m3"]), initial_volume * float(config["gates"]["mass_creation_relative_fraction"]))
    flags = []
    if runtime_error:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "numerical-error")
    if float(final["year"]) < 5.0:
        flags.append("checkpoint-incomplete")
    if abs(float(final["volume_drift_fraction"])) > float(config["gates"]["volume_drift_fraction_max"]):
        flags.append("volume-drift-over-0.1pct")
    if float(final.get("unresolved_negative_tributary_creation_m3", 0.0)) > limit:
        flags.append("unresolved-negative-tributary-creation")
    if float(final.get("negative_section_clip_creation_m3", 0.0)) > limit:
        flags.append("negative-section-clip-mass-creation")
    if abs(float(final.get("root_domain_outflow_m3", 0.0))) > limit:
        flags.append("root-domain-outflow")
    if abs(float(final.get("closure_residual_m3", 0.0))) > limit:
        flags.append("budget-closure-residual")
    if float(final["root_terminal_thickness_m"]) > float(config["gates"]["root_terminal_thickness_m_max"]):
        flags.append("root-terminal-thickness")
    if not final["finite_geometry"] or int(final["negative_section_count"]):
        flags.append("invalid-geometry")
    summary = {
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "experiment": config["experiment"], "system_id": int(config["system_id"]),
        "solver": "ConservativeOneWayFluxBasedModel", "mass_balance": "ScalarMassBalance(0.0)",
        "flowline_count": len(initial_flowlines), "initial_volume_m3": initial_volume,
        "completed_checkpoint_year": float(final["year"]), "final": final,
        "numerical_mass_creation_limit_m3": limit, "review_flags": flags,
        "runtime_error": runtime_error, "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "steady_50y_run": False, "historical_run": False,
    }
    _write_json(output_root / "zero_mb_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
