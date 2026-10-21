"""Run an isolated fixed-thickness turnover-equivalence pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from oggm import cfg, tasks, workflow
from oggm.core.massbalance import MonthlyTIModel


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE0_SOURCE = REPO_ROOT / "src" / "stage0_bed"
sys.path.insert(0, str(STAGE0_SOURCE))

from build_modern_gdirs_pilot import configure_oggm, resolve_path  # noqa: E402
from run_inversion_pilot import (  # noqa: E402
    load_hugonnet,
    system_modeled_mb,
)


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
LOGGER = logging.getLogger("stage1_turnover_pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "turnover_pilot.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def clone_gdir(source: Path, target: Path) -> None:
    ignore_auxiliary_tar_metadata = shutil.ignore_patterns("*.properties")
    if target.exists():
        required = (
            target / "climate_historical.nc",
            target / "inversion_flowlines.pkl",
            target / "glacier_grid.json",
        )
        marker = target / ".clone_incomplete"
        if marker.exists():
            LOGGER.info("Resuming incomplete clone %s", target)
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=ignore_auxiliary_tar_metadata,
            )
            marker.unlink()
        elif not all(path.exists() for path in required):
            raise RuntimeError(f"Incomplete existing clone: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=False)
    marker = target / ".clone_incomplete"
    marker.touch()
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=ignore_auxiliary_tar_metadata,
    )
    marker.unlink()


def key_file_hashes(gdir_root: Path, rgi_ids: list[str], filesuffix: str) -> dict:
    names = (
        "mb_calib.json",
        "inversion_flowlines.pkl",
        f"inversion_output{filesuffix}.pkl",
    )
    records = {}
    for rgi_id in sorted(rgi_ids):
        directory = gdir_path(gdir_root, rgi_id)
        records[rgi_id] = {
            name: sha256(directory / name)
            for name in names
            if (directory / name).exists()
        }
    return records


def fixed_candidate_values(
    grid: list[float], baseline: float
) -> list[tuple[str, float]]:
    result = [("stage0-baseline", float(baseline))]
    for value in sorted(float(item) for item in grid):
        if not np.isclose(value, baseline):
            result.append((f"grid-{value:g}", value))
    return result


def adaptive_candidate_values(
    lower: float, upper: float, baseline: float, count: int
) -> list[tuple[str, float]]:
    if count < 2:
        raise ValueError("equivalence_candidate_count must be at least 2")
    result = [("stage0-baseline", float(baseline))]
    for index, value in enumerate(np.linspace(lower, upper, count)):
        if not np.isclose(value, baseline, rtol=0, atol=1e-10):
            result.append((f"ridge-{index:03d}", float(value)))
    return result


def solve_precip_multiplier(
    gdir_by_id,
    member_ids: list[str],
    hugonnet: pd.DataFrame,
    p0_by_id: dict[str, float],
    melt_f: float,
    temp_bias: float,
    target_mb: float,
    years: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, float, dict[str, float]]:
    def residual(multiplier: float) -> float:
        modeled, _ = system_modeled_mb(
            gdir_by_id,
            member_ids,
            hugonnet,
            p0_by_id,
            melt_f,
            multiplier,
            temp_bias,
            years,
        )
        return modeled - target_mb

    residual_lower = residual(lower)
    residual_upper = residual(upper)
    if np.isclose(residual_lower, 0.0, atol=1e-10):
        multiplier = lower
    elif np.isclose(residual_upper, 0.0, atol=1e-10):
        multiplier = upper
    elif residual_lower * residual_upper > 0:
        raise ValueError(
            "No modern-MB root inside precipitation prior: "
            f"residuals={residual_lower:.3f},{residual_upper:.3f}"
        )
    else:
        multiplier = float(optimize.brentq(residual, lower, upper, xtol=1e-10))
    modeled, member_values = system_modeled_mb(
        gdir_by_id,
        member_ids,
        hugonnet,
        p0_by_id,
        melt_f,
        multiplier,
        temp_bias,
        years,
    )
    return multiplier, modeled, member_values


def find_feasible_melt_interval(
    gdir_by_id,
    member_ids: list[str],
    hugonnet: pd.DataFrame,
    p0_by_id: dict[str, float],
    baseline_melt: float,
    temp_bias: float,
    target_mb: float,
    years: np.ndarray,
    precip_lower: float,
    precip_upper: float,
    melt_lower: float,
    melt_upper: float,
) -> tuple[float, float]:
    def residual(melt_f: float, precip_multiplier: float) -> float:
        modeled, _ = system_modeled_mb(
            gdir_by_id,
            member_ids,
            hugonnet,
            p0_by_id,
            melt_f,
            precip_multiplier,
            temp_bias,
            years,
        )
        return modeled - target_mb

    def root_including_endpoints(
        precip_multiplier: float, left: float, right: float
    ) -> float:
        residual_left = residual(left, precip_multiplier)
        residual_right = residual(right, precip_multiplier)
        endpoint_tolerance = 1e-7
        if abs(residual_left) <= endpoint_tolerance:
            return left
        if abs(residual_right) <= endpoint_tolerance:
            return right
        if residual_left * residual_right > 0:
            raise ValueError(
                "Melt-factor root is not bracketed: "
                f"residuals={residual_left:.6g},{residual_right:.6g}"
            )
        return float(
            optimize.brentq(
                lambda value: residual(value, precip_multiplier),
                left,
                right,
                xtol=1e-10,
            )
        )

    try:
        solve_precip_multiplier(
            gdir_by_id,
            member_ids,
            hugonnet,
            p0_by_id,
            melt_lower,
            temp_bias,
            target_mb,
            years,
            precip_lower,
            precip_upper,
        )
        feasible_lower = melt_lower
    except ValueError:
        feasible_lower = root_including_endpoints(
            precip_lower, melt_lower, baseline_melt
        )

    try:
        solve_precip_multiplier(
            gdir_by_id,
            member_ids,
            hugonnet,
            p0_by_id,
            melt_upper,
            temp_bias,
            target_mb,
            years,
            precip_lower,
            precip_upper,
        )
        feasible_upper = melt_upper
    except ValueError:
        feasible_upper = root_including_endpoints(
            precip_upper, baseline_melt, melt_upper
        )

    if feasible_lower > baseline_melt or feasible_upper < baseline_melt:
        raise RuntimeError("Stage 0 baseline is outside the feasible MB interval")
    return feasible_lower, feasible_upper


def interval_residual(modeled: float, low: float, high: float) -> float:
    floor = 1e-6
    modeled = max(float(modeled), floor)
    low = max(float(low), floor)
    high = max(float(high), low)
    if modeled < low:
        return math.log(modeled / low)
    if modeled > high:
        return math.log(modeled / high)
    return 0.0


def evaluate_candidate_speeds(
    system_id: int,
    candidate_id: str,
    gdir_by_id,
    gate_rows: pd.DataFrame,
    p0_by_id: dict[str, float],
    melt_f: float,
    precip_multiplier: float,
    temp_bias: float,
    years: np.ndarray,
    qc_weights: dict[str, float],
    surface_factor: float,
) -> tuple[list[dict], list[dict]]:
    records = []
    failures = []
    for rgi_id, member_gates in gate_rows.groupby("rgi_id"):
        try:
            gdir = gdir_by_id[rgi_id]
            prcp_fac = float(
                np.clip(
                    p0_by_id[rgi_id] * precip_multiplier,
                    cfg.PARAMS["prcp_fac_min"],
                    cfg.PARAMS["prcp_fac_max"],
                )
            )
            model = MonthlyTIModel(
                gdir,
                melt_f=melt_f,
                prcp_fac=prcp_fac,
                temp_bias=temp_bias,
                check_calib_params=False,
            )
            tasks.apparent_mb_from_any_mb(gdir, mb_model=model, mb_years=years)
            tasks.prepare_for_inversion(gdir)
            inversion_input = gdir.read_pickle("inversion_input")
            for gate in member_gates.itertuples(index=False):
                line = inversion_input[int(gate.main_flowline_list_index)]
                flux = float(line["flux"][int(gate.gate_flowline_index)])
                section = float(gate.fixed_section_m2)
                if not np.isfinite(section) or section <= 0:
                    raise ValueError("Invalid fixed-thickness cross-section")
                integrated_speed = flux / section * SECONDS_PER_YEAR
                surface_speed = integrated_speed * surface_factor
                residual = interval_residual(
                    surface_speed,
                    gate.observed_speed_low_m_per_year,
                    gate.observed_speed_high_m_per_year,
                )
                weight = float(qc_weights[gate.qc_status])
                records.append(
                    {
                        "lia_unique_id": system_id,
                        "candidate_id": candidate_id,
                        "rgi_id": rgi_id,
                        "qc_status": gate.qc_status,
                        "weight": weight,
                        "main_flowline_list_index": int(
                            gate.main_flowline_list_index
                        ),
                        "gate_flowline_index": int(gate.gate_flowline_index),
                        "fixed_section_m2": section,
                        "candidate_flux_m3_s": flux,
                        "modeled_integrated_speed_m_per_year": integrated_speed,
                        "modeled_surface_speed_m_per_year": surface_speed,
                        "observed_speed_low_m_per_year": float(
                            gate.observed_speed_low_m_per_year
                        ),
                        "observed_speed_median_m_per_year": float(
                            gate.observed_speed_median_m_per_year
                        ),
                        "observed_speed_high_m_per_year": float(
                            gate.observed_speed_high_m_per_year
                        ),
                        "observed_to_modeled_surface_ratio": (
                            float(gate.observed_speed_median_m_per_year)
                            / surface_speed
                            if surface_speed > 0
                            else math.nan
                        ),
                        "interval_log_residual": residual,
                        "inside_observation_interval": residual == 0.0,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "lia_unique_id": system_id,
                    "candidate_id": candidate_id,
                    "rgi_id": rgi_id,
                    "stage": "velocity-operator",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return records, failures


def summarize_candidate(
    system_id: int,
    candidate_id: str,
    candidate_kind: str,
    melt_f: float,
    precip_multiplier: float,
    temp_bias: float,
    target_mb: float,
    modeled_mb: float,
    precip_lower: float,
    precip_upper: float,
    gate_records: list[dict],
) -> dict:
    gates = pd.DataFrame(gate_records)
    if gates.empty:
        loss = math.nan
        inside = 0
        median_ratio = math.nan
    else:
        weights = gates["weight"].to_numpy(float)
        residuals = gates["interval_log_residual"].to_numpy(float)
        loss = float(np.sum(weights * residuals**2) / np.sum(weights))
        inside = int(gates["inside_observation_interval"].sum())
        median_ratio = float(
            gates["observed_to_modeled_surface_ratio"].median()
        )
    return {
        "lia_unique_id": system_id,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "melt_f": melt_f,
        "precip_prior_multiplier": precip_multiplier,
        "temp_bias": temp_bias,
        "target_mb_kg_m2_yr": target_mb,
        "modeled_mb_kg_m2_yr": modeled_mb,
        "mb_residual_kg_m2_yr": modeled_mb - target_mb,
        "precip_at_prior_bound": bool(
            np.isclose(precip_multiplier, precip_lower)
            or np.isclose(precip_multiplier, precip_upper)
        ),
        "evaluated_gate_count": len(gates),
        "usable_gate_count": int((gates["qc_status"] == "usable").sum())
        if not gates.empty
        else 0,
        "weak_slow_gate_count": int((gates["qc_status"] == "weak-slow").sum())
        if not gates.empty
        else 0,
        "inside_observation_interval_count": inside,
        "weighted_interval_loss": loss,
        "median_observed_to_modeled_ratio": median_ratio,
        "status": "feasible-modern-mb-equivalence-candidate",
    }


def run_system(
    system_id: int,
    sample_role: str,
    config: dict,
    manifest: pd.DataFrame,
    members: pd.DataFrame,
    stage0_systems: pd.DataFrame,
    hugonnet: pd.DataFrame,
    gates: pd.DataFrame,
    baseline_gates: pd.DataFrame,
    gdir_by_id,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    system_row = stage0_systems.loc[system_id]
    manifest_row = manifest.loc[system_id]
    member_rows = members[members["lia_unique_id"] == system_id]
    member_ids = member_rows["rgi_id"].astype(str).tolist()
    p0_by_id = member_rows.set_index("rgi_id")[
        "winter_prcp_fac_center"
    ].astype(float).to_dict()
    target_mb = float(system_row.target_mb_kg_m2_yr)
    target_err = float(system_row.target_error_upper_kg_m2_yr)
    temp_bias = float(system_row.temp_bias)
    baseline_melt = float(system_row.melt_f)
    baseline_precip = float(system_row.precip_prior_multiplier)
    precip_lower = float(manifest_row.precip_prior_multiplier_p025)
    precip_upper = float(manifest_row.precip_prior_multiplier_p975)
    year_start = int(config["mb_period"][:4])
    year_end = int(config["mb_period"][11:15])
    years = np.arange(year_start, year_end)
    strong_classes = set(config["strong_velocity_qc"])
    system_all_gates = gates[gates["lia_unique_ids"].astype(str).str.split(";").apply(
        lambda values: str(system_id) in values
    )]
    system_baseline_gates = baseline_gates[
        baseline_gates["lia_unique_id"] == system_id
    ]
    strong_gate_ids = set(
        system_all_gates.loc[
            system_all_gates["qc_status"].isin(strong_classes), "rgi_id"
        ]
    )
    evaluable_gates = system_baseline_gates[
        system_baseline_gates["rgi_id"].isin(strong_gate_ids)
        & system_baseline_gates["qc_status"].isin(strong_classes)
    ].copy()
    missing_section_ids = sorted(strong_gate_ids - set(evaluable_gates["rgi_id"]))
    failures = [
        {
            "lia_unique_id": system_id,
            "candidate_id": "all",
            "rgi_id": rgi_id,
            "stage": "fixed-section-operator",
            "error_type": "MissingValidFixedSection",
            "error": "Strong velocity gate has no positive Farinotti section",
        }
        for rgi_id in missing_section_ids
    ]
    generation = config.get("candidate_generation", "fixed-grid")
    if generation == "adaptive-equivalence-grid":
        melt_lower, melt_upper = map(
            float, config["melt_f_bounds_kg_m2_day_k"]
        )
        feasible_melt_lower, feasible_melt_upper = find_feasible_melt_interval(
            gdir_by_id,
            member_ids,
            hugonnet,
            p0_by_id,
            baseline_melt,
            temp_bias,
            target_mb,
            years,
            precip_lower,
            precip_upper,
            melt_lower,
            melt_upper,
        )
        candidates_to_evaluate = adaptive_candidate_values(
            feasible_melt_lower,
            feasible_melt_upper,
            baseline_melt,
            int(config["equivalence_candidate_count"]),
        )
    elif generation == "fixed-grid":
        feasible_melt_lower = math.nan
        feasible_melt_upper = math.nan
        candidates_to_evaluate = fixed_candidate_values(
            config["melt_f_grid_kg_m2_day_k"], baseline_melt
        )
    else:
        raise ValueError(f"Unknown candidate_generation: {generation}")

    candidate_rows = []
    gate_rows = []
    for candidate_id, melt_f in candidates_to_evaluate:
        LOGGER.info(
            "System %d candidate=%s melt_f=%.6g",
            system_id,
            candidate_id,
            melt_f,
        )
        candidate_kind = (
            "stage0-baseline" if candidate_id == "stage0-baseline" else "grid"
        )
        try:
            if candidate_kind == "stage0-baseline":
                precip_multiplier = baseline_precip
                modeled_mb, _ = system_modeled_mb(
                    gdir_by_id,
                    member_ids,
                    hugonnet,
                    p0_by_id,
                    melt_f,
                    precip_multiplier,
                    temp_bias,
                    years,
                )
            else:
                precip_multiplier, modeled_mb, _ = solve_precip_multiplier(
                    gdir_by_id,
                    member_ids,
                    hugonnet,
                    p0_by_id,
                    melt_f,
                    temp_bias,
                    target_mb,
                    years,
                    precip_lower,
                    precip_upper,
                )
            speed_rows, speed_failures = evaluate_candidate_speeds(
                system_id,
                candidate_id,
                gdir_by_id,
                evaluable_gates,
                p0_by_id,
                melt_f,
                precip_multiplier,
                temp_bias,
                years,
                config["velocity_qc_weights"],
                float(config["surface_to_integrated_speed_factor"]),
            )
            failures.extend(speed_failures)
            gate_rows.extend(speed_rows)
            candidate_summary = summarize_candidate(
                system_id,
                candidate_id,
                candidate_kind,
                melt_f,
                precip_multiplier,
                temp_bias,
                target_mb,
                modeled_mb,
                precip_lower,
                precip_upper,
                speed_rows,
            )
            candidate_rows.append(candidate_summary)
            LOGGER.info(
                "System %d candidate=%s precip=%.6g mb_residual=%.3g "
                "gates=%d loss=%s",
                system_id,
                candidate_id,
                precip_multiplier,
                candidate_summary["mb_residual_kg_m2_yr"],
                candidate_summary["evaluated_gate_count"],
                candidate_summary["weighted_interval_loss"],
            )
        except Exception as exc:
            LOGGER.warning(
                "System %d candidate=%s infeasible: %s: %s",
                system_id,
                candidate_id,
                type(exc).__name__,
                exc,
            )
            failures.append(
                {
                    "lia_unique_id": system_id,
                    "candidate_id": candidate_id,
                    "rgi_id": "",
                    "stage": "mb-equivalence-candidate",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    evaluated = candidates[candidates["evaluated_gate_count"] > 0]
    baseline_reproduction_error = math.nan
    baseline_speed_pass = None
    if not evaluable_gates.empty:
        baseline_candidate_gates = pd.DataFrame(gate_rows)
        baseline_candidate_gates = baseline_candidate_gates[
            baseline_candidate_gates["candidate_id"] == "stage0-baseline"
        ]
        comparison = baseline_candidate_gates.merge(
            evaluable_gates[
                ["rgi_id", "modeled_surface_speed_m_per_year"]
            ].rename(
                columns={
                    "modeled_surface_speed_m_per_year": "saved_baseline_speed"
                }
            ),
            on="rgi_id",
            validate="one_to_one",
        )
        relative = np.abs(
            comparison["modeled_surface_speed_m_per_year"]
            / comparison["saved_baseline_speed"]
            - 1
        )
        baseline_reproduction_error = (
            float(relative.max()) if not relative.empty else math.nan
        )
        baseline_speed_pass = bool(
            len(comparison) == len(evaluable_gates)
            and np.isfinite(baseline_reproduction_error)
            and baseline_reproduction_error
            <= float(config["baseline_speed_reproduction_relative_tolerance"])
        )

    mb_pass = bool(
        not candidates.empty
        and (
            candidates["mb_residual_kg_m2_yr"].abs()
            <= float(config["mb_residual_tolerance_kg_m2_yr"])
        ).all()
    )
    if evaluated.empty:
        best = None
        status = "weak-identifiability-no-evaluable-strong-gate"
    else:
        best = evaluated.sort_values(
            ["weighted_interval_loss", "candidate_id"]
        ).iloc[0]
        status = config.get(
            "evaluated_status", "pilot-evaluated-not-frozen-posterior"
        )
    semantic_pass = mb_pass and baseline_speed_pass is not False
    summary = {
        "lia_unique_id": system_id,
        "sample_role": sample_role,
        "member_count": len(member_ids),
        "all_gate_count": len(system_all_gates),
        "strong_gate_count": len(strong_gate_ids),
        "evaluable_strong_gate_count": len(evaluable_gates),
        "missing_fixed_section_count": len(missing_section_ids),
        "candidate_generation": generation,
        "feasible_melt_f_lower": feasible_melt_lower,
        "feasible_melt_f_upper": feasible_melt_upper,
        "feasible_candidate_count": len(candidates),
        "failed_candidate_count": len(
            {
                row["candidate_id"]
                for row in failures
                if row["stage"] == "mb-equivalence-candidate"
            }
        ),
        "baseline_speed_max_relative_error": baseline_reproduction_error,
        "baseline_speed_reproduction_pass": baseline_speed_pass,
        "mb_equivalence_pass": mb_pass,
        "semantic_validation_pass": semantic_pass,
        "best_candidate_id": None if best is None else best.candidate_id,
        "best_melt_f": None if best is None else float(best.melt_f),
        "best_precip_prior_multiplier": None
        if best is None
        else float(best.precip_prior_multiplier),
        "best_temp_bias": None if best is None else float(best.temp_bias),
        "best_weighted_interval_loss": None
        if best is None
        else float(best.weighted_interval_loss),
        "best_inside_observation_interval_count": 0
        if best is None
        else int(best.inside_observation_interval_count),
        "stage0_identifiability_status": system_row.identifiability_status,
        "target_mb_kg_m2_yr": target_mb,
        "target_mb_error_kg_m2_yr": target_err,
        "status": status,
    }
    return candidate_rows, gate_rows, summary, failures


def resolve_run_system_ids(
    config: dict,
    stage0_systems: pd.DataFrame,
    eligible_system_ids: set[int] | None,
) -> list[int]:
    configured = config["lia_system_ids"]
    if configured == "stage0-successful":
        return sorted(int(value) for value in stage0_systems.index)
    if configured == "eligible-systems":
        if eligible_system_ids is None:
            raise ValueError(
                "eligible-systems requires eligible_systems_path"
            )
        return sorted(eligible_system_ids)
    if not isinstance(configured, list):
        raise ValueError(
            "lia_system_ids must be a list, 'stage0-successful', or "
            "'eligible-systems'"
        )
    return [int(value) for value in configured]


def sample_role(config: dict, system_id: int) -> str:
    return config.get("sample_roles", {}).get(
        str(system_id), config.get("default_sample_role", "turnover-scan")
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve_path(REPO_ROOT, config["output_dir"])
    configure_logging(output_dir)
    started = time.time()
    LOGGER.info("Starting %s", config["pilot_id"])

    stage0_config_path = resolve_path(REPO_ROOT, config["stage0_config_path"])
    stage0_config = read_json(stage0_config_path)
    stage0_output = resolve_path(REPO_ROOT, config["stage0_output_dir"])
    source_gdir_root = stage0_output / "oggm_workdir" / "per_glacier"
    pilot_workdir = output_dir / config["oggm_workdir_name"]
    target_gdir_root = pilot_workdir / "per_glacier"
    manifest_path = resolve_path(REPO_ROOT, config["manifest_path"])
    gate_path = resolve_path(REPO_ROOT, config["velocity_gate_path"])
    baseline_path = resolve_path(REPO_ROOT, config["velocity_baseline_path"])
    manifest = pd.read_csv(manifest_path).set_index("lia_unique_id")
    members = pd.read_csv(stage0_output / "mb_member_parameters.csv")
    stage0_systems = pd.read_csv(
        stage0_output / "mb_system_candidates.csv"
    ).set_index("lia_unique_id")
    gates = pd.read_csv(gate_path)
    baseline_gates = pd.read_csv(baseline_path)
    eligibility_path = None
    eligible_system_ids = None
    if config.get("eligible_systems_path"):
        eligibility_path = resolve_path(REPO_ROOT, config["eligible_systems_path"])
        eligibility = pd.read_csv(eligibility_path)
        eligible_system_ids = set(
            int(value) for value in eligibility["lia_unique_id"].tolist()
        )
    system_ids = resolve_run_system_ids(
        config, stage0_systems, eligible_system_ids
    )
    missing_systems = sorted(set(system_ids) - set(stage0_systems.index.astype(int)))
    if missing_systems:
        raise ValueError(f"Pilot systems absent from Stage 0 MB output: {missing_systems}")
    selected_members = members[members["lia_unique_id"].isin(system_ids)].copy()
    rgi_ids = selected_members["rgi_id"].astype(str).tolist()

    filesuffix = config["fixed_thickness_filesuffix"]
    source_hashes_before = key_file_hashes(source_gdir_root, rgi_ids, filesuffix)
    input_hashes = {
        "manifest": sha256(manifest_path),
        "velocity_gates": sha256(gate_path),
        "velocity_baseline": sha256(baseline_path),
        "stage0_system_candidates": sha256(
            stage0_output / "mb_system_candidates.csv"
        ),
        "stage0_member_parameters": sha256(
            stage0_output / "mb_member_parameters.csv"
        ),
    }
    if eligibility_path is not None:
        input_hashes["eligible_systems"] = sha256(eligibility_path)
    current_config_hash = sha256(config_path)
    current_code_hash = sha256(Path(__file__).resolve())
    existing_manifest_path = output_dir / "run_manifest.json"
    existing_manifest = (
        read_json(existing_manifest_path)
        if existing_manifest_path.exists()
        else None
    )
    if existing_manifest is not None:
        resume_checks = {
            "config_sha256": current_config_hash,
            "code_sha256": current_code_hash,
            "input_hashes": input_hashes,
            "source_key_hashes_before": source_hashes_before,
        }
        mismatched = [
            key
            for key, expected in resume_checks.items()
            if existing_manifest.get(key) != expected
        ]
        if mismatched:
            raise RuntimeError(
                "Refusing to resume a changed run; mismatched fields: "
                + ", ".join(mismatched)
            )

    LOGGER.info("Cloning %d Stage 0 glacier directories", len(rgi_ids))
    for index, rgi_id in enumerate(rgi_ids, start=1):
        clone_gdir(
            gdir_path(source_gdir_root, rgi_id),
            gdir_path(target_gdir_root, rgi_id),
        )
        if index % 20 == 0 or index == len(rgi_ids):
            LOGGER.info("Clone progress %d/%d", index, len(rgi_ids))

    run_config = dict(stage0_config)
    run_config["oggm_workdir_path"] = str(pilot_workdir)
    run_config["oggm_workdir_name"] = config["oggm_workdir_name"]
    run_config["use_multiprocessing"] = bool(config["use_multiprocessing"])
    run_config["mp_processes"] = int(config["mp_processes"])
    dem_path = resolve_path(REPO_ROOT, stage0_config["surface_dem_path"])
    configure_oggm(run_config, output_dir, dem_path)
    cfg.PARAMS["baseline_climate"] = "GSWP3_W5E5"
    cfg.PARAMS["use_winter_prcp_fac"] = True
    cfg.PARAMS["auto_skip_task"] = False
    gdirs = workflow.init_glacier_directories(rgi_ids, reset=False)
    gdir_by_id = {gdir.rgi_id: gdir for gdir in gdirs}
    hugonnet = load_hugonnet(stage0_config, config["mb_period"])

    run_manifest = {
        "pilot_id": config["pilot_id"],
        "run_kind": config.get("run_kind", "pilot"),
        "method_version": config["method_version"],
        "created_at_utc": (
            existing_manifest["created_at_utc"]
            if existing_manifest is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "config_path": str(config_path),
        "config_sha256": current_config_hash,
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": current_code_hash,
        "stage0_output_dir": str(stage0_output),
        "fixed_thickness_product": config.get(
            "fixed_thickness_product", "farinotti2019"
        ),
        "stage0_workdir_read_only": True,
        "pilot_workdir": str(pilot_workdir),
        "system_ids": system_ids,
        "member_count": len(rgi_ids),
        "input_hashes": input_hashes,
        "forbidden_inputs": [
            "LIA geometry",
            "reconstructed climate",
            "forward terminal residual",
        ],
        "source_key_hashes_before": source_hashes_before,
    }
    if existing_manifest is None:
        write_json_atomic(existing_manifest_path, run_manifest)

    resume_enabled = bool(config.get("resume", True))
    all_systems = (
        load_existing_records(output_dir / "system_summary.csv")
        if resume_enabled
        else []
    )
    if len(all_systems) != len(
        {int(row["lia_unique_id"]) for row in all_systems}
    ):
        raise RuntimeError("Existing system summary contains duplicate systems")
    completed_system_ids = {
        int(row["lia_unique_id"]) for row in all_systems
    }
    all_candidates = [
        row
        for row in (
            load_existing_records(output_dir / "candidate_summary.csv")
            if resume_enabled
            else []
        )
        if int(row["lia_unique_id"]) in completed_system_ids
    ]
    all_gate_rows = [
        row
        for row in (
            load_existing_records(output_dir / "gate_residuals.csv")
            if resume_enabled
            else []
        )
        if int(row["lia_unique_id"]) in completed_system_ids
    ]
    all_failures = [
        row
        for row in (
            load_existing_records(output_dir / "failures.csv")
            if resume_enabled
            else []
        )
        if int(row["lia_unique_id"]) in completed_system_ids
    ]
    if not completed_system_ids.issubset(set(system_ids)):
        raise RuntimeError("Existing result contains systems outside this run")
    if len(completed_system_ids) == len(system_ids):
        existing_summary = read_json(output_dir / "run_summary.json")
        LOGGER.info("Run is already complete; no artifacts rewritten")
        print(json.dumps(existing_summary, indent=2, ensure_ascii=True))
        return
    if completed_system_ids:
        LOGGER.info(
            "Resuming with %d/%d systems complete",
            len(completed_system_ids),
            len(system_ids),
        )

    for index, system_id in enumerate(system_ids, start=1):
        if system_id in completed_system_ids:
            continue
        LOGGER.info("System %d (%d/%d)", system_id, index, len(system_ids))
        try:
            candidate_rows, gate_rows, summary, failures = run_system(
                system_id,
                sample_role(config, system_id),
                config,
                manifest,
                selected_members,
                stage0_systems,
                hugonnet,
                gates,
                baseline_gates,
                gdir_by_id,
            )
        except Exception as exc:
            LOGGER.exception("System %d failed", system_id)
            candidate_rows = []
            gate_rows = []
            failures = [
                {
                    "lia_unique_id": system_id,
                    "candidate_id": "all",
                    "rgi_id": "",
                    "stage": "system",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
            summary = {
                "lia_unique_id": system_id,
                "sample_role": sample_role(config, system_id),
                "member_count": int(
                    (selected_members["lia_unique_id"] == system_id).sum()
                ),
                "all_gate_count": 0,
                "strong_gate_count": 0,
                "evaluable_strong_gate_count": 0,
                "missing_fixed_section_count": 0,
                "candidate_generation": config.get(
                    "candidate_generation", "fixed-grid"
                ),
                "feasible_melt_f_lower": math.nan,
                "feasible_melt_f_upper": math.nan,
                "feasible_candidate_count": 0,
                "failed_candidate_count": 0,
                "baseline_speed_max_relative_error": math.nan,
                "baseline_speed_reproduction_pass": False,
                "mb_equivalence_pass": False,
                "semantic_validation_pass": False,
                "best_candidate_id": None,
                "best_melt_f": None,
                "best_precip_prior_multiplier": None,
                "best_temp_bias": None,
                "best_weighted_interval_loss": None,
                "best_inside_observation_interval_count": 0,
                "stage0_identifiability_status": stage0_systems.loc[
                    system_id, "identifiability_status"
                ],
                "target_mb_kg_m2_yr": float(
                    stage0_systems.loc[system_id, "target_mb_kg_m2_yr"]
                ),
                "target_mb_error_kg_m2_yr": float(
                    stage0_systems.loc[
                        system_id, "target_error_upper_kg_m2_yr"
                    ]
                ),
                "status": "failed-system-retained-for-review",
            }
        all_candidates.extend(candidate_rows)
        all_gate_rows.extend(gate_rows)
        all_systems.append(summary)
        all_failures.extend(failures)
        pd.DataFrame(all_candidates).to_csv(
            output_dir / "candidate_summary.csv", index=False
        )
        pd.DataFrame(all_gate_rows).to_csv(
            output_dir / "gate_residuals.csv", index=False
        )
        pd.DataFrame(all_systems).to_csv(
            output_dir / "system_summary.csv", index=False
        )
        pd.DataFrame(
            all_failures,
            columns=[
                "lia_unique_id",
                "candidate_id",
                "rgi_id",
                "stage",
                "error_type",
                "error",
            ],
        ).to_csv(output_dir / "failures.csv", index=False)
        write_json_atomic(
            output_dir / "progress.json",
            {
                "pilot_id": config["pilot_id"],
                "completed_system_count": len(all_systems),
                "total_system_count": len(system_ids),
                "last_system_id": system_id,
                "last_system_status": summary["status"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        LOGGER.info(
            "System %d status=%s feasible=%d strong=%d",
            system_id,
            summary["status"],
            summary["feasible_candidate_count"],
            summary["evaluable_strong_gate_count"],
        )

    candidate_frame = pd.DataFrame(all_candidates)
    gate_frame = pd.DataFrame(all_gate_rows)
    system_frame = pd.DataFrame(all_systems)
    failure_frame = pd.DataFrame(
        all_failures,
        columns=[
            "lia_unique_id",
            "candidate_id",
            "rgi_id",
            "stage",
            "error_type",
            "error",
        ],
    )
    candidate_frame.to_csv(output_dir / "candidate_summary.csv", index=False)
    gate_frame.to_csv(output_dir / "gate_residuals.csv", index=False)
    system_frame.to_csv(output_dir / "system_summary.csv", index=False)
    failure_frame.to_csv(output_dir / "failures.csv", index=False)

    source_hashes_after = key_file_hashes(source_gdir_root, rgi_ids, filesuffix)
    source_immutable = source_hashes_before == source_hashes_after
    semantic_pass = bool(system_frame["semantic_validation_pass"].all())
    summary = {
        "pilot_id": config["pilot_id"],
        "run_kind": config.get("run_kind", "pilot"),
        "status": "completed-"
        + config.get("run_kind", "pilot")
        + "-not-frozen-posterior",
        "system_count": len(system_frame),
        "member_count": len(rgi_ids),
        "candidate_count": len(candidate_frame),
        "gate_residual_count": len(gate_frame),
        "failure_count": len(failure_frame),
        "systems_with_evaluable_strong_gate": int(
            (system_frame["evaluable_strong_gate_count"] > 0).sum()
        ),
        "systems_without_evaluable_strong_gate": int(
            (system_frame["evaluable_strong_gate_count"] == 0).sum()
        ),
        "source_stage0_key_files_immutable": source_immutable,
        "semantic_validation_pass": semantic_pass,
        "scale_up_gate_pass": source_immutable and semantic_pass,
        "scale_up_note": (
            "A passing gate validates operator semantics only. Velocity "
            "reachability remains a scientific diagnostic."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    run_manifest["source_key_hashes_after"] = source_hashes_after
    run_manifest["source_stage0_key_files_immutable"] = source_immutable
    write_json_atomic(existing_manifest_path, run_manifest)
    write_json_atomic(output_dir / "run_summary.json", summary)
    LOGGER.info("Completed: %s", json.dumps(summary, ensure_ascii=True))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
