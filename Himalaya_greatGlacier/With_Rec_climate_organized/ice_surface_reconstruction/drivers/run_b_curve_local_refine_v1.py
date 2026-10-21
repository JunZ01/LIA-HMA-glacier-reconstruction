"""Refine B-window curve candidates with real adaptive OGGM chains.

The experiment continues only parameter branches that already place the
forward trajectory inside the declared LIA-RGI corridor.  It jointly perturbs
``melt_f`` and ``delta_t`` while keeping all other controls frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import _gdir_path


SYSTEM_IDS = (1044, 3584)
SPINUP_WINDOW = (1441, 1470)
FORWARD_START = 1470
SPINUP_MAX_YEARS = 1500
SPINUP_MIN_YEARS = 500
SPINUP_CHECKPOINT_YEARS = 100
SPINUP_STABILITY_YEARS = 200
SPINUP_STABILITY_TOLERANCE = 0.02


SEED_BRANCHES = (
    {
        "branch": "1044_low_melt",
        "system_id": 1044,
        "melt_values": (4.55, 4.66600143825807, 4.78),
        "delta_t_values": (1.188, 1.193, 1.198),
        "reuse": {
            (4.66600143825807, 1.193): (
                "layer_B_melt_delta_boundary_refine_1044_round2_v1/system_1044/"
                "mf_4p666_dt_p1p193"
            )
        },
    },
    {
        "branch": "1044_mid_melt",
        "system_id": 1044,
        "melt_values": (5.10, 5.24925161804032, 5.40),
        "delta_t_values": (0.90, 0.9375, 0.975),
        "reuse": {
            (5.24925161804032, 0.9375): (
                "layer_B_melt_delta_boundary_refine_v1/system_1044/"
                "mf_5p249_dt_p0p9375"
            )
        },
    },
    {
        "branch": "1044_high_melt",
        "system_id": 1044,
        "melt_values": (5.65, 5.83250179782258, 6.00),
        "delta_t_values": (0.59, 0.625, 0.66),
        "reuse": {
            (5.83250179782258, 0.625): (
                "layer_B_melt_delta_boundary_refine_v1/system_1044/"
                "mf_5p833_dt_p0p625"
            )
        },
    },
    {
        "branch": "3584_endpoint_shape",
        "system_id": 3584,
        "melt_values": (2.15, 2.3116274251298465, 2.47),
        "delta_t_values": (0.70, 0.75, 0.80),
        "reuse": {
            (2.3116274251298465, 0.75): (
                "layer_B_adaptive_delta_profile_v1/system_3584/dt_p0p75"
            )
        },
    },
)


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _value_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _source_root(template: Path, system_id: int) -> Path:
    return Path(str(template).format(system_id=system_id))


def _load_preflight(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[int, dict]]:
    refs = pd.read_csv(args.references)
    stage2 = pd.read_csv(args.stage2_members)
    pixels = pd.read_csv(args.climate_pixels)
    audit = pd.read_csv(args.elevation_band_root / "audit" / "core191_elevation_band_audit.csv")
    rows: list[dict] = []
    frozen: dict[int, dict] = {}

    for system_id in SYSTEM_IDS:
        ref_rows = refs[refs["lia_unique_id"].astype(int).eq(system_id)]
        if len(ref_rows) != 1:
            raise ValueError(f"reference-row-count:{system_id}:{len(ref_rows)}")
        ref = ref_rows.iloc[0]
        rgi_id = str(ref["rgi_id"])

        parameter_rows = stage2[
            stage2["lia_unique_id"].astype(int).eq(system_id)
            & stage2["rgi_id"].astype(str).eq(rgi_id)
            & stage2["ensemble_role"].astype(str).eq("mid-turnover")
        ]
        if len(parameter_rows) != 1:
            raise ValueError(f"stage2-parameter-row-count:{system_id}:{len(parameter_rows)}")
        parameter = parameter_rows.iloc[0]

        pixel_rows = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
        ]
        if len(pixel_rows) != 1:
            raise ValueError(f"climate-pixel-row-count:{system_id}:{len(pixel_rows)}")

        audit_rows = audit[
            audit["lia_unique_id"].astype(int).eq(system_id)
            & audit["rgi_id"].astype(str).eq(rgi_id)
        ]
        if len(audit_rows) != 1:
            raise ValueError(f"geometry-audit-row-count:{system_id}:{len(audit_rows)}")
        audit_row = audit_rows.iloc[0]
        mismatch = abs(float(audit_row["corrected_length_error_fraction"]))
        if not bool(audit_row["all_geometry_gates_pass"]) or mismatch > 0.10:
            raise ValueError(f"geometry-gate-failed:{system_id}:{mismatch}")

        source_root = _source_root(args.source_gdir_root_template, system_id)
        source_gdir = _gdir_path(source_root, rgi_id)
        flowline = args.elevation_band_root / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
        for required in (source_gdir, source_gdir / "mb_calib.json", flowline):
            if not required.exists():
                raise FileNotFoundError(required)

        mb_calib = json.loads((source_gdir / "mb_calib.json").read_text(encoding="utf-8"))
        temp_melt = float(mb_calib["mb_global_params"]["temp_melt"])
        if not _same(temp_melt, -1.0):
            raise ValueError(f"temp-melt-conflict:{system_id}:{temp_melt}")

        frozen[system_id] = {
            "system_id": system_id,
            "rgi_id": rgi_id,
            "glen_a": float(parameter["glen_a_pa_3_s_1"]),
            "temp_melt": temp_melt,
            "prcp_fac": float(parameter["prcp_fac"]),
            "base_temp_bias": float(parameter["temp_bias"]),
            "lia_target_m": float(ref["lia_registered_length_m"]),
            "rgi_target_m": float(ref["rgi_registered_length_m"]),
            "source_root": source_root,
        }
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "lia_main_branch_index": ref.get("lia_main_branch_index", np.nan),
                "lia_spatial_length_target_m": float(ref["lia_registered_length_m"]),
                "rgi_root_flowline_index": ref.get("rgi_root_flowline_index", np.nan),
                "rgi_spatial_length_target_m": float(ref["rgi_registered_length_m"]),
                "elevation_band_target_length_m": float(audit_row["target_main_flowline_length_m"]),
                "initial_model_length_m": float(audit_row["pseudo_flowline_length_m"]),
                "node_spacing_m": float(audit_row.get("dx_meter", np.nan)),
                "initial_length_mismatch_fraction": mismatch,
                "length_definition": (
                    "Core-191 corrected elevation-band model length versus registered "
                    "spatial main-flowline endpoints"
                ),
                "geometry_gates_pass": True,
                "climate_id": "ensemble_rec_ref",
                "precipitation_anomaly": "off",
                "spinup_window": "1441-1470",
                "forward_period": "1470-2000",
                "adaptive_spinup": True,
                "spinup_max_years": SPINUP_MAX_YEARS,
                "spinup_stability_window_years": SPINUP_STABILITY_YEARS,
                "spinup_stability_tolerance_fraction": SPINUP_STABILITY_TOLERANCE,
                "prcp_fac": float(parameter["prcp_fac"]),
                "glen_a_pa_3_s_1": float(parameter["glen_a_pa_3_s_1"]),
                "temp_melt_c": temp_melt,
                "source_gdir": str(source_gdir.resolve()),
                "flowline_file": str(flowline.resolve()),
            }
        )
    return pd.DataFrame(rows), frozen


def _validate_source(
    source_dir: Path,
    system_id: int,
    melt_f: float,
    delta_t: float,
    frozen: dict,
) -> None:
    required = (
        source_dir / "summary.json",
        source_dir / "timeseries.csv",
        source_dir / "spinup_state.pkl.gz",
        source_dir / "state_2000.pkl.gz",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(required[0].read_text(encoding="utf-8"))
    parameters = summary["parameters"]
    checks = {
        "system_id": int(summary["lia_unique_id"]) == system_id,
        "rgi_id": str(summary["rgi_id"]) == frozen["rgi_id"],
        "layer": summary.get("layer") == "B",
        "window": list(summary["spinup_window"]) == list(SPINUP_WINDOW),
        "start_year": int(summary["start_year"]) == FORWARD_START,
        "melt_f": _same(parameters["melt_f"], melt_f),
        "delta_t": _same(parameters["delta_t_c"], delta_t),
        "prcp_fac_multiplier": _same(parameters["prcp_fac_multiplier"], 1.0),
        "glen_a": _same(parameters["glen_a_pa_3_s_1"], frozen["glen_a"]),
        "temp_melt": _same(parameters["temp_melt_c"], frozen["temp_melt"]),
        "adaptive_spinup": summary["spinup_control"]["mode"] == "adaptive",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"reuse-contract-mismatch:{source_dir}:{','.join(failed)}")


def _run_chain(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        payload["python_exe"],
        "-m",
        "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_lhs_single_chain",
        "--system-id", str(payload["system_id"]),
        "--references", payload["references"],
        "--stage2-members", payload["stage2_members"],
        "--climate-pixels", payload["climate_pixels"],
        "--reconstruction-climate", payload["reconstruction_climate"],
        "--source-gdir-root", payload["source_gdir_root"],
        "--elevation-band-root", payload["elevation_band_root"],
        "--workdir", payload["workdir"],
        "--output-dir", payload["output_dir"],
        "--ensemble-role", "mid-turnover",
        "--climate-id", "ensemble_rec_ref",
        "--prcp-mode", "off",
        "--spinup-window-start", str(SPINUP_WINDOW[0]),
        "--spinup-window-end", str(SPINUP_WINDOW[1]),
        "--start-year", str(FORWARD_START),
        "--spinup-years", str(SPINUP_MAX_YEARS),
        "--spinup-checkpoint-years", str(SPINUP_CHECKPOINT_YEARS),
        "--adaptive-spinup",
        "--spinup-min-years", str(SPINUP_MIN_YEARS),
        "--spinup-stability-window-years", str(SPINUP_STABILITY_YEARS),
        "--spinup-stability-tolerance", repr(SPINUP_STABILITY_TOLERANCE),
        "--delta-t", repr(payload["delta_t"]),
        "--melt-f", repr(payload["melt_f"]),
        "--prcp-fac-multiplier", "1.0",
        "--glen-a", repr(payload["glen_a"]),
        "--temp-melt", repr(payload["temp_melt"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = payload["project_root"]
    try:
        completed = subprocess.run(
            command,
            cwd=payload["project_root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=payload["timeout_seconds"],
        )
    except Exception as exc:
        (output_dir / "run.log").write_text(f"worker-exception: {exc}\n", encoding="utf-8")
        return {**payload, "run_status": "crash", "error": f"worker-exception:{exc}"}
    (output_dir / "run.log").write_text(
        (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0 or not (output_dir / "summary.json").exists():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
        return {
            **payload,
            "run_status": "crash",
            "returncode": completed.returncode,
            "error": tail[0] if tail else "missing-summary",
        }
    return {**payload, "run_status": "complete", "returncode": 0, "error": ""}


def _score(source_dir: Path, branch: str, source_kind: str) -> dict:
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    series = pd.read_csv(source_dir / "timeseries.csv")
    spin = series[series["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError(f"missing-phase:{source_dir}")

    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    peak_row = forward.loc[forward["length_m"].idxmax()]
    minimum = float(forward["length_m"].min())
    end_row = forward.iloc[-1]
    end = float(end_row["length_m"])
    peak = float(peak_row["length_m"])
    minimum_at_end = end <= minimum + 1e-9

    gap = abs(lia - rgi)
    corridor_low = min(lia, rgi) - 0.20 * gap
    corridor_high = max(lia, rgi) + 0.20 * gap
    lengths = forward["length_m"].to_numpy(float)
    outside = np.maximum.reduce(
        [corridor_low - lengths, lengths - corridor_high, np.zeros_like(lengths)]
    )
    peak_error = abs(peak - lia) / lia
    endpoint_error = abs(end - rgi) / rgi
    corridor_error = float(np.sqrt(np.mean((outside / max(gap, 1.0)) ** 2)))
    curve_score = peak_error + endpoint_error + corridor_error

    final_spin = spin.iloc[-1]
    previous_year = float(final_spin["year"]) - SPINUP_STABILITY_YEARS
    previous = spin.iloc[(spin["year"].astype(float) - previous_year).abs().argmin()]
    stability_span = float(final_spin["year"] - previous["year"])
    stability = (
        abs(float(final_spin["volume_m3"]) - float(previous["volume_m3"]))
        / float(previous["volume_m3"])
        if stability_span >= SPINUP_STABILITY_YEARS and float(previous["volume_m3"]) > 0
        else math.inf
    )
    finite = bool(
        np.isfinite(series[["length_m", "area_m2", "volume_m3"]].to_numpy(float)).all()
    )
    corridor_pass = float(outside.max()) <= 1e-6
    curve_pass = (
        peak_error <= 0.10
        and endpoint_error <= 0.10
        and minimum_at_end
        and corridor_pass
    )
    physical_pass = (
        curve_pass
        and finite
        and summary.get("status") == "ok"
        and bool(summary["spinup_control"]["converged"])
        and stability <= SPINUP_STABILITY_TOLERANCE
    )
    parameters = summary["parameters"]
    return {
        "system_id": int(summary["lia_unique_id"]),
        "rgi_id": str(summary["rgi_id"]),
        "branch": branch,
        "source_kind": source_kind,
        "source_dir": str(source_dir.resolve()),
        "delta_t_c": float(parameters["delta_t_c"]),
        "melt_f": float(parameters["melt_f"]),
        "prcp_fac_multiplier": float(parameters["prcp_fac_multiplier"]),
        "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "spinup_actual_years": int(summary["spinup_control"]["actual_years"]),
        "spinup_converged": bool(summary["spinup_control"]["converged"]),
        "spinup_volume_change_last_200_pct": 100.0 * stability,
        "peak_year": int(peak_row["year"]),
        "peak_length_km": peak / 1000.0,
        "lia_target_km": lia / 1000.0,
        "length_2000_km": end / 1000.0,
        "rgi_target_km": rgi / 1000.0,
        "minimum_length_km": minimum / 1000.0,
        "minimum_at_2000": minimum_at_end,
        "e_peak_lia_pct": 100.0 * peak_error,
        "e_2000_rgi_pct": 100.0 * endpoint_error,
        "max_corridor_exceedance_km": float(outside.max()) / 1000.0,
        "curve_score_pct": 100.0 * curve_score,
        "corridor_pass": corridor_pass,
        "curve_pass": curve_pass,
        "physical_pass": physical_pass,
        "numerical_status": str(summary.get("status", "unknown")),
        "finite": finite,
        "runtime_s": float(summary.get("timing_seconds", {}).get("total", math.nan)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root-template", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--existing-results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.processes <= 18:
        raise ValueError("processes must be between 1 and 18")
    for required in (
        args.references,
        args.stage2_members,
        args.climate_pixels,
        args.reconstruction_climate,
        args.elevation_band_root,
        args.existing_results_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    preflight, frozen = _load_preflight(args)
    preflight.to_csv(args.output_root / "geometry_climate_preflight.csv", index=False)

    project_root = str(Path(__file__).resolve().parents[4])
    design_rows: list[dict] = []
    payloads: list[dict] = []
    sources: dict[tuple[str, float, float], tuple[Path, str]] = {}

    for seed in SEED_BRANCHES:
        branch = str(seed["branch"])
        system_id = int(seed["system_id"])
        fixed = frozen[system_id]
        for melt_f in seed["melt_values"]:
            for delta_t in seed["delta_t_values"]:
                key = (branch, float(melt_f), float(delta_t))
                design_rows.append(
                    {
                        "branch": branch,
                        "system_id": system_id,
                        "melt_f": melt_f,
                        "delta_t_c": delta_t,
                        "prcp_fac_multiplier": 1.0,
                        "glen_a_pa_3_s_1": fixed["glen_a"],
                        "temp_melt_c": fixed["temp_melt"],
                    }
                )
                reuse_relative = next(
                    (
                        relative
                        for (reuse_melt, reuse_delta), relative in seed["reuse"].items()
                        if _same(reuse_melt, melt_f) and _same(reuse_delta, delta_t)
                    ),
                    None,
                )
                if reuse_relative is not None:
                    source_dir = args.existing_results_root / reuse_relative
                    _validate_source(source_dir, system_id, melt_f, delta_t, fixed)
                    sources[key] = (source_dir, "reused-seed")
                    continue

                output_dir = (
                    args.output_root
                    / branch
                    / f"mf_{_value_tag(melt_f)}_dt_{_value_tag(delta_t)}"
                )
                completed = (
                    output_dir / "summary.json",
                    output_dir / "timeseries.csv",
                    output_dir / "spinup_state.pkl.gz",
                    output_dir / "state_2000.pkl.gz",
                )
                if all(path.exists() for path in completed):
                    _validate_source(output_dir, system_id, melt_f, delta_t, fixed)
                    sources[key] = (output_dir, "resumed")
                    continue

                payloads.append(
                    {
                        "branch": branch,
                        "system_id": system_id,
                        "delta_t": float(delta_t),
                        "melt_f": float(melt_f),
                        "glen_a": float(fixed["glen_a"]),
                        "temp_melt": float(fixed["temp_melt"]),
                        "python_exe": sys.executable,
                        "project_root": project_root,
                        "references": str(args.references.resolve()),
                        "stage2_members": str(args.stage2_members.resolve()),
                        "climate_pixels": str(args.climate_pixels.resolve()),
                        "reconstruction_climate": str(args.reconstruction_climate.resolve()),
                        "source_gdir_root": str(fixed["source_root"].resolve()),
                        "elevation_band_root": str(args.elevation_band_root.resolve()),
                        "output_dir": str(output_dir.resolve()),
                        "workdir": str(
                            (
                                args.work_root
                                / branch
                                / f"mf_{_value_tag(melt_f)}_dt_{_value_tag(delta_t)}"
                            ).resolve()
                        ),
                        "timeout_seconds": int(args.timeout_seconds),
                    }
                )

    pd.DataFrame(design_rows).to_csv(args.output_root / "parameter_design.csv", index=False)
    if args.dry_run:
        print(f"design={len(design_rows)} new={len(payloads)} reused={len(sources)}")
        return 0

    run_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_chain, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            run_records.append(record)
            print(
                f"completed {index}/{len(futures)} branch={record['branch']} "
                f"melt_f={record['melt_f']:.6g} delta_t={record['delta_t']:+.6g} "
                f"status={record['run_status']}",
                flush=True,
            )
            if record["run_status"] == "complete":
                key = (
                    str(record["branch"]),
                    float(record["melt_f"]),
                    float(record["delta_t"]),
                )
                sources[key] = (Path(record["output_dir"]), "new")
    pd.DataFrame(run_records).to_csv(args.output_root / "run_records.csv", index=False)

    expected = {
        (str(seed["branch"]), float(melt_f), float(delta_t))
        for seed in SEED_BRANCHES
        for melt_f in seed["melt_values"]
        for delta_t in seed["delta_t_values"]
    }
    missing = sorted(expected - set(sources))
    if missing:
        pd.DataFrame(missing, columns=["branch", "melt_f", "delta_t_c"]).to_csv(
            args.output_root / "failed_or_missing_chains.csv", index=False
        )

    scores = pd.DataFrame(
        _score(source_dir, branch, source_kind)
        for (branch, _, _), (source_dir, source_kind) in sorted(sources.items())
    )
    scores = scores.sort_values(
        ["system_id", "physical_pass", "curve_pass", "curve_score_pct"],
        ascending=[True, False, False, True],
    )
    scores.to_csv(args.output_root / "local_curve_scores.csv", index=False)
    scores.groupby("branch", as_index=False).head(3).to_csv(
        args.output_root / "top3_by_branch.csv", index=False
    )
    scores.groupby("system_id", as_index=False).head(5).to_csv(
        args.output_root / "top5_by_system.csv", index=False
    )

    manifest = {
        "experiment": "B-window local melt_f-delta_t curve refinement",
        "experiment_question": (
            "Can local joint melt_f-delta_t perturbations improve existing in-corridor "
            "curves while preserving adaptive spin-up convergence?"
        ),
        "python_executable": sys.executable,
        "processes": args.processes,
        "precipitation_anomaly": "off",
        "spinup_window": list(SPINUP_WINDOW),
        "forward_period": [FORWARD_START, 2000],
        "adaptive_spinup": {
            "minimum_years": SPINUP_MIN_YEARS,
            "maximum_years": SPINUP_MAX_YEARS,
            "stability_window_years": SPINUP_STABILITY_YEARS,
            "stability_tolerance_fraction": SPINUP_STABILITY_TOLERANCE,
        },
        "design_count": len(design_rows),
        "scheduled_new_chain_count": len(payloads),
        "completed_chain_count": len(scores),
        "reused_seed_count": sum(1 for _, kind in sources.values() if kind == "reused-seed"),
        "resumed_count": sum(1 for _, kind in sources.values() if kind == "resumed"),
        "physical_pass_count": int(scores["physical_pass"].sum()) if len(scores) else 0,
        "missing_count": len(missing),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    columns = [
        "system_id",
        "branch",
        "melt_f",
        "delta_t_c",
        "e_peak_lia_pct",
        "e_2000_rgi_pct",
        "minimum_at_2000",
        "spinup_volume_change_last_200_pct",
        "physical_pass",
    ]
    print(scores.head(12)[columns].to_string(index=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
