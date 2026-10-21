"""Test temp_melt seasonality effects on the B-window shape of system 3584."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .curve_objective_v3_peak_rgi import score_curve_peak_rgi
from .run_b_curve_local_refine_v1 import (
    FORWARD_START,
    SPINUP_MAX_YEARS,
    SPINUP_MIN_YEARS,
    SPINUP_STABILITY_TOLERANCE,
    SPINUP_STABILITY_YEARS,
    SPINUP_WINDOW,
    _load_preflight,
    _run_chain,
    _same,
    _value_tag,
)


SYSTEM_ID = 3584
MELT_F = 2.3116274251298465
TEMP_MELT_DELTA_GRIDS = {
    -1.5: (0.15, 0.25, 0.35),
    -1.0: (0.70, 0.75, 0.80),
    -0.5: (1.15, 1.25, 1.35),
}


def _score_v3(source_dir: Path, source_kind: str) -> dict:
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    series = pd.read_csv(source_dir / "timeseries.csv")
    targets = summary["targets"]
    score = score_curve_peak_rgi(
        series,
        lia_length_m=float(targets["lia_length_m"]),
        rgi_length_m=float(targets["rgi_length_m"]),
    )
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    start_length = float(forward.iloc[0]["length_m"])
    parameters = summary["parameters"]
    return {
        "system_id": int(summary["lia_unique_id"]),
        "source_kind": source_kind,
        "source_dir": str(source_dir.resolve()),
        "delta_t_c": float(parameters["delta_t_c"]),
        "melt_f": float(parameters["melt_f"]),
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
        "spinup_actual_years": int(summary["spinup_control"]["actual_years"]),
        "forward_start_length_m": start_length,
        "forward_end_minus_start_m": float(score["length_2000_m"]) - start_length,
        **score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root-template", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--melt-delta-results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.processes <= 12:
        raise ValueError("processes must be between 1 and 12")
    for required in (
        args.references,
        args.stage2_members,
        args.climate_pixels,
        args.reconstruction_climate,
        args.elevation_band_root,
        args.melt_delta_results_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    preflight, frozen_all = _load_preflight(args)
    preflight = preflight[preflight["lia_unique_id"].astype(int).eq(SYSTEM_ID)]
    preflight.to_csv(args.output_root / "geometry_climate_preflight.csv", index=False)
    frozen = frozen_all[SYSTEM_ID]
    project_root = str(Path(__file__).resolve().parents[4])

    design_rows: list[dict] = []
    payloads: list[dict] = []
    sources: dict[tuple[float, float], tuple[Path, str]] = {}
    for temp_melt, delta_values in TEMP_MELT_DELTA_GRIDS.items():
        for delta_t in delta_values:
            key = (float(temp_melt), float(delta_t))
            design_rows.append(
                {
                    "system_id": SYSTEM_ID,
                    "temp_melt_c": temp_melt,
                    "delta_t_c": delta_t,
                    "delta_t_minus_temp_melt_c": delta_t - temp_melt,
                    "melt_f": MELT_F,
                    "glen_a_pa_3_s_1": frozen["glen_a"],
                }
            )
            if _same(temp_melt, -1.0):
                source_dir = (
                    args.melt_delta_results_root
                    / "3584_endpoint_shape"
                    / f"mf_{_value_tag(MELT_F)}_dt_{_value_tag(delta_t)}"
                )
                if not source_dir.exists():
                    raise FileNotFoundError(source_dir)
                sources[key] = (source_dir, "reused-melt-delta")
                continue

            output_dir = (
                args.output_root
                / f"tm_{_value_tag(temp_melt)}_dt_{_value_tag(delta_t)}"
            )
            complete = (
                output_dir / "summary.json",
                output_dir / "timeseries.csv",
                output_dir / "spinup_state.pkl.gz",
                output_dir / "state_2000.pkl.gz",
            )
            if all(path.exists() for path in complete):
                summary = json.loads(complete[0].read_text(encoding="utf-8"))
                parameters = summary["parameters"]
                if not (
                    _same(parameters["melt_f"], MELT_F)
                    and _same(parameters["delta_t_c"], delta_t)
                    and _same(parameters["temp_melt_c"], temp_melt)
                ):
                    raise ValueError(f"resume-contract-mismatch:{output_dir}")
                sources[key] = (output_dir, "resumed")
                continue

            payloads.append(
                {
                    "branch": "3584_temp_melt_shape",
                    "system_id": SYSTEM_ID,
                    "delta_t": float(delta_t),
                    "melt_f": MELT_F,
                    "glen_a": float(frozen["glen_a"]),
                    "temp_melt": float(temp_melt),
                    "python_exe": sys.executable,
                    "project_root": project_root,
                    "references": str(args.references.resolve()),
                    "stage2_members": str(args.stage2_members.resolve()),
                    "climate_pixels": str(args.climate_pixels.resolve()),
                    "reconstruction_climate": str(args.reconstruction_climate.resolve()),
                    "source_gdir_root": str(frozen["source_root"].resolve()),
                    "elevation_band_root": str(args.elevation_band_root.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "workdir": str(
                        (
                            args.work_root
                            / f"tm_{_value_tag(temp_melt)}_dt_{_value_tag(delta_t)}"
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
                f"completed {index}/{len(futures)} temp_melt={record['temp_melt']:+.2f} "
                f"delta_t={record['delta_t']:+.2f} status={record['run_status']}",
                flush=True,
            )
            if record["run_status"] == "complete":
                sources[(record["temp_melt"], record["delta_t"])] = (
                    Path(record["output_dir"]),
                    "new",
                )
    pd.DataFrame(run_records).to_csv(args.output_root / "run_records.csv", index=False)

    expected = {
        (float(temp_melt), float(delta_t))
        for temp_melt, delta_values in TEMP_MELT_DELTA_GRIDS.items()
        for delta_t in delta_values
    }
    missing = sorted(expected - set(sources))
    if missing:
        pd.DataFrame(missing, columns=["temp_melt_c", "delta_t_c"]).to_csv(
            args.output_root / "failed_or_missing_chains.csv", index=False
        )

    scores = pd.DataFrame(
        _score_v3(source_dir, source_kind)
        for _, (source_dir, source_kind) in sorted(sources.items())
    ).sort_values(["accepted_curve_v3", "objective_v3"], ascending=[False, True])
    scores.to_csv(args.output_root / "temp_melt_shape_scores_v3.csv", index=False)
    manifest = {
        "experiment": "B-window system-3584 temp_melt shape test",
        "experiment_question": (
            "Does temp_melt alter the forward response shape after Delta-T is "
            "recentered, or is it effectively redundant for this branch?"
        ),
        "python_executable": sys.executable,
        "processes": args.processes,
        "temp_melt_delta_grids": {
            str(key): list(values) for key, values in TEMP_MELT_DELTA_GRIDS.items()
        },
        "melt_f": MELT_F,
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
        "completed_chain_count": len(scores),
        "accepted_v3_count": int(scores["accepted_curve_v3"].sum()),
        "missing_count": len(missing),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        scores[
            [
                "temp_melt_c",
                "delta_t_c",
                "objective_v3",
                "failure_reasons_v3",
                "e_peak_lia_v3",
                "e_2000_rgi_v3",
                "spinup_volume_change_last_200",
                "forward_end_minus_start_m",
                "minimum_at_2000",
            ]
        ].to_string(index=False)
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
