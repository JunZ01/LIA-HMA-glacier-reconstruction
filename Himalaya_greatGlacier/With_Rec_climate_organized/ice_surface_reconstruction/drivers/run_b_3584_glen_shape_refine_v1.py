"""Test whether glen_a changes the B-window forward shape of system 3584."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

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
    _score,
    _validate_source,
    _value_tag,
)


SYSTEM_ID = 3584
MELT_F = 2.3116274251298465
GLEN_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DELTA_T_VALUES = (0.70, 0.75, 0.80)


def _shape_metrics(source_dir: Path) -> dict:
    series = pd.read_csv(source_dir / "timeseries.csv")
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    start = float(forward.iloc[0]["length_m"])
    end = float(forward.iloc[-1]["length_m"])
    return {
        "forward_start_length_km": start / 1000.0,
        "forward_end_minus_start_km": (end - start) / 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=12)
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
    if not 1 <= args.processes <= 18:
        raise ValueError("processes must be between 1 and 18")
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
    base_glen = float(frozen["glen_a"])
    project_root = str(Path(__file__).resolve().parents[4])

    design_rows: list[dict] = []
    payloads: list[dict] = []
    sources: dict[tuple[float, float], tuple[Path, str]] = {}
    for multiplier in GLEN_MULTIPLIERS:
        glen_a = base_glen * multiplier
        for delta_t in DELTA_T_VALUES:
            key = (float(multiplier), float(delta_t))
            design_rows.append(
                {
                    "system_id": SYSTEM_ID,
                    "glen_a_multiplier": multiplier,
                    "glen_a_pa_3_s_1": glen_a,
                    "melt_f": MELT_F,
                    "delta_t_c": delta_t,
                }
            )
            if _same(multiplier, 1.0):
                source_dir = (
                    args.melt_delta_results_root
                    / "3584_endpoint_shape"
                    / f"mf_{_value_tag(MELT_F)}_dt_{_value_tag(delta_t)}"
                )
                if not source_dir.exists() and _same(delta_t, 0.75):
                    raise FileNotFoundError(source_dir)
                if source_dir.exists():
                    _validate_source(source_dir, SYSTEM_ID, MELT_F, delta_t, frozen)
                    sources[key] = (source_dir, "reused-melt-delta")
                    continue

            output_dir = (
                args.output_root
                / f"glen_x{_value_tag(multiplier)}_dt_{_value_tag(delta_t)}"
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
                    and _same(parameters["glen_a_pa_3_s_1"], glen_a)
                ):
                    raise ValueError(f"resume-contract-mismatch:{output_dir}")
                sources[key] = (output_dir, "resumed")
                continue

            payloads.append(
                {
                    "glen_a_multiplier": float(multiplier),
                    "branch": "3584_glen_shape",
                    "system_id": SYSTEM_ID,
                    "delta_t": float(delta_t),
                    "melt_f": MELT_F,
                    "glen_a": glen_a,
                    "temp_melt": float(frozen["temp_melt"]),
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
                            / f"glen_x{_value_tag(multiplier)}_dt_{_value_tag(delta_t)}"
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
                f"completed {index}/{len(futures)} glen_x={record['glen_a_multiplier']:.3g} "
                f"delta_t={record['delta_t']:+.2f} status={record['run_status']}",
                flush=True,
            )
            if record["run_status"] == "complete":
                sources[(record["glen_a_multiplier"], record["delta_t"])] = (
                    Path(record["output_dir"]),
                    "new",
                )
    pd.DataFrame(run_records).to_csv(args.output_root / "run_records.csv", index=False)

    expected = {
        (float(multiplier), float(delta_t))
        for multiplier in GLEN_MULTIPLIERS
        for delta_t in DELTA_T_VALUES
    }
    missing = sorted(expected - set(sources))
    if missing:
        pd.DataFrame(missing, columns=["glen_a_multiplier", "delta_t_c"]).to_csv(
            args.output_root / "failed_or_missing_chains.csv", index=False
        )

    score_rows: list[dict] = []
    for (multiplier, _), (source_dir, source_kind) in sorted(sources.items()):
        row = _score(source_dir, "3584_glen_shape", source_kind)
        row["glen_a_multiplier"] = multiplier
        row.update(_shape_metrics(source_dir))
        score_rows.append(row)
    scores = pd.DataFrame(score_rows).sort_values(
        ["physical_pass", "curve_pass", "curve_score_pct"],
        ascending=[False, False, True],
    )
    scores.to_csv(args.output_root / "glen_shape_scores.csv", index=False)
    scores.head(10).to_csv(args.output_root / "top10.csv", index=False)

    manifest = {
        "experiment": "B-window system-3584 glen_a shape refinement",
        "experiment_question": (
            "Can a glen_a change make the 2000 endpoint the forward minimum "
            "without losing the LIA peak, RGI endpoint, corridor, or spin-up stability?"
        ),
        "python_executable": sys.executable,
        "processes": args.processes,
        "glen_a_multipliers": list(GLEN_MULTIPLIERS),
        "delta_t_values_c": list(DELTA_T_VALUES),
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
        "physical_pass_count": int(scores["physical_pass"].sum()),
        "missing_count": len(missing),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        scores[
            [
                "glen_a_multiplier",
                "delta_t_c",
                "e_peak_lia_pct",
                "e_2000_rgi_pct",
                "forward_end_minus_start_km",
                "minimum_at_2000",
                "spinup_volume_change_last_200_pct",
                "physical_pass",
            ]
        ].head(12).to_string(index=False)
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
