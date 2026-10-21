"""Run the three-system layer-1 Delta-T profile with frozen Stage-2 parameters.

Layer 1 estimates only the spin-up temperature offset.  Forward trajectories
are retained by the delegated chain runner, but they never enter the layer-1
score or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import _gdir_path


SYSTEM_IDS = (1044, 1545, 3584)
DELTA_T_VALUES = tuple(np.arange(-3.0, 1.5 + 0.001, 0.5))
ENSEMBLE_ROLE = "mid-turnover"
SPINUP_WINDOW = (1807, 1836)
SPINUP_YEARS = 500
SPINUP_STABILITY_START = 300
TEMP_MELT_EXPECTED = -1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tag(value: float) -> str:
    sign = "m" if value < 0 else "p"
    return f"dt_{sign}{abs(value):.1f}".replace(".", "p")


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _source_root_for_system(root: Path, system_id: int) -> Path:
    """Resolve an optional per-system source-root template."""
    text = str(root)
    if "{system_id}" in text:
        return Path(text.format(system_id=system_id))
    return root


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = pd.read_csv(args.references)
    members = pd.read_csv(args.stage2_members)
    pixels = pd.read_csv(args.climate_pixels)
    audit = pd.read_csv(args.elevation_band_root / "audit" / "core191_elevation_band_audit.csv")

    preflight_rows: list[dict] = []
    frozen_rows: list[dict] = []
    for system_id in args.system_ids:
        ref_rows = refs[refs["lia_unique_id"].astype(int).eq(system_id)]
        if len(ref_rows) != 1:
            raise ValueError(f"reference-row-count:{system_id}:{len(ref_rows)}")
        ref = ref_rows.iloc[0]
        rgi_id = str(ref["rgi_id"])

        member_rows = members[
            members["lia_unique_id"].astype(int).eq(system_id)
            & members["rgi_id"].astype(str).eq(rgi_id)
            & members["ensemble_role"].astype(str).eq(ENSEMBLE_ROLE)
        ]
        if len(member_rows) != 1:
            raise ValueError(f"stage2-member-row-count:{system_id}:{rgi_id}:{len(member_rows)}")
        member = member_rows.iloc[0]

        pixel_rows = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
        ]
        if len(pixel_rows) != 1:
            raise ValueError(f"climate-pixel-row-count:{system_id}:{rgi_id}:{len(pixel_rows)}")

        audit_rows = audit[
            audit["lia_unique_id"].astype(int).eq(system_id)
            & audit["rgi_id"].astype(str).eq(rgi_id)
        ]
        if len(audit_rows) != 1:
            raise ValueError(f"geometry-audit-row-count:{system_id}:{rgi_id}:{len(audit_rows)}")
        audit_row = audit_rows.iloc[0]
        if not bool(audit_row["all_geometry_gates_pass"]):
            raise ValueError(f"geometry-gate-failed:{system_id}:{rgi_id}")
        mismatch = abs(float(audit_row["corrected_length_error_fraction"]))
        if mismatch > 0.10:
            raise ValueError(f"initial-length-mismatch:{system_id}:{mismatch}")

        flowline_path = args.elevation_band_root / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
        source_root = _source_root_for_system(args.source_gdir_root, system_id)
        source_gdir = _gdir_path(source_root, rgi_id)
        mb_calib_path = source_gdir / "mb_calib.json"
        for required in (flowline_path, source_gdir, mb_calib_path):
            if not required.exists():
                raise FileNotFoundError(required)
        mb_calib = json.loads(mb_calib_path.read_text(encoding="utf-8"))
        temp_melt = float(mb_calib["mb_global_params"]["temp_melt"])
        if not _same(temp_melt, TEMP_MELT_EXPECTED):
            raise ValueError(f"temp-melt-conflict:{system_id}:{temp_melt}")

        preflight_rows.append({
            "lia_unique_id": system_id,
            "rgi_id": rgi_id,
            "lia_length_target_m": float(ref["lia_registered_length_m"]),
            "rgi_length_target_m": float(ref["rgi_registered_length_m"]),
            "length_definition": "Core-191 corrected elevation-band model length vs registered spatial main-flowline target",
            "initial_model_length_m": float(audit_row["pseudo_flowline_length_m"]),
            "initial_length_mismatch_fraction": mismatch,
            "geometry_gates_pass": True,
            "climate_id": "ensemble_rec_ref",
            "precipitation_anomaly": "off",
            "spinup_window": "1807-1836",
            "spinup_years": SPINUP_YEARS,
            "melt_f": float(member["melt_f"]),
            "prcp_fac": float(member["prcp_fac"]),
            "base_temp_bias_c": float(member["temp_bias"]),
            "glen_a_pa_3_s_1": float(member["glen_a_pa_3_s_1"]),
            "fs": float(member["fs"]),
            "temp_melt_c": temp_melt,
            "source_gdir": str(source_gdir.resolve()),
            "flowline_file": str(flowline_path.resolve()),
        })
        frozen_rows.append({
            "system_id": system_id,
            "rgi_id": rgi_id,
            "melt_f": float(member["melt_f"]),
            "prcp_fac": float(member["prcp_fac"]),
            "base_temp_bias_c": float(member["temp_bias"]),
            "glen_a": float(member["glen_a_pa_3_s_1"]),
            "temp_melt": temp_melt,
            "lia_target_m": float(ref["lia_registered_length_m"]),
            "rgi_target_m": float(ref["rgi_registered_length_m"]),
        })

    return pd.DataFrame(preflight_rows), pd.DataFrame(frozen_rows)


def _validate_reuse(source_dir: Path, frozen: pd.Series, delta_t: float) -> None:
    summary_path = source_dir / "summary.json"
    timeseries_path = source_dir / "timeseries.csv"
    if not summary_path.exists() or not timeseries_path.exists():
        raise FileNotFoundError(f"incomplete-reuse-source:{source_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    params = summary["parameters"]
    targets = summary["targets"]
    checks = {
        "system_id": int(summary["lia_unique_id"]) == int(frozen["system_id"]),
        "rgi_id": str(summary["rgi_id"]) == str(frozen["rgi_id"]),
        "spinup_window": list(summary["spinup_window"]) == list(SPINUP_WINDOW),
        "spinup_years": int(summary["spinup_years"]) == SPINUP_YEARS,
        "delta_t": _same(params["delta_t_c"], delta_t),
        "melt_f": _same(params["melt_f"], frozen["melt_f"]),
        "prcp_fac": _same(params["prcp_fac"], frozen["prcp_fac"]),
        "glen_a": _same(params["glen_a_pa_3_s_1"], frozen["glen_a"]),
        "temp_melt": _same(params["temp_melt_c"], frozen["temp_melt"]),
        "lia_target": _same(targets["lia_length_m"], frozen["lia_target_m"]),
        "rgi_target": _same(targets["rgi_length_m"], frozen["rgi_target_m"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"reuse-contract-mismatch:{source_dir}:{','.join(failed)}")
    ts = pd.read_csv(timeseries_path)
    required = ts[
        ts["phase"].astype(str).eq("spinup")
        & ts["year"].astype(int).isin([SPINUP_STABILITY_START, SPINUP_YEARS])
    ]
    if set(required["year"].astype(int)) != {SPINUP_STABILITY_START, SPINUP_YEARS}:
        raise ValueError(f"reuse-missing-spinup-checkpoints:{source_dir}")


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
        "--ensemble-role", ENSEMBLE_ROLE,
        "--climate-id", "ensemble_rec_ref",
        "--prcp-mode", "off",
        "--spinup-window-start", str(SPINUP_WINDOW[0]),
        "--spinup-window-end", str(SPINUP_WINDOW[1]),
        "--start-year", "1836",
        "--spinup-years", str(SPINUP_YEARS),
        "--spinup-checkpoint-years", "100",
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
    return {**payload, "run_status": "complete", "error": ""}


def _score_chain(source_dir: Path, source_kind: str, frozen: pd.Series, delta_t: float) -> dict:
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    ts = pd.read_csv(source_dir / "timeseries.csv")
    spin = ts[ts["phase"].astype(str).eq("spinup")].copy()
    row_300 = spin[spin["year"].astype(int).eq(SPINUP_STABILITY_START)].iloc[0]
    row_500 = spin[spin["year"].astype(int).eq(SPINUP_YEARS)].iloc[0]
    numeric_columns = ["area_m2", "volume_m3", "length_m", "minimum_thickness_m"]
    finite = bool(np.isfinite(ts[numeric_columns].to_numpy(dtype=float)).all())
    positive_v300 = float(row_300["volume_m3"]) > 0
    numerical_clean = finite and positive_v300 and summary.get("status") != "crash"
    e_lia = abs(float(row_500["length_m"]) - float(frozen["lia_target_m"])) / float(frozen["lia_target_m"])
    e_spin = (
        abs(float(row_500["volume_m3"]) - float(row_300["volume_m3"])) / float(row_300["volume_m3"])
        if positive_v300 else math.inf
    )
    objective = (e_lia / 0.10) ** 2 + (e_spin / 0.02) ** 2
    return {
        "system_id": int(frozen["system_id"]),
        "rgi_id": str(frozen["rgi_id"]),
        "delta_t_c": float(delta_t),
        "effective_temp_bias_c": float(frozen["base_temp_bias_c"]) + float(delta_t),
        "melt_f": float(frozen["melt_f"]),
        "prcp_fac": float(frozen["prcp_fac"]),
        "glen_a_pa_3_s_1": float(frozen["glen_a"]),
        "temp_melt_c": float(frozen["temp_melt"]),
        "source_kind": source_kind,
        "source_dir": str(source_dir.resolve()),
        "chain_status": str(summary.get("status", "unknown")),
        "numerical_clean": numerical_clean,
        "l_spinup_500_m": float(row_500["length_m"]),
        "lia_target_m": float(frozen["lia_target_m"]),
        "e_lia": e_lia,
        "v_spinup_300_m3": float(row_300["volume_m3"]),
        "v_spinup_500_m3": float(row_500["volume_m3"]),
        "e_spin_300_500": e_spin,
        "a_spinup_500_m2": float(row_500["area_m2"]),
        "recorded_only_l_2000_m": float(summary["metrics"]["l_2000_m"]),
        "recorded_only_e_2000": float(summary["metrics"]["e_2000"]),
        "layer1_objective": objective,
        "lia_gate_pass": e_lia <= 0.10,
        "spin_gate_pass": e_spin <= 0.02,
        "layer1_accept": numerical_clean and e_lia <= 0.10 and e_spin <= 0.02,
        "forward_used_for_selection": False,
        "runtime_s": float(summary.get("timing_seconds", {}).get("total", math.nan)),
    }


def _plot_system(group: pd.DataFrame, output_path: Path) -> None:
    group = group.sort_values("delta_t_c")
    best = group[group["numerical_clean"]].sort_values("layer1_objective").iloc[0]
    target_km = float(group["lia_target_m"].iloc[0]) / 1000.0
    x = group["delta_t_c"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True, constrained_layout=True)

    axes[0].plot(x, group["l_spinup_500_m"] / 1000.0, "o-", color="#1f5a7a", lw=1.8)
    axes[0].axhspan(0.9 * target_km, 1.1 * target_km, color="#74a66a", alpha=0.18)
    axes[0].axhline(target_km, color="#376b32", lw=1.2, label="LIA target")
    axes[0].set_ylabel("500-year length (km)")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(x, 100.0 * group["e_lia"], "o-", color="#b24a3b", lw=1.8)
    axes[1].axhline(10.0, color="#333333", ls="--", lw=1.0)
    axes[1].set_ylabel("LIA length error (%)")

    axes[2].plot(x, 100.0 * group["e_spin_300_500"], "o-", color="#65508f", lw=1.8)
    axes[2].axhline(2.0, color="#333333", ls="--", lw=1.0)
    axes[2].set_ylabel("Volume change 300-500 (%)")
    axes[2].set_xlabel("Additional Delta T (K)")

    for axis in axes:
        axis.grid(True, color="#d7d7d7", lw=0.6, alpha=0.8)
        axis.axvline(float(best["delta_t_c"]), color="#d08a1e", ls=":", lw=1.2)
    fig.suptitle(
        f"LIA system {int(group['system_id'].iloc[0])}: layer-1 Delta T response\n"
        f"best grid point {float(best['delta_t_c']):+.1f} K; forward not used",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-ids", nargs="+", type=int, default=list(SYSTEM_IDS))
    parser.add_argument("--delta-t-values", nargs="+", type=float, default=list(DELTA_T_VALUES))
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--reuse-delta-zero-root", type=Path)
    parser.add_argument("--reuse-delta-plus1p5-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    if tuple(args.system_ids) != SYSTEM_IDS:
        raise ValueError(f"layer1-system-contract:{args.system_ids}")
    if tuple(float(value) for value in args.delta_t_values) != DELTA_T_VALUES:
        raise ValueError(f"layer1-delta-t-contract:{args.delta_t_values}")
    if not 1 <= args.processes <= 12:
        raise ValueError("processes must be between 1 and 12")
    for required in (
        args.references,
        args.stage2_members,
        args.climate_pixels,
        args.reconstruction_climate,
        args.elevation_band_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    for system_id in args.system_ids:
        source_root = _source_root_for_system(args.source_gdir_root, system_id)
        if not source_root.exists():
            raise FileNotFoundError(source_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight, frozen = _load_inputs(args)
    preflight.to_csv(args.output_root / "preflight_frozen_parameters.csv", index=False)

    project_root = str(Path(__file__).resolve().parents[4])
    payloads: list[dict] = []
    sources: dict[tuple[int, float], tuple[Path, str]] = {}
    for frozen_row in frozen.itertuples(index=False):
        frozen_series = pd.Series(frozen_row._asdict())
        for delta_t in DELTA_T_VALUES:
            reuse_root = None
            if _same(delta_t, 0.0):
                reuse_root = args.reuse_delta_zero_root
            elif _same(delta_t, 1.5):
                reuse_root = args.reuse_delta_plus1p5_root
            if reuse_root is not None:
                source_dir = reuse_root / f"system_{frozen_row.system_id}"
                _validate_reuse(source_dir, frozen_series, delta_t)
                sources[(int(frozen_row.system_id), float(delta_t))] = (source_dir, "reused")
                continue

            output_dir = args.output_root / f"system_{frozen_row.system_id}" / _tag(delta_t)
            payloads.append({
                "system_id": int(frozen_row.system_id),
                "delta_t": float(delta_t),
                "melt_f": float(frozen_row.melt_f),
                "glen_a": float(frozen_row.glen_a),
                "temp_melt": float(frozen_row.temp_melt),
                "python_exe": sys.executable,
                "project_root": project_root,
                "references": str(args.references.resolve()),
                "stage2_members": str(args.stage2_members.resolve()),
                "climate_pixels": str(args.climate_pixels.resolve()),
                "reconstruction_climate": str(args.reconstruction_climate.resolve()),
                "source_gdir_root": str(
                    _source_root_for_system(args.source_gdir_root, int(frozen_row.system_id)).resolve()
                ),
                "elevation_band_root": str(args.elevation_band_root.resolve()),
                "output_dir": str(output_dir.resolve()),
                "workdir": str((args.output_root / "work" / f"system_{frozen_row.system_id}" / _tag(delta_t)).resolve()),
                "timeout_seconds": int(args.timeout_seconds),
            })

    run_records: list[dict] = []
    if payloads:
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = {pool.submit(_run_chain, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                run_records.append(record)
                print(
                    f"completed {index}/{len(futures)} system={record['system_id']} "
                    f"delta_t={record['delta_t']:+.1f} status={record['run_status']}",
                    flush=True,
                )
                if record["run_status"] == "complete":
                    sources[(int(record["system_id"]), float(record["delta_t"]))] = (
                        Path(record["output_dir"]), "new"
                    )
    pd.DataFrame(run_records).to_csv(args.output_root / "run_records.csv", index=False)

    expected = {(system_id, float(delta_t)) for system_id in SYSTEM_IDS for delta_t in DELTA_T_VALUES}
    missing = sorted(expected - set(sources))
    if missing:
        pd.DataFrame(missing, columns=["system_id", "delta_t_c"]).to_csv(
            args.output_root / "failed_or_missing_chains.csv", index=False
        )

    scored_rows: list[dict] = []
    for (system_id, delta_t), (source_dir, source_kind) in sorted(sources.items()):
        frozen_row = frozen[frozen["system_id"].astype(int).eq(system_id)].iloc[0]
        scored_rows.append(_score_chain(source_dir, source_kind, frozen_row, delta_t))
    results = pd.DataFrame(scored_rows).sort_values(["system_id", "delta_t_c"])
    if len(results):
        results["layer1_rank"] = (
            results.groupby("system_id")["layer1_objective"].rank(method="first").astype(int)
        )
    results.to_csv(args.output_root / "layer1_results.csv", index=False)

    best_rows: list[pd.Series] = []
    for system_id, group in results.groupby("system_id"):
        clean = group[group["numerical_clean"]]
        if clean.empty:
            continue
        best_rows.append(clean.sort_values("layer1_objective").iloc[0])
        _plot_system(group, args.output_root / f"layer1_response_system_{int(system_id)}.png")
    best = pd.DataFrame(best_rows)
    best.to_csv(args.output_root / "layer1_best_by_system.csv", index=False)

    manifest = {
        "experiment": "three-system-layer1-delta-t-profile",
        "python_executable": sys.executable,
        "system_ids": list(SYSTEM_IDS),
        "delta_t_values_c": list(DELTA_T_VALUES),
        "selection_terms": ["e_lia", "e_spin_300_500"],
        "forward_used_for_selection": False,
        "processes": args.processes,
        "new_chain_count": len(payloads),
        "reused_chain_count": len(sources) - sum(1 for value in sources.values() if value[1] == "new"),
        "completed_chain_count": len(results),
        "input_sha256": {
            "references": _sha256(args.references),
            "stage2_members": _sha256(args.stage2_members),
            "climate_pixels": _sha256(args.climate_pixels),
            "reconstruction_climate": _sha256(args.reconstruction_climate),
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(best[["system_id", "delta_t_c", "e_lia", "e_spin_300_500", "layer1_accept"]].to_string(index=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
