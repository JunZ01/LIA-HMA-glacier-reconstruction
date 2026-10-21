"""Profile melt_f with Delta-T re-optimization for three LIA systems.

Experiment question: after the Delta-T-only layer failed, can a directed
melt-factor change produce a 500-year state that satisfies both the LIA length
and 300-500-year volume-stability gates? Forward output is retained but is not
used for this layer's selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lia_delta_t_layer1_batch import (
    ENSEMBLE_ROLE,
    SPINUP_WINDOW,
    SPINUP_YEARS,
    SYSTEM_IDS,
    _load_inputs,
    _run_chain,
    _same,
    _score_chain,
    _sha256,
    _source_root_for_system,
    _validate_reuse,
)


MELT_MULTIPLIERS = {
    1044: (0.6, 0.7, 0.8, 0.9, 1.0),
    1545: (0.6, 0.7, 0.8, 0.9, 1.0),
    3584: (1.0, 1.25, 1.5, 1.75, 2.0),
}
DELTA_T_GRIDS = {
    1044: (0.5, 0.75, 1.0, 1.25, 1.5),
    1545: (0.5, 0.75, 1.0, 1.25, 1.5),
    3584: (0.0, 0.25, 0.5, 0.75, 1.0),
}


def _value_tag(prefix: str, value: float) -> str:
    sign = "m" if value < 0 else "p"
    return f"{prefix}_{sign}{abs(value):.3f}".replace(".", "p")


def _modified_frozen(row: pd.Series, melt_multiplier: float) -> pd.Series:
    result = row.copy()
    result["melt_f_base"] = float(row["melt_f"])
    result["melt_f_multiplier"] = float(melt_multiplier)
    result["melt_f"] = float(row["melt_f"]) * float(melt_multiplier)
    return result


def _plot_system(results: pd.DataFrame, profile: pd.DataFrame, output_path: Path) -> None:
    results = results.sort_values(["melt_f_multiplier", "delta_t_c"])
    profile = profile.sort_values("melt_f_multiplier")
    target_km = float(results["lia_target_m"].iloc[0]) / 1000.0
    best = results[results["numerical_clean"]].sort_values("layer1_objective").iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, results["melt_f_multiplier"].nunique()))
    for color, (multiplier, group) in zip(colors, results.groupby("melt_f_multiplier")):
        group = group.sort_values("delta_t_c")
        label = f"melt_f x {float(multiplier):g}"
        axes[0].plot(group["delta_t_c"], group["l_spinup_500_m"] / 1000.0, "o-", color=color, label=label)
        axes[1].plot(group["delta_t_c"], 100.0 * group["e_spin_300_500"], "o-", color=color, label=label)

    axes[0].axhspan(0.9 * target_km, 1.1 * target_km, color="#74a66a", alpha=0.18)
    axes[0].axhline(target_km, color="#376b32", lw=1.2)
    axes[0].set_ylabel("500-year length (km)")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].grid(True, color="#d7d7d7", lw=0.6)

    axes[1].axhline(2.0, color="#333333", ls="--", lw=1.0)
    axes[1].set_ylabel("Volume change 300-500 (%)")
    axes[1].set_xlabel("Additional Delta T (K)")
    axes[1].set_yscale("symlog", linthresh=2.0)
    axes[1].grid(True, color="#d7d7d7", lw=0.6)

    axes[2].plot(profile["melt_f_multiplier"], profile["profiled_objective"], "o-", color="#8a4f35")
    axes[2].axvline(float(best["melt_f_multiplier"]), color="#d08a1e", ls=":", lw=1.2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Stage-2 melt_f multiplier")
    axes[2].set_ylabel("Profiled layer-2 objective")
    axes[2].grid(True, color="#d7d7d7", lw=0.6)

    system_id = int(results["system_id"].iloc[0])
    fig.suptitle(
        f"LIA system {system_id}: melt_f profile with Delta T re-optimization\n"
        f"best melt_f x {float(best['melt_f_multiplier']):g}, Delta T {float(best['delta_t_c']):+.2f} K",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--layer1-results", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.system_ids = list(SYSTEM_IDS)

    if not 1 <= args.processes <= 12:
        raise ValueError("processes must be between 1 and 12")
    for required in (
        args.references,
        args.stage2_members,
        args.climate_pixels,
        args.reconstruction_climate,
        args.elevation_band_root,
        args.layer1_results,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    for system_id in SYSTEM_IDS:
        source_root = _source_root_for_system(args.source_gdir_root, system_id)
        if not source_root.exists():
            raise FileNotFoundError(source_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    preflight, frozen = _load_inputs(args)
    preflight.to_csv(args.output_root / "preflight_frozen_parameters.csv", index=False)
    layer1 = pd.read_csv(args.layer1_results)
    project_root = str(Path(__file__).resolve().parents[4])

    design_rows: list[dict] = []
    payloads: list[dict] = []
    sources: dict[tuple[int, float, float], tuple[Path, str]] = {}
    frozen_by_system = {
        int(row["system_id"]): row for _, row in frozen.iterrows()
    }

    for system_id in SYSTEM_IDS:
        base = frozen_by_system[system_id]
        for melt_multiplier in MELT_MULTIPLIERS[system_id]:
            current = _modified_frozen(base, melt_multiplier)
            for delta_t in DELTA_T_GRIDS[system_id]:
                design_rows.append({
                    "system_id": system_id,
                    "rgi_id": str(current["rgi_id"]),
                    "melt_f_base": float(current["melt_f_base"]),
                    "melt_f_multiplier": melt_multiplier,
                    "melt_f": float(current["melt_f"]),
                    "delta_t_c": delta_t,
                    "effective_temp_bias_c": float(current["base_temp_bias_c"]) + delta_t,
                    "glen_a_pa_3_s_1": float(current["glen_a"]),
                    "prcp_fac": float(current["prcp_fac"]),
                    "temp_melt_c": float(current["temp_melt"]),
                })
                key = (system_id, float(melt_multiplier), float(delta_t))
                if _same(melt_multiplier, 1.0):
                    reuse = layer1[
                        layer1["system_id"].astype(int).eq(system_id)
                        & np.isclose(layer1["delta_t_c"].astype(float), delta_t)
                    ]
                    if len(reuse) == 1:
                        source_dir = Path(str(reuse.iloc[0]["source_dir"]))
                        _validate_reuse(source_dir, current, delta_t)
                        sources[key] = (source_dir, "reused-layer1")
                        continue

                output_dir = (
                    args.output_root
                    / f"system_{system_id}"
                    / _value_tag("melt", melt_multiplier)
                    / _value_tag("dt", delta_t)
                )
                completed_files = (
                    output_dir / "summary.json",
                    output_dir / "timeseries.csv",
                    output_dir / "spinup_state.pkl.gz",
                    output_dir / "state_2000.pkl.gz",
                )
                if all(path.exists() for path in completed_files):
                    _validate_reuse(output_dir, current, delta_t)
                    sources[key] = (output_dir, "resumed-layer2")
                    continue
                workdir = (
                    args.work_root
                    / f"s{system_id}"
                    / _value_tag("m", melt_multiplier)
                    / _value_tag("t", delta_t)
                )
                payloads.append({
                    "system_id": system_id,
                    "melt_f_multiplier": float(melt_multiplier),
                    "delta_t": float(delta_t),
                    "melt_f": float(current["melt_f"]),
                    "glen_a": float(current["glen_a"]),
                    "temp_melt": float(current["temp_melt"]),
                    "python_exe": sys.executable,
                    "project_root": project_root,
                    "references": str(args.references.resolve()),
                    "stage2_members": str(args.stage2_members.resolve()),
                    "climate_pixels": str(args.climate_pixels.resolve()),
                    "reconstruction_climate": str(args.reconstruction_climate.resolve()),
                    "source_gdir_root": str(
                        _source_root_for_system(args.source_gdir_root, system_id).resolve()
                    ),
                    "elevation_band_root": str(args.elevation_band_root.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "workdir": str(workdir.resolve()),
                    "timeout_seconds": int(args.timeout_seconds),
                })

    design = pd.DataFrame(design_rows)
    design.to_csv(args.output_root / "parameter_design.csv", index=False)
    if args.dry_run:
        print(f"dry-run design={len(design)} new={len(payloads)} reused={len(sources)}")
        return 0

    run_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_chain, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            run_records.append(record)
            print(
                f"completed {index}/{len(futures)} system={record['system_id']} "
                f"melt_x={record['melt_f_multiplier']:.3f} "
                f"delta_t={record['delta_t']:+.2f} status={record['run_status']}",
                flush=True,
            )
            if record["run_status"] == "complete":
                key = (
                    int(record["system_id"]),
                    float(record["melt_f_multiplier"]),
                    float(record["delta_t"]),
                )
                sources[key] = (Path(record["output_dir"]), "new")

    pd.DataFrame(run_records).to_csv(args.output_root / "run_records.csv", index=False)
    expected = {
        (system_id, float(multiplier), float(delta_t))
        for system_id in SYSTEM_IDS
        for multiplier in MELT_MULTIPLIERS[system_id]
        for delta_t in DELTA_T_GRIDS[system_id]
    }
    missing = sorted(expected - set(sources))
    if missing:
        pd.DataFrame(
            missing, columns=["system_id", "melt_f_multiplier", "delta_t_c"]
        ).to_csv(args.output_root / "failed_or_missing_chains.csv", index=False)

    scored_rows: list[dict] = []
    for (system_id, melt_multiplier, delta_t), (source_dir, source_kind) in sorted(sources.items()):
        current = _modified_frozen(frozen_by_system[system_id], melt_multiplier)
        scored = _score_chain(source_dir, source_kind, current, delta_t)
        scored["melt_f_base"] = float(current["melt_f_base"])
        scored["melt_f_multiplier"] = melt_multiplier
        scored_rows.append(scored)
    results = pd.DataFrame(scored_rows).sort_values(
        ["system_id", "melt_f_multiplier", "delta_t_c"]
    )
    results.to_csv(args.output_root / "layer2_results.csv", index=False)

    profile_rows: list[pd.Series] = []
    best_rows: list[pd.Series] = []
    for system_id, group in results.groupby("system_id"):
        clean = group[group["numerical_clean"]]
        if clean.empty:
            continue
        for multiplier, melt_group in clean.groupby("melt_f_multiplier"):
            row = melt_group.sort_values("layer1_objective").iloc[0].copy()
            row["profiled_objective"] = float(row["layer1_objective"])
            profile_rows.append(row)
        best_rows.append(clean.sort_values("layer1_objective").iloc[0])

    profile = pd.DataFrame(profile_rows).sort_values(["system_id", "melt_f_multiplier"])
    best = pd.DataFrame(best_rows).sort_values("system_id")
    profile.to_csv(args.output_root / "layer2_profile_by_melt.csv", index=False)
    best.to_csv(args.output_root / "layer2_best_by_system.csv", index=False)
    for system_id, group in results.groupby("system_id"):
        _plot_system(
            group,
            profile[profile["system_id"].astype(int).eq(int(system_id))],
            args.output_root / f"layer2_response_system_{int(system_id)}.png",
        )

    manifest = {
        "experiment": "three-system-layer2-melt-profile",
        "python_executable": sys.executable,
        "system_ids": list(SYSTEM_IDS),
        "melt_f_multipliers": {str(k): list(v) for k, v in MELT_MULTIPLIERS.items()},
        "delta_t_grids_c": {str(k): list(v) for k, v in DELTA_T_GRIDS.items()},
        "selection_terms": ["e_lia", "e_spin_300_500"],
        "forward_used_for_selection": False,
        "processes": args.processes,
        "design_count": len(design),
        "scheduled_new_chain_count": len(payloads),
        "newly_completed_chain_count": sum(
            1 for source in sources.values() if source[1] == "new"
        ),
        "reused_layer1_chain_count": sum(
            1 for source in sources.values() if source[1] == "reused-layer1"
        ),
        "resumed_layer2_chain_count": sum(
            1 for source in sources.values() if source[1] == "resumed-layer2"
        ),
        "completed_chain_count": len(results),
        "input_sha256": {
            "references": _sha256(args.references),
            "stage2_members": _sha256(args.stage2_members),
            "climate_pixels": _sha256(args.climate_pixels),
            "reconstruction_climate": _sha256(args.reconstruction_climate),
            "layer1_results": _sha256(args.layer1_results),
        },
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        best[
            [
                "system_id",
                "melt_f_multiplier",
                "melt_f",
                "delta_t_c",
                "e_lia",
                "e_spin_300_500",
                "layer1_accept",
            ]
        ].to_string(index=False)
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())