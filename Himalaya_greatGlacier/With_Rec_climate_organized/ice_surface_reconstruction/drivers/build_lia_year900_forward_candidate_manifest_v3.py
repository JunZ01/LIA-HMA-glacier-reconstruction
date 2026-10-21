"""Build the user-selected year-900 forward candidate manifest.

This is a read-only rescore of the completed v10.203/v10.216 spin-up
trajectories.  It never runs OGGM or rewrites a source result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TX = BASE / "lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
OUT = BASE / "lia_191_year900_forward_candidate_manifest_v3"

VOLUME_CYCLE_MAX = 0.03
VOLUME_90_MAX = 0.03
LENGTH_CYCLE_MAX_M = 300.0
LENGTH_90_MAX_M = 300.0
BOUNDARIES = (300, 600, 900)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def window_metrics(by_year: dict[int, dict], years: list[int]) -> dict | None:
    if any(year not in by_year for year in years):
        return None
    states = [by_year[year] for year in years]
    volumes = np.asarray([state["volume_m3"] for state in states], dtype=float)
    lengths = np.asarray([state["length_m"] for state in states], dtype=float)
    volume_cycles = np.abs(np.diff(volumes)) / np.maximum(np.abs(volumes[:-1]), 1e-30)
    volume_90 = abs(volumes[-1] - volumes[0]) / max(abs(volumes[0]), 1e-30)
    length_cycles = np.abs(np.diff(lengths))
    length_90 = abs(lengths[-1] - lengths[0])
    passed = bool(
        np.all(volume_cycles <= VOLUME_CYCLE_MAX)
        and volume_90 <= VOLUME_90_MAX
        and np.all(length_cycles <= LENGTH_CYCLE_MAX_M)
        and length_90 <= LENGTH_90_MAX_M
    )
    return {
        "years": years,
        "volume_cycle_max": float(np.max(volume_cycles)),
        "volume_90": float(volume_90),
        "length_cycle_max_m": float(np.max(length_cycles)),
        "length_90_m": float(length_90),
        "passed": passed,
    }


def first_confirmed_boundary(by_year: dict[int, dict]):
    for year in BOUNDARIES:
        first = window_metrics(by_year, [year - 90, year - 60, year - 30, year])
        confirm = window_metrics(by_year, [year - 60, year - 30, year, year + 30])
        if first and confirm and first["passed"] and confirm["passed"]:
            return year, first, confirm
    return None, None, None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUT}")

    full = pd.read_csv(TX / "level1_matrix.csv").set_index("cell_ordinal", drop=False)
    production = pd.read_csv(TX / "production_manifest.csv")
    reused = pd.read_csv(TX / "reused_v10_203_converged_results.csv")

    result_paths: dict[int, Path] = {}
    result_sources: dict[int, str] = {}
    for row in production.itertuples(index=False):
        ordinal = int(row.cell_ordinal)
        result_paths[ordinal] = ROOT / "w216/m" / row.member_key / "c/result.json"
        result_sources[ordinal] = "v10.216-new-result"
    for row in reused.itertuples(index=False):
        ordinal = int(row.cell_ordinal)
        result_paths[ordinal] = Path(row.result_path)
        result_sources[ordinal] = "v10.203-validated-reused-result"
    if set(result_paths) != set(range(1, 1338)):
        raise RuntimeError("source result coverage is not exactly 1,337 cells")

    candidates = []
    excluded = []
    numerical_counts = {year: 0 for year in BOUNDARIES}
    numerical_systems = {year: set() for year in BOUNDARIES}

    for ordinal in range(1, 1338):
        meta = full.loc[ordinal]
        result_path = result_paths[ordinal]
        result = read_json(result_path)
        diagnostics = read_jsonl(result_path.parent / "diagnostics.jsonl")
        by_year = {int(row["year"]): row for row in diagnostics}
        if 0 not in by_year:
            raise RuntimeError(f"year zero missing: {result_path}")
        first_year, first_metrics, confirmation_metrics = first_confirmed_boundary(by_year)

        for cutoff in BOUNDARIES:
            if first_year is not None and first_year <= cutoff:
                numerical_counts[cutoff] += 1
                numerical_systems[cutoff].add(int(meta.lia_unique_id))

        if first_year is None or first_year > 900:
            continue
        if result["status"] != "converged":
            final_year = max(by_year)
            final = by_year[final_year]
            initial = by_year[0]
            excluded.append({
                "cell_ordinal": ordinal,
                "lia_unique_id": int(meta.lia_unique_id),
                "delta_t_eq_degC": float(meta.delta_t_eq_degC),
                "first_confirmed_boundary_year_3pct": int(first_year),
                "terminal_original_contract_status": result["status"],
                "final_diagnostic_year": int(final_year),
                "final_volume_ratio_to_year0": float(final["volume_m3"] / initial["volume_m3"]),
                "final_length_ratio_to_year0": float(final["length_m"] / initial["length_m"]),
                "final_maximum_thickness_m": float(final["maximum_thickness_m"]),
                "exclusion_reason": "short-lived-3pct-stability-without-original-strict-terminal-convergence",
            })
            continue

        confirmation_year = int(result["current_year"])
        checkpoint_dir = result_path.parent / "checkpoints" / f"year-{confirmation_year:04d}"
        checkpoint_manifest = checkpoint_dir / "checkpoint.json"
        checkpoint_state = checkpoint_dir / "state.pkl.gz"
        if not checkpoint_manifest.is_file() or not checkpoint_state.is_file():
            raise FileNotFoundError(f"confirmation checkpoint missing: {checkpoint_dir}")
        checkpoint = read_json(checkpoint_manifest)
        if int(checkpoint["year"]) != confirmation_year:
            raise RuntimeError(f"checkpoint year mismatch: {checkpoint_manifest}")

        initial = by_year[0]
        state_900 = by_year.get(900, by_year[max(year for year in by_year if year <= 900)])
        candidates.append({
            "forward_candidate_ordinal": len(candidates) + 1,
            "cell_ordinal": ordinal,
            "lia_unique_id": int(meta.lia_unique_id),
            "rgi_id": str(meta.rgi_id),
            "delta_t_eq_degC": float(meta.delta_t_eq_degC),
            "base_temp_bias_degC": float(meta.base_temp_bias_degC),
            "effective_temp_bias_degC": float(meta.effective_temp_bias_degC),
            "melt_f": float(meta.melt_f),
            "prcp_fac": float(meta.prcp_fac),
            "glen_a_pa_3_s_1": float(meta.glen_a_pa_3_s_1),
            "fs": float(meta.fs),
            "temp_melt_degC": float(meta.temp_melt_degC),
            "first_confirmed_boundary_year_3pct": int(first_year),
            "first_window_volume_cycle_max": first_metrics["volume_cycle_max"],
            "first_window_volume_90": first_metrics["volume_90"],
            "first_window_length_cycle_max_m": first_metrics["length_cycle_max_m"],
            "first_window_length_90_m": first_metrics["length_90_m"],
            "confirmation_window_volume_cycle_max": confirmation_metrics["volume_cycle_max"],
            "confirmation_window_volume_90": confirmation_metrics["volume_90"],
            "confirmation_window_length_cycle_max_m": confirmation_metrics["length_cycle_max_m"],
            "confirmation_window_length_90_m": confirmation_metrics["length_90_m"],
            "original_strict_confirmation_state_year": confirmation_year,
            "original_strict_first_pass_year": int(result["first_pass_year"]),
            "result_source": result_sources[ordinal],
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "confirmation_checkpoint_manifest_path": str(checkpoint_manifest.resolve()),
            "confirmation_checkpoint_manifest_sha256": sha256(checkpoint_manifest),
            "confirmation_checkpoint_state_path": str(checkpoint_state.resolve()),
            "confirmation_checkpoint_state_sha256": sha256(checkpoint_state),
            "year900_diagnostic_volume_ratio_to_year0": float(state_900["volume_m3"] / initial["volume_m3"]),
            "year900_diagnostic_length_ratio_to_year0": float(state_900["length_m"] / initial["length_m"]),
            "year900_diagnostic_maximum_thickness_m": float(state_900["maximum_thickness_m"]),
            "forward_launch_authorized": False,
        })

    manifest = pd.DataFrame(candidates).sort_values(["lia_unique_id", "delta_t_eq_degC"])
    manifest["forward_candidate_ordinal"] = np.arange(1, len(manifest) + 1)
    excluded_frame = pd.DataFrame(excluded).sort_values(["lia_unique_id", "delta_t_eq_degC"])
    if len(manifest) != 530 or manifest["lia_unique_id"].nunique() != 187 or len(excluded_frame) != 5:
        raise RuntimeError(
            f"unexpected selection: candidates={len(manifest)}, "
            f"systems={manifest['lia_unique_id'].nunique()}, excluded={len(excluded_frame)}"
        )

    OUT.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(OUT / "forward_candidate_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    excluded_frame.to_csv(OUT / "excluded_short_lived_3pct_stability.csv", index=False)
    system_counts = manifest.groupby("lia_unique_id").size().rename("forward_candidate_count").reset_index()
    system_counts.to_csv(OUT / "system_forward_candidate_counts.csv", index=False)

    summary = {
        "schema": "lia-191-year900-forward-candidate-manifest-v3",
        "status": "complete-user-selected-forward-input-no-simulation-launched",
        "source_parameter_cells": 1337,
        "source_systems": 191,
        "selection_rule": {
            "first_confirmed_boundary_at_or_before_year": 900,
            "volume_each_30y_relative_max": VOLUME_CYCLE_MAX,
            "volume_90y_relative_max": VOLUME_90_MAX,
            "length_each_30y_max_m": LENGTH_CYCLE_MAX_M,
            "length_90y_max_m": LENGTH_90_MAX_M,
            "one_additional_30y_confirmation_required": True,
            "original_strict_contract_terminal_convergence_required": True,
            "short_lived_relaxed_only_stability_excluded": True,
        },
        "retrospective_numerically_stable_cells": {
            str(year): numerical_counts[year] for year in BOUNDARIES
        },
        "retrospective_systems_with_at_least_one_stable_cell": {
            str(year): len(numerical_systems[year]) for year in BOUNDARIES
        },
        "forward_candidate_cells": int(len(manifest)),
        "forward_candidate_systems": int(manifest["lia_unique_id"].nunique()),
        "excluded_short_lived_cells": int(len(excluded_frame)),
        "no_candidate_systems": sorted(set(full.lia_unique_id.astype(int)) - set(manifest.lia_unique_id.astype(int))),
        "spinup_recomputation_authorized": False,
        "forward_simulation_launched": False,
        "forward_launch_authorized_by_this_manifest": False,
        "next_legal_action": "build and validate a separate versioned 530-cell forward transaction from these checkpoints; never rerun spin-up",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = sha256(path)
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
