from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
TX = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
MILESTONE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_milestone_balance_diagnostics_v1/cell_milestone_states.csv"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_year900_forward_candidate_manifest_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if OUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUT}")
    OUT.mkdir(parents=True)

    states = pd.read_csv(MILESTONE)
    states = states[states.milestone_year == 900].copy()
    full = pd.read_csv(TX / "level1_matrix.csv").set_index("cell_ordinal", drop=False)
    production = pd.read_csv(TX / "production_manifest.csv")
    reused = pd.read_csv(TX / "reused_v10_203_converged_results.csv")
    result_paths = {}
    sources = {}
    for row in production.itertuples(index=False):
        ordinal = int(row.cell_ordinal)
        result_paths[ordinal] = ROOT / "w216/m" / row.member_key / "c/result.json"
        sources[ordinal] = "v10.216-new-result"
    for row in reused.itertuples(index=False):
        ordinal = int(row.cell_ordinal)
        result_paths[ordinal] = Path(row.result_path)
        sources[ordinal] = "v10.203-validated-reused-result"

    eligible = states[
        (states.numerical_group == "confirmed-stable")
        & (states.terminal_original_contract_status == "converged")
        & (states.first_confirmed_boundary_year <= 900)
    ].copy()
    relaxed_false_stable = states[
        (states.numerical_group == "confirmed-stable")
        & (states.terminal_original_contract_status == "max-duration-nonconverged")
    ].copy()

    rows = []
    for s in eligible.itertuples(index=False):
        ordinal = int(s.cell_ordinal)
        meta = full.loc[ordinal]
        result_path = result_paths[ordinal]
        result = read_json(result_path)
        if result["status"] != "converged":
            raise RuntimeError(f"invalid year-900 candidate terminal result: {result_path}")
        confirmation_year = int(result["current_year"])
        checkpoint_dir = result_path.parent / "checkpoints" / f"year-{confirmation_year:04d}"
        checkpoint_manifest = checkpoint_dir / "checkpoint.json"
        checkpoint_state = checkpoint_dir / "state.pkl.gz"
        if not checkpoint_manifest.is_file() or not checkpoint_state.is_file():
            raise FileNotFoundError(f"confirmation checkpoint missing: {checkpoint_dir}")
        checkpoint = read_json(checkpoint_manifest)
        if int(checkpoint["year"]) != confirmation_year:
            raise RuntimeError(f"checkpoint year mismatch: {checkpoint_manifest}")
        rows.append({
            "forward_candidate_ordinal": len(rows) + 1,
            "cell_ordinal": ordinal,
            "lia_unique_id": int(meta.lia_unique_id),
            "rgi_id": meta.rgi_id,
            "delta_t_eq_degC": float(meta.delta_t_eq_degC),
            "base_temp_bias_degC": float(meta.base_temp_bias_degC),
            "effective_temp_bias_degC": float(meta.effective_temp_bias_degC),
            "melt_f": float(meta.melt_f),
            "prcp_fac": float(meta.prcp_fac),
            "glen_a_pa_3_s_1": float(meta.glen_a_pa_3_s_1),
            "fs": float(meta.fs),
            "temp_melt_degC": float(meta.temp_melt_degC),
            "first_pass_boundary_year": int(result["first_pass_year"]),
            "confirmation_state_year": confirmation_year,
            "confirmation_state_after_year900": confirmation_year > 930,
            "result_source": sources[ordinal],
            "result_path": str(result_path),
            "result_sha256": sha256(result_path),
            "confirmation_checkpoint_manifest_path": str(checkpoint_manifest),
            "confirmation_checkpoint_manifest_sha256": sha256(checkpoint_manifest),
            "confirmation_checkpoint_state_path": str(checkpoint_state),
            "confirmation_checkpoint_state_sha256": sha256(checkpoint_state),
            "year900_diagnostic_volume_ratio_to_year0": float(s.volume_ratio_to_year0),
            "year900_diagnostic_length_ratio_to_year0": float(s.length_ratio_to_year0),
            "year900_diagnostic_maximum_thickness_m": float(s.maximum_thickness_m),
            "forward_launch_authorized": False,
        })

    manifest = pd.DataFrame(rows).sort_values(["lia_unique_id", "delta_t_eq_degC"])
    manifest.to_csv(OUT / "forward_candidate_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    relaxed_false_stable[[
        "cell_ordinal", "lia_unique_id", "delta_t_eq_degC", "first_confirmed_boundary_year",
        "volume_ratio_to_year0", "length_ratio_to_year0", "maximum_thickness_m",
        "terminal_original_contract_status",
    ]].to_csv(OUT / "excluded_relaxed_short_lived_stability.csv", index=False)

    late = states[
        (states.terminal_original_contract_status == "converged")
        & (states.first_confirmed_boundary_year > 900)
    ][["cell_ordinal", "lia_unique_id", "delta_t_eq_degC", "first_confirmed_boundary_year"]].copy()
    late.to_csv(OUT / "held_after_year900_boundary.csv", index=False)

    system_counts = manifest.groupby("lia_unique_id").size().rename("forward_candidate_count").reset_index()
    system_counts.to_csv(OUT / "system_forward_candidate_counts.csv", index=False)

    summary = {
        "schema": "lia-191-year900-forward-candidate-manifest-v2",
        "status": "complete-prospective-forward-input-no-launch",
        "source_parameter_cells": 1337,
        "source_systems": 191,
        "year900_relaxed_numerically_stable_cells": int((states.numerical_group == "confirmed-stable").sum()),
        "forward_candidate_cells": int(len(manifest)),
        "forward_candidate_systems": int(manifest.lia_unique_id.nunique()),
        "excluded_relaxed_short_lived_stability_cells": int(len(relaxed_false_stable)),
        "held_original_convergence_after_year900_cells": int(len(late)),
        "no_candidate_systems": sorted(set(full.lia_unique_id.astype(int)) - set(manifest.lia_unique_id.astype(int))),
        "selection_rule": {
            "first_pass_boundary_at_or_before_year": 900,
            "requires_original_strict_contract_terminal_confirmation": True,
            "confirmation_checkpoint_reused_without_spinup_recomputation": True,
            "already_computed_post900_terminal_checkpoint_allowed": True,
            "relaxed_only_pass_with_observed_later_reactivation_excluded": True,
        },
        "candidate_geometry_diagnostics": {
            "maximum_thickness_m_max": float(manifest.year900_diagnostic_maximum_thickness_m.max()),
            "length_ratio_to_year0_max": float(manifest.year900_diagnostic_length_ratio_to_year0.max()),
            "volume_ratio_to_year0_max": float(manifest.year900_diagnostic_volume_ratio_to_year0.max()),
        },
        "spinup_recomputation_authorized": False,
        "forward_launch_authorized": False,
        "next_legal_action": "review this immutable candidate manifest and create a separate versioned forward transaction; do not rerun spinup",
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = sha256(path)
    with (OUT / "artifact_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"hashes": hashes}, indent=2))


if __name__ == "__main__":
    main()

