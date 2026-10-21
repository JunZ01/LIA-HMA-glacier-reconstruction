"""Build the immutable representative/source manifest for structural cycle 1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
MASTER = BASE / "lia_191_controller_v1" / "diagnostics" / "unresolved_master_router_v1" / "unresolved_master_queue.csv"
REFERENCES = BASE / "lia_191_controller_v1" / "core191_length_references.csv"
OUTPUT = BASE / "lia_191_continuous_class_controller_v1"
DESIGN = {
    3850: ("Q1_neither_endpoint", "q1_preflight", "monotonic_regularized_cross_section"),
    4369: ("Q1_neither_endpoint", "q1_representative", "monotonic_regularized_cross_section"),
    2086: ("Q1_neither_endpoint", "q1_representative", "monotonic_regularized_cross_section"),
    2729: ("Q1_neither_endpoint", "q1_representative", "monotonic_regularized_cross_section"),
    2925: ("resolution_subclass", "resolution_preflight", "conservative_25m_terminal_remap"),
    214: ("Q2_rgi_only", "q2_representative", "conservative_25m_terminal_remap"),
    584: ("Q2_rgi_only", "q2_representative", "conservative_25m_terminal_remap"),
    1120: ("Q2_rgi_only", "q2_representative", "conservative_25m_terminal_remap"),
    2806: ("Q3_lia_only", "q3_rgi_short_representative", "conservative_25m_terminal_remap"),
    1012: ("Q3_lia_only", "q3_rgi_short_representative", "conservative_25m_terminal_remap"),
    843: ("Q3_lia_only", "q3_rgi_long_representative", "conservative_25m_terminal_remap"),
    1044: ("Q4_no_stable_valid_candidate", "source_rescore", "read_only"),
    2711: ("Q4_no_stable_valid_candidate", "convergence_preflight", "read_only"),
    3053: ("Q4_no_stable_valid_candidate", "convergence_secondary", "read_only"),
    2387: ("Q4_no_stable_valid_candidate", "zero_mb_preflight", "read_only"),
    5508: ("Q4_no_stable_valid_candidate", "zero_mb_secondary", "read_only"),
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER)
    references = pd.read_csv(REFERENCES)
    rows: list[dict] = []
    for system_id, (queue, role, mechanism) in DESIGN.items():
        reference = references[references["lia_unique_id"].astype(int).eq(system_id)]
        if len(reference) != 1:
            raise RuntimeError(f"reference-row-count:{system_id}:{len(reference)}")
        current = master[master["lia_unique_id"].astype(int).eq(system_id)]
        item = current.iloc[0].to_dict() if len(current) == 1 else {}
        ref = reference.iloc[0]
        rows.append({
            "lia_unique_id": system_id,
            "rgi_id": str(ref["rgi_id"]),
            "queue": queue,
            "role": role,
            "mechanism": mechanism,
            "lia_target_m": float(ref["lia_registered_length_m"]),
            "rgi_target_m": float(ref["rgi_registered_length_m"]),
            "separation_m": abs(float(ref["lia_registered_length_m"]) - float(ref["rgi_registered_length_m"])),
            "current_experiment": item.get("experiment", ""),
            "current_source_dir": item.get("source_dir", ""),
            "current_output_dir": item.get("output_dir", ""),
            "spinup_delta_t_c": item.get("spinup_delta_t_c", ""),
            "forward_delta_t_c": item.get("forward_delta_t_c", ""),
            "spinup_stability_fraction": item.get("spinup_stability_fraction", ""),
            "current_forward_peak_m": item.get("forward_peak_m", ""),
            "current_forward_2000_m": item.get("forward_2000_m", ""),
            "current_e_lia_sep": item.get("e_lia_sep", ""),
            "current_e_rgi_sep": item.get("e_rgi_sep", ""),
            "current_objective_sep": item.get("objective_sep", ""),
            "release_state": "read_only_audit" if mechanism == "read_only" else "geometry_preflight_pending",
        })
    manifest = pd.DataFrame(rows).sort_values(["queue", "role", "lia_unique_id"])
    manifest.to_csv(OUTPUT / "cycle1_representative_manifest.csv", index=False)
    summary = {
        "representative_count": int(len(manifest)),
        "with_reusable_current_candidate": int(manifest["current_source_dir"].fillna("").ne("").sum()),
        "geometry_preflight_pending": int(manifest["release_state"].eq("geometry_preflight_pending").sum()),
        "read_only_audit": int(manifest["release_state"].eq("read_only_audit").sum()),
        "launched_oggm": False,
    }
    (OUTPUT / "cycle1_representative_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
