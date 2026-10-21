from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "historical_summary.json").read_text(encoding="utf-8"))

    lia = float(config["lia_registered_length_m"])
    rgi = float(config["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    initial = summary.get("initial") or {}
    peak = summary.get("peak") or {}
    final = summary.get("final") or {}

    e_lia_total = finite(summary.get("e_lia_total"))
    e_rgi_total = finite(summary.get("e_rgi_total"))
    e_lia_sep = finite(summary.get("e_lia_separation"))
    e_rgi_sep = finite(summary.get("e_rgi_separation"))
    total_pass = bool(
        e_lia_total is not None
        and e_rgi_total is not None
        and e_lia_total <= 0.1
        and e_rgi_total <= 0.1
    )
    separation_pass = bool(
        e_lia_sep is not None
        and e_rgi_sep is not None
        and e_lia_sep <= 0.2
        and e_rgi_sep <= 0.2
    )

    domain_reasons = []
    for label, state in (("initial", initial), ("peak", peak), ("final", final)):
        if state.get("finite_geometry") is not True:
            domain_reasons.append(f"{label}-nonfinite")
        negative_count = finite(state.get("negative_section_count"))
        if negative_count is None or int(negative_count) != 0:
            domain_reasons.append(f"{label}-negative-section")
        boundary_ice = finite(state.get("root_array_end_thickness_m"))
        if boundary_ice is None or boundary_ice > 1e-9:
            domain_reasons.append(f"{label}-root-boundary-ice")
        free = finite(state.get("remaining_free_downstream_m"))
        if free is None or free < 0:
            domain_reasons.append(f"{label}-observation-domain-overflow")

    final_year = finite(final.get("year"))
    numerical_pass = bool(
        summary.get("runtime_error") in (None, "")
        and final_year is not None
        and int(final_year) == 2000
        and (root / "historical_1470_2000.csv").is_file()
        and (root / "historical_1500_2000.csv").is_file()
    )
    initial_length = finite(initial.get("length_m"))
    peak_joint_upper = min(lia * 1.1, lia + 0.2 * separation)
    initial_peak_bound_pass = bool(
        initial_length is not None and initial_length <= peak_joint_upper
    )

    flags = []
    if not total_pass:
        flags.append("total-length-gate")
    if not separation_pass:
        flags.append("separation-length-gate")
    flags.extend(domain_reasons)
    if not numerical_pass:
        flags.append("numerical-or-completion-gate")
    if not initial_peak_bound_pass:
        flags.append("initial-peak-hard-bound")

    formal = {
        **summary,
        "status": "historical-pass" if not flags else "historical-fail",
        "formal_scoring_version": "root-coarsen2x-formal-v2",
        "formal_total_length_pass": total_pass,
        "formal_separation_length_pass": separation_pass,
        "formal_observation_domain_pass": not domain_reasons,
        "formal_numerical_completion_pass": numerical_pass,
        "formal_initial_peak_hard_bound_pass": initial_peak_bound_pass,
        "formal_peak_joint_upper_m": peak_joint_upper,
        "formal_review_flags": list(dict.fromkeys(flags)),
        "supersedes_scoring_only": str(root / "historical_summary_formal.json"),
    }
    (root / "historical_summary_formal_v2.json").write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": formal["status"],
        "formal_review_flags": formal["formal_review_flags"],
        "formal_observation_domain_pass": formal["formal_observation_domain_pass"],
        "formal_numerical_completion_pass": formal["formal_numerical_completion_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
