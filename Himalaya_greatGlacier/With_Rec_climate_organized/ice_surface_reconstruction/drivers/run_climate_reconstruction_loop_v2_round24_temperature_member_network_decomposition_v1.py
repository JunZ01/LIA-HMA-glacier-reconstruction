"""Decompose the corrected LIA temperature conflict by method and proxy network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat(); digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def summarize(group: pd.DataFrame, label: str, category: str) -> dict[str, object]:
    outside_above = group.mean_degC > group.reference_family_max_degC
    outside_below = group.mean_degC < group.reference_family_min_degC
    differences = group.mean_degC - group.reference_family_central_mean_degC
    standardized = differences / group.reference_family_std_degC
    return {
        "category": category,
        "label": label,
        "lia_window_count": int(len(group)),
        "above_reference_range_count": int(outside_above.sum()),
        "below_reference_range_count": int(outside_below.sum()),
        "inside_reference_range_count": int((~outside_above & ~outside_below).sum()),
        "median_minus_reference_central_degC": float(differences.median()),
        "median_absolute_family_standardized_difference": float(np.median(np.abs(standardized))),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_24_temperature_member_network_decomposition_v1"
    member_path = loop / "round_23_corrected_reconstruction_reference_envelope_v1/member_50year_means.csv"
    reference_path = loop / "round_23_corrected_reconstruction_reference_envelope_v1/corrected_reconstruction_reference_50year_windows.csv"
    paths = {"member_windows": member_path, "reference_windows": reference_path, "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    members = pd.read_csv(member_path)
    reference = pd.read_csv(reference_path)
    reference = reference[reference.is_lia_window_1400_1899].copy()
    members = members.merge(reference[["start_year", "end_year", "reference_family_central_mean_degC", "reference_family_min_degC", "reference_family_max_degC", "reference_family_std_degC"]], on=["start_year", "end_year"], how="inner", validate="many_to_one")
    members["method"] = members.member.str.split("_").str[0]
    members["network"] = "proxy" + members.member.str.split("_").str[1]
    if len(members) != 60:
        raise RuntimeError(f"member-lia-row-count-failed:{len(members)}")

    summaries: list[dict[str, object]] = []
    for member, group in members.groupby("member"):
        summaries.append(summarize(group, member, "member"))
    aggregate_rows: list[dict[str, object]] = []
    for category, column in (("method", "method"), ("network", "network")):
        for label, group in members.groupby(column):
            averaged = group.groupby(["start_year", "end_year"], as_index=False).agg({
                "mean_degC": "mean",
                "reference_family_central_mean_degC": "first",
                "reference_family_min_degC": "first",
                "reference_family_max_degC": "first",
                "reference_family_std_degC": "first",
            })
            aggregate_rows.extend([{category: label, **row} for row in averaged.to_dict("records")])
            summaries.append(summarize(averaged, label, category))
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "member_method_network_summary.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output / "method_network_50year_means.csv", index=False)

    member_summary = summary_table[summary_table.category == "member"]
    shared_conflict = bool(
        np.all(member_summary.median_minus_reference_central_degC > 0)
        and np.all(member_summary.above_reference_range_count >= 8)
    )
    network_table = summary_table[summary_table.category == "network"].set_index("label")
    network9_minus_network6 = float(network_table.loc["proxy9", "median_minus_reference_central_degC"] - network_table.loc["proxy6", "median_minus_reference_central_degC"])
    result = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "member_count": 6,
        "all_members_positive_median_reference_difference": bool(np.all(member_summary.median_minus_reference_central_degC > 0)),
        "all_members_above_range_in_at_least_8_of_10_lia_windows": bool(np.all(member_summary.above_reference_range_count >= 8)),
        "shared_upstream_warm_conflict": shared_conflict,
        "member_median_reference_difference_range_degC": [float(member_summary.median_minus_reference_central_degC.min()), float(member_summary.median_minus_reference_central_degC.max())],
        "proxy9_minus_proxy6_median_reference_difference_degC": network9_minus_network6,
        "decision": "close_member_reweighting_and_separate_prior_from_assimilation_update" if shared_conflict else "permit_branch_specific_blocked_validation_only",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    member_order = sorted(members.member.unique())
    positions = np.arange(len(member_order))
    for index, member in enumerate(member_order):
        group = members[members.member == member]
        difference = group.mean_degC - group.reference_family_central_mean_degC
        axis.scatter(np.full(len(group), index), difference, s=35, alpha=0.75)
        axis.hlines(float(difference.median()), index - 0.25, index + 0.25, color="#111111", linewidth=2)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(positions, member_order)
    axis.set(ylabel="Member minus reference central (degC)", title="LIA temperature conflict by method and proxy network")
    fig.savefig(output / "temperature_member_network_decomposition.png", dpi=180)
    plt.close(fig)

    lines = ["# Temperature Member/Network Decomposition", ""]
    for row in member_summary.itertuples(index=False):
        lines.append(f"- {row.label}: above range `{row.above_reference_range_count}/10`, median difference `{row.median_minus_reference_central_degC:+.3f} degC`.")
    lines.extend(["", f"Shared upstream warm conflict: `{str(shared_conflict).lower()}`.", f"Proxy9 minus proxy6 median difference: `{network9_minus_network6:+.3f} degC`.", "", f"Decision: `{result['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
