"""Quantify 50-year pre-instrumental consensus across temperature families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = ("local_enkf", "lmr", "neukom", "phyda")


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


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_22_temperature_reference_preinstrumental_consensus_v1"
    curves_path = loop / "round_21_temperature_reference_field_v1/hma_temperature_reference_curves.csv"
    netcdf_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    paths = {"curves": curves_path, "netcdf": netcdf_path, "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    curves = pd.read_csv(curves_path)
    years = curves.year.to_numpy(int)
    if np.any((years[(years >= 1921) & (years <= 1939)] != np.arange(1921, 1940))):
        raise RuntimeError("year-coordinate-failed")
    locked_columns = ["family_balanced_hma_degC", "member_equal_hma_degC", *[f"{name}_hma_degC" for name in FAMILIES]]
    locked = (years >= 1921) & (years <= 1939)
    if not curves.loc[locked, locked_columns].isna().all().all():
        raise RuntimeError("locked-values-present")

    windows = [(start, start + 49) for start in range(1000, 1900, 50)] + [(1940, 1989)]
    rows: list[dict[str, object]] = []
    for start, end in windows:
        selected = curves[(curves.year >= start) & (curves.year <= end)]
        if len(selected) != 50:
            raise RuntimeError(f"incomplete-window:{start}-{end}:{len(selected)}")
        family_means = {name: float(selected[f"{name}_hma_degC"].mean()) for name in FAMILIES}
        family_values = np.asarray(list(family_means.values()), dtype=float)
        central = float(selected.family_balanced_hma_degC.mean())
        control = float(selected.member_equal_hma_degC.mean())
        spread = float(np.std(family_values, ddof=1))
        cold_count = int(np.sum(family_values < 0))
        warm_count = int(np.sum(family_values > 0))
        row = {
            "start_year": start,
            "end_year": end,
            "is_lia_window_1400_1899": bool(start >= 1400 and end <= 1899),
            **{f"{name}_mean_degC": value for name, value in family_means.items()},
            "family_balanced_mean_degC": central,
            "member_equal_mean_degC": control,
            "family_balanced_minus_member_equal_degC": central - control,
            "family_window_std_degC": spread,
            "family_window_min_degC": float(np.min(family_values)),
            "family_window_max_degC": float(np.max(family_values)),
            "cold_family_count": cold_count,
            "warm_family_count": warm_count,
            "majority_cold": cold_count >= 3,
            "all_four_cold": cold_count == 4,
            "amplitude_resolved": bool(abs(central) >= spread),
            "absolute_signal_to_family_spread": float(abs(central) / spread) if spread > 0 else np.inf,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "temperature_50year_family_consensus.csv", index=False)
    lia = table[table.is_lia_window_1400_1899].copy()
    majority_fraction = float(lia.majority_cold.mean())
    all_cold_fraction = float(lia.all_four_cold.mean())
    resolved_fraction = float(lia.amplitude_resolved.mean())
    coldest = lia.loc[lia.family_balanced_mean_degC.idxmin()]
    family_coldest = {}
    for family in FAMILIES:
        row = lia.loc[lia[f"{family}_mean_degC"].idxmin()]
        family_coldest[family] = {"start_year": int(row.start_year), "end_year": int(row.end_year), "mean_degC": float(row[f"{family}_mean_degC"])}
    gates = {
        "lia_majority_cold_fraction_at_least_70pct": majority_fraction >= 0.70,
        "lia_amplitude_resolved_fraction_at_least_50pct": resolved_fraction >= 0.50,
    }
    central_supported = gates["lia_majority_cold_fraction_at_least_70pct"]
    deterministic_supported = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "lia_window_count": int(len(lia)),
        "lia_majority_cold_fraction": majority_fraction,
        "lia_all_four_cold_fraction": all_cold_fraction,
        "lia_amplitude_resolved_fraction": resolved_fraction,
        "lia_median_family_balanced_minus_member_equal_degC": float(lia.family_balanced_minus_member_equal_degC.median()),
        "lia_median_family_window_std_degC": float(lia.family_window_std_degC.median()),
        "coldest_family_balanced_lia_window": {"start_year": int(coldest.start_year), "end_year": int(coldest.end_year), "mean_degC": float(coldest.family_balanced_mean_degC), "family_std_degC": float(coldest.family_window_std_degC)},
        "family_specific_coldest_lia_windows": family_coldest,
        "gates": gates,
        "central_lia_reference_supported": central_supported,
        "deterministic_forcing_supported": deterministic_supported,
        "decision": "use_family_balanced_central_reference_with_family_uncertainty" if central_supported and not deterministic_supported else ("permit_separate_deterministic_reference_forcing_review" if deterministic_supported else "do_not_treat_family_balance_as_lia_consensus"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lia_plot = table[table.start_year < 1900]
    centers = (lia_plot.start_year + lia_plot.end_year) / 2
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.fill_between(centers, lia_plot.family_window_min_degC, lia_plot.family_window_max_degC, color="#9ecae1", alpha=0.5, label="Four-family range")
    axis.plot(centers, lia_plot.family_balanced_mean_degC, color="#005f73", marker="o", linewidth=2, label="Family balanced")
    axis.plot(centers, lia_plot.member_equal_mean_degC, color="#555555", marker="s", linewidth=1.4, label="Member equal")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.axvspan(1400, 1900, color="#dddddd", alpha=0.25)
    axis.set(xlabel="50-year window center", ylabel="HMA temperature anomaly (degC)", title="Pre-instrumental temperature family consensus")
    axis.legend(frameon=False)
    fig.savefig(output / "temperature_50year_family_consensus.png", dpi=180)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Pre-Instrumental Temperature Consensus\n\n"
        f"- LIA majority-cold windows: `{majority_fraction:.1%}`; all-four-cold: `{all_cold_fraction:.1%}`.\n"
        f"- LIA amplitude-resolved windows: `{resolved_fraction:.1%}`.\n"
        f"- Median family-balanced minus member-equal LIA anomaly: `{summary['lia_median_family_balanced_minus_member_equal_degC']:+.3f} degC`.\n"
        f"- Median family-window spread: `{summary['lia_median_family_window_std_degC']:.3f} degC`.\n"
        f"- Coldest central window: `{int(coldest.start_year)}-{int(coldest.end_year)}`, `{coldest.family_balanced_mean_degC:+.3f} degC`.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
