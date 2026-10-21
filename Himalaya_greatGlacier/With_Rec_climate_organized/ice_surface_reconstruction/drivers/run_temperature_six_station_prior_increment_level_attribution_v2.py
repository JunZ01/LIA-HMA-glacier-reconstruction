"""Correct v1 aggregate attribution using exactly additive equal-station means."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERIODS = ("1880-1900", "1901-1920", "1940-1950")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "temperature_six_station_prior_increment_level_attribution_v1"
    output = loop / "temperature_six_station_prior_increment_level_attribution_v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")

    inputs = {
        "v1_metrics": source / "station_period_level_trend_attribution.csv",
        "v1_annual": source / "station_ensemble_annual_components.csv",
        "v1_summary": source / "summary.json",
        "v1_manifest": source / "source_manifest.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in inputs.items()}
    v1_summary = json.loads(inputs["v1_summary"].read_text(encoding="utf-8"))
    if v1_summary["locked_1921_1939_accessed"] or not v1_summary["source_states_unchanged"]:
        raise RuntimeError("v1-boundary-failed")

    metrics = pd.read_csv(inputs["v1_metrics"])
    annual = pd.read_csv(inputs["v1_annual"])
    if annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-in-derived-input")

    exact_rows: list[dict[str, object]] = []
    dominance_rows: list[dict[str, object]] = []
    for (period, target), group in metrics.groupby(["period", "target"], sort=False):
        eligible = group[group.matched_years >= 5].copy()
        if not len(eligible):
            continue
        level_residual = float(eligible.posterior_minus_station_degC.mean())
        level_background = float(eligible.background_minus_station_degC.mean())
        level_increment = float(eligible.increment_mean_degC.mean())
        trend_residual = float(eligible.trend_residual_degC_per_century.mean())
        trend_background = float(eligible.background_minus_station_trend_degC_per_century.mean())
        trend_increment = float(eligible.increment_trend_degC_per_century.mean())
        exact_rows.append({
            "period": period,
            "target": target,
            "eligible_station_count": int(len(eligible)),
            "equal_station_mean_posterior_minus_station_degC": level_residual,
            "equal_station_mean_background_minus_station_degC": level_background,
            "equal_station_mean_increment_degC": level_increment,
            "level_closure_abs_degC": abs(level_residual - level_background - level_increment),
            "background_fraction_of_positive_level_residual": level_background / level_residual if level_residual > 0 else np.nan,
            "increment_fraction_of_positive_level_residual": level_increment / level_residual if level_residual > 0 else np.nan,
            "equal_station_mean_trend_residual_degC_per_century": trend_residual,
            "equal_station_mean_background_minus_station_trend_degC_per_century": trend_background,
            "equal_station_mean_increment_trend_degC_per_century": trend_increment,
            "trend_closure_abs_degC_per_century": abs(trend_residual - trend_background - trend_increment),
            "median_posterior_minus_station_degC": float(eligible.posterior_minus_station_degC.median()),
            "median_trend_residual_degC_per_century": float(eligible.trend_residual_degC_per_century.median()),
        })
        for row in eligible.itertuples(index=False):
            dominance_rows.append({
                "station_id": row.station_id,
                "station_name": row.station_name,
                "period": period,
                "target": target,
                "matched_years": int(row.matched_years),
                "posterior_minus_station_degC": row.posterior_minus_station_degC,
                "background_minus_station_degC": row.background_minus_station_degC,
                "increment_degC": row.increment_mean_degC,
                "dominant_absolute_level_term": (
                    "background_minus_station"
                    if abs(row.background_minus_station_degC) >= abs(row.increment_mean_degC)
                    else "assimilation_increment"
                ),
                "warm_residual": bool(row.posterior_minus_station_degC > 0),
            })

    exact = pd.DataFrame(exact_rows)
    dominance = pd.DataFrame(dominance_rows)
    exact.to_csv(output / "exact_equal_station_attribution.csv", index=False)
    dominance.to_csv(output / "station_level_dominance.csv", index=False)

    early = exact[(exact.period == "1901-1920") & (exact.target == "qcf")].iloc[0]
    early_stations = dominance[(dominance.period == "1901-1920") & (dominance.target == "qcf")]
    warm = early_stations[early_stations.warm_residual]
    background_dominant_all = int((early_stations.dominant_absolute_level_term == "background_minus_station").sum())
    background_dominant_warm = int((warm.dominant_absolute_level_term == "background_minus_station").sum())
    closure_max = float(max(exact.level_closure_abs_degC.max(), exact.trend_closure_abs_degC_per_century.max()))
    if closure_max > 1e-10:
        raise RuntimeError(f"aggregate-closure-failed:{closure_max}")
    if early.equal_station_mean_posterior_minus_station_degC <= 0:
        decision = "no_common_warm_level_residual"
    elif early.background_fraction_of_positive_level_residual > 0.5 and background_dominant_warm >= 4:
        decision = "prior_background_disagreement_is_primary_but_increment_adds_warming"
    else:
        decision = "assimilation_increment_or_mixed_mechanism_requires_followup"

    after = {name: digest(path) for name, path in inputs.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("derived-source-changed")
    summary = {
        "experiment": "temperature_six_station_prior_increment_level_attribution_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes_aggregate_interpretation_of": "temperature_six_station_prior_increment_level_attribution_v1",
        "reason_for_v2": "Separate station medians are not algebraically additive; v2 uses equal-station means after exact station-level closure.",
        "locked_1921_1939_accessed": False,
        "qcf_1901_1920": {
            "eligible_station_count": int(early.eligible_station_count),
            "equal_station_mean_posterior_minus_station_degC": float(early.equal_station_mean_posterior_minus_station_degC),
            "equal_station_mean_background_minus_station_degC": float(early.equal_station_mean_background_minus_station_degC),
            "equal_station_mean_increment_degC": float(early.equal_station_mean_increment_degC),
            "background_fraction_of_positive_level_residual": float(early.background_fraction_of_positive_level_residual),
            "increment_fraction_of_positive_level_residual": float(early.increment_fraction_of_positive_level_residual),
            "background_dominant_station_count": background_dominant_all,
            "warm_residual_station_count": int(len(warm)),
            "background_dominant_warm_station_count": background_dominant_warm,
            "median_posterior_minus_station_degC": float(early.median_posterior_minus_station_degC),
        },
        "maximum_equal_station_aggregate_closure": closure_max,
        "decision": decision,
        "interpretation_boundary": "This identifies where the archived posterior-station discrepancy enters the exact decomposition. It does not by itself prove that the model prior is physically wrong across HMA or authorize a production correction.",
        "derived_sources_unchanged": unchanged,
        "original_sources_reopened": False,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"inputs_before_sha256": before, "inputs_after_sha256": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )

    qcf = exact[exact.target.eq("qcf")].set_index("period").loc[list(PERIODS)]
    positions = np.arange(len(PERIODS))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    for axis, suffix, ylabel, title in [
        (axes[0], "", "Equal-station mean level term (degC)", "QCF level residual: exact additive attribution"),
        (axes[1], "_trend", "Equal-station mean trend term (degC/century)", "QCF trend residual: exact additive attribution"),
    ]:
        if suffix:
            background = qcf.equal_station_mean_background_minus_station_trend_degC_per_century.to_numpy(float)
            increment = qcf.equal_station_mean_increment_trend_degC_per_century.to_numpy(float)
            residual = qcf.equal_station_mean_trend_residual_degC_per_century.to_numpy(float)
        else:
            background = qcf.equal_station_mean_background_minus_station_degC.to_numpy(float)
            increment = qcf.equal_station_mean_increment_degC.to_numpy(float)
            residual = qcf.equal_station_mean_posterior_minus_station_degC.to_numpy(float)
        axis.bar(positions, background, color="#00798C", label="background - station")
        axis.bar(positions, increment, bottom=background, color="#7A5195", label="assimilation increment")
        axis.plot(positions, residual, color="#111111", marker="o", lw=1.5, label="posterior - station")
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set(xticks=positions, xticklabels=PERIODS, ylabel=ylabel, title=title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "exact_six_station_level_trend_attribution.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Six-station prior/increment attribution v2\n\n"
        "V2 corrects the non-additive separate-median aggregation in v1. Every station closes first; the six station terms are then averaged with equal station weight.\n\n"
        f"- 1901-1920 qcf posterior-minus-station: `{early.equal_station_mean_posterior_minus_station_degC:+.3f} degC`.\n"
        f"- Background-minus-station term: `{early.equal_station_mean_background_minus_station_degC:+.3f} degC` (`{100 * early.background_fraction_of_positive_level_residual:.1f}%`).\n"
        f"- Assimilation-increment term: `{early.equal_station_mean_increment_degC:+.3f} degC` (`{100 * early.increment_fraction_of_positive_level_residual:.1f}%`).\n"
        f"- Background is the larger absolute term at `{background_dominant_all}/6` stations and `{background_dominant_warm}/{len(warm)}` warm-residual stations.\n"
        f"- Exact aggregate closure: `{early.level_closure_abs_degC:.3e} degC`.\n\n"
        f"Decision: `{decision}`.\n\n"
        "The direct-station result points first to the archived prior/background level at these western stations. The EnKF increment is not innocent: it adds a positive warm contribution, but it is secondary in the 1901-1920 level residual.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
