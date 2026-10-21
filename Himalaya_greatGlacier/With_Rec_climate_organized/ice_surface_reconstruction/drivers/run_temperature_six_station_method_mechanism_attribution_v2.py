"""Correct the method-route interpretation after inspecting all six members."""

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


def root_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "temperature_six_station_method_mechanism_attribution_v1"
    output = loop / "temperature_six_station_method_mechanism_attribution_v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "members": source / "member_level_mechanism_summary.csv",
        "stations": source / "member_station_period_level_terms.csv",
        "v1_summary": source / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    old = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    if old["locked_1921_1939_accessed"] or not old["derived_inputs_unchanged"]:
        raise RuntimeError("v1-boundary-failed")
    members = pd.read_csv(paths["members"])
    stations = pd.read_csv(paths["stations"])
    early = members[(members.period == "1901-1920") & (members.target == "qcf")].copy()
    if len(early) != 6:
        raise RuntimeError("six-member-gate-failed")

    all_warm = bool((early.equal_station_mean_posterior_minus_station_degC > 0).all())
    all_increments_warm = bool((early.equal_station_mean_increment_degC > 0).all())
    all_background_terms_positive = bool((early.equal_station_mean_background_minus_station_degC > 0).all())
    background_dominates_all = bool(
        (early.equal_station_mean_background_minus_station_degC.abs() > early.equal_station_mean_increment_degC.abs()).all()
    )
    network = early.groupby("proxy_network", as_index=False)[[
        "equal_station_mean_posterior_minus_station_degC",
        "equal_station_mean_background_minus_station_degC",
        "equal_station_mean_increment_degC",
    ]].mean()
    method = early.groupby("method", as_index=False)[[
        "equal_station_mean_posterior_minus_station_degC",
        "equal_station_mean_background_minus_station_degC",
        "equal_station_mean_increment_degC",
    ]].mean()
    network.to_csv(output / "network_level_mechanism_1901_1920.csv", index=False)
    method.to_csv(output / "method_level_mechanism_1901_1920.csv", index=False)
    early.to_csv(output / "member_level_mechanism_1901_1920.csv", index=False)

    station_early = stations[(stations.period == "1901-1920") & (stations.target == "qcf") & (stations.matched_years >= 5)]
    station_sign = station_early.groupby(["station_id", "station_name"], as_index=False).agg(
        member_count=("member", "size"),
        positive_posterior_member_count=("posterior_minus_station_degC", lambda values: int((values > 0).sum())),
        positive_increment_member_count=("increment_mean_degC", lambda values: int((values > 0).sum())),
        mean_posterior_minus_station_degC=("posterior_minus_station_degC", "mean"),
        mean_background_minus_station_degC=("background_minus_station_degC", "mean"),
        mean_increment_degC=("increment_mean_degC", "mean"),
    )
    station_sign.to_csv(output / "station_member_sign_consistency_1901_1920.csv", index=False)
    if all_warm and all_increments_warm and all_background_terms_positive and background_dominates_all:
        decision = "common_background_level_mismatch_primary_and_all_assimilation_updates_reinforce_warming"
    else:
        decision = "method_or_member_specific_route_requires_followup"

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("derived-input-changed")
    summary = {
        "experiment": "temperature_six_station_method_mechanism_attribution_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes_interpretation_of": "temperature_six_station_method_mechanism_attribution_v1",
        "correction": "V1 anticipated ordinary cooling increments, but the calculated table shows positive increments for all six archived members.",
        "locked_1921_1939_accessed": False,
        "all_six_member_posterior_residuals_warm": all_warm,
        "all_six_member_increment_terms_warm": all_increments_warm,
        "all_six_member_background_minus_station_terms_positive": all_background_terms_positive,
        "background_term_dominates_increment_in_all_six_members": background_dominates_all,
        "posterior_residual_member_range_degC": [float(early.equal_station_mean_posterior_minus_station_degC.min()), float(early.equal_station_mean_posterior_minus_station_degC.max())],
        "increment_member_range_degC": [float(early.equal_station_mean_increment_degC.min()), float(early.equal_station_mean_increment_degC.max())],
        "background_term_member_range_degC": [float(early.equal_station_mean_background_minus_station_degC.min()), float(early.equal_station_mean_background_minus_station_degC.max())],
        "network6_minus_network9_posterior_residual_degC": float(network.loc[network.proxy_network == 6, "equal_station_mean_posterior_minus_station_degC"].iloc[0] - network.loc[network.proxy_network == 9, "equal_station_mean_posterior_minus_station_degC"].iloc[0]),
        "station_all_six_members_warm_count": int((station_sign.positive_posterior_member_count == 6).sum()),
        "station_all_six_increments_warm_count": int((station_sign.positive_increment_member_count == 6).sum()),
        "decision": decision,
        "interpretation_boundary": "For ordinary O members the prior background anomaly is fixed at zero. A positive background-minus-station term therefore means that the cold station anomaly is absent from the stationary background; the positive update then moves in the wrong level direction for this station target. Adaptive background recovery remains deterministic rather than archived member-index exact.",
        "derived_inputs_unchanged": unchanged,
        "original_sources_reopened": False,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    plot = early.set_index("member").loc[["AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"]]
    x = np.arange(len(plot))
    background = plot.equal_station_mean_background_minus_station_degC.to_numpy(float)
    increment = plot.equal_station_mean_increment_degC.to_numpy(float)
    residual = plot.equal_station_mean_posterior_minus_station_degC.to_numpy(float)
    fig, axis = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    axis.bar(x, background, color="#00798C", label="background - station")
    axis.bar(x, increment, bottom=background, color="#7A5195", label="assimilation increment")
    axis.plot(x, residual, color="#111111", marker="o", lw=1.5, label="posterior - station")
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set(xticks=x, xticklabels=plot.index, ylabel="Equal-station mean anomaly term (degC)", title="1901-1920 QCF: every member has a positive update")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output / "corrected_member_mechanism_1901_1920.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Corrected six-member mechanism result\n\n"
        "V1's calculated values were valid, but its anticipated route label was not. V2 supersedes that interpretation.\n\n"
        f"- Posterior warm residual in all six members: `{str(all_warm).lower()}`; range `{early.equal_station_mean_posterior_minus_station_degC.min():+.3f}` to `{early.equal_station_mean_posterior_minus_station_degC.max():+.3f} degC`.\n"
        f"- Positive assimilation increment in all six members: `{str(all_increments_warm).lower()}`; range `{early.equal_station_mean_increment_degC.min():+.3f}` to `{early.equal_station_mean_increment_degC.max():+.3f} degC`.\n"
        f"- Background-minus-station is positive and larger than the increment in all six members: `{str(background_dominates_all).lower()}`; range `{early.equal_station_mean_background_minus_station_degC.min():+.3f}` to `{early.equal_station_mean_background_minus_station_degC.max():+.3f} degC`.\n"
        f"- Network-6 minus network-9 posterior residual: `{summary['network6_minus_network9_posterior_residual_degC']:+.3f} degC`; network choice is secondary.\n\n"
        f"Decision: `{decision}`.\n\n"
        "The immediate mechanism is now specific: the stationary/adaptive background does not contain enough of the observed cold anomaly, and every archived update adds warming rather than correcting it. The next experiment should test proxy-innovation and gain sign/amplitude at these six coordinates, while keeping the archived background fixed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
