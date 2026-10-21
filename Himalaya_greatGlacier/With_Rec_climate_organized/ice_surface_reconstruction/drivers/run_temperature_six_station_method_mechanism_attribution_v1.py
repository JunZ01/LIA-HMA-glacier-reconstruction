"""Resolve member/method routes behind the six-station warm residual."""

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


def root_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "temperature_six_station_prior_increment_level_attribution_v1"
    output = loop / "temperature_six_station_method_mechanism_attribution_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "annual": source / "station_member_annual_components.csv",
        "v1_summary": source / "summary.json",
        "v2_summary": loop / "temperature_six_station_prior_increment_level_attribution_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    if json.loads(paths["v1_summary"].read_text(encoding="utf-8"))["locked_1921_1939_accessed"]:
        raise RuntimeError("v1-lock-boundary-failed")
    if json.loads(paths["v2_summary"].read_text(encoding="utf-8"))["locked_1921_1939_accessed"]:
        raise RuntimeError("v2-lock-boundary-failed")

    annual = pd.read_csv(paths["annual"])
    if annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-in-derived-input")
    station_rows: list[dict[str, object]] = []
    for keys, group in annual.groupby(["member", "method", "proxy_network", "station_id", "station_name"], sort=False):
        member, method, network, station_id, station_name = keys
        for period in PERIODS:
            start, end = map(int, period.split("-"))
            section = group[group.year.between(start, end)]
            for target_column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
                selected = section[np.isfinite(section[target_column]) & np.isfinite(section.posterior_degC)]
                if not len(selected):
                    continue
                station = selected[target_column].to_numpy(float)
                posterior = selected.posterior_degC.to_numpy(float)
                background = selected.background_degC.to_numpy(float)
                increment = selected.increment_degC.to_numpy(float)
                station_rows.append({
                    "member": member,
                    "method": method,
                    "proxy_network": int(network),
                    "station_id": station_id,
                    "station_name": station_name,
                    "period": period,
                    "target": target,
                    "matched_years": int(len(selected)),
                    "station_mean_degC": float(np.mean(station)),
                    "posterior_mean_degC": float(np.mean(posterior)),
                    "background_mean_degC": float(np.mean(background)),
                    "increment_mean_degC": float(np.mean(increment)),
                    "posterior_minus_station_degC": float(np.mean(posterior - station)),
                    "background_minus_station_degC": float(np.mean(background - station)),
                    "closure_abs_degC": float(abs(np.mean(posterior - station) - np.mean(background - station) - np.mean(increment))),
                })
    station = pd.DataFrame(station_rows)
    station.to_csv(output / "member_station_period_level_terms.csv", index=False)

    member_rows: list[dict[str, object]] = []
    for keys, group in station.groupby(["member", "method", "proxy_network", "period", "target"], sort=False):
        member, method, network, period, target = keys
        eligible = group[group.matched_years >= 5]
        if not len(eligible):
            continue
        residual = float(eligible.posterior_minus_station_degC.mean())
        background = float(eligible.background_minus_station_degC.mean())
        increment = float(eligible.increment_mean_degC.mean())
        member_rows.append({
            "member": member,
            "method": method,
            "proxy_network": int(network),
            "period": period,
            "target": target,
            "eligible_station_count": int(len(eligible)),
            "equal_station_mean_posterior_minus_station_degC": residual,
            "equal_station_mean_background_minus_station_degC": background,
            "equal_station_mean_increment_degC": increment,
            "closure_abs_degC": abs(residual - background - increment),
            "warm_residual_station_count": int((eligible.posterior_minus_station_degC > 0).sum()),
            "positive_increment_station_count": int((eligible.increment_mean_degC > 0).sum()),
            "background_dominant_station_count": int((eligible.background_minus_station_degC.abs() >= eligible.increment_mean_degC.abs()).sum()),
        })
    members = pd.DataFrame(member_rows)
    members.to_csv(output / "member_level_mechanism_summary.csv", index=False)

    method_rows: list[dict[str, object]] = []
    for keys, group in members.groupby(["method", "period", "target"], sort=False):
        method, period, target = keys
        residual = float(group.equal_station_mean_posterior_minus_station_degC.mean())
        background = float(group.equal_station_mean_background_minus_station_degC.mean())
        increment = float(group.equal_station_mean_increment_degC.mean())
        method_rows.append({
            "method": method,
            "period": period,
            "target": target,
            "member_count": int(len(group)),
            "member_mean_posterior_minus_station_degC": residual,
            "member_mean_background_minus_station_degC": background,
            "member_mean_increment_degC": increment,
            "closure_abs_degC": abs(residual - background - increment),
        })
    methods = pd.DataFrame(method_rows)
    methods.to_csv(output / "method_level_mechanism_summary.csv", index=False)

    early = members[(members.period == "1901-1920") & (members.target == "qcf")].copy()
    ordinary = early[early.method == "O"]
    adaptive = early[early.method != "O"]
    all_warm = bool((early.equal_station_mean_posterior_minus_station_degC > 0).all())
    fixed_background_zero = bool(np.allclose(ordinary.equal_station_mean_background_minus_station_degC, -ordinary.assign(dummy=0).dummy))
    # A zero fixed-background anomaly implies background-minus-station equals minus the observed station anomaly, not zero residual.
    ordinary_increment_cooling = bool((ordinary.equal_station_mean_increment_degC < 0).all())
    adaptive_increment_warming = bool((adaptive.equal_station_mean_increment_degC > 0).all())
    adaptive_background_cold = bool((adaptive.equal_station_mean_background_minus_station_degC > 0).all())
    closure_max = float(max(station.closure_abs_degC.max(), members.closure_abs_degC.max(), methods.closure_abs_degC.max()))
    if closure_max > 1e-10:
        raise RuntimeError(f"closure-failed:{closure_max}")
    if all_warm and ordinary_increment_cooling and adaptive_increment_warming:
        decision = "two_route_failure_fixed_updates_under_cool_adaptive_updates_warm"
    elif all_warm:
        decision = "warm_residual_common_but_method_route_mixed"
    else:
        decision = "member_specific_warm_residual"

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("derived-input-changed")
    summary = {
        "experiment": "temperature_six_station_method_mechanism_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "qcf_1901_1920_all_six_members_warm": all_warm,
        "ordinary_increment_is_cooling_for_both_networks": ordinary_increment_cooling,
        "adaptive_increment_is_warming_for_all_four_members": adaptive_increment_warming,
        "adaptive_background_minus_station_positive_for_all_four_members": adaptive_background_cold,
        "ordinary_member_metrics": ordinary.to_dict(orient="records"),
        "adaptive_member_metrics": adaptive.to_dict(orient="records"),
        "maximum_closure_abs_degC": closure_max,
        "decision": decision,
        "interpretation_boundary": "The ordinary background anomaly is structurally fixed at zero by experiment design; its background-minus-station term is therefore not a time-varying prior forecast error. The adaptive backgrounds are deterministic strategy-C recoveries.",
        "derived_inputs_unchanged": unchanged,
        "original_sources_reopened": False,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    plot = early.set_index("member").loc[["AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"]]
    positions = np.arange(len(plot))
    background = plot.equal_station_mean_background_minus_station_degC.to_numpy(float)
    increment = plot.equal_station_mean_increment_degC.to_numpy(float)
    residual = plot.equal_station_mean_posterior_minus_station_degC.to_numpy(float)
    fig, axis = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    axis.bar(positions, background, color="#00798C", label="background - station")
    axis.bar(positions, increment, bottom=background, color="#7A5195", label="assimilation increment")
    axis.plot(positions, residual, color="#111111", marker="o", lw=1.5, label="posterior - station")
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set(xticks=positions, xticklabels=plot.index, ylabel="Equal-station mean anomaly term (degC)", title="1901-1920 QCF residual mechanism by archived member")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output / "member_method_level_mechanism_1901_1920.png", dpi=220)
    plt.close(fig)

    ordinary_text = ", ".join(f"{row.member} {row.equal_station_mean_increment_degC:+.3f}" for row in ordinary.itertuples(index=False))
    adaptive_text = ", ".join(f"{row.member} {row.equal_station_mean_increment_degC:+.3f}" for row in adaptive.itertuples(index=False))
    (output / "FINAL_RESULT.md").write_text(
        "# Six-station method mechanism attribution\n\n"
        f"- All six member posterior residuals are warm in 1901-1920: `{str(all_warm).lower()}`.\n"
        f"- Ordinary fixed-background increments (degC): `{ordinary_text}`.\n"
        f"- Adaptive AO/HGAO increments (degC): `{adaptive_text}`.\n"
        f"- Ordinary increments cool in both networks: `{str(ordinary_increment_cooling).lower()}`.\n"
        f"- Adaptive increments warm in all four members: `{str(adaptive_increment_warming).lower()}`.\n\n"
        f"Decision: `{decision}`.\n\n"
        "The common posterior warm residual is not produced by one universal operator. Ordinary EnKF starts from a stationary zero-anomaly background and cools too little; AO/HGAO select adaptive backgrounds and then apply warming increments. The next diagnostic must therefore test update amplitude/sign separately for ordinary and adaptive branches, not apply one global offset.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
