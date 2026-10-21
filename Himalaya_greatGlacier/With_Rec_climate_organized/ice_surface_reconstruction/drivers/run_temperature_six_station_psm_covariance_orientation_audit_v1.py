"""Audit whether PSM-sign or prior-covariance orientation drives warm O updates."""

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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sign_label(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([numeric > 0, numeric < 0, numeric == 0], ["positive", "negative", "zero"], default="nonfinite"), index=values.index)


def aggregate(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for dimension in dimensions:
        for (network, group), values in frame.groupby(["network", dimension], dropna=False):
            contribution = values.equal_station_mean_contribution_degC
            rows.append({
                "network": int(network),
                "dimension": dimension,
                "group": str(group),
                "record_count": int(len(values)),
                "net_contribution_degC": float(contribution.sum()),
                "positive_contribution_degC": float(contribution[contribution > 0].sum()),
                "negative_contribution_degC": float(contribution[contribution < 0].sum()),
                "absolute_contribution_degC": float(contribution.abs().sum()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "temperature_six_station_oenkf_innovation_gain_attribution_v2"
    output = loop / "temperature_six_station_psm_covariance_orientation_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "contributions": source / "all_proxy_six_station_contributions.csv",
        "groups": source / "proxy_group_contributions.csv",
        "attribution_summary": source / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in paths.items()}
    upstream = json.loads(paths["attribution_summary"].read_text(encoding="utf-8"))
    if upstream["status"] != "complete" or upstream["locked_1921_1939_accessed"]:
        raise RuntimeError("upstream-attribution-gate-failed")

    data = pd.read_csv(paths["contributions"])
    data["b1_sign"] = sign_label(data.b1_temperature)
    data["gain_sign"] = sign_label(data.mean_six_station_gain)
    data["innovation_anomaly_sign"] = sign_label(data.innovation_1901_1920_minus_1961_1990_proxy_units)
    data["contribution_sign"] = sign_label(data.equal_station_mean_contribution_degC)
    finite_orientation = data.b1_sign.isin(["positive", "negative"]) & data.gain_sign.isin(["positive", "negative"])
    data["b1_gain_orientation"] = np.where(
        finite_orientation,
        np.where(data.b1_sign == data.gain_sign, "aligned", "opposed"),
        "unresolved",
    )
    finite_product = data.gain_sign.isin(["positive", "negative"]) & data.innovation_anomaly_sign.isin(["positive", "negative"])
    data["gain_innovation_product_sign"] = np.where(
        finite_product,
        np.where(data.gain_sign == data.innovation_anomaly_sign, "positive", "negative"),
        "unresolved",
    )
    data["formal_b1_innovation_product_sign"] = np.where(
        data.b1_sign.isin(["positive", "negative"]) & data.innovation_anomaly_sign.isin(["positive", "negative"]),
        np.where(data.b1_sign == data.innovation_anomaly_sign, "positive", "negative"),
        "unresolved",
    )
    data.to_csv(output / "proxy_orientation_diagnostics.csv", index=False)

    dimensions = [
        "proxy_location_group", "b1_sign", "gain_sign", "innovation_anomaly_sign",
        "b1_gain_orientation", "gain_innovation_product_sign", "formal_b1_innovation_product_sign",
    ]
    if "network_role" in data.columns:
        dimensions.append("network_role")
    grouped = aggregate(data, dimensions)
    grouped.to_csv(output / "orientation_group_contributions.csv", index=False)

    network_rows = []
    for network, values in data.groupby("network"):
        positive = values[values.equal_station_mean_contribution_degC > 0]
        total_positive = float(positive.equal_station_mean_contribution_degC.sum())
        total_net = float(values.equal_station_mean_contribution_degC.sum())
        west_net = float(values.loc[values.proxy_location_group == "west_lt80", "equal_station_mean_contribution_degC"].sum())
        opposed_positive = float(positive.loc[positive.b1_gain_orientation == "opposed", "equal_station_mean_contribution_degC"].sum())
        aligned_positive = float(positive.loc[positive.b1_gain_orientation == "aligned", "equal_station_mean_contribution_degC"].sum())
        product_match = values.gain_innovation_product_sign == values.contribution_sign
        unresolved_net = float(values.loc[values.get("network_role", pd.Series(index=values.index, dtype=object)) == "unresolved", "equal_station_mean_contribution_degC"].sum()) if "network_role" in values else np.nan
        network_rows.append({
            "network": int(network),
            "net_increment_degC": total_net,
            "west_net_contribution_degC": west_net,
            "west_fraction_of_net_increment": west_net / total_net,
            "positive_contribution_total_degC": total_positive,
            "positive_contribution_from_b1_gain_aligned_degC": aligned_positive,
            "positive_contribution_from_b1_gain_opposed_degC": opposed_positive,
            "opposed_fraction_of_positive_contribution": opposed_positive / total_positive,
            "aligned_fraction_of_positive_contribution": aligned_positive / total_positive,
            "gain_innovation_product_sign_match_fraction": float(product_match.mean()),
            "unresolved_role_net_contribution_degC": unresolved_net,
            "unresolved_role_fraction_of_net_increment": unresolved_net / total_net if np.isfinite(unresolved_net) else np.nan,
        })
    network_table = pd.DataFrame(network_rows)
    network_table.to_csv(output / "network_orientation_summary.csv", index=False)

    both_covariance_opposed_material = bool((network_table.opposed_fraction_of_positive_contribution >= 0.40).all())
    both_gain_innovation_explain = bool((network_table.gain_innovation_product_sign_match_fraction >= 0.85).all())
    both_west_dominant = bool((network_table.west_fraction_of_net_increment >= 0.70).all())
    if both_covariance_opposed_material:
        decision = "prior_covariance_orientation_conflict_material"
    elif both_gain_innovation_explain and both_west_dominant:
        decision = "western_proxy_innovation_common_mode_with_covariance_consistent_gains"
    else:
        decision = "mixed_psm_covariance_orientation"

    after = {name: digest(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("derived-input-changed")
    summary = {
        "experiment": "temperature_six_station_psm_covariance_orientation_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1901, 1920],
        "baseline": [1961, 1990],
        "locked_1921_1939_accessed": False,
        "network_metrics": network_rows,
        "both_networks_b1_gain_opposition_material": both_covariance_opposed_material,
        "both_networks_gain_innovation_sign_explains_contribution": both_gain_innovation_explain,
        "both_networks_western_contribution_dominant": both_west_dominant,
        "decision": decision,
        "interpretation_boundary": "The b1-versus-gain comparison is directional. A bivariate PSM's prior covariance gain is not required to equal the formal local b1 sign, and exact contributions contain pool/year covariance beyond products of means.",
        "derived_inputs_unchanged": unchanged,
        "original_sources_reopened": False,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    plot = grouped[grouped.dimension.isin(["proxy_location_group", "b1_gain_orientation"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for axis, dimension, title in zip(axes, ("proxy_location_group", "b1_gain_orientation"), ("Contribution by proxy longitude band", "Contribution by PSM/gain orientation")):
        selected = plot[plot.dimension == dimension]
        labels = list(dict.fromkeys(selected.group.tolist()))
        x = np.arange(len(labels))
        for offset, network, color in ((-0.18, 6, "#00798C"), (0.18, 9, "#B23A48")):
            series = selected[selected.network == network].set_index("group").net_contribution_degC.reindex(labels).fillna(0)
            axis.bar(x + offset, series, width=0.36, color=color, label=f"proxy{network}")
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set(xticks=x, xticklabels=labels, ylabel="Net contribution (degC)", title=title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.savefig(output / "psm_covariance_orientation_groups.png", dpi=220)
    plt.close(fig)

    lines = ["# Six-station PSM/covariance orientation audit", ""]
    for row in network_table.itertuples(index=False):
        lines.append(f"- proxy{row.network}: west contribution `{row.west_fraction_of_net_increment:.1%}` of net update; b1/gain-opposed records provide `{row.opposed_fraction_of_positive_contribution:.1%}` of positive contribution; gain-innovation mean-sign agreement with exact contribution `{row.gain_innovation_product_sign_match_fraction:.1%}`.")
    lines += ["", f"Decision: `{decision}`.", ""]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
