"""Analyze site, PSM, and season structure of the western increment attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_increment_proxy_psm_groups_1861_1900_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def concentration(values: np.ndarray) -> dict[str, float | int]:
    absolute = np.sort(np.abs(values[np.abs(values) > 1e-15]))[::-1]
    positive = np.sort(values[values > 0])[::-1]
    aw = absolute / absolute.sum() if absolute.sum() else np.asarray([])
    pw = positive / positive.sum() if positive.sum() else np.asarray([])
    return {
        "nonzero_site_count": int(len(absolute)),
        "positive_site_count": int(len(positive)),
        "top5_absolute_site_share": float(aw[:5].sum()) if len(aw) else float("nan"),
        "effective_absolute_site_count": float(1.0 / np.sum(aw**2)) if len(aw) else float("nan"),
        "positive_sites_for_50percent": int(np.searchsorted(np.cumsum(pw), 0.5) + 1) if len(pw) else 0,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Aggregate exact repeated proxy coordinates before calling the signal broad or concentrated.\n"
        "- Quantify location, PSM, documented-variable, role, and season shares without removing records.\n"
        "- A single-season explanation requires one season group to exceed 50% of positive net contribution.\n"
        "- Do not read climate years; analyze only frozen attribution tables.\n",
        encoding="utf-8",
    )
    paths = {
        "summary": SOURCE / "summary.json",
        "groups": SOURCE / "proxy_psm_group_attribution.csv",
        "proxy6": SOURCE / "proxy6_west_trend_contribution_ranking.csv",
        "proxy9": SOURCE / "proxy9_west_trend_contribution_ranking.csv",
    }
    before = {name: sha256(path) for name, path in paths.items()}
    attribution = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if not attribution["interpretation_authorized_for_six_member_west_increment"]:
        raise RuntimeError("Upstream interpretation gate failed")
    groups = pd.read_csv(paths["groups"])
    network_total = {int(row["network"]): float(row["recovered_increment_trend_degC_per_century"]) for row in attribution["network_metrics"]}

    site_tables = []
    site_metrics = {}
    rankings = {}
    for network in (6, 9):
        ranking = pd.read_csv(paths[f"proxy{network}"])
        rankings[network] = ranking
        sites = (
            ranking.groupby(["latitude", "longitude"], as_index=False)
            .agg(
                record_count=("record_index_1based", "size"),
                net_trend_contribution_degC_per_century=("trend_contribution_degC_per_century", "sum"),
                absolute_record_contribution_sum_degC_per_century=("absolute_trend_contribution_degC_per_century", "sum"),
            )
        )
        sites["network"] = network
        sites["absolute_net_site_contribution_degC_per_century"] = sites.net_trend_contribution_degC_per_century.abs()
        sites = sites.sort_values("absolute_net_site_contribution_degC_per_century", ascending=False)
        site_tables.append(sites)
        site_metrics[str(network)] = concentration(sites.net_trend_contribution_degC_per_century.to_numpy(float))
    sites_all = pd.concat(site_tables, ignore_index=True)
    sites_all.to_csv(OUTPUT / "site_cluster_attribution.csv", index=False)

    share_rows = []
    for row in groups.itertuples(index=False):
        total = network_total[int(row.network)]
        share_rows.append({
            **row._asdict(),
            "share_of_network_net_trend": float(row.trend_contribution_sum_degC_per_century / total),
        })
    shares = pd.DataFrame(share_rows)
    shares.to_csv(OUTPUT / "proxy_psm_group_attribution_with_shares.csv", index=False)

    key_shares = {}
    for network in (6, 9):
        local = shares.loc[(shares.network == network) & (shares.group_dimension == "proxy_location_group") & (shares.group == "west_lt80"), "share_of_network_net_trend"]
        bivariate = shares.loc[(shares.network == network) & (shares.group_dimension == "psm_method") & (shares.group == "bivariate"), "share_of_network_net_trend"]
        key_shares[str(network)] = {
            "west_location_share_of_net": float(local.iloc[0]),
            "bivariate_psm_share_of_net": float(bivariate.iloc[0]),
        }
    season = shares.loc[(shares.network == 9) & (shares.group_dimension == "documented_seasonality")].copy()
    positive_season = season.loc[season.trend_contribution_sum_degC_per_century > 0]
    positive_total = float(positive_season.trend_contribution_sum_degC_per_century.sum())
    season["share_of_positive_season_net"] = np.where(
        season.trend_contribution_sum_degC_per_century > 0,
        season.trend_contribution_sum_degC_per_century / positive_total,
        0.0,
    )
    season.to_csv(OUTPUT / "proxy9_documented_season_attribution.csv", index=False)
    leading_season = season.sort_values("trend_contribution_sum_degC_per_century", ascending=False).iloc[0]

    concentrated_sites = {
        network: bool(site_metrics[str(network)]["top5_absolute_site_share"] >= 0.50 or site_metrics[str(network)]["effective_absolute_site_count"] <= 10.0)
        for network in (6, 9)
    }
    single_season_dominates = bool(leading_season.share_of_positive_season_net > 0.50)
    broad_local_bivariate = bool(
        all(key_shares[str(network)]["west_location_share_of_net"] >= 0.80 for network in (6, 9))
        and all(key_shares[str(network)]["bivariate_psm_share_of_net"] >= 0.90 for network in (6, 9))
        and not all(concentrated_sites.values())
    )
    summary = {
        "status": "complete",
        "network_net_trends_degC_per_century": network_total,
        "key_group_shares": key_shares,
        "site_concentration": site_metrics,
        "site_concentration_flags": concentrated_sites,
        "leading_documented_season": {
            "group": str(leading_season.group),
            "trend_contribution_degC_per_century": float(leading_season.trend_contribution_sum_degC_per_century),
            "share_of_positive_season_net": float(leading_season.share_of_positive_season_net),
        },
        "single_documented_season_dominates": single_season_dominates,
        "broad_local_bivariate_mechanism": broad_local_bivariate,
        "decision": "test_network_wide_bivariate_temperature_psm_stability_not_single_record_or_season_deletion" if broad_local_bivariate else "retain_site_or_season_specific_audit",
        "input_hashes_unchanged": before == {name: sha256(path) for name, path in paths.items()},
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    colors = {6: "#276FBF", 9: "#B92D27"}
    for network, offset in [(6, -0.18), (9, 0.18)]:
        values = sites_all.loc[sites_all.network.eq(network)].head(12)
        axes[0].bar(np.arange(len(values)) + offset, values.net_trend_contribution_degC_per_century, width=0.36, color=colors[network], label=f"proxy{network}")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set(title="Top coordinate-cluster contributions", xlabel="Within-network site rank", ylabel="Trend contribution (degC/century)", xticks=np.arange(0, 12, 2), xticklabels=np.arange(1, 13, 2))
    axes[0].legend(frameon=False)
    season_plot = season.sort_values("trend_contribution_sum_degC_per_century", ascending=True)
    axes[1].barh(season_plot.group.astype(str), season_plot.trend_contribution_sum_degC_per_century, color="#6A8E5E")
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set(title="proxy9 documented-season net contributions", xlabel="Trend contribution (degC/century)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "temperature_west_increment_site_psm_season_attribution.png", dpi=240)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
