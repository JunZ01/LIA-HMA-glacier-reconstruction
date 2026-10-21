"""Audit direct precipitation-pathway effects of significant positive-b2 PSMs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIOD = (1901, 1920)
BRANCHES = [
    "baseline",
    "combined_rebuild",
    "temperature_only",
    "precipitation_only",
    "zero_selected_anomaly",
]
TOLERANCE = 1e-12


def project_root() -> Path:
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


def matlab_string(handle: h5py.File, reference: h5py.Reference) -> str:
    return "".join(chr(int(value)) for value in np.asarray(handle[reference]).reshape(-1))


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(years.astype(float), values.astype(float), 1)[0] * 100.0)


def main() -> int:
    root = project_root()
    operator_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1"
    stability_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_psm_temporal_stability_v2"
    common_mode_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    result_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1"
    inputs = {
        "operator_decomposition": operator_dir / "operator_decomposition.mat",
        "operator_summary": operator_dir / "operator_decomposition_summary.json",
        "source_audit": common_mode_dir / "proxy_source_trend_table.csv",
        "pc1_loadings": common_mode_dir / "innovation_pc1_loadings.csv",
        "pc1_yearly": common_mode_dir / "innovation_pc1_yearly.csv",
        "stability": stability_dir / "psm_temporal_stability_proxy_level.csv",
        "drift": stability_dir / "psm_temporal_coefficient_drift_proxy_level.csv",
    }
    if missing := [str(path) for path in inputs.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if result_dir.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{result_dir}")
    source_hashes_before = {name: sha256(path) for name, path in inputs.items()}

    with h5py.File(inputs["operator_decomposition"], "r") as handle:
        branches = [matlab_string(handle, ref) for ref in handle["branch_names"][()].reshape(-1)]
        if branches != BRANCHES:
            raise RuntimeError(f"branch-contract-failed:{branches}")
        years = np.asarray(handle["run_years"]).reshape(-1).astype(int)
        total_increment = np.asarray(handle["branch_total_increment"]).T
        contribution = np.asarray(handle["branch_contribution"]).transpose(1, 2, 0)
        selected = np.asarray(handle["selected"]).reshape(-1).astype(int)
        closure = {
            "combined_operator": float(np.asarray(handle["combined_closure"]).squeeze()),
            "archived_baseline": float(np.asarray(handle["baseline_closure"]).squeeze()),
            "innovation_invariance": float(np.asarray(handle["innovation_invariance"]).squeeze()),
            "increment": float(np.max(np.abs(np.asarray(handle["branch_closure_error"])))),
        }
    if not np.array_equal(years, np.arange(PERIOD[0], PERIOD[1] + 1)):
        raise RuntimeError("period-contract-failed")
    if len(selected) != 65 or len(np.unique(selected)) != 65:
        raise RuntimeError("selected-proxy-contract-failed")
    if any(value > TOLERANCE for value in closure.values()):
        raise RuntimeError("operator-closure-contract-failed:" + json.dumps(closure, sort_keys=True))

    audit = pd.read_csv(inputs["source_audit"]).set_index("proxy_index")
    chosen = audit.loc[selected].copy()
    chosen = chosen.loc[(chosen["psm_significance"] == "p<=0.05") & (chosen["coefficient_b2"] >= 0)].copy()
    if len(chosen) != 60:
        raise RuntimeError(f"positive-b2-selection-contract-failed:{len(chosen)}")
    chosen_indices = chosen.index.to_numpy(dtype=int)
    p_branch = BRANCHES.index("precipitation_only")
    zero_branch = BRANCHES.index("zero_selected_anomaly")
    p_effect_by_proxy = contribution[:, :, p_branch] - contribution[:, :, zero_branch]
    p_effect_total = total_increment[:, p_branch] - total_increment[:, zero_branch]
    direct_positive = p_effect_by_proxy[:, chosen_indices - 1].sum(axis=1)
    direct_other = p_effect_total - direct_positive

    stability = pd.read_csv(inputs["stability"])
    cross_period = stability.pivot(index="proxy_index", columns="split", values=["test_r", "test_rmse", "external_1901_1920_r", "external_1901_1920_rmse"]) 
    cross_period.columns = ["_".join(map(str, column)).strip() for column in cross_period.columns]
    drift = pd.read_csv(inputs["drift"]).set_index("proxy_index")
    pc1 = pd.read_csv(inputs["pc1_loadings"]).set_index("proxy_index")

    proxy_rows: list[dict[str, object]] = []
    for proxy_index in chosen_indices:
        effect = p_effect_by_proxy[:, proxy_index - 1]
        row = chosen.loc[proxy_index].to_dict()
        row.update(
            {
                "proxy_index": int(proxy_index),
                "precipitation_pathway_effect_mean_mm_day": float(effect.mean()),
                "precipitation_pathway_effect_start_1901_mm_day": float(effect[0]),
                "precipitation_pathway_effect_end_1920_mm_day": float(effect[-1]),
                "precipitation_pathway_effect_trend_mm_day_per_century": trend_per_century(years, effect),
                "precipitation_pathway_effect_abs_trend_mm_day_per_century": abs(trend_per_century(years, effect)),
                "b2_early": float(drift.loc[proxy_index, "b2_early"]),
                "b2_late": float(drift.loc[proxy_index, "b2_late"]),
                "b2_early_minus_late": float(drift.loc[proxy_index, "b2_early_minus_late"]),
                "b2_sign_agreement_early_late": bool(drift.loc[proxy_index, "b2_early"] * drift.loc[proxy_index, "b2_late"] > 0),
                "pc1_loading": float(pc1.loc[proxy_index, "pc1_loading"]),
            }
        )
        row.update(cross_period.loc[proxy_index].to_dict())
        proxy_rows.append(row)
    proxy_table = pd.DataFrame(proxy_rows).sort_values("precipitation_pathway_effect_trend_mm_day_per_century")

    total_p_trend = trend_per_century(years, p_effect_total)
    direct_positive_trend = trend_per_century(years, direct_positive)
    other_trend = trend_per_century(years, direct_other)
    negative = proxy_table.loc[proxy_table["precipitation_pathway_effect_trend_mm_day_per_century"] < 0].copy()
    negative_mass = float(negative["precipitation_pathway_effect_abs_trend_mm_day_per_century"].sum())
    top_ten = negative.nsmallest(10, "precipitation_pathway_effect_trend_mm_day_per_century").copy()
    top_ten_share = float(top_ten["precipitation_pathway_effect_abs_trend_mm_day_per_century"].sum() / negative_mass)
    top_ten_unstable = int((~top_ten["b2_sign_agreement_early_late"]).sum())
    all_unstable = int((~proxy_table["b2_sign_agreement_early_late"]).sum())
    pc1_majority_sign = 1 if float(proxy_table["pc1_loading"].median()) >= 0 else -1
    top_ten_pc1_majority = int((np.sign(top_ten["pc1_loading"]) == pc1_majority_sign).sum())
    p_effect_pc1_correlation = float(np.corrcoef(proxy_table["precipitation_pathway_effect_trend_mm_day_per_century"], proxy_table["pc1_loading"])[0, 1])
    p_effect_abs_b2_drift_correlation = float(np.corrcoef(proxy_table["precipitation_pathway_effect_trend_mm_day_per_century"], np.abs(proxy_table["b2_early_minus_late"]))[0, 1])

    yearly = pd.DataFrame(
        {
            "year": years,
            "total_precipitation_pathway_effect_mm_day": p_effect_total,
            "direct_positive_b2_effect_mm_day": direct_positive,
            "all_other_effect_mm_day": direct_other,
        }
    )
    pc_years = pd.read_csv(inputs["pc1_yearly"])
    if not np.array_equal(pc_years["year"].to_numpy(dtype=int), years):
        raise RuntimeError("pc1-period-contract-failed")
    yearly = yearly.merge(pc_years, on="year", how="left", validate="one_to_one")

    result_dir.mkdir(parents=True)
    proxy_table.to_csv(result_dir / "positive_b2_proxy_mechanism_table.csv", index=False)
    yearly.to_csv(result_dir / "positive_b2_pathway_yearly.csv", index=False)
    top_ten.to_csv(result_dir / "top_ten_negative_positive_b2_proxies.csv", index=False)

    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1",
        "period": list(PERIOD),
        "locked_final_period_accessed": False,
        "selection": "65 archived-significant HMA bivariate tree PSMs, restricted to production b2 >= 0",
        "positive_b2_proxy_count": int(len(proxy_table)),
        "operator_closure_max_abs_errors": closure,
        "trend_mm_day_per_century": {
            "full_precipitation_pathway_effect": total_p_trend,
            "direct_positive_b2_proxy_effect": direct_positive_trend,
            "all_other_proxy_and_covariance_effect": other_trend,
        },
        "concentration": {
            "negative_direct_effect_proxy_count": int(len(negative)),
            "top_ten_share_of_negative_direct_effect_absolute_trend": top_ten_share,
            "top_ten_b2_sign_disagreement_count": top_ten_unstable,
            "all_positive_b2_b2_sign_disagreement_count": all_unstable,
        },
        "common_mode": {
            "top_ten_pc1_majority_sign_count": top_ten_pc1_majority,
            "top_ten_count": int(len(top_ten)),
            "proxy_effect_trend_pc1_loading_correlation": p_effect_pc1_correlation,
            "proxy_effect_trend_abs_b2_drift_correlation": p_effect_abs_b2_drift_correlation,
            "pc1_sign_note": "PC1 sign is oriented by the selected positive-b2 median loading; sign itself is otherwise arbitrary.",
        },
        "interpretation_limit": "Read-only mechanism audit. It does not refit, reject, replace, or downweight PSMs, and it does not produce a revised reconstruction.",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8})
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 9.4), constrained_layout=True)
    axes[0].plot(years, p_effect_total, color="#1b1b1b", linewidth=2, label="full P-pathway effect")
    axes[0].plot(years, direct_positive, color="#0072B2", linewidth=1.8, label="direct positive-b2 rows")
    axes[0].plot(years, direct_other, color="#D55E00", linewidth=1.5, label="other rows and covariance response")
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[0].set_ylabel("Increment effect (mm day$^{-1}$)")
    axes[0].set_title("Where the precipitation-pathway effect enters")
    axes[0].legend(frameon=False, ncol=2)

    for _, row in top_ten.iterrows():
        proxy_index = int(row["proxy_index"])
        axes[1].plot(years, p_effect_by_proxy[:, proxy_index - 1], linewidth=1.25, label=str(proxy_index))
    axes[1].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[1].set_ylabel("Direct effect (mm day$^{-1}$)")
    axes[1].set_title("Top 10 negative direct positive-b2 proxy effects")
    axes[1].legend(title="proxy index", ncol=5, frameon=False, loc="best")

    stability_color = np.where(proxy_table["b2_sign_agreement_early_late"], "#0072B2", "#D55E00")
    axes[2].scatter(np.abs(proxy_table["b2_early_minus_late"]), proxy_table["precipitation_pathway_effect_trend_mm_day_per_century"], c=stability_color, s=32, alpha=0.9)
    for _, row in top_ten.iterrows():
        axes[2].annotate(str(int(row["proxy_index"])), (abs(row["b2_early_minus_late"]), row["precipitation_pathway_effect_trend_mm_day_per_century"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[2].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[2].set_xlabel("Absolute early-minus-late b2 drift")
    axes[2].set_ylabel("Direct P-effect trend (mm day$^{-1}$ century$^{-1}$)")
    axes[2].set_title("Direct effect versus b2 temporal drift (blue: same split sign; orange: sign change)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result_dir / "positive_b2_proxy_mechanism_audit.png", dpi=180)
    plt.close(figure)

    report = [
        "# Positive-b2 Proxy Mechanism Audit",
        "",
        "Period: 1901-1920 only. The locked 1921-1939 final test period was not accessed.",
        "",
        "## Main Result",
        "",
        f"- Full precipitation-pathway trend: `{total_p_trend:.6f}` mm day^-1 century^-1.",
        f"- Direct positive-b2 proxy trend: `{direct_positive_trend:.6f}` mm day^-1 century^-1.",
        f"- Other-proxy and covariance-response trend: `{other_trend:.6f}` mm day^-1 century^-1.",
        f"- Top 10 share of negative direct-effect magnitude: `{top_ten_share:.2%}`.",
        f"- Top-10 b2 split-sign disagreements: `{top_ten_unstable}/{len(top_ten)}`; all positive-b2 disagreements: `{all_unstable}/{len(proxy_table)}`.",
        f"- Top-10 PC1-majority-sign count: `{top_ten_pc1_majority}/{len(top_ten)}`.",
        "",
        "This is a read-only mechanism audit, not a PSM validation or reconstruction replacement decision.",
    ]
    (result_dir / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    source_hashes_after = {name: sha256(path) for name, path in inputs.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (result_dir / "source_manifest.json").write_text(
        json.dumps({"before": source_hashes_before, "after": source_hashes_after, "analysis_driver_sha256": sha256(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
