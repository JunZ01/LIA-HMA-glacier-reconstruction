"""Reporting correction for Round 142 transfer-failure attribution."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "round_142_south_tp_transfer_failure_attribution_v1"
    output = loop / "round_142_south_tp_transfer_failure_attribution_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round142_south_tp_transfer_failure_attribution_v2_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "v1_summary": source / "summary.json",
        "target": source / "instrumental_target_coherence.csv",
        "proxy": source / "proxy_early_late_correlations.csv",
        "stability": source / "proxy_temporal_stability_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    v1 = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    target = pd.read_csv(paths["target"])
    proxy = pd.read_csv(paths["proxy"])
    stability = pd.read_csv(paths["stability"])

    product_rows = []
    for season in ("pSep_cJun", "pJul_cApr"):
        for period in ("early", "late"):
            gpcc = proxy[(proxy["season"] == season) & (proxy["product"] == "GPCC")].set_index("proxy_index")[period]
            cru = proxy[(proxy["season"] == season) & (proxy["product"] == "CRU")].set_index("proxy_index")[period]
            common = gpcc.index.intersection(cru.index)
            left = gpcc.loc[common].to_numpy(dtype=float)
            right = cru.loc[common].to_numpy(dtype=float)
            valid = np.isfinite(left) & np.isfinite(right)
            product_rows.append({
                "season": season,
                "period": period,
                "finite_proxy_count": int(valid.sum()),
                "proxy_correlation_vector_gpcc_cru_correlation": correlation(left[valid], right[valid]),
                "proxy_sign_agreement_fraction_across_products": float(np.mean(np.sign(left[valid]) == np.sign(right[valid]))),
            })
    product = pd.DataFrame(product_rows)
    if product.isna().any().any():
        raise RuntimeError("corrected-product-metrics-still-nan")

    target_coherent = bool((target.gpcc_cru_correlation >= 0.80).all() and target.trend_sign_agreement.all())
    both_fail = {
        season: bool((~stability[stability.season == season].passes_temporal_stability).all())
        for season in ("pSep_cJun", "pJul_cApr")
    }
    proxy_instability = bool(any(both_fail.values()))
    if not target_coherent:
        decision = "resolve_instrumental_target_disagreement_before_psm_redesign"
    elif proxy_instability:
        decision = "advance_to_proxy_specific_cross_validated_seasonal_psm_redesign"
    else:
        decision = "failure_not_attributed_stop_before_method_change"
    if decision != v1["decision"]:
        raise RuntimeError("v1-decision-not-reproduced")

    shutil.copy2(paths["target"], output / "instrumental_target_coherence.csv")
    shutil.copy2(paths["proxy"], output / "proxy_early_late_correlations.csv")
    shutil.copy2(paths["stability"], output / "proxy_temporal_stability_summary.csv")
    product.to_csv(output / "proxy_product_agreement.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    for column, season in enumerate(("pSep_cJun", "pJul_cApr")):
        for product_name in ("GPCC", "CRU"):
            rows = proxy[(proxy["season"] == season) & (proxy["product"] == product_name)]
            axes[column].scatter(rows.early, rows.late, s=32, alpha=0.75, color=colors[product_name], label=product_name)
        axes[column].axhline(0, color="#999999", linewidth=0.7)
        axes[column].axvline(0, color="#999999", linewidth=0.7)
        axes[column].plot([-1, 1], [-1, 1], color="#555555", linestyle="--", linewidth=0.8)
        axes[column].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="proxy-target r, 1941-1970", ylabel="proxy-target r, 1971-2000")
        axes[column].set_title(f"{season}: individual proxy stability")
        axes[column].grid(color="#e2e2e2", linewidth=0.5)
        axes[column].spines[["top", "right"]].set_visible(False)
        axes[column].legend(frameon=False)
    fig.savefig(output / "round142_proxy_temporal_stability_scatter.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_142_south_tp_transfer_failure_attribution_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "v1_core_metrics_changed": False,
        "v1_decision_reproduced": True,
        "target_metrics": target.to_dict(orient="records"),
        "proxy_stability": stability.to_dict(orient="records"),
        "proxy_product_agreement": product.to_dict(orient="records"),
        "instrumental_targets_coherent": target_coherent,
        "both_products_fail_proxy_stability_by_season": both_fail,
        "proxy_temporal_instability_diagnosed": proxy_instability,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 142 Transfer-Failure Attribution V2\n\n"
        + "\n".join(
            f"- {row.season} {row.period}: GPCC/CRU proxy-correlation-vector r=`{row.proxy_correlation_vector_gpcc_cru_correlation:.3f}`, sign agreement=`{row.proxy_sign_agreement_fraction_across_products:.1%}`."
            for row in product.itertuples()
        )
        + f"\n- Instrumental targets coherent: `{target_coherent}`.\n"
        + f"- Proxy temporal instability diagnosed: `{proxy_instability}`.\n"
        + f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
