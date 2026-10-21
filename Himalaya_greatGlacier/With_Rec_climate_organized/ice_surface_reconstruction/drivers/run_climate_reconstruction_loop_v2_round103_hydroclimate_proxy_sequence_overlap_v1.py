"""Test whether Round 102 hydroclimate evidence duplicates frozen proxy sequences."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round102_tree_ring_hydroclimate_evidence_v1 as round102


OUTPUT = round102.LOOP / "round_103_hydroclimate_proxy_sequence_overlap_v1"
ROUND102 = round102.LOOP / "round_102_tree_ring_hydroclimate_evidence_v4"
MANIFEST27 = round102.LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def standardized(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if finite.sum() < 2 or np.ptp(values[finite]) <= 1e-12:
        return result
    result[finite] = (values[finite] - np.mean(values[finite])) / np.std(values[finite], ddof=1)
    return result


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(left) & np.isfinite(right)
    n = int(valid.sum())
    if n < 5:
        return {"overlap_years": n, "pearson_r": np.nan, "spearman_r": np.nan, "z_rmse_best_sign": np.nan, "z_max_abs_best_sign": np.nan, "best_sign": np.nan}
    a = standardized(left[valid]); b = standardized(right[valid])
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 5:
        return {"overlap_years": n, "pearson_r": np.nan, "spearman_r": np.nan, "z_rmse_best_sign": np.nan, "z_max_abs_best_sign": np.nan, "best_sign": np.nan}
    a = a[finite]; b = b[finite]
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    sign = 1.0 if pearson >= 0 else -1.0
    residual = a - sign * b
    return {"overlap_years": int(finite.sum()), "pearson_r": pearson, "spearman_r": spearman, "z_rmse_best_sign": float(np.sqrt(np.mean(residual**2))), "z_max_abs_best_sign": float(np.max(np.abs(residual))), "best_sign": int(sign)}


def evidence_curves() -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    curves = pd.read_csv(ROUND102 / "allowed_period_standardized_curves.csv")
    for period, years in round102.PERIODS.items():
        mada = curves[(curves.period == period) & (curves.evidence == "MADA_JJA_PDSI") & (curves.region == "hma") & (curves.series == "evidence")].sort_values("year")
        if not np.array_equal(mada.year.to_numpy(int), years):
            raise RuntimeError(f"mada-year-axis-failed:{period}")
        result[("MADA_JJA_PDSI", period)] = mada.raw_value.to_numpy(float)
    for name, item in round102.EVIDENCE.items():
        values_by_year, _ = round102.parse_allowed_text(round102.EXTERNAL / item["file"], int(item["value_index"]))
        for period, years in round102.PERIODS.items():
            result[(name, period)] = np.asarray([values_by_year.get(int(year), np.nan) for year in years], dtype=float)
    return result


def main() -> int:
    manifest27 = json.loads(MANIFEST27.read_text(encoding="utf-8"))
    proxy9_path = Path(manifest27["sources_before"]["proxy9"]["path"])
    identity_path = round102.PROXY_IDENTITY
    sources = {"proxy9": proxy9_path, "proxy_identity": identity_path, "round27_manifest": MANIFEST27, "round102_curves": ROUND102 / "allowed_period_standardized_curves.csv", "round102_catalog": ROUND102 / "evidence_catalog_and_overlap.csv", "driver": Path(__file__).resolve()}
    for name, item in round102.EVIDENCE.items():
        sources[f"external_{name}"] = round102.EXTERNAL / item["file"]
    for path in sources.values():
        if not path.exists(): raise FileNotFoundError(path)
    if state(proxy9_path)["sha256"] != manifest27["sources_before"]["proxy9"]["sha256"]:
        raise RuntimeError("proxy9-hash-conflict")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text("\n".join(["# Round 103 Contract", "", "- Compare Round 102 evidence only with frozen proxy9 observations over 1901-1920 and 1940-1950.", "- Do not read, calculate, or export 1921-1939 values.", "- Sequence correlation is overlap evidence, not proof of source identity unless an identity crosswalk also exists.", "- Do not alter proxy, climate, or reconstruction files and do not run EnKF or OGGM."]) + "\n", encoding="utf-8")
    before = {name: state(path) for name, path in sources.items()}
    identity = pd.read_csv(identity_path)
    curves = evidence_curves()
    rows: list[dict[str, object]] = []
    proxy_values: dict[str, np.ndarray] = {}
    for period, years in round102.PERIODS.items():
        _, observations, locked_reads = round25.read_proxy_allowed(proxy9_path, years)
        if locked_reads != 0 or observations.shape[0] != len(identity):
            raise RuntimeError(f"proxy-read-contract-failed:{period}")
        proxy_values[period] = observations
        for evidence_name in ["MADA_JJA_PDSI", *round102.EVIDENCE.keys()]:
            evidence = curves[(evidence_name, period)]
            for proxy_index in range(observations.shape[0]):
                row = compare(evidence, observations[proxy_index])
                rows.append({"period": period, "evidence": evidence_name, "proxy_record_index_1based": proxy_index + 1, "proxy_latitude": float(identity.iloc[proxy_index].latitude), "proxy_longitude": float(identity.iloc[proxy_index].longitude), "proxy_type": identity.iloc[proxy_index].proxy_type, "psm_method": identity.iloc[proxy_index].psm_method, "canonical_source": identity.iloc[proxy_index].canonical_combined_name, **row})
    ranking = pd.DataFrame(rows)
    ranking["absolute_pearson_r"] = ranking.pearson_r.abs()
    ranking["sequence_class"] = "no_match"
    ranking.loc[(ranking.overlap_years >= 10) & (ranking.absolute_pearson_r >= 0.90), "sequence_class"] = "strong_sequence_similarity"
    ranking.loc[(ranking.overlap_years >= 10) & (ranking.absolute_pearson_r >= 0.995) & (ranking.z_max_abs_best_sign <= 0.05), "sequence_class"] = "near_exact_standardized_sequence"
    ranking = ranking.sort_values(["period", "evidence", "absolute_pearson_r"], ascending=[True, True, False])
    ranking.to_csv(OUTPUT / "all_proxy_sequence_comparisons.csv", index=False)
    top = ranking.groupby(["period", "evidence"], as_index=False, group_keys=False).head(10).copy()
    top.to_csv(OUTPUT / "top10_proxy_sequence_matches.csv", index=False)
    summaries = []
    for (period, evidence_name), group in ranking.groupby(["period", "evidence"]):
        best = group.iloc[0]
        summaries.append({"period": period, "evidence": evidence_name, "best_proxy_record_index_1based": int(best.proxy_record_index_1based), "best_proxy_latitude": float(best.proxy_latitude), "best_proxy_longitude": float(best.proxy_longitude), "best_pearson_r": float(best.pearson_r), "best_spearman_r": float(best.spearman_r), "best_overlap_years": int(best.overlap_years), "best_z_rmse": float(best.z_rmse_best_sign), "best_z_max_abs": float(best.z_max_abs_best_sign), "best_sequence_class": best.sequence_class, "strong_match_count": int((group.sequence_class == "strong_sequence_similarity").sum()), "near_exact_match_count": int((group.sequence_class == "near_exact_standardized_sequence").sum())})
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(OUTPUT / "evidence_sequence_overlap_summary.csv", index=False)

    fig, axes = plt.subplots(5, 2, figsize=(14, 14), constrained_layout=True)
    evidence_names = ["MADA_JJA_PDSI", *round102.EVIDENCE.keys()]
    for row, evidence_name in enumerate(evidence_names):
        for col, (period, years) in enumerate(round102.PERIODS.items()):
            ax = axes[row, col]
            best = summary_table[(summary_table.period == period) & (summary_table.evidence == evidence_name)].iloc[0]
            proxy_index = int(best.best_proxy_record_index_1based) - 1
            evidence_z = standardized(curves[(evidence_name, period)])
            proxy_z = standardized(proxy_values[period][proxy_index])
            sign = 1 if best.best_pearson_r >= 0 else -1
            ax.plot(years, evidence_z, color="#111111", marker="o", linewidth=2.2, label=evidence_name)
            ax.plot(years, sign * proxy_z, color="#0072B2", marker="s", linewidth=1.5, label=f"proxy {proxy_index + 1} x {sign:+d}")
            ax.axhline(0, color="#888888", linewidth=0.7)
            ax.set_title(f"{period}: r={best.best_pearson_r:.3f}, {best.best_sequence_class}")
            ax.set_ylabel("z")
    axes[0, 0].legend(frameon=False, fontsize=8); axes[-1, 0].set_xlabel("Year"); axes[-1, 1].set_xlabel("Year")
    fig.suptitle("Hydroclimate evidence and its closest frozen proxy sequence", fontsize=15)
    fig.savefig(OUTPUT / "closest_proxy_sequence_curves.png", dpi=220)
    plt.close(fig)

    early = summary_table[summary_table.period == "early_1901_1920"].to_dict(orient="records")
    summary = {"status": "complete", "periods": {name: [int(years[0]), int(years[-1])] for name, years in round102.PERIODS.items()}, "locked_1921_1939_accessed": False, "proxy_record_count": len(identity), "early_sequence_results": early, "decision": "downgrade_near_exact_or_strong_matches_to_internal_consistency_evidence"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Round 103 Hydroclimate Proxy-Sequence Overlap", "", "Sequence similarity is not by itself proof of source identity, but strong or near-exact matches cannot be treated as independent validation.", ""]
    for row in early:
        lines.append(f"- {row['evidence']}: best proxy {row['best_proxy_record_index_1based']} at ({row['best_proxy_latitude']:.2f}, {row['best_proxy_longitude']:.2f}), r={row['best_pearson_r']:.3f}, n={row['best_overlap_years']}, class={row['best_sequence_class']}.")
    lines.extend(["", "Decision: downgrade any strong/near-exact match to internal consistency evidence; retain unmatched records as external but seasonally mismatched phase evidence."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in sources.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "locked_1921_1939_accessed": False}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]: raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
