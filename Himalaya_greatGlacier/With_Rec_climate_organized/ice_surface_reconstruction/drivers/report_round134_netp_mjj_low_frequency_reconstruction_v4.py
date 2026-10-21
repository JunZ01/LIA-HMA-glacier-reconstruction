"""Final date-label correction and report for Round 134."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": digest}


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    v2 = loop / "round_134_netp_mjj_low_frequency_reconstruction_v2"
    v3 = loop / "round_134_netp_mjj_low_frequency_reconstruction_v3_final_audit"
    output = loop / "round_134_netp_mjj_low_frequency_reconstruction_v4_final_report"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round134_netp_mjj_low_frequency_reconstruction_v4_final_report_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "v2_validation": v2 / "qt_prec_external_validation_metrics.csv",
        "v2_oof": v2 / "blocked_oof_summary.csv",
        "v2_summary": v2 / "summary.json",
        "v3_summary": v3 / "summary.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    v2_summary = json.loads(paths["v2_summary"].read_text(encoding="utf-8"))
    v3_summary = json.loads(paths["v3_summary"].read_text(encoding="utf-8"))
    validation = pd.read_csv(paths["v2_validation"])
    oof = pd.read_csv(paths["v2_oof"]).set_index("product")
    effective = {
        "1500_1920": {"effective_first_year": 1515, "effective_last_year": 1900, "effective_sample_count": 386},
        "1800_1920": {"effective_first_year": 1800, "effective_last_year": 1900, "effective_sample_count": 101},
    }
    for period, item in effective.items():
        expected = item["effective_last_year"] - item["effective_first_year"] + 1
        observed = set(validation.loc[validation.period == period, "sample_count"].astype(int))
        if expected != item["effective_sample_count"] or observed != {expected}:
            raise RuntimeError(f"effective-period-count-failed:{period}:{expected}:{observed}")
    if v2_summary["decision"] != "close_equal_weight_netp_temporal_composite" or v3_summary["decision"] != v2_summary["decision"]:
        raise RuntimeError("decision-closure-failed")
    marginal = validation[validation.branch == "marginalized"].set_index("period")
    summary = {
        "experiment_id": "round_134_netp_mjj_low_frequency_reconstruction_v4_final_report",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model_rerun": False,
        "metric_values_changed": False,
        "date_label_corrected": True,
        "locked_1921_1939_accessed": False,
        "effective_validation_periods": effective,
        "oof": oof.reset_index().to_dict(orient="records"),
        "marginalized_qt_metrics": marginal.reset_index().to_dict(orient="records"),
        "gates": v3_summary["gates"],
        "passes_all_gates": False,
        "decision": "close_equal_weight_netp_temporal_composite",
    }
    after = {name: fingerprint(path) for name, path in paths.items()}
    summary["source_fingerprints_unchanged"] = before == after
    if not summary["source_fingerprints_unchanged"]:
        raise RuntimeError("source-fingerprint-changed")
    pd.DataFrame([{"requested_period": period, **item} for period, item in effective.items()]).to_csv(output / "effective_validation_periods.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": True}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 134 NETP MJJ Reconstruction Final Report\n\n"
        "- Effective QT-Prec comparison periods: `1515-1900` (386 years) and `1800-1900` (101 years).\n"
        f"- GPCC/CRU blocked-OOF correlations: `{oof.loc['GPCC','oof_correlation']:.3f}` / `{oof.loc['CRU','oof_correlation']:.3f}`.\n"
        f"- Marginalized QT-Prec correlations: `{marginal.loc['1500_1920','qt_correlation']:.3f}` / `{marginal.loc['1800_1920','qt_correlation']:.3f}`.\n"
        f"- Changes versus Round 133 baseline: `{marginal.loc['1500_1920','correlation_delta_vs_round133']:+.3f}` / `{marginal.loc['1800_1920','correlation_delta_vs_round133']:+.3f}`.\n"
        "- Decision: `close_equal_weight_netp_temporal_composite`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
