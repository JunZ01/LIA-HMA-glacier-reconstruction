"""Final interpretation audit for the completed Round 134 V2 reconstruction."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "round_134_netp_mjj_low_frequency_reconstruction_v2"
    output = loop / "round_134_netp_mjj_low_frequency_reconstruction_v3_final_audit"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round134_netp_mjj_low_frequency_reconstruction_v3_final_audit_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "v2_summary": source / "summary.json",
        "v2_curves": source / "netp_mjj_reconstruction_curves.csv",
        "v2_validation": source / "qt_prec_external_validation_metrics.csv",
        "v2_oof": source / "blocked_oof_summary.csv",
        "v2_fold": source / "blocked_oof_fold_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    v2 = json.loads(paths["v2_summary"].read_text(encoding="utf-8"))
    if v2["status"] != "complete" or v2["locked_1921_1939_accessed"] or not v2["source_fingerprints_unchanged"]:
        raise RuntimeError("v2-contract-failed")
    curves = pd.read_csv(paths["v2_curves"])
    validation = pd.read_csv(paths["v2_validation"])
    oof = pd.read_csv(paths["v2_oof"]).set_index("product")
    folds = pd.read_csv(paths["v2_fold"])

    requested_raw = curves.year.between(1500, 1920)
    raw_finite = requested_raw & curves.marginalized_mjj_mm_day.notna()
    corrected_minimum_active_groups = int(curves.loc[raw_finite, "active_group_count"].min())
    corrected_raw_years = [int(curves.loc[raw_finite, "year"].min()), int(curves.loc[raw_finite, "year"].max())]
    effective_rows = []
    for period in ("1500_1920", "1800_1920"):
        start, end = map(int, period.split("_"))
        use = curves.year.between(start, end) & curves.marginalized_31yr.notna()
        effective_rows.append({
            "requested_period": period,
            "effective_first_year": int(curves.loc[use, "year"].min()),
            "effective_last_year": int(curves.loc[use, "year"].max()),
            "effective_year_count": int(use.sum()),
        })
    effective = pd.DataFrame(effective_rows)
    validation = validation.merge(effective, left_on="period", right_on="requested_period", validate="many_to_one").drop(columns="requested_period")
    marginal = validation[validation.branch == "marginalized"].set_index("period")

    gates = dict(v2["gates"])
    gates["minimum_five_active_groups_on_full_window_support"] = corrected_minimum_active_groups >= 5
    gates.pop("minimum_five_active_groups_1500_1920", None)
    passed = bool(all(gates.values()))
    decision = "advance_to_spatial_netp_mjj_reconstruction" if passed else "close_equal_weight_netp_temporal_composite"
    if decision != "close_equal_weight_netp_temporal_composite":
        raise RuntimeError("unexpected-v3-advance")

    validation.to_csv(output / "qt_prec_external_validation_metrics_effective_periods.csv", index=False)
    effective.to_csv(output / "effective_validation_periods.csv", index=False)
    pd.DataFrame([{"corrected_raw_first_year": corrected_raw_years[0], "corrected_raw_last_year": corrected_raw_years[1], "minimum_active_group_count": corrected_minimum_active_groups}]).to_csv(output / "corrected_coverage_gate.csv", index=False)
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_134_netp_mjj_low_frequency_reconstruction_v3_final_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model_rerun": False,
        "v2_values_changed": False,
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "corrected_raw_support_years": corrected_raw_years,
        "corrected_minimum_active_group_count": corrected_minimum_active_groups,
        "effective_validation_periods": effective_rows,
        "oof": oof.reset_index().to_dict(orient="records"),
        "all_fold_correlations_positive": bool((folds.test_correlation > 0).all()),
        "marginalized_qt_metrics": marginal.reset_index().to_dict(orient="records"),
        "gates": gates,
        "passes_all_gates": passed,
        "decision": decision,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 134 NETP MJJ Reconstruction Final Audit\n\n"
        f"- Corrected raw support: `{corrected_raw_years[0]}-{corrected_raw_years[1]}` with minimum `{corrected_minimum_active_groups}` active groups.\n"
        "- Effective 31-year comparison periods: `1515-1900` and `1800-1900`.\n"
        f"- GPCC/CRU blocked-OOF correlations: `{oof.loc['GPCC','oof_correlation']:.3f}` / `{oof.loc['CRU','oof_correlation']:.3f}`.\n"
        f"- Marginalized QT-Prec correlations: `{marginal.loc['1500_1920','qt_correlation']:.3f}` / `{marginal.loc['1800_1920','qt_correlation']:.3f}`.\n"
        f"- Correlation changes versus Round 133 baseline: `{marginal.loc['1500_1920','correlation_delta_vs_round133']:+.3f}` / `{marginal.loc['1800_1920','correlation_delta_vs_round133']:+.3f}`.\n"
        f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
