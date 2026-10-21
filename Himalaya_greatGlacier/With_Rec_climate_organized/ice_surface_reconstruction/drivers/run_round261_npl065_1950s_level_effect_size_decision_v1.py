from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NEAR_BASELINE_BAND_DEGC = 0.15
MIN_PROXY_COOL_GAP_DEGC = 0.30


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_261_npl065_1950s_level_effect_size_decision_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "round260_summary": loop / "round_260_npl065_1950s_temperature_product_level_audit_v1/summary.json",
        "round258_summary": loop / "round_258_npl065_precalibration_holdout_v2/summary.json",
        "round259_summary": loop / "round_259_npl065_cru_early_station_count_audit_v1/summary.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    source = json.loads(paths["round260_summary"].read_text(encoding="ascii"))
    shifts = {row["series"]: row["holdout_1952_1959_mean_anomaly_degC"] for row in source["series_metrics"]}
    product_names = ("CRU_TS401_Tmin", "CRU_TS401_Tmean", "Berkeley_Tmean")
    proxy_names = ("signal_free_prediction", "modnegex_prediction")
    product_values = np.asarray([shifts[name] for name in product_names], dtype=float)
    product_min = float(product_values.min())
    product_max = float(product_values.max())
    product_median = float(np.median(product_values))
    product_near_baseline = bool(np.max(np.abs(product_values)) <= NEAR_BASELINE_BAND_DEGC)
    proxy_rows = []
    for name in proxy_names:
        value = float(shifts[name])
        gap_below_nearest_product = product_min - value
        proxy_rows.append({
            "series": name,
            "holdout_anomaly_degC": value,
            "gap_below_product_median_degC": product_median - value,
            "gap_below_nearest_product_degC": gap_below_nearest_product,
            "materially_colder_than_every_product": gap_below_nearest_product >= MIN_PROXY_COOL_GAP_DEGC,
        })
    materially_colder = all(row["materially_colder_than_every_product"] for row in proxy_rows)
    if product_near_baseline and materially_colder:
        decision = "instrumental_products_place_1950s_near_baseline_proxy_chronologies_overcool_low_frequency_level"
    elif materially_colder:
        decision = "proxy_chronologies_colder_than_products_but_product_baseline_not_equivalent"
    else:
        decision = "no_robust_proxy_overcooling_effect"
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "frozen_thresholds": {
            "near_baseline_absolute_band_degC": NEAR_BASELINE_BAND_DEGC,
            "minimum_proxy_gap_below_every_product_degC": MIN_PROXY_COOL_GAP_DEGC,
        },
        "instrumental_product_shifts_degC": {name: shifts[name] for name in product_names},
        "product_shift_min_median_max_degC": [product_min, product_median, product_max],
        "all_products_within_near_baseline_band": product_near_baseline,
        "proxy_effects": proxy_rows,
        "both_proxies_materially_colder_than_every_product": materially_colder,
        "decision": decision,
        "interpretation": "signal_free_recovers_modern_high_frequency_skill_but_not_low_frequency_level_transport",
        "locked_1921_1939_values_accessed": False,
        "paleo_proxy_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "rcsigfree_binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    lines = [
        "# Round 261 NPL065 1950s level effect-size decision",
        "",
        f"- Product min/median/max anomaly: `{product_min:+.3f}/{product_median:+.3f}/{product_max:+.3f} degC`.",
    ]
    for row in proxy_rows:
        lines.append(f"- {row['series']}: `{row['holdout_anomaly_degC']:+.3f} degC`; gap below nearest product `{row['gap_below_nearest_product_degC']:.3f} degC`.")
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
