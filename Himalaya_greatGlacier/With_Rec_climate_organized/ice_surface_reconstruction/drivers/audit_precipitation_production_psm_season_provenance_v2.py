"""V2 production PSM provenance with the archived glacier-ice exception."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from audit_precipitation_production_psm_season_provenance_v1 import (
    project_root,
    require,
    state,
)


EXPERIMENT_ID = "climate_reconstruction_precipitation_production_psm_season_provenance_v2"


def main() -> int:
    root = project_root()
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "fixed_experiment": phyda / "fix_proxys_experiments.m",
        "three_method_entry": phyda / "daa_three_method_hma.m",
        "proxy_loader": phyda / "load_prxs_v5.m",
        "psm7_implementation": phyda / "psm_uni.m",
        "proxy6_metadata": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver_v1_failed_contract": Path(__file__).with_name("audit_precipitation_production_psm_season_provenance_v1.py"),
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}
    fixed = sources["fixed_experiment"].read_text(encoding="utf-8", errors="replace")
    entry = sources["three_method_entry"].read_text(encoding="utf-8", errors="replace")
    loader = sources["proxy_loader"].read_text(encoding="utf-8", errors="replace")
    psm = sources["psm7_implementation"].read_text(encoding="utf-8", errors="replace")

    evidence = {
        "psm_option": require(fixed, r"psm_ot_list\s*=\s*\[7\]\s*;", "psm-option-7"),
        "calibration_period": require(fixed, r"clb_prd_list\s*=\s*\{\s*\[1940\s*,\s*2000\]\s*\}\s*;", "calibration-1940-2000"),
        "month_start": require(fixed, r"mon_avg_o\s*=\s*1\s*;", "month-start-1"),
        "month_end": require(fixed, r"mon_avg_f\s*=\s*12\s*;", "month-end-12"),
        "entry_month_start": require(entry, r"mon_avg_o\s*=\s*1\s*;", "entry-month-start-1"),
        "entry_month_end": require(entry, r"mon_avg_f\s*=\s*12\s*;", "entry-month-end-12"),
        "psm7_method": require(entry, r"if\s+psm_ot\s*==\s*7\s*[\s\S]{0,120}?method\s*=\s*'bi'\s*;", "psm7-bi-method"),
        "temperature_product": require(loader, r"obs_dtp1\s*=\s*'bearth'\s*;", "berkeley-earth-product"),
        "precipitation_product": require(loader, r"obs_dtp2\s*=\s*'gpcp'\s*;", "gpcp-product-label"),
        "psm7_function": require(loader, r"\[proxy_t\]\s*=\s*psm_uni\(", "psm-uni-call"),
        "temperature_annual_loader": require(psm, r"load_obs_S\(obs_dtp1,\s*mnavg_p_o,\s*mnavg_p_f\)", "temperature-annual-loader"),
        "precipitation_annual_loader": require(psm, r"load_obs_S\(obs_dtp2,\s*mnavg_p_o,\s*mnavg_p_f\)", "precipitation-annual-loader"),
        "precipitation_divisor": require(psm, r"O_p\s*=\s*O_p\s*/\s*30\s*;", "precipitation-divisor-30"),
        "precipitation_reference_period": require(psm, r"ref_period\s*=\s*1961\s*:\s*1990\s*;", "precip-reference-1961-1990"),
        "linear_design": require(psm, r"X\s*=\s*\[ones\(sum\(valid\),\s*1\),\s*o_t\(valid\)',\s*o_p\(valid\)'\]\s*;", "bivariate-design"),
        "linear_regression": require(psm, r"\[b_tmp,\s*~,\s*r_tmp\]\s*=\s*regress\(pd\(valid\),\s*X\)\s*;", "bivariate-regression"),
    }
    seasonal_terms = {
        term: int(len(re.findall(rf"\b{term}\b", psm, flags=re.IGNORECASE)))
        for term in ("jja", "djf", "mamjja", "jjason")
    }
    if any(seasonal_terms.values()):
        raise RuntimeError(f"unexpected-explicit-season-in-production-psm:{seasonal_terms}")

    metadata = pd.read_csv(sources["proxy6_metadata"])
    exceptions = metadata[(metadata["proxy_type"] != "tree") | (metadata["coefficient_count"] != 3)].copy()
    counts = {
        "proxy_count": int(len(metadata)),
        "tree_bivariate_three_coefficient_count": int(((metadata["proxy_type"] == "tree") & (metadata["coefficient_count"] == 3)).sum()),
        "exception_count": int(len(exceptions)),
        "archived_significant_p_le_0_05_count_all_types": int((metadata["psm_pvalue"] <= 0.05).sum()),
    }
    if counts["proxy_count"] != 202 or counts["tree_bivariate_three_coefficient_count"] != 201 or counts["exception_count"] != 1:
        raise RuntimeError(f"archived-proxy-contract-conflict:{counts}")
    exception = exceptions.iloc[0]
    if int(exception["proxy_index"]) != 202 or exception["proxy_type"] != "glacier ice" or int(exception["coefficient_count"]) != 2:
        raise RuntimeError(f"unexpected-proxy-exception:{exception.to_dict()}")

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "production_tree_psm": {
            "option": 7,
            "method": "bivariate linear",
            "calibration_period": [1940, 2000],
            "climate_aggregation_months": [1, 12],
            "climate_aggregation_label": "calendar-year annual mean/aggregate",
            "temperature_product": "Berkeley Earth (bearth)",
            "precipitation_product_source_label": "gpcp",
            "precipitation_conversion": "divide loaded annual field by fixed 30 before 1961-1990 anomaly",
            "record_specific_season_selection": False,
        },
        "archived_proxy_counts": counts,
        "non_tree_exception": {
            "proxy_index": 202,
            "proxy_type": "glacier ice",
            "location": [float(exception["latitude"]), float(exception["longitude"])],
            "coefficient_count": 2,
            "pvalue": float(exception["psm_pvalue"]),
        },
        "explicit_named_season_terms_in_psm_uni": seasonal_terms,
        "source_evidence": evidence,
        "failed_v1_contract_correction": "V1 expected all 202 operators to be three-coefficient tree PSMs and stopped before output; V2 records the one glacier-ice exception.",
        "decision": "open_predeclared_seasonal_tree_psm_rebuild_without_using_1901_1920_for_selection",
        "interpretation": "MADA JJA PDSI is phase evidence, not an annual precipitation amplitude target; the production code nevertheless confirms that proxy-specific seasonality is absent for all 201 tree PSMs.",
    }
    after = {name: state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"evidence": key, "source_expression": value} for key, value in evidence.items()]).to_csv(output / "production_psm_source_evidence.csv", index=False)
    exceptions.to_csv(output / "non_tree_psm_exception.csv", index=False)
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Production PSM Season Provenance V2 Contract\n\nRead source code and archived metadata only. Do not fit a model, inspect 1921-1939, alter production inputs, or run OGGM.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Production Precipitation PSM Season Provenance V2\n\n"
        "The production proxy6 branch contains 201 tree proxies using PSM option 7: bivariate linear proxy-on-temperature-and-precipitation regression calibrated over 1940-2000. Both predictors are aggregated over January-December, and the code has no proxy-specific season selection.\n\n"
        "One additional operator is a glacier-ice record at 28.38 N, 85.72 E with two coefficients; it is outside the seasonal tree-PSM repair scope.\n\n"
        "The next rebuild should compare a small preregistered season set using blocked post-1940 validation only. The 1901-1920 MADA comparison remains an untouched diagnostic rather than a season-selection target.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-provenance-audit")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
