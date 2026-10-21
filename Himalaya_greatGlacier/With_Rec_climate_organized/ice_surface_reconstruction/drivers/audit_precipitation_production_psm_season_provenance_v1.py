"""Freeze the production precipitation PSM season/calibration provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "climate_reconstruction_precipitation_production_psm_season_provenance_v1"


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


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def require(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"source-contract-not-found:{label}")
    return match.group(0).strip()


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
        for term in ("jja", "djf", "mamjja", "jjason", "season")
    }
    if any(seasonal_terms[term] for term in ("jja", "djf", "mamjja", "jjason")):
        raise RuntimeError(f"unexpected-explicit-season-in-production-psm:{seasonal_terms}")

    metadata = pd.read_csv(sources["proxy6_metadata"])
    counts = {
        "proxy_count": int(len(metadata)),
        "tree_proxy_count": int(metadata["proxy_type"].eq("tree").sum()),
        "bivariate_coefficient_count_3": int(metadata["coefficient_count"].eq(3).sum()),
        "archived_significant_p_le_0_05_count": int((metadata["psm_pvalue"] <= 0.05).sum()),
    }
    if counts["proxy_count"] != 202 or counts["bivariate_coefficient_count_3"] != 202:
        raise RuntimeError(f"archived-proxy-contract-conflict:{counts}")

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "production_psm": {
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
        "explicit_season_terms_in_psm_uni": seasonal_terms,
        "source_evidence": evidence,
        "decision": "open_predeclared_seasonal_psm_rebuild_without_using_1901_1920_for_selection",
        "interpretation": "The MADA JJA-PDSI conflict is not an exact target-to-target validation of the annual production PSM, but the production code confirms that documented or proxy-specific seasonality is not represented.",
    }
    after = {name: state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"evidence": key, "source_expression": value} for key, value in evidence.items()]).to_csv(output / "production_psm_source_evidence.csv", index=False)
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Production PSM Season Provenance Contract\n\n"
        "Read source code and archived metadata only. Do not fit a model, inspect 1921-1939, alter production inputs, or run OGGM.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Production Precipitation PSM Season Provenance\n\n"
        "The production proxy6 precipitation branch uses PSM option 7: a bivariate linear proxy-on-temperature-and-precipitation regression calibrated over 1940-2000. Both climate predictors are aggregated from January through December. The code contains no record-specific season selection.\n\n"
        "All 202 archived proxy6 operators have three coefficients. The precipitation predictor is loaded under `gpcp`, divided by a fixed 30, and expressed relative to 1961-1990 before fitting.\n\n"
        "This establishes a real repair target: test predeclared seasonal PSMs with blocked post-1940 validation, without using 1901-1920 to choose the season. MADA remains phase evidence rather than an amplitude calibration target.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-provenance-audit")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
