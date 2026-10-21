from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2"
)
OUT = BASE / "round_201_east_asia_temperature_field_admission_v1"
SOURCES = {
    "round197": BASE / "round_197_east_asia_temperature_metadata_v2/summary.json",
    "round198": BASE / "round_198_east_asia_remote_data_inventory_v1/summary.json",
    "round199": BASE / "round_199_east_asia_proxy_identity_metadata_v3/summary.json",
    "round200": BASE / "round_200_east_asia_xlsread_metadata_closure_v4/summary.json",
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)
    evidence = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCES.items()}

    r197, r198, r199, r200 = (evidence[key] for key in ("round197", "round198", "round199", "round200"))
    model_count = len(r198["model_member_names"])
    strong_fraction = r199["external_with_strong_metadata_match"] / r199["external_proxy_count"]
    hma_long_strong_fraction = r199["hma_long_with_strong_project_match"] / r199["hma_long_candidate_count"]

    gates = {
        "hma_and_1500_1899_domain": bool(r197["hma_temperature_gate"]),
        "distinct_method_family": True,
        "independent_absolute_scale": False,
        "independent_proxy_evidence": False,
        "deposited_code_reproducible": False,
        "safe_to_use_as_independent_warm_bias_target": False,
    }
    summary = {
        "round": 201,
        "record": "10.5281/zenodo.18477496",
        "declared_domain": "East Asia 15-55N, 70-150E, 1400-2000, annual, 1 degree",
        "method": "proxy-conditioned improved analog selection from 37 CMIP6/PMIP4 model files",
        "absolute_unit_path": "selected standardized model field multiplied by CRU 1902-1977 cellwise standard deviation and shifted by CRU cellwise mean",
        "model_file_count": model_count,
        "external_proxy_count": r199["external_proxy_count"],
        "strong_project_proxy_match_count": r199["external_with_strong_metadata_match"],
        "strong_project_proxy_match_fraction": strong_fraction,
        "hma_long_proxy_count": r199["hma_long_candidate_count"],
        "hma_long_strong_project_match_count": r199["hma_long_with_strong_project_match"],
        "hma_long_strong_project_match_fraction": hma_long_strong_fraction,
        "numeric_metadata_max_abs_error": r200["numeric_max_abs_error"],
        "metadata_id_match_count": r200["id_match_count"],
        "metadata_type_match_count_raw": r200["proxy_type_match_count"],
        "metadata_type_match_count_after_trim": 423,
        "deposited_proxy_year_expression_count": r200["proxy_year_expression_count"],
        "deposited_proxy_year_expression_first": r200["proxy_year_expression_first"],
        "deposited_proxy_year_expression_last": r200["proxy_year_expression_last"],
        "deposited_loop_requested_rows": 2019,
        "code_conflict": "num(15:end,1) yields years 2-2019 and 2018 rows, while Step1 loops over 2019 rows",
        "gates": gates,
        "results_zip_downloaded": False,
        "data_zip_fully_downloaded": False,
        "climate_or_proxy_values_read": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": "close_as_independent_absolute_temperature_validation_field",
        "permitted_future_role": "comparison-only after corrected-code and source-lineage clarification; not current mainline",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in SOURCES.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Round 201 East Asia field admission decision

- The field passes the domain gate: East Asia, annual, 1 degree, 1400-2000.
- Its method is distinct: an improved analog selects fields from `{model_count}` CMIP6/PMIP4 model files using a 423-record proxy ensemble.
- It does **not** provide an independent absolute scale. Physical degrees Celsius are restored directly with CRU 1902-1977 cellwise means and standard deviations.
- Proxy independence also fails: `{r199['external_with_strong_metadata_match']}/{r199['external_proxy_count']}` ({100 * strong_fraction:.1f}%) of all proxies and `{r199['hma_long_with_strong_project_match']}/{r199['hma_long_candidate_count']}` ({100 * hma_long_strong_fraction:.1f}%) of HMA long proxies strongly match the current project ledger.
- MATLAB R2024b closes numeric metadata exactly and all 423 IDs; the two raw type mismatches are trailing spaces only. However, the deposited expression `num(15:end,1)` yields 2018 rows spanning 2-2019, while the next loop requests 2019 rows. The public code/input pair therefore does not reproduce as deposited.
- `Results.zip` was not downloaded. No annual climate/proxy value, locked-period value, or Jionglacuo value was accessed.

Decision: `close_as_independent_absolute_temperature_validation_field`.

The product may be reconsidered only as comparison-only evidence after corrected code and source-lineage clarification. It cannot currently adjudicate or calibrate the HMA 1500-1899 warm-bias magnitude.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
