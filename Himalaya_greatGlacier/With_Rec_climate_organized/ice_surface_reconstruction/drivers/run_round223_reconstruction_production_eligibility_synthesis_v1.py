from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_223_reconstruction_production_eligibility_synthesis_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
ROUND134_DRIVER = DRIVERS / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py"
METADATA = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {
        138: LOOP / "round_138_netp_structural_sensitivity_envelope_v1/summary.json",
        217: LOOP / "round_217_temperature_bias_phase_three_record_synthesis_v1/summary.json",
        218: LOOP / "round_218_netp_mjj_prior_covariance_spatial_oof_v2/summary.json",
        219: LOOP / "round_219_ccsm4_netp_mjj_prior_preflight_v1/summary.json",
        220: LOOP / "round_220_netp_mjj_cross_prior_spatial_oof_v1/summary.json",
        221: LOOP / "round_221_regen_netp_mjj_data_recovery_v1/summary.json",
        222: LOOP / "round_222_regen_netp_mjj_spatial_adjudication_v1/summary.json",
    }
    paths = {"facts": ROOT / "LIA_PROJECT_FACTS.md", "contract": CONTRACT, "driver": Path(__file__).resolve(), "round134_driver": ROUND134_DRIVER, "metadata": METADATA}
    paths.update({f"round_{number}_summary": path for number, path in summaries.items()})
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}
    data = {number: json.loads(path.read_text(encoding="utf-8")) for number, path in summaries.items()}

    expected = {
        217: "uniform_historical_warm_bias_rejected_early20_conflict_is_season_dependent",
        219: "admit_ccsm4_as_independent_netp_mjj_prior_covariance",
        220: "close_two_prior_netp_mjj_spatialization_retain_temporal_curve",
        221: "admit_regen_as_source_correlated_independent_gridding_diagnostic",
        222: "rank1_spatialization_failure_supported_by_two_of_three_gridding_methods",
    }
    for number, decision in expected.items():
        if data[number]["decision"] != decision:
            raise RuntimeError(f"Round {number} decision mismatch")

    round134 = load_module("round134_for_round223", ROUND134_DRIVER)
    metadata = pd.read_csv(METADATA)
    keys, groups = round134.groups_from_metadata(metadata)
    group_frame = pd.DataFrame(
        [{"group_index": i, "latitude": lat, "longitude": lon, "record_count": len(group)} for i, ((lat, lon), group) in enumerate(zip(keys, groups))]
    )
    group_frame.to_csv(OUT / "netp_training_group_spatial_support.csv", index=False)
    east_count = int((group_frame["longitude"] >= 100.0).sum())
    support = {
        "training_group_count": int(len(group_frame)),
        "training_longitude_range": [float(group_frame.longitude.min()), float(group_frame.longitude.max())],
        "training_latitude_range": [float(group_frame.latitude.min()), float(group_frame.latitude.max())],
        "groups_at_or_east_of_100E": east_count,
        "target_longitude_range": [93.0, 107.0],
        "target_latitude_range": [35.0, 41.0],
        "second_spatial_mode_support_adequate": False,
    }

    rows = [
        {"component": "uniform_temperature_translation", "status": "rejected", "production_eligible": False, "evidence": "Round 217: all three independent records reject uniform historical cooling"},
        {"component": "common_1901_1920_temperature_phase_correction", "status": "unresolved_season_dependent", "production_eligible": False, "evidence": "Round 217: seasonal records cool while independent annual Haizi and project warm"},
        {"component": "netp_mjj_regional_temporal_curve_and_envelope", "status": "retain_as_regional_research_product", "production_eligible": False, "evidence": "Round 138: stable temporal curve; not a gridded HMA state"},
        {"component": "netp_mjj_rank1_gridded_field", "status": "rejected", "production_eligible": False, "evidence": "Rounds 220/222: two priors and REGEN confirm spatial-pattern failure"},
        {"component": "netp_mjj_two_mode_gridded_field", "status": "not_admissible_with_current_proxy_geometry", "production_eligible": False, "evidence": f"15 groups span 97-100.25E; only {east_count} groups are at/east of 100E for a 93-107E target"},
        {"component": "production_climate_reconstruction_rerun", "status": "not_authorized_by_current_evidence", "production_eligible": False, "evidence": "No validated temperature correction or spatial precipitation replacement exists"},
        {"component": "oggm_rerun_from_climate_improvement_branch", "status": "not_authorized", "production_eligible": False, "evidence": "No production climate replacement has passed"},
    ]
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT / "production_eligibility_matrix.csv", index=False)
    decision = "retain_current_production_climate_and_regional_netp_temporal_research_product_only"
    next_requirement = "new_independent_spatially_distributed_central_eastern_hma_evidence_before_any_gridded_replacement"
    summary = {
        "round": 223,
        "decision": decision,
        "next_requirement": next_requirement,
        "netp_proxy_spatial_support": support,
        "production_reconstruction_rerun_authorized": False,
        "oggm_rerun_authorized": False,
        "source_states_unchanged": True,
        "locked_1901_1939_values_read": False,
        "paleo_proxy_values_read": False,
        "external_validation_reserve_opened": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("A source changed")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 223 Production Eligibility\n\n"
        f"`{decision}`\n\n"
        "The current evidence does not authorize a production reconstruction or OGGM rerun. "
        "Keep the existing production temperature and precipitation fields unchanged. Retain the NETP MJJ temporal curve and structural envelope only as a regional research product.\n\n"
        f"The next admissible requirement is `{next_requirement}`. The 15 current training groups occupy only "
        f"{support['training_longitude_range'][0]:.2f}-{support['training_longitude_range'][1]:.2f} E within a 93-107 E target, "
        f"so they cannot independently identify the missing spatial degree of freedom.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
