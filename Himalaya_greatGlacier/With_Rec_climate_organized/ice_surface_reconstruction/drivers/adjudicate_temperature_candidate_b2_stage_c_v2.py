from pathlib import Path

import adjudicate_temperature_candidate_b2_stage_c_v1 as impl


ROOT = Path(__file__).resolve().parents[3]
impl.ROOT = ROOT
impl.LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
impl.EVALUATION = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v3"
impl.REAL_RUN = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3"
impl.OUTPUT = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_adjudication_v2"
impl.CURVES = impl.EVALUATION / "candidate_b2_long_term_hma_curves_allowed.csv"
impl.AGGREGATE = impl.EVALUATION / "instrumental_aggregate_metrics.csv"
impl.REGIONAL = impl.EVALUATION / "instrumental_west_central_east_metrics.csv"
impl.EVAL_SUMMARY = impl.EVALUATION / "summary.json"
impl.REAL_SUMMARY = impl.REAL_RUN / "summary.json"


if __name__ == "__main__":
    impl.main()
