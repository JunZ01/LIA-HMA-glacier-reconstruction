from pathlib import Path

import decompose_temperature_candidate_b2_stage_c_increment_v1 as impl


ROOT = Path(__file__).resolve().parents[4]
impl.ROOT = ROOT
impl.LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
impl.STATIC = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
impl.REAL = impl.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3/candidate_b2_stage_c_real_proxy_output_v1.npz"
impl.OUTPUT = impl.LOOP / "temp_b2_stage_c_increment_decomp_v2"


if __name__ == "__main__":
    impl.main()
