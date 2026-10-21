# Modern static checkpoint 1, 2026-10-21

Human-authorized experiment; no LIA or dynamic reconstruction.

- Supplied thin-plate patch: original commit 7d3a7e5; local d184a92.
- Selection accounting uses all 22 frozen systems, including missing/error cases. Each result is saved immediately. Numerical method and gates unchanged.
- The squared-Laplacian thickness operator is a simplified surrogate, not an exact ANUDEM/Topo to Raster implementation; Zhang 2024 full methods remain unverified.
- All 10 extension tests passed; all 22 systems locally solved in 48.105 seconds using oggm_env2 and original NPZ inputs.
- Fixed rule: inclusive ratio 0.7-1.3, at least 80% of all 22 systems. Passed 21/22 (95.45%); 3053 failed with ratio 1.475902337.
- Harmonic median ratio 0.611153276; thin-plate median 0.857096625; thin-plate total-volume ratio 0.874728578.
- The supplied method clips raw negative thickness overshoots. Per-system fractions retained; maximum 7.735006%.
- Reference volume is a model product; this checkpoint is not independent observed-volume validation or proof of LIA transferability.
- Facts 1505 (authorization) and 1506 (completion). Prior A/volume-tau route stays closed. No step 2/191 launch or acceptance change. Multi-member gap 156/191 remains.

This export contains small metadata and results. NPZ inputs and original source data remain local. Canonical controller state belongs to the local project; do not launch from this export.
