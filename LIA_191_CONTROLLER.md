# HMA 191-system controller

`LIA_PROJECT_FACTS.md` is the scientific contract. This document only defines
how the 191-system search is scheduled, resumed, and recorded.

## Controller chain

```text
canonical facts
  -> same-definition LIA/RGI reference build
  -> 191-system input and geometry preflight
  -> explicit parameter block manifest
  -> isolated OGGM single-chain subprocesses (maximum 12)
  -> separation-normalized curve scoring
  -> accepted / unresolved_target_curve / technical_failure
```

The controller delegates every real evolution run to `lia_lhs_single_chain.py`.
It does not contain a second OGGM physics implementation.

## Safety state on 2026-08-09

- The mandatory single-chain preflight passed after moving temporary work to `w191/<fingerprint>`. The initial block completed 1860/1860 chains with 12 workers and zero technical failures; PID and logs are recorded in `batch_process.json` and `batch_controller.*.log`.
- The final endpoint tolerance normalized by the LIA-RGI separation is still
  pending an explicit decision, so the controller cannot silently accept new
  systems.
- The former `climate_pixel_registry_330.csv` was a 12-system Stage-3 pilot
  registry and must not be used as a Core-191 availability table. The controller
  now builds `core191_climate_pixels_ensemble_rec_ref.csv` from all 191 main
  RGI geometries. All 191 rows pass the complete-1000-2000 finite-data and
  100 km distance gates; the maximum selected-pixel distance is 36.271 km.
- The corrected dry-run has 189 input-ready systems, two dynamic-review systems,
  and 186 systems eligible for the initial 10-point delta-T block. It creates
  1860 planned tasks. The former 90-task/177-input-blocked dry-run is obsolete.
- Systems 2387 and 5508 remain blocked for explicit dynamic review because the
  five-year zero-MB gate failed.
- Existing B-window decisions for 1044, 1545, and 3584 are imported and are not
  rerun by the initial block.

## Files written by a dry run

- `core191_length_references.csv`: LIA target length and unique modern root
  centerline spatial length in the same declared definition.
- `core191_length_reference_audit.csv`: root selection and reference failures.
- `core191_climate_pixels_ensemble_rec_ref.csv`: one nearest all-time-valid
  reconstructed-climate pixel for each Core-191 main RGI geometry.
- `preflight_manifest.csv`: all input and geometry gates for every system.
- `task_manifest.csv` and `.json`: deduplicated parameter combinations.
- `lia_191_master_status.csv`: one current state per LIA system.
- `state/system_<id>.json`: resumable per-system state.
- `dry_run_summary.json`: population counts and execution lock state.
- Temporary cloned glacier directories use the short path `w191/<fingerprint>` to remain below the Windows path limit.

## Commands

Run from the project root with the verified OGGM environment:

```powershell
D:\Anaconda\envs\oggm_env2\python.exe -m Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_controller --config Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_191_controller_v1.json prepare --rebuild-references --rebuild-climate-pixels
```

Read current state:

```powershell
D:\Anaconda\envs\oggm_env2\python.exe -m Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_controller --config Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_191_controller_v1.json status
```

The Core-191 climate-pixel registry is resolved and the required one-chain
preflight is numerically clean. The initial block may run with at most 12
workers. Acceptance remains `pending-tolerance`, so runs produce diagnostics
but cannot be automatically accepted.

## Iteration rule

Only the first declared block is planned. The next physical parameter block is
not selected automatically. After a block finishes, the system becomes
`needs_parameter_block_review`; response evidence then determines whether the
next approved block is `melt_f-delta_t`, `prcp_fac`, `temp_melt`, or `glen_a`.
Exhausting explicitly declared blocks without a joint curve solution produces
`unresolved_target_curve`, with the best real curve and parameters preserved.
