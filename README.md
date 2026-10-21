# LIA HMA glacier reconstruction

Research source snapshot, prepared on 2026-10-21.

## Contents

- `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/`: surface reconstruction, flowline/network solvers, configurations, drivers and tests.
- `Himalaya_greatGlacier/With_Rec_climate_organized/src/`: organized OGGM preparation, inversion and analysis workflows.
- `analysis/` and root Python scripts: analysis, diagnostics and reporting tools.
- `polim_cpu/`, `paleoice_v1/`, `paleoice_v0.4.2/`: research solvers and earlier workflows.
- `LIA_PROJECT_FACTS.md`, `LIA_EXECUTION_CONTRACT.md`, `LIA_CURRENT_STATE.json`: frozen scientific contract and execution context.

This is a source snapshot, not a standalone reproducibility package. Original scientific datasets, DEMs, inventories, climate products, run outputs, checkpoints, environments, external result directories and nested Git history are excluded. Configuration files retain local paths that require configuration on another machine. Notebook outputs were cleared in this export only.

## Scientific status

Read the project contract before running LIA analysis or simulations. This export does not authorize or start model runs. The copied state describes the original local project, not running processes in a fresh checkout.

The legacy 191-system Himalayan static reconstruction has a known comparability problem: modern effective stress was calibrated on one-dimensional inversion thickness, while LIA volume was integrated from a two-dimensional zero-boundary harmonic extension. Three modern-reference interpolation controls showed volume biases of -28.04% to -36.40% despite exact thickness at sampled constraints. These controls are not a complete legacy workflow replay, do not quantify the bias of all 191 systems, and do not provide repaired LIA volumes. Apparent regional thickening from that comparison is pending reconstruction repair and cannot be treated as a verified physical change.

## Dependencies and checks

Requirements differ between model branches; consult their requirements files. Production simulations require external inputs and the execution-state validator. No simulation or model test suite was run for this export.

`EXPORT_MANIFEST.json` records per-file hashes and notebook transformations. `UPLOAD_PREPARATION.json` records preparation and credential-pattern scan status. No new license is assigned by this snapshot.

## Extension operator research follow-up (2026-10-21)

The local research branch adds candidate harmonic-surface and Poisson-thickness extensions. Nine pytest cases pass and seven supplied synthetic cases reproduce. Three modern-reference controls improve aggregate volume consistency with operator A, but surface interpolation error and negative-thickness clipping remain. Boundary propagation accounts for a substantial part of the legacy deficit. Candidates are not connected to production. See `docs/extension_operator_controls_20261021/report.html` and the accompanying CSV/JSON files. These modern controls are not independent validation of reconstructed LIA thickness.

The follow-up residual-analysis mail patch was integrated without rerunning its new variants. `export_fullchain_inputs_npz_20261021.py` exports native flowline topology, both grid frames and unified-bed samples for offline modern replay; schema and review notes are in `docs/extension_operator_controls_20261021/fullchain_input_export_v1/`. Native main-flowline endpoints retain about 15-18 m of reference thickness, whereas the existing network solver starts unseen outlets at zero thickness; endpoint conventions must be disclosed in a replay. No new tau/PP/LIA/dynamic solve or production promotion has been performed.

The subsequent supplied full-chain modern replay is integrated in `replay_modern_fullchain_20261021.py` and `docs/extension_operator_controls_20261021/fullchain_replay_v1/`. Eight network tests passed locally; the supplied replay itself was not rerun locally. The optional outlet surface preserves the network solver's default behavior. Local review and calibration-target comparison are in `docs/extension_operator_controls_20261021/fullchain_review_v1/`, registered as project fact1499. Modern-volume calibration is the primary experimental target for a proposed next pilot, with profile fitting retained as structural sensitivity. Curve-interpolated volume roots are not completed volume fits, and modern closure is calibration rather than independent validation. No production route is switched or new glacier-model batch launched by this integration.

Pilot preparation (fact1500) adds supplied volume-root calibration and stratified-pilot tools. The unmodified selector actually yields22 systems (West9/Central9/East4), not27: East has only4 single-member systems. Multi-member systems are156/191 (81.7%) and remain outside this exporter. Local preparation exported22 NumPy inputs with0failures, without running the pilot or a new glacier solve. The runner now counts missing exports in the full selected denominator; gate thresholds are unchanged. Coverage, input manifests and handoff notes are in `docs/extension_operator_controls_20261021/pilot_input_export_v1/`. These singleton inputs do not establish applicability to all191 systems.

The modern pilot is now complete (facts1501/1502): 19 new local float64 calculations plus3reused supplied controls, with12/22passing (54.545%). Five systems lack a volume root within the fixed bounds, four fail the profile-error ratio gate, and one fails the surface-error increase gate. The80%progression rule is not met: the proposed191rerun is stopped. Results and the checkpointed controller are in `docs/extension_operator_controls_20261021/pilot_execution_v1/`. No LIA volume-change result or production promotion follows from this calibration pilot.

The human has explicitly closed this static extension/volume-tau branch (fact1503). Do not repair A, extend multi-member exports, enlarge the pilot or promote A to production. Affected old191 static volume-change conclusions remain withdrawn. Closure policy and registration are in `docs/extension_operator_controls_20261021/branch_closure_v1/`. The requested next handoff reads an existing accepted-mainline dynamic snapshot; it does not launch or calibrate a new model. The original local dynamic sample is not uploaded into this source repository.

The human subsequently authorized a separate thin-plate modern checkpoint only (facts1505/1506). All22 systems solved locally;21/22 (95.45%) pass the fixed0.7–1.3 volume-ratio band, exceeding80%. System3053 fails at1.475902337. The operator is a simplified squared-Laplacian surrogate, not an exact ANUDEM implementation. Protocol, per-system results and limitations are in [static checkpoint1](docs/extension_operator_controls_20261021/static_samemethod_ck1_v1/README.md). The prior A route stays closed; no LIA run,191promotion or independent observational validation follows from this modern model-reference comparison.
