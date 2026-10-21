# LIA HMA glacier reconstruction

Private research source snapshot, prepared on 2026-10-21.

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
