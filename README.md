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
