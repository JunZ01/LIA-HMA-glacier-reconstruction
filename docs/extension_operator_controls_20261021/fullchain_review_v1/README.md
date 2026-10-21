# Local review of the supplied modern replay

The supplied f85b576 mail patch was applied to the research branch. Its full-chain
results were reviewed, not independently rerun locally. Eight network unit tests
passed locally. The archived tau, RMSE and point counts in the supplied harness
table match the frozen NPZ metadata exactly for all three controls. This metadata
check does not independently reproduce the optimization. The script's stop guard
checks a 0.05 kPa tau tolerance; it does not enforce bitwise equality of every metric.

## Calibration choice for the next research pilot

Use modern system volume as the primary calibration target because the intended
estimand is LIA-to-modern volume change. Keep profile-thickness fitting as a
separate structural sensitivity and diagnostic. This is an experimental choice,
not promotion of operator A or replacement of the 191-glacier production route.
Use the same bed, footprint, surface solver and discretization on both calibration
branches; do not repair a volume mismatch afterwards with a scaling factor.

Volume closure is obtained by calibration and therefore is not an independent
validation result. Both targets derive from the same redistributed OGGM reference.
They are dependent model-based constraints. Their difference is not a confidence
interval, posterior sample, or guaranteed bracket for truth. Reference total-volume
error, spatial bed error, DEM/outline errors and temporal transfer remain separate
uncertainties. Tau is an effective parameter with the shape factor fixed; it can
absorb discrepancies in the bed or model and must not be interpreted as a direct
basal-stress observation or climate indicator.

| System | Profile-fit tau (kPa) | Volume-zero tau from curve (kPa) |
|---|---:|---:|
| 3632 | 92.870 | 92.055 |
| 3131 | 105.861 | 152.063 |
| 2906 | 107.231 | 119.209 |

The volume-zero values are linear interpolations between supplied 20 kPa grid
samples, not completed volume calibrations. Actual evaluations at those values
and convergence/closure residuals remain to be obtained in a separately registered
research experiment. No such experiment was launched in this review.

## Interpretation limits and corrections to the supplied narrative

- On 3131 the profile objective is weakly identifying in the tested range and
  minimum RMSE remains about 59 m. This flags incompatibility between the selected
  model and reference spatial structure. It does not uniquely show that the
  reference redistribution is wrong. Uniform tau, network assumptions and the bed
  definition can all contribute. Retain and flag the glacier rather than discard it.
- The old operator is 33-47% low at the reported archived/profile-fit solutions.
  The supplied broader curves also cross zero old-operator volume bias. Thus these
  controls demonstrate a poor joint fit, not mathematical impossibility of volume
  calibration or invalidity of every previously calibrated system.
- The outlet effect is small for these three modern networks at fixed tau; it is
  not a universal boundary-condition result. The machine-readable review pairs
  H0 and Href at the same tau over the entire supplied scan.
- Constraint density was not isolated in this full-chain comparison. Sparse nodes
  are a plausible contributor, not a measured causal share of the volume deficit.
- Along-line spacing is 124 m (3632), 122 m (3131), 134 m (2906), not universally
  122 m. Profile fitting uses the archived 132 m exclusion; raster constraints use
  100 m. Operator A receives grid-cell bed plus network thickness to ensure equal
  thickness constraints, rather than the unadjusted bilinear-bed network surface.
- A_surface_rmse_vs_dem_m is based on raw interpolated surface, before clipping
  negative thickness. It must not be relabelled as the realized clipped surface RMSE.
- Modern replay tests internal consistency, not temporal tau transfer to the LIA.
  The supplied runtime and replay success are external result provenance.

## Next evidence, before regional promotion

In a separately versioned 20-30-system pilot, stratify across mountain regions,
size, relief, branching/confluences and single/multiple modern-member systems;
retain poorly identified cases. Fit both objectives, report volume closure,
profile and surface errors, clipping extent and tau sensitivity together. On 3131,
compare original 1-D constraints, redistributed field and available independent
thickness data before assigning a cause. Independent thickness and geomorphic
surface constraints, when available, should be held out from calibration.

Regional method selection still requires the project's original-paper comparison,
including Li/Zhang methods; this code review does not fill missing full-text or
supplementary evidence. No new validation threshold is imposed here. This pilot
design is a proposal, not an executed or launch-authorized batch.
