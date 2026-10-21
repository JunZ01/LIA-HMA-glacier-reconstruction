# Section and width operator audit

**Status:** executable read-only diagnostic v0; not a thickness correction and
not a velocity posterior

## Scope

This audit starts from the 21-system union classified as
`residual-outside-envelope` by either Farinotti or Millan in the turnover
attribution preflight. Twelve systems carry that median-based classification
in both thickness scenarios. The audit does not change this frozen parent
classification; it resolves its sign and tests whether the registered velocity
uncertainty, alternate thickness product, or an outline-width proxy can close
the mismatch.

## Fixed section decomposition

For each selected member gate and thickness scenario:

```text
section = inversion_output[flowline]["volume"][gate] / dx
shape_factor = section / (flowline_width * product_thickness)
```

The audit reproduces the section saved in the parent turnover result and checks
that the inversion-output width matches the width used to build the ITS_LIVE
gate. Farinotti and Millan share the same surface flowline width; their section
difference is therefore decomposed into product thickness and section-shape
effects.

The independent-looking width field is not an external width observation.
`local_width_m` is the OGGM inversion-flowline width. A second, coarse proxy is
derived from the RGI-outline intersection used by the velocity sampler:

```text
outline_width_proxy = gate_intersection_area / gate_along_length
```

This 480 m slab-average is a diagnostic geometry proxy only. It is not allowed
to overwrite the flowline width.

## Signed reachability

The parent preflight used whether the observed median falls inside the model
envelope. This audit preserves that field for continuity and adds the stricter
structural test: the full registered observation interval must not overlap the
model envelope.

```text
interval_overlap = observed_high >= model_lower
                   and observed_low <= model_upper
```

Only a non-overlap receives a signed structural residual:

- `strict-too-fast`: model lower bound exceeds the observed upper bound;
- `strict-too-slow`: model upper bound is below the observed lower bound;
- `interval-overlap`: no section change is required within registered
  observation and imbalance uncertainty.

For a strict residual, the boundary section multiplier is:

```text
strict-too-fast: section_factor_min = model_lower / observed_high  (> 1)
strict-too-slow: section_factor_max = model_upper / observed_low   (< 1)
```

Velocity alone cannot distinguish section error from flux error. The output
therefore uses neutral routing labels such as
`larger-section-or-lower-flux-required` rather than declaring thickness wrong.

## Structural sensitivities

Two deterministic sensitivities hold the candidate flux fixed:

1. replace the current section with the alternate Farinotti/Millan section;
2. scale section by `outline_width_proxy / flowline_width`, holding thickness
   and section shape fixed.

They are reachability tests, not corrections. A sensitivity that overlaps the
observation interval identifies a plausible structural direction for later
testing.

## Outputs and acceptance

- `gate_section_audit.csv`: section decomposition and signed gate diagnostics;
- `system_scenario_audit.csv`: per-scenario gate conflict and closure routing;
- `system_consensus_audit.csv`: 21-system cross-scenario consensus;
- `priority_strict_residuals.csv`: systems with at least one strict residual;
- `diagnostic_summary.json`, `run_manifest.json`, `progress.json`, and
  `failures.csv`.

Acceptance requires 21 consensus systems, exact parent-section reproduction,
zero unregistered worker losses, and unchanged source pickle hashes. No LIA
geometry, reconstructed climate, parameter file, or OGGM source directory is
written.
