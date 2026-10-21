# Stage 3 initial-state search pilot

**Status:** executable pilot specification v0; no terminal validation

## Boundary

Stage 3 identifies a centennial initial state after the Stage 2 parameter
freeze. It may read, but may never rewrite, the frozen `glen_a`, `fs`,
`melt_f`, `prcp_fac`, or `temp_bias` values. ITS_LIVE velocity remains a
routing flag and later Stage 4 diagnostic. It has zero weight in this search.

The search coordinate is an additional temperature offset used only to create
the state before the LIA observation window. It is not a calibrated mass-
balance parameter. The offset is present at the end of the constant-climate
spinup and tapers linearly to zero before 1050 CE. It is therefore zero at
least 150 years before the outer LIA window begins in 1200 CE. The transient
forcing and the frozen annual-scale parameters are unmodified throughout the
observation window.

Every model run occurs in a Stage 3 copy of the Stage 0 glacier directories.
Source directories and the Stage 2 CSV files are protected by before/after
SHA-256 checks.

The heavy OGGM work directory uses the short E-drive path `../../s3w/p0`.
This keeps NetCDF geometry paths below the Windows/HDF5 path limit while all
ledgers, summaries, and manifests remain in the registered organized output
directory. No Stage 3 artifact is written to C drive.

## Observation and simulation units

The observation unit is one LIA system. A system may contain several modern
RGI glaciers. The simulation runs each member glacier on its own Farinotti
bed, then aggregates member geometry before evaluating the system observation.

The primary geometry components are registered before any dynamic run:

1. LIA polygon area;
2. LIA reach along the main RGI glacier's model flowline and downstream bed;
3. the LIA surface-elevation distribution sampled from the registered DEM.

For a simulated year, the component residuals are the log area ratio, log
main-flowline-length ratio, and the weighted RMSE of the 10/25/50/75/90%
surface-elevation quantiles. The composite geometry score is a weighted
geometric mean of Gaussian component scores. Missing components are permitted
only when their reason is registered; available component weights are then
renormalized. Area is always required.

Timing is applied after geometry:

```text
weighted_score(t) = timing_weight(t) * geometry_score(t)
state_score = max_t weighted_score(t)
```

The timing weight is the frozen regional trapezoid from
`05_lia_timing_data_decision.md`. Timing cannot rescue a geometric miss: a
successful state must also satisfy every available component tolerance at the
selected year.

## Search operator

For each system, Stage 2 turnover role, and reconstructed-climate member:

1. create a zero-ice spinup on the fixed Farinotti bed;
2. run a constant climate centred on 1015 CE with an initial-state offset;
3. continue with reconstructed climate from 1000 CE while tapering the offset
   to zero by 1050 CE;
4. evaluate annual system geometry from 1200 through 1900 CE;
5. use system area to establish the cold/warm bracket and bisect it;
6. select the evaluated candidate with the best full geometry/timing score.

The constant-climate spinup starts at 600 years and extends in 300-year steps
to at most 1200 years when its final-window area has not converged. Failure to
converge at the cap is retained as a candidate-level identifiability failure.
Each duration writes a distinct NetCDF suffix (`y600`, `y900`, or `y1200`),
avoiding unsafe in-place replacement through the Windows NetCDF/HDF5 stack.

Area is the bisection coordinate because it is monotonic under the temperature
offset in the expected regime. Length and hypsometry validate the resulting
state; they do not become extra tuning directions. A cold endpoint that fills
the downstream domain is retained as a right-censored upper-area candidate so
the search may continue toward warmer offsets; it can never be selected as a
restart state. A non-monotonic response, a selected state at the domain edge,
or a geometry-tolerance miss is a classified task failure. An unbracketed area
target changes the identifiability label but does not discard an otherwise
valid full-geometry hit. None authorizes a Stage 0-2 parameter change.

The initial bracket is -4 to +1 K. If both finite residuals have the same
sign, the bracket expands in 1 K steps within -6 to +4 K. A state at an
unexpanded artificial boundary is diagnostic only. Selection normally
requires a positive/negative area-residual bracket. The sole exception is the
physical zero-ice lower bound: if the warm spinup ends with zero area and its
transient LIA geometry is within all registered tolerances, the result is
accepted as `zero-state-lower-bound`, not as a point estimate of temperature
offset. This identifiability flag must accompany the restart state into Stage
4.

When a cold candidate is domain-censored and a warmer candidate is finite,
the operator first bisects that feasibility boundary to 0.25 K. The area root
is a search aid, not an extra acceptance rule: after the feasible range has
been sampled, any converged candidate satisfying all registered geometry
tolerances belongs to the admissible set. The presence or absence of an area
sign bracket determines the identifiability label, not whether a genuine
geometry hit is discarded.

If at least three converged, geometry-compatible candidates span at least 2 K
in initial-state offset but converge to indistinguishable LIA-window geometry,
the task is accepted as `trajectory-insensitive-compatible`. The offset is not
reported as a point estimate. A deterministic median representative restart
is retained only because the LIA states have converged; the admissible offset
range and sensitivity spans remain in the task ledger.

## Pilot waves

The pilot contains 12 systems: three LIA-area representatives per region,
with `2692`, `2907`, and `4637` forced into their regional sets. Selection is
deterministic and is written to `pilot_selection_12.csv`.

Execution is gated in three waves:

1. **operator smoke:** four small systems, one per region, using the
   mid-turnover role and `ensemble_rec_ref` only;
2. **turnover pilot:** all 12 systems and all three turnover roles using
   `ensemble_rec_ref`;
3. **climate replication:** the three permanent review systems, the
   mid-turnover role, and all six registered reconstruction files.

The six reconstruction files are registered as six working individuals. The
pilot reference member is an execution gate only and is not declared the
preferred scientific reconstruction. A full Stage 3 scale-up is therefore
`191 systems x 3 turnover roles x 6 climate individuals`, subject to the
pilot gates.

## Acceptance and outputs

The preflight gate requires 12 selected core systems, four represented
regions, the three permanent review systems, six climate hashes, complete
Stage 2 parameter rows, valid LIA area targets, and no writes to frozen inputs.

The dynamic scale-up gate additionally requires:

- reproducible restart from saved candidate state;
- a continuous initial-state offset at the spinup/transient boundary;
- zero additional offset throughout 1200-1900 CE;
- classified rather than discarded failures;
- source hashes unchanged;
- no Stage 4 modern-geometry residual in the search objective.

Pilot artifacts are written under
`outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage3_initial_state_pilot_v0`.
The preflight writes geometry targets, climate and source registries, the
pilot ledger, failures, summary, and a run manifest. Dynamic search traces and
restart states are added only by the run phase.

Each completed offset candidate is atomically checkpointed before the next
candidate begins. Resume validates the candidate's diagnostic, flowline, and
geometry NetCDF files for every system member before reusing it. A process or
terminal interruption therefore re-enters at candidate granularity rather
than restarting the system.

All OGGM model files also include an immutable attempt tag. A new attempt may
read frozen Stage 0 and preflight inputs but may never overwrite NetCDF files
from an earlier attempt.
