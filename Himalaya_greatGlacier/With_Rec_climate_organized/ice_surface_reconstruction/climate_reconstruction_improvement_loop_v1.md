# HMA Climate Reconstruction Improvement Loop v1

Date: 2026-08-15

## 1. Objective

Build a new, independently versioned HMA temperature and precipitation reconstruction (`HMA_reconstruction_v2`) that is reproducible, uncertainty-aware, and validated without modifying or replacing any original source, reconstruction, MATLAB, NetCDF, CSV, or MAT file.

The loop is production-oriented. An experiment is permitted only when its result can promote, reject, or materially redesign a production candidate. Narrow diagnostics that cannot change a candidate decision are out of scope.

The final deliverable must include:

- native-grid temperature and precipitation reconstructions;
- all retained ensemble/member outputs, not only a median or mean;
- uncertainty quantiles and ensemble spread;
- the frozen 0.5-degree output product and its tested transformation operator;
- source, code, configuration, seed, and checksum manifests;
- blocked-validation and final-test reports;
- a changelog against the existing reconstruction.

## 2. Non-negotiable constraints

1. `LIA_PROJECT_FACTS.md` is the canonical project contract and must be read before every climate batch.
2. All original inputs and existing reconstruction outputs are read-only.
3. Every round uses a new independent result directory. A failed candidate is retained and labelled; it is not overwritten.
4. MATLAB work uses R2024b only.
5. `1921-1939` is a locked final test period. It must not be read, plotted, summarized, or used in any decision until code, candidate, weights, and acceptance rules are frozen and the user explicitly approves opening it.
6. `1901-1920` is a support-aware stress test only. It must not determine proxy deletion, regional correction, ensemble weights, or tuning parameters.
7. CRU, GPCC, and reference reconstructions are benchmarks with known coverage and uncertainty, not error-free truth.
8. OGGM results cannot tune climate reconstruction parameters or ensemble weights. Glacier simulations may be used only as a declared downstream physical diagnostic after the climate candidate is frozen.
9. A result inferred from one experiment, proxy network, or EnKF method cannot be generalized to the production six-member ensemble without a matching ensemble-level test.
10. If a result conflicts with `LIA_PROJECT_FACTS.md`, stop the affected round and resolve the conflict before continuing.

## 3. Evaluation split

### Development evidence

- Use the currently approved instrumental calibration domain (`1940-2000`) through contiguous blocked temporal cross-validation.
- Add spatial or station holdout where the underlying observations permit it.
- Fit every scaling factor, PSM choice, covariance setting, and ensemble weight inside the training fold only.
- Keep temperature and precipitation scorecards separate before any joint ranking.

### Diagnostic evidence

- Use `1901-1920` only after a candidate has been scored out of fold.
- Report direct-support, partial-support, and unsupported grid cells separately.
- A stress-test improvement cannot rescue a candidate that fails blocked validation.

### Final evidence

- Keep `1921-1939` sealed through model selection.
- Open it once, after the candidate registry and acceptance thresholds are frozen and the user approves.
- Do not revise the selected candidate after seeing the final-test result. A failure starts a new versioned loop with a newly locked final test.

## 4. Loop state machine

Each candidate moves through these states:

`proposed -> preflight_passed -> evaluated -> promoted | closed | blocked`

- `promoted`: passes all frozen gates and becomes the control for the next round.
- `closed`: fails a gate or does not improve a production decision; no more variants in that mechanism family unless new independent evidence is declared.
- `blocked`: input identity, scope, or reproducibility is unresolved; no scientific interpretation is allowed.

Only one promoted candidate exists for temperature and one for precipitation at a time. The legacy reconstruction remains the permanent external control.

## 5. Stage gates

### Loop 0: Production-definition freeze

Purpose: establish exactly what is being reconstructed and reproduced.

Actions:

- freeze source identities, units, calendars, grids, proxy6/proxy9 membership, PSM definitions, prior members, seeds, and method labels;
- reproduce the six archived production members: `AO`, `HGAO`, and `O`, each for proxy networks 6 and 9;
- reproduce the existing equal-weight six-member product over its complete available period;
- freeze and test the native-grid to 0.5-degree transformation separately from reconstruction skill.

Gate 0:

- all six members have unambiguous lineage;
- archived-member and ensemble reproduction closes at declared numerical tolerance;
- hashes show all sources are unchanged;
- no result is promoted if method, network, experiment, or ensemble scope is ambiguous.

### Loop 1: Legacy baseline scorecard

Purpose: determine what actually performs well or poorly before changing the system.

Actions:

- score the six members and their current equal-weight ensemble with identical blocked folds;
- evaluate temperature and precipitation separately;
- report correlation, RMSE, bias, trend error, ensemble coverage, and CRPS where ensembles exist;
- stratify by west, central, and east HMA, season, elevation/support class where available;
- bootstrap differences against the legacy equal-weight ensemble.

Gate 1:

- publish a complete out-of-fold baseline scorecard;
- freeze numerical promotion/no-regression thresholds before testing repairs;
- provisional advancement rule: at least 5% improvement in the primary aggregate error or a bootstrap 95% interval excluding no improvement, with no region worsening by more than 5% in the same primary error;
- thresholds may be revised only once from baseline scale information, before Loop 2 begins.

### Loop 2: Network and operator repair

Purpose: repair only mechanisms demonstrated by the baseline to limit production skill.

Allowed mechanism families:

- proxy6 versus proxy9 composition, overlap, and PSM seasonality;
- PSM calibration stability across blocked periods;
- proxy error covariance and correlated-record handling;
- prior covariance, localization, or inflation;
- observation-support-aware treatment that is defined before seeing diagnostic-period curves.

Rules:

- one mechanism family per round;
- one legacy/promoted control plus at most three variants;
- no record-by-record deletion based on a target curve;
- no manual regional trend correction;
- no parameter is tuned on `1901-1920`;
- every variant is judged with the Loop 1 folds and metrics.

Gate 2:

- promote only a variant that passes the frozen improvement and no-regression rules for its target variable;
- close a mechanism family after two consecutive rounds fail to pass;
- a mechanistically interesting result that does not improve validation is documented but not promoted.

### Loop 3: Ensemble construction

Purpose: replace unvalidated equal weighting only if out-of-fold evidence supports it.

Candidates:

- legacy equal six-member mean;
- validation-weighted method/network mean;
- constrained nonnegative stacking or ridge weights;
- separate network ensembles when proxy6/proxy9 disagreement cannot be defensibly collapsed.

Rules:

- learn weights using nested blocked cross-validation only;
- weights must be nonnegative and sum to one unless a different constraint is physically and statistically justified in advance;
- never infer weights from `1901-1920`, glacier curves, or visual agreement with a reference reconstruction;
- report fold-to-fold weight stability and single-member dominance.

Gate 3:

- the selected ensemble must beat the legacy equal mean out of fold under the frozen primary rule;
- it must satisfy regional no-regression and uncertainty-coverage gates;
- unstable weights trigger simplification, not manual adjustment.

### Loop 4: Frozen full reconstruction

Purpose: generate the production candidate only after the method is selected.

Actions:

- freeze code commit, configuration, weights, source hashes, random seeds, and software versions;
- run the declared full experiment/member set for temperature and precipitation over the complete reconstruction period;
- preserve member-level native-grid outputs and uncertainty summaries;
- apply the independently verified 0.5-degree transformation;
- rerun one duplicate member to verify numerical reproducibility.

Gate 4:

- source hashes remain unchanged;
- duplicate execution closes at declared tolerance;
- all years, cells, units, calendars, and missing-value policies pass QA;
- the production candidate can be rebuilt from its manifest without hidden manual steps.

### Loop 5: Validation and locked final test

Purpose: decide whether the frozen candidate is releasable.

Actions before opening the lock:

- consolidate blocked-validation results;
- run the support-stratified `1901-1920` stress test;
- compare independent pre-instrumental references without treating them as calibration targets;
- test temperature/precipitation physical coherence, spatial behavior, temporal variance, and extremes;
- optionally run declared glacier-response diagnostics without feeding results back into climate tuning.

Gate 5A:

- freeze the candidate, report template, and accept/reject criteria;
- record the user's explicit approval before accessing `1921-1939`.

Gate 5B:

- evaluate `1921-1939` once;
- accept or reject exactly under the frozen criteria;
- do not tune the candidate after the result is known.

### Loop 6: Release

Purpose: publish a traceable v2 product while preserving v1.

Release package:

- native-grid and 0.5-degree NetCDF products;
- retained member fields and uncertainty quantiles;
- machine-readable source/configuration/checksum manifests;
- validation scorecards and figures;
- known-limitations statement and v1-to-v2 changelog;
- a frozen release identifier.

## 6. Per-round contract

Every round must declare before execution:

1. one production decision it will change;
2. one falsifiable hypothesis;
3. the control and no more than three variants;
4. exact training, validation, diagnostic, and forbidden periods;
5. primary metric, no-regression metrics, and thresholds;
6. expected files and numerical closure tests;
7. promotion, closure, and stop conditions.

Every result directory must contain:

- `ROUND_CONTRACT.md`
- `config.json`
- `source_manifest.json`
- `scores.csv`
- `summary.json`
- `FINAL_RESULT.md`
- only decision-relevant figures

The candidate registry is append-only. It records candidate id, parent, variable, mechanism, state, result directory, decision, and evidence.

## 7. Stop rules

- Stop a mechanism after two failed rounds.
- Stop immediately on source mutation, period leakage, scope mismatch, unit/calendar inconsistency, or failed reproduction closure.
- Do not add a diagnostic because a result is surprising unless it can distinguish between competing production decisions.
- Do not delete individual proxies or apply geographic corrections to improve one historical curve.
- Do not continue an experiment whose outcome cannot promote or close a candidate.
- Do not run OGGM until a climate candidate is frozen for downstream diagnosis.

## 8. Initial execution queue

### Round 00: Full production lineage and reproduction

Deliverable: exact all-period lineage for the six legacy members and equal-weight ensemble, including temperature and precipitation, with a source/checksum manifest.

Decision: whether the current reconstruction is reproducible enough to serve as the baseline.

### Round 01: Blocked-CV baseline scorecard

Deliverable: a common-fold scorecard for all six members and the equal-weight ensemble for temperature and precipitation, including regional no-regression metrics and uncertainty behavior.

Decision: which variable, network, method, and region are limiting production skill; freeze the numerical promotion thresholds.

### Round 02: First bounded repair

Default hypothesis: the proxy6/proxy9 conflict, especially the central-HMA precipitation disagreement, is a network/operator problem that degrades out-of-fold skill.

This round runs only if Round 01 confirms that the conflict exists out of fold. Otherwise, Round 01 selects the highest-impact failed mechanism and records why the default was skipped.

Decision: promote one network/operator variant or close this mechanism family.

### Round 03: Nested-CV ensemble selection

Run only after at least one repair family is closed or promoted. Compare the surviving members/candidates with the legacy equal mean using nested blocked CV.

Decision: retain equal weighting, promote a validated weighted ensemble, or preserve separate network products.

## 9. Directory layout

Loop root:

`validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/`

Planned round directories:

- `round_00_production_reproduction/`
- `round_01_blocked_cv_baseline/`
- `round_02_network_operator_repair/`
- `round_03_nested_ensemble_selection/`
- `round_04_frozen_full_reconstruction/`
- `round_05_validation/`
- `round_06_release/`

Production candidate outputs must be written to a new versioned location outside all original source directories. The existing reconstruction remains untouched.

## 10. Immediate decision

The previous experiment-1-only and `1901-1920` diagnostics are retained as supporting evidence, but they are no longer the active workstream. The next executable task is Round 00 followed by the Round 01 baseline scorecard. No additional narrow proxy audit is authorized unless the baseline scorecard shows that it can change a production candidate decision.
