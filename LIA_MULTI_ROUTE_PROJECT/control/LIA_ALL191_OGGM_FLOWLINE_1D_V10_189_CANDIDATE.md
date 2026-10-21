# LIA all-191 OGGM flowline 1-D transaction v10.189 — candidate

Status: `candidate-not-authorized-not-launched`  
Transaction ID: `lia-all191-oggm-flowline-1d-v10.189-candidate`  
Model family: `oggm-flowline-1d`  
Accepted-count effect now: `0` (`accepted=58`, `active=0`, `queued=0`)

This candidate supersedes the design intent of `shared/ONE_D_PARAMETER_MATRIX_V10_189_DRAFT.md`. It does not include, authorize, or inherit any GPU or planform-2D transaction.

## 1. Authority and population reset

- Canonical base is controller state `2026-09-01-v10.188-discovery15-stage-b-terminal-retry-decision-required`; its validator passed with `launch_authorized=false`.
- Population is exactly the 191 unique identities in `LIA_ALL191_OGGM_FLOWLINE_1D_V10_189_MEMBER_LEDGER.csv` (SHA-256 `FD11EB121DD7F6E4BA07906C76707485BAF6D3F043E11D891FAB32DCC2699D87`).
- Every ledger row begins `not-started`. The legacy 58 accepted results are not inherited as v10.189 results; all 191 must be rerun under this exact 1-D contract before any new-route acceptance claim.
- The retained 32-curve evidence is sample prior only. It may define preregistered gates but may not seed a state, skip a member, or transfer pass/accepted status.
- This draft does not modify `LIA_CURRENT_STATE.json`, `LIA_PROJECT_FACTS.md`, the route snapshot, or the canonical accepted count.

## 2. Latest retained-32 standard and forward hard gate

Latest landed version: `validation_data/results/paleo_surface_validation_v0/lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json`, SHA-256 `1A6E63B5F0F6B5D1B486C7E7B458002DD913D7352DCB93529B37482AB47F3A76`.

It contains an explicit forward hard gate. A technically valid 1470–2000 forward trajectory must pass all four v3 scientific groups:

1. endpoint bounds: LIA peak and year-2000 total-length errors `<=0.10`, and both separation-normalized errors `<=0.20`;
2. lagged internal response: 21-year smoothing, `B(t)` to later `dlnL/dt` over inclusive lags 0–100 years, correlation `>=0.30`, sign consistency `>=0.55`, with a best lag at 100 years held for review;
3. bidirectional multistage response: derivative deadband `0.25 SD`, neutral bridge at most 5 years, phase duration at least 20 years, at least two persistent growth and two persistent retreat phases, and sustained-variation share `>=0.60`;
4. artifact exclusion: terminal endpoint drop or numerical high-frequency oscillation is a failure.

Passing these four groups is only a provisional forward-curve pass. Formal acceptance also requires inherited identity, geometry, source-hash, numerical, boundary, and observation-domain gates plus an independent exact-contract reproduction and a separate canonical controller transaction.

## 3. Four-level adaptive parameter matrix

`base_temp_bias` remains immutable provenance; effective equilibrium bias is `base_temp_bias + delta_t_eq`. Length is an output, never a spin-up fitting target.

1. Level 1 — `delta_t_eq = [-1.5,-1.0,-0.5,0.0,+0.5,+1.0,+1.5] degC`, with all other parameters at each member's frozen baseline.
2. Level 2 — only unresolved Level-1 branches: `glen_a = baseline × [0.5,1.0,2.0,3.0]`.
3. Level 3 — only unresolved Level-2 branches: `(prcp_fac,melt_f) = baseline × ([0.8,1.0,1.2] × [0.8,1.0,1.2])`.
4. Level 4 — only unresolved Level-3 branches: `temp_melt = [-1.5,-1.0,-0.5] degC`.

Each declared cell is enumerated before launch. A converged cell freezes immediately and is not expanded. A failed or unresolved branch can open only its next registered level. No unconditional Cartesian search, bound extension, glacier-specific manual rescue, or post-result threshold change is permitted.

## 4. Adaptive spin-up and checkpoints

- Repeated equilibrium climate period: 30 years.
- Advance only in atomic 300-year blocks, with checkpoints at years `300,600,...,3000`.
- At every checkpoint, compare the final three 30-year cycles only at identical climate phase and store raw and normalized length, area, volume, mass-balance, domain, boundary, and numerical drift.
- Hard maximum is 3000 years. A state not converged by then is `max-duration-nonconverged` and cannot enter forward evaluation.
- A checkpoint is eligible for resume only when state, manifest, fingerprint, and checksum are complete and mutually consistent. Resume uses the identical fingerprint and writes to the same versioned cell root without overwriting committed evidence.

## 5. CPU multi-process short preflight

No production cell may start before one versioned preflight passes.

- Fixed roster, three identities per frozen area quartile: `214,252,313,7,84,130,512,705,709,14,79,334`.
- A non-scientific process/resource fixture evaluates worker candidates `1,2,4,8,12` without advancing model time.
- The selected count is the largest count with deterministic process completion, no file/root collision, no oversubscription, and peak committed memory below 80% of available RAM; the production cap remains 12.
- At that one selected worker count, each fixed member runs exactly one short exact-contract chain: static/hash validation, 0.1-year and 5-year zero-MB numerical gates, then one 30-year repeated-climate timing segment. These are preflight-only and cannot be labeled converged or accepted.
- Any collision, source mutation, nondeterministic identity/fingerprint, unhandled exception, numerical gate failure, or memory threshold breach stops the preflight and authorizes no production cell.

## 6. Versioned roots, fingerprints, and outputs

Reserved output root (currently absent): `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_v10_189/`.

Every cell root is immutable and keyed by a SHA-256 fingerprint over: transaction/version, `lia_unique_id`, RGI/component identity, source-file hashes, geometry and observation-domain versions, model family, complete parameter vector and baseline provenance, spin-up/forward forcing IDs, solver and dependency versions, thresholds, time interval, checkpoint cadence, CPU worker contract, and driver hash. An existing fingerprint is `duplicate-not-launched`.

Required per-cell evidence includes manifest and fingerprint, stdout/stderr and process metadata, atomic checkpoint plus checksum, annual/checkpoint length-area-volume-MB series, step and runtime metrics, boundary/domain/mass-closure diagnostics, terminal class, and immutable summary. Forward roots must additionally contain the source spin-up state hash and baseline reproduction result.

## 7. Forward release, reproduction, and stop conditions

- Only a converged and hash-validated 1-D state may be proposed for forward release.
- Forward is a separate 1470–2000 transaction stage; it cannot refit or mutate its source state or parameter vector.
- The source branch must first reproduce its saved state within the frozen reproduction tolerances. Failure stops all dependent forward cells.
- Stop the affected cell/block on unreadable authority, validator failure, authorization false, ledger mismatch, missing/changed hash, duplicate fingerprint, colliding root, checkpoint mismatch, nonfinite/negative state, mass-closure, CFL, boundary or observation-domain failure, exhausted 3000-year ceiling, invalid-rate ceiling, loss of a registered response bracket, or any requested change outside this contract.
- A member failure is isolated to its declared branch; it cannot be imputed, silently retried, hand-tuned, or counted as accepted. One versioned technical retry is possible only after cause identification and separate explicit authorization.

## 8. Sole blocking item

`BLOCK-V10.189-01`: the v3 retained-32 forward standard is explicitly non-canonical, and the exact same-phase spin-up convergence, checkpoint reproduction, boundary, and invalid-rate numerical tolerance bundle is not frozen. These must be adopted with exact values in one canonical versioned transaction before any preflight or model launch; until then `launch_authorized=false`.

