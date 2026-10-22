# Checkpoint 3: 191 systems / 950 modern members

Completed under the human-fixed rules (facts1510-1514). Both fixed gates pass; no formal mainline acceptance or dynamic run.

## Results

- Export: 950/950 member NPZs, zero failures; 191 main members; native-volume sums match pairing within the prescribed check.
- Modern: 191/191 systems complete under the 5% area policy. 853/950 nominal member solutions; 97 failed secondary members in 52 systems. Maximum failed-area share 2.108337%; missing volumes were not replaced with reference volumes. Own-tau sensitivity solutions: 653 members.
- LIA: 191/191 paired endpoints complete; 190 success and one success with the pre-existing documented small-component exclusion. Runtime284.092s (modern103.567s).
- Primary unflagged187 systems:186 lose volume (99.465241%); LIA669.204428252km3, modern430.296527516km3, change -238.907900736km3 (-35.700286885% of LIA).
- All191 complete systems: change -240.381322938km3 (-35.489479266%).
- Low-tau flags (<=45kPa):1856/2387/3053/3137. Retained in data, excluded only from primary totals. A heuristic flag is not a physical diagnosis.
- The unflagged volume gain is1412;3053also gains in the flagged set. No parameter rescue or post-result rule change.
- CK2 consistency:22/22 finite overlaps; maximum absolute dV difference8.881784197001252e-16km3.
- All6421 frozen source/code hashes checked unchanged; identity, arithmetic and gate checks pass.

## Integration details

- System tau is the production main member profile-fit tau, shared by all modern members and the LIA endpoint; own-member tau is sensitivity only.
- Fixed5% failed-member area rule,45kPa flag and80% gates unchanged.
- Exporter repairs: avoid duplicate main_rgi_id suffixes, use production selection, support an all-success error-column case, retain member progress/area metadata and hash the actual config path.
- Modern runner repairs: enforce main-member nominal failure, attempt secondary nominal solves independently of own-tau sensitivity success, preserve known areas on fitting failure, list failed IDs, save fingerprinted phase checkpoints and verify NPZ hashes.
- LIA runner adds identity checks, full-population regional totals and finite CK2-overlap diagnostics. Production numerical driver unchanged; new output and grid-cache directories.
- These are static model estimates: incomplete member volumes remain unknown, tau transfer is not independently validated, and this discrete thin-plate operator is not an exact ANUDEM implementation. The old mixed-method conclusions remain withdrawn.
- No low-tau rescue investigation or mainline32 dynamic comparison started by this task. All launch permissions closed on completion; accepted count unchanged.

The requested five raw files and their ZIP are included. Member/system audit CSVs support completeness accounting. Full NPZ data, NetCDFs and full input hashes remain local; no raw spatial arrays are uploaded. The complete canonical controller operates in the local data workspace, not this documentation export.
