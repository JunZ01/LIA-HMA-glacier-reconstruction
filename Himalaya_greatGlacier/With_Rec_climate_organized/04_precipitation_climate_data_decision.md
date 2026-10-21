# 04 Precipitation and Modern-Climate Data Decision

Status: **confirmed decision; climate files registered but intentionally not processed before Stage 0**

Decision date: 2026-07-19

## 1. Compatibility Audit

The existing notebook uses:

- `baseline_climate = CRU`;
- `use_winter_prcp_fac = False`;
- fixed `prcp_fac = 8`;
- `mb_calibration_from_scalar_mb` against the 2000-2020 geodetic MB period.

That configuration does not use OGGM's native winter-precipitation prior. It fixes one of the three annual-scale MB directions before calibration and is retained only as the historical baseline.

The installed OGGM version is `1.6.3.dev3+g865475d`. Its `decide_winter_precip_factor` implementation explicitly rejects any baseline whose name does not contain `W5E5`. The implementation also requires all 287 northern-hemisphere winter months from January 1979 through December 2019.

Therefore:

> The native OGGM winter-prcp-fac coefficients are not compatible with CRU. They must not be copied into a CRU run.

## 2. Baseline Decision

Use **GSWP3_W5E5, OGGM flattened release 2023.2 monthly, 1901-2019** as the primary modern calibration baseline.

- W5E5 part: 1979-2019
- GSWP3 extension: 1901-1978
- Native winter-prcp-fac period: 1979-2019
- Geodetic MB calibration period: 2000-01-01 to 2020-01-01, corresponding to climate years 2000-2019

This choice closes both the precipitation-prior and modern-climate entry gates. It also provides the exact 1979-2019 period expected by the OGGM implementation.

CRU remains a separate structural climate scenario. A CRU run must be independently calibrated and must not inherit:

- the existing fixed `prcp_fac=8` as a W5E5 parameter;
- the W5E5 winter-prcp-fac regression coefficients;
- a W5E5-calibrated `melt_f` or `temp_bias`.

For the LIA transient sequence, reconstructed anomalies must be bias-anchored to GSWP3_W5E5 over the shared modern period. The switch into the transient series must remain continuous; no instantaneous baseline jump is allowed.

## 3. Registered Climate Files

OGGM's installed `shop/w5e5.py` registers the following release under:

`https://cluster.klima.uni-bremen.de/~oggm/climate/gswp3-w5e5/flattened/2023.2/monthly/`

| Variable | File |
|---|---|
| Invariant fields | `gswp3-w5e5_glacier_invariant_flat.nc` |
| Monthly temperature | `gswp3-w5e5_obsclim_tas_global_monthly_1901_2019_flat_glaciers.nc` |
| Monthly temperature standard deviation | `gswp3-w5e5_obsclim_temp_std_global_monthly_1901_2019_flat_glaciers.nc` |
| Monthly precipitation | `gswp3-w5e5_obsclim_pr_global_monthly_1901_2019_flat_glaciers.nc` |

The files are registered but not downloaded or processed in this decision stage. Processing before the remaining 960 glacier directories exist would create an incomplete cache and invite silent mixed-baseline reuse.

Before any OGGM climate task runs, set:

- `cfg.PATHS['working_dir']` to the organized `E:` workspace;
- `cfg.PATHS['dl_cache_dir']` to `data_raw/climate/cache` on `E:`;
- `TEMP`, `TMP`, and `CPL_TMPDIR` to `_tmp` on `E:`.

Any climate file or cache entry found under `C:` is invalid for this project run.

## 4. Native Prior Center

For each glacier, compute uncorrected mean daily precipitation over the seven northern-hemisphere winter months (October-April) for 1979-2019, then use the installed OGGM relationship:

`p0 = -1.0614 * ln(winter_daily_mean_prcp) + 3.9200`

Clip `p0` to the physical range `[0.1, 10]`.

`p0` is the climate-source-specific center of the prior. It is not a globally transferable parameter and is not itself the final calibrated `prcp_fac`.

## 5. Regional Prior Width

The native OGGM informed three-step calibration allows precipitation to vary between 0.8 and 1.2 times `p0`. This is retained as the standard prior scale.

| Prior class | RGI mask | Count | Approximate 95% prior interval |
|---|---|---:|---|
| `standard-w5e5` | all targets except O1=14, O2=3 | 799 | `0.8 p0` to `1.2 p0` |
| `upper-indus-wide` | O1=14, O2=3 | 406 | `0.5 p0` to `2.0 p0` |

The Upper Indus/Karakoram class is deliberately wider. Immerzeel et al. (2015) found that high-altitude precipitation required to close glacier mass balance and runoff in the Upper Indus was more than twice prior estimates on average and up to ten times higher in extreme cases: `https://doi.org/10.5194/hess-19-4673-2015`.

The factor-of-two interval is applied around the W5E5-informed center `p0`, not around raw valley-gauge precipitation. The physical optimizer may still explore `[0.1, 10]`; a prior-interval crossing is a posterior penalty and diagnostic, not an artificial no-solution wall.

For implementation, use a split-normal penalty in log space:

- standard: lower `sigma_ln=0.113849`, upper `sigma_ln=0.093021`;
- Upper Indus: lower and upper `sigma_ln=0.353647`.

## 6. Identifiability and Failure Rules

The precipitation prior is one soft constraint in the three-parameter Layer 1 problem. It does not independently identify `prcp_fac`, `melt_f`, and `temp_bias`.

- Geodetic MB constrains the long-term balance combination.
- The precipitation prior supplies the accumulation-side direction.
- ITS_LIVE velocity supplies the turnover direction once.
- `temp_bias` remains a climate-axis parameter and must not absorb an unresolved precipitation product mismatch without a source-stamped diagnostic.

If the posterior lies outside the nominal prior interval but inside `[0.1, 10]`, retain the glacier with a `prior-tension` flag and wider posterior. If it hits `[0.1, 10]`, mark `calibration-boundary-hit` and stop that glacier before Stage 2.

No LIA geometry, LIA timing, centennial path, or modern endpoint residual may modify this prior or the calibrated annual-scale parameters.

## 7. Validation Hooks

Before Stage 1 execution, every glacier must pass:

1. `climate_historical.nc` reports a W5E5/GSWP3_W5E5 baseline source.
2. Monthly coverage includes all required 1979-2019 winter months.
3. The computed `p0` is finite and the pre-clip value is recorded.
4. The RGI prior class matches the frozen 1,205-row registry.
5. No parameter value is copied from a CRU `mb_calib.json`.
6. Climate cache and output paths resolve under `E:`.

## 8. Artifacts

- `data_raw/climate/precipitation_prior_registry.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/precipitation_work/precipitation_prior_registry_1205.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/precipitation_work/precipitation_prior_summary.json`

This decision replaces the inventory placeholders for both precipitation prior and modern calibration baseline.
