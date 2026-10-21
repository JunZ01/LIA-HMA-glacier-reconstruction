# Round 117 Documented-Moisture Precipitation Residual OOF Contract

## Question

Can the 25 Round 110 proxy9 evidence groups with documented moisture response
improve annual HMA precipitation around the equal-six reconstruction under the
full Round 113 modern gates?

## Frozen Design

- Years and outer blocks: 1940-2000 and the five Round 113 blocks.
- Predictors: proxy9 groups with
  `documented_network_role_v4 == documented_moisture` and
  `operator_metadata_ready_v4 == True`. No outcome-based proxy selection.
- Observations: CRU and GPCC annual precipitation anomalies. Use only HMA state
  cells where both products are finite for every validation year. Report but do
  not fill all-product-missing cells.
- Background and legacy comparator: `pr_ensemble_mean_rec_result`.
- Target in each fit: the training residual
  `unweighted(CRU, GPCC) mean - equal-six`.
- Reuse the Round 115/116 training-only standardization, mean imputation,
  area-weighted EOF algorithm, minimum 20 finite training years, inner blocked
  selection, rank grid 1/2/3/5/8, ridge grid 0.1/1/10/100/1000, and deterministic
  tie rule.
- No blend factor, time trend, localization, product weight, proxy deletion,
  or threshold adjustment is permitted.

## Gates And Boundary

Apply every required Round 113 modern relative, absolute, annual-correlation,
and spatial gate against equal-six and zero anomaly. No 1901-1920 value is used
for fitting or selection; 1921-1939 is never requested; the Round 111/114 paleo
reserve remains unopened. A full pass advances to a separately contracted
1000-2000 precipitation candidate. Failure closes annual linear residual
correction with the documented-moisture network and requires a season-aware
operator before any further precipitation candidate. OGGM remains prohibited.
