# Volume-tau pilot input handoff — 2026-10-21

Steps 0–2 are complete. This package contains **22 systems, not 27**. The supplied
selection rule was executed unchanged; no repeated or substitute systems were added.
There are 35 single-member systems in the 191-system population: Central Himalaya
19, East Himalaya 4, West Himalaya 12. The rule selects 9, 4 and 9 respectively.
All three existing controls (3632, 3131, 2906) are included.

The excluded multi-member population is 156/191 = 81.7%: Central 70, East 14,
West 72. A single-member pilot cannot establish performance for all 191 systems.
Multi-member export/replay support is needed before the proposed full-population
rerun. The actual 22-system selection is disclosed, not relabelled as the intended
balanced 27-system design; no route-level proceed/stop decision was made here.

## Files

- `npz/`: 22 `fullchain_inputs_<uid>.npz` files, exporter manifest, frozen selection
  and the exact batch exporter. All exports succeeded; NumPy `allow_pickle=False`
  loads all arrays, with no object arrays. SHA256 values are in the manifest.
- `selection/`: exact selection CSV and population-coverage JSON.
- `export_QC.json`: local data checks. Existing controls have identical numerical
  array values to the previous full-chain export. The new source manifest omits
  the additional earlier-control NPZ dependency; shared source hashes agree.
- `code/`: pilot runner, replay/calibration scripts and selection/export tools.
  The complete package imports remain in the GitHub research branch.
- `protocol_snapshot.json`: supplied gates and actual preparation scope.

No local tau optimization, network/PP reconstruction, LIA reconstruction, climate
or dynamic run was performed. The volume-tau results included in the received
patch are externally supplied results, not locally reproduced calculations.

## Continue from the research repository

Update `research/extension-operators-20261021`, then run with SciPy/NumPy/pandas
and the repository imports available (locally, use `oggm_env2`):

```
python run_volume_tau_pilot_20261021.py H:/ToYangxin/result_yx/volume_tau_pilot_20261021/npz H:/ToYangxin/result_yx/volume_tau_pilot_20261021/pilot
```

The pilot runner has been corrected to use all IDs in `npz/pilot_systems.csv`
as its denominator. Missing/failed exports count as failures instead of silently
disappearing. The numerical gate thresholds were not changed. This input package
does not contain the three requested pilot result files, because step 3 has not run.

Volume closure remains calibration rather than independent validation. The three
existing controls have already informed development; the thresholds are frozen
for the new cases, not retrospectively independent of those earlier controls.
The surface gate currently uses the raw interpolated surface RMSE before clipping,
as in the supplied replay. There is no production promotion in this preparation.
