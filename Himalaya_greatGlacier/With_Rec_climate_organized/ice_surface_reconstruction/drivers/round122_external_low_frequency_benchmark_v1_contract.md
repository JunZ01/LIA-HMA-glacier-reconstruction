# Round 122 External Low-Frequency Benchmark Contract

## Question

Do newly acquired, independently published Tibetan Plateau temperature and
precipitation reconstructions support or contradict the regional low-frequency
phase and trend of the current equal-six HMA reconstruction?

## Frozen Design

- Read the current public reconstruction only at 1500-1920 and 1940-2000.
  Never read 1921-1939 values from either the current field or a benchmark.
- Use `tas_ensemble_mean_rec_result` and `pr_ensemble_mean_rec_result` without
  changing the production files.
- Compare eastern benchmarks to the frozen `east >= 90E` regional mean and
  central benchmarks to `80E <= central < 90E`. Keep the same HMA mask and
  area weighting used by Round 115/116.
- Temperature benchmarks: 2019 NETP annual and SETP JJA reconstructions, 2025
  eastern Tibetan June-September reconstruction, 2026 southeastern Tibetan JJA
  maximum-temperature reconstruction, and the Aweng Co, Heihai Lake, and
  Ngamring Co low-resolution temperature reconstructions.
- Precipitation benchmarks: the 845-2023 northwestern Qinghai-Tibet May-July
  reconstruction and the Shenqi Cave precipitation index.
- Seasonal or index benchmarks are compared by correlation, standardized phase,
  and trend direction only. They are not interpreted as annual absolute bias or
  converted to the production units.
- Annual series receive a centered 31-year low-pass comparison. Sparse records
  are compared only at their dated samples. No interpolation of a benchmark is
  used to manufacture annual observations.

## Boundary And Decision

This is an external consistency diagnostic, not fitting, tuning, or promotion.
No benchmark enters EnKF, no gate changes, no localization, and no OGGM run.
The output must report each benchmark separately; disagreement may identify a
regional/time-scale problem but cannot by itself assign error to one source.
