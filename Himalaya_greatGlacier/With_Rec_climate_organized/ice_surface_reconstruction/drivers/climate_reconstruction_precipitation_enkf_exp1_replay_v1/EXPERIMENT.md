# HMA Precipitation EnKF Experiment-1 Replay v1

This independent no-OGGM experiment replays exactly one archived branch:

- state: precipitation in mm/day;
- proxy network: proxy6;
- method: deterministic square-root EnKF;
- experiment: 1 of 50;
- years: 1000-2000;
- prior: the frozen R2024b `336 x 848` precipitation pool from the preceding
  recovery preflight;
- members: the archived precipitation experiment-1 indices;
- proxy observations and `HXb`: frozen `proxy_data_6.mat`;
- residual completion: the previously recovered exact proxy6 RegEM matrix;
- update: the original `M_update.m`, `infl=1`, no localization;
- diagnostics: posterior mean, standard deviation, and 5/50/95 percentiles.

The experiment compares all 1001 years and 336 state cells against the
archived precipitation `EnKF_results.mat`. It does not run AOEnKF or HGAOEnKF,
change any parameter, use `rec_ref`, or validate the precipitation product.

Passing means that the surviving current-loader chain is sufficiently close
for reconstruction redevelopment. It does not establish bitwise identity with
the missing May 17 loader/RNG snapshot.

