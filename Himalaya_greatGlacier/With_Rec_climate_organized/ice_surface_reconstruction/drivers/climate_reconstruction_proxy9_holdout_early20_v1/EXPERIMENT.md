# Proxy9 Holdout Early-Twentieth-Century Experiment

This independent experiment tests whether the controlled proxy9 reconstruction
at LIA system 3584 is stable to removal of selected proxy records.

The recovered experiment-1 prior, fixed 500-member indices, state coordinates,
EnKF implementation, proxy observations, proxy operators, and RegEM residual
matrix are unchanged. Each holdout removes the named rows from the proxy9
observation system and recomputes the complete simultaneous EnKF update. It is
not an algebraic subtraction of saved contribution curves.

The no-removal case must reproduce the saved controlled proxy9 posterior before
any holdout is run. Original source files and the controlled baseline result are
read-only. `rec_ref` is comparison-only, no OGGM run is permitted, and these
single-prior results are not a final climate reconstruction product.
