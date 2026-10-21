# Controlled EnKF Early-Twentieth-Century Experiment

This experiment creates a new, fully specified reconstruction from the
recovered fixed experiment-1 prior. Proxy6 and proxy9 use the same 500 prior
members and the same EnKF mean-update implementation. Their archived proxy
operators, observations, and separately verified RegEM residual matrices are
the only proxy-set-specific inputs.

The experiment saves innovations, solved observation-space weights, target and
regional per-proxy increments, and complete closure diagnostics. `rec_ref` is
used only as a comparison curve. No glacier trajectory, OGGM parameter, or old
proxy9 posterior is used to tune the reconstruction.

This is a mechanism diagnostic, not a replacement climate product. Expansion
to all 50 fixed-prior experiments requires a separate decision after the
experiment-1 closure and proxy ranking are reviewed.
