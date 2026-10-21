# Proxy9 Key-Record Audit

This independent experiment audits proxy9 records 203 and 241 and recomputes
target/HMA EnKF curves for bounded combinations of the verified nearby proxy9-
added records. It does not modify the preceding controlled or holdout runs.

The experiment asks whether the 3584 sensitivity is dominated by records 203
and 241, by the other nearby records, or by their simultaneous covariance.
Original inputs remain read-only, `rec_ref` is comparison-only, and no OGGM
simulation is permitted.
