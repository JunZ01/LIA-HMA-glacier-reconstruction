# Early-Twentieth-Century Proxy Increment Attribution

Question: can the 1901-1950 low-frequency phase mismatch near LIA system 3584
be traced through exact EnKF proxy-update contributions for both proxy sets?

This experiment first audits replay provenance. It does not run OGGM, modify
source reconstruction files, or tune climate against glacier curves.

Only after each proxy set independently closes its experiment-1 posterior may
the next stage calculate `K x innovation` terms. A member or proxy-set mean is
not treated as a causal per-proxy attribution.
