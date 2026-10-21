# Round 163 Temperature Prior-Mode PSM Observability Contract v2

This contract preserves every scientific input, threshold, diagnostic, and
interpretation rule in v1. V1 was not run after preflight found that it performed
two separate SVD calls when pairing prior-mode truth energy with mode
observability. V2 uses one SVD result for `U`, singular values, and `Vt`, preventing
any ambiguous rotation within numerically degenerate modes.

No proxy, member, PSM, residual variance, region, threshold, or scientific gate is
changed. The full v1 contract remains incorporated by reference.

