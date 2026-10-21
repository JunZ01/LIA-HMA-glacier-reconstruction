# Six-station ordinary-EnKF innovation/gain attribution contract

- Explain the positive 1901-1920 assimilation increment at the six admitted GHCN-M station coordinates.
- Analyze archived ordinary-EnKF proxy6 and proxy9 members only; keep the fixed prior, pools, proxy observations, PSM operators, and RegEM error variances unchanged.
- Reconstruct all 50 archived pool contributions and require agreement with the archived O-member increment before interpreting proxy rankings.
- Center each proxy contribution on 1961-1990, then aggregate 1901-1920 only over each station's finite qcf years and give each admitted station equal weight.
- Rank and group contributions without deleting, tuning, or reweighting any proxy.
- Read only 1901-1920 and 1961-1990. Never request or access 1921-1939.
- This is a read-only mechanism diagnostic, not an EnKF rerun or production correction.

