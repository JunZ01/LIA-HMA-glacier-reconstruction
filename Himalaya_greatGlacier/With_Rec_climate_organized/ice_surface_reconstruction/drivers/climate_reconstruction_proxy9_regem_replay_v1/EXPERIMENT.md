# Proxy9 RegEM Replay

This standalone, read-only experiment completes the archived proxy9 residual
matrix using the recovered original RegEM implementation. It is a prerequisite
to, not a substitute for, a later proxy9 EnKF experiment-1 replay.

Success requires zero missing output residuals and no change greater than
`1e-12` in any originally observed residual. No OGGM model or glacier curve
is involved.
