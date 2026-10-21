# Proxy9 EnKF Experiment-1 Replay

This independent stage reuses the recovered frozen prior and exact MATLAB
replay function, but replaces only the proxy observation operator, completed
RegEM residual matrix, and archived proxy9 posterior target. It runs no OGGM
model and does not alter source archives.

Only a pass against the documented posterior-closure gates permits later
per-proxy `K x innovation` attribution.
