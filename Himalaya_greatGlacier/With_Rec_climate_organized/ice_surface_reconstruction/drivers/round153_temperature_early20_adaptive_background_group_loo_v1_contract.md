# Round 153 Adaptive-Background Group Leave-Out Contract

## Question

Is the proxy-value-driven adaptive-background warming identified in Round 152
concentrated in one pre-defined proxy group, or distributed across the network?

## Frozen groups

Evaluate proxy6 and proxy9 separately. For each network, retain the original
1901--1920 observations and HXb and omit one complete group at a time:

- longitude: west `<80E`, central `80--90E`, east `>=90E`;
- temperature PSM coefficient: `b1<0`, `b1>=0`;
- precipitation PSM coefficient: `b2<0`, `b2>=0`;
- archived PSM significance: `p<=0.05`, `p>0.05`.

Groups from different families overlap and their effects must not be summed.
For each omission, recompute only the deterministic AO/HGAO adaptive background.
Keep the other proxy network and the fixed O-EnKF background unchanged. Define
the group warming effect as `full six-member trend - leave-group-out trend`.

## Gates and boundary

- The no-omission proxy6/proxy9 trends must reproduce Round 152 within `1e-10`.
- Metadata row counts and record order must match the archived proxy arrays.
- Never request or decode 1921--1939; do not access Jionglacuo.
- This is a nonlinear sensitivity diagnostic. Leave-out effects are not additive
  causal contributions and do not authorize deleting proxies or changing PSMs.
- No EnKF rerun, parameter fit, production reconstruction, or OGGM run.
