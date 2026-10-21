# Round 152 Adaptive-Background Factorial Contract

## Question

Why does the archived adaptive AO/HGAO background create a positive eastern-HMA
temperature trend during 1901--1920?

## Frozen 2 x 2 design

For proxy6 and proxy9 separately, use the original recovered prior, the original
50 experiment pools, archived HXb, and only proxy observations from 1901--1920.
Construct four deterministic adaptive-background branches:

1. `dynamic_support_dynamic_values`: archived annual availability and values.
2. `common_support_dynamic_values`: proxies finite in every one of the 20 years,
   with their archived annual values.
3. `dynamic_support_frozen_values`: archived annual availability, but each
   available proxy is assigned its own 1901--1920 mean value.
4. `common_support_frozen_values`: common support with each proxy's fixed mean.

The last branch is the constant-pattern control. Define exact factorial terms:

- support = branch 3 - branch 4;
- values = branch 2 - branch 4;
- interaction = branch 1 - branch 2 - branch 3 + branch 4.

Their sum plus branch 4 must reproduce branch 1 annually and in trend. Combine
proxy6, proxy9, and the fixed O-EnKF background with the archived six-member
weights, and require the full branch to close to Round 151 within `1e-10`.

## Boundary

- Never request or decode 1921--1939.
- Do not change proxy values, HXb, prior members, experiment pools, spatial
  masks, or branch weights outside the four declared counterfactuals.
- This is a nonlinear selection-mechanism diagnostic, not a replacement prior,
  PSM validation, EnKF rerun, parameter fit, or production candidate.
- Do not open Jionglacuo and do not run OGGM.
