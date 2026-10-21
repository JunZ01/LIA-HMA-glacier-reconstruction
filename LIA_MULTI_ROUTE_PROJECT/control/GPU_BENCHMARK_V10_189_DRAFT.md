# GPU benchmark transaction v10.189 — UNAUTHORIZED DRAFT

Status: `draft-only`; `launch_authorized=false`; no model run is authorized by this file.

## Frozen scientific member

- sole representative member/system: `705`
- `melt_f=10.072768920083137`
- `prcp_fac=3.672361860575085`
- `glen_a=3.4103970553475186e-23`
- `fs=0`
- `temp_melt=-1.0 degC`
- `base_temp_bias=0`
- `delta_t=0.1682996858927377`
- precipitation anomalies disabled; equilibrium-centred `1441–1470 MonthlyTI`
- solver contract: source-half / masked-frozen-diffusion / source-half; timestep limits `0.03 / 0.001 / 86400 s`; adaptive error ceiling `0.05`

No member or scientific parameter may change inside this transaction. Batch lanes are identical technical replicas, not additional scientific members.

## Exact scope and order

1. Five sequential short consistency, memory, and throughput batches: batch sizes `1`, `2`, `4`, `8`, `16`; duration `0.1 model year` each.
2. Exactly one `30 model year` performance batch at the largest short-test batch size that passes every numerical gate while peak VRAM remains below `90%`. If none passes, stop.
3. Exactly one `300 model year` production-grade checkpoint/resume validation at batch size `1`, interrupted at year `150`, restored from that checkpoint, and continued to year `300`.

This is `7` logical batches (`5 + 1 + 1`) and `8` execution segments because the 300-year chain has pre- and post-checkpoint segments.

## Gates, downgrade, isolation, and metrics

- Numerical consistency: all values finite; thickness non-negative; thickness outside the glacier mask exactly zero; adaptive error `<=0.05`; relative cumulative mass residual `<=1e-10`; versus batch-size-1 reference, final thickness max absolute difference `<=1e-10 m` and final area/volume relative difference `<=1e-12`; elapsed time and minimum-step counts must match exactly. Checkpoint restore must reproduce saved state bitwise before continuation.
- OOM rule: an OOM at batch size `N` fails `N`, skips every larger size, and retains only the largest smaller passing size. No automatic retry. OOM at batch size `1` stops the transaction. If the selected 30-year size OOMs, allow one deterministic downgrade to the largest smaller short-test pass; a second OOM stops the transaction.
- Failure isolation: a failed lane invalidates only its current technical batch and cannot promote results; failure of the 30-year or 300-year single chain stops that stage and authorizes no downstream work.
- Capture per segment: device/model and software versions, input/config hashes, batch size and lane count, wall time, model-years/s, lane-years/s, peak and reserved VRAM, GPU utilization, adaptive-step statistics, mass residual, final area/volume, thickness extrema, checkpoint hashes, exit status, and stderr/OOM classification.
- Stop immediately on any numerical-gate failure, NaN/Inf, negative thickness, mask leakage, hash/config drift, checkpoint mismatch, unexpected member count, VRAM `>=90%`, unhandled exception, or scope expansion.

Explicitly excluded: parameter matrices, 3000-year full runs, forward runs, 50-year steady-state runs, and historical chains.

## Approval boundary

Only an explicit user approval of this exact versioned transaction may change its authorization state. Approval must not be inferred from this draft or from controller-validator success.
