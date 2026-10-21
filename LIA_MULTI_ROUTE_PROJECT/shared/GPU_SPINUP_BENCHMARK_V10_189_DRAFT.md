# GPU spin-up benchmark v10.189 draft

Status: proposed, not launch-authorized

Model family: `modified-planform2d-gpu`

Purpose: determine the highest stable GPU batch size before any spin-up matrix
or forward production launch. This benchmark has no accepted-count effect.

## Fixed members

Use only the 13 existing Stage B technical passes:

`988, 3185, 801, 4959, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559`

Systems `2086`, `2534`, and `4588` are excluded from this benchmark. Their
registered technical/scientific classifications are unchanged.

## Execution stages

1. Technical consistency: fixed states and fixed scientific parameters; test
   requested batch sizes `1, 2, 4, 8, 16` over the shortest existing numerical
   fixture. Batch size 16 is an effective batch of 13 members.
2. Throughput: run the same 13 members for 30 model years at every technically
   valid batch size.
3. Production-grade checkpoint test: run one 300-year block only at the selected
   batch size, restore its atomic checkpoint, and verify continuation identity.

## Selection metric

Select the batch size with the greatest completed model-years per wall-clock
hour subject to all consistency, memory, checkpoint, and member-isolation gates.
Single-chain speed is secondary.

Record wall time, completed model-years, throughput, peak GPU memory, mean and
peak GPU utilization, CPU utilization, accepted/rejected solver steps, minimum
timestep, per-member status, checkpoint hash, and restored-state differences.

## Hard rules

- Do not change climate, geometry, solver, timestep policy, precision, physical
  parameters, scientific gates, or member identities between batch sizes.
- A member failure is isolated and reported; it must not erase completed sibling
  evidence.
- On CUDA OOM, record the failed batch size and continue once at the next lower
  registered size. No hidden retry or parameter change is allowed.
- Numerical consistency thresholds must use the existing model-family fixture
  contract. If that threshold is absent or ambiguous, stop before launch.
- Do not launch the parameter matrix, a 3000-year spin-up, forward runs, 50-year
  steady runs, historical runs, or the unresolved retries for 2086/2534.
- Every output root and fingerprint is immutable and versioned.

## Stop conditions

Stop immediately on controller-validator failure, missing fixture tolerance,
input-hash mismatch, checkpoint restoration failure, non-isolated batch failure,
or any attempted scientific-contract drift.

## Authorization boundary

This draft grants no launch authority. A canonical versioned transaction and an
explicit user approval are required before the first numerical fixture starts.
