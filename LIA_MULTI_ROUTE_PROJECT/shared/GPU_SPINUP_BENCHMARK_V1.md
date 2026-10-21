# Minimal GPU Spin-Up Benchmark V1

Status: proposed next-version transaction; not authorized and not launched.

## Frozen Scope

- Route: `two_d_gpu_validation`
- Model family: `modified-planform2d-gpu`
- Unique scientific member list: `[705]`
- Batch sizes, in order: `[1, 2, 4, 8, 16]`
- Short technical duration: `0.1 model year` per batch size
- Production-grade validation: one batch-size-1 chain from year 0 to year 300,
  with a planned year-150 interruption, exact checkpoint restore, and
  continuation to year 300
- No parameter matrix, 3000-year run, forward run, 50-year steady run,
  historical chain, accepted-count action, or member substitution

Batch size means identical vector lanes inside one GPU process. It does not
mean concurrent independent processes. Every lane uses the same system-705
state, forcing, grid, solver, and scientific parameters. Extra lanes are
technical replicas only and create no additional scientific sample or
fingerprint.

## Why System 705

System 705 passed both registered Stage B phases and every input, numerical,
mass, exterior-zero, checkpoint, and source-immutability gate. Its observed
Stage B peak PyTorch reserved memory was 662 MiB, the largest among the 13
passing members. It is therefore the single conservative capacity
representative. Systems 2086 and 2534 remain unresolved technical failures;
system 4588 remains the Stage A scientific failure. None is included.

## Frozen Inputs and Scientific Parameters

- Adapter shape: `1208 x 1560`
- Geology SHA256:
  `356ADD8554F8F265CB7285E0B01C22DA81DA9B6CC06996A40928761BA45A946D`
- Source thickness SHA256:
  `A1623FD453EEE6B4011F3E93FD9402C9CCE8C97677706FC4CAC5B7BD27403893`
- Adapter initial thickness SHA256:
  `8E9AA611A7A22D0D7CA01E0B8FC28B90B9C13F55739001D368F95D0A386A1120`
- Parent climate: exact equilibrium-centered 1441-1470 MonthlyTI package,
  precipitation anomalies off
- Climate-package SHA256:
  `74EC208F15CE15F5B31DF46203D91B14083293748B6E2F530AD0F802403FF8AF`
- `melt_f=10.072768920083137`
- `prcp_fac=3.672361860575085`
- `glen_a=3.4103970553475186e-23 Pa-3 s-1`
- `fs=0`
- `temp_melt=-1.0 degC`
- `base_temp_bias=0 degC`
- Equilibrium correction `delta_t=0.1682996858927377 degC`
- Operator:
  `fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1`
- Solver core: source-half / masked-frozen-diffusion / source-half
- Initial/minimum/maximum timestep: `0.03 / 0.001 / 86400 s`
- Accepted adaptive error: relative L2 at most `0.05`

No dtype, grid, mask, thickness, physical constant, forcing, solver, timestep,
or gate may change inside this transaction.

## Phase T: Five Short Technical Batches

Run batch sizes 1, 2, 4, 8, and 16 sequentially. Each starts from the same
immutable source-derived thickness and runs the first 0.1 model year of the
same equilibrium-centered parent climate. Batch 1 is the numerical reference.

Required consistency thresholds:

- source and configuration hashes: exact;
- finite and nonnegative thickness: required in every lane;
- exterior-domain thickness: exactly zero;
- maximum accepted adaptive error: at most `0.05`;
- relative cumulative mass-budget residual: at most `1e-10`;
- maximum absolute final-thickness difference between any lane and batch-1:
  at most `1e-10 m`;
- relative final volume and area difference from batch-1: at most `1e-12`;
- accepted/rejected step counts, final elapsed time, and minimum accepted
  timestep: exact across identical lanes and batch sizes;
- per-lane initial and final state hashes: recorded; identical lanes must have
  identical hashes after canonical serialization.

The short outputs are technical benchmark evidence only. They may not be
scored against LIA/RGI observations or used as spin-up convergence evidence.

## GPU Memory and Throughput Rules

Collect for every batch:

- wall time, warm-up time, and timed-section wall time;
- lane-years per hour, accepted lane-steps per second, and aggregate throughput
  gain relative to batch 1;
- per-lane accepted/rejected steps, minimum timestep, and maximum accepted
  error;
- PyTorch peak allocated and reserved memory;
- external GPU used/total memory, utilization, power, temperature, clock, and
  telemetry sampling cadence;
- GPU model, driver, CUDA and PyTorch versions;
- CPU utilization, process RSS, host free memory, PID, command, start/end
  times, stderr bytes, output root, and all input/output hashes.

GPU memory is a hard capacity gate at 90% of physical VRAM. Throughput has no
post-hoc promotion threshold: report the measured curve and retain the largest
numerically passing batch size below the memory gate only as a benchmark
result, not as production authorization.

## OOM Degradation and Failure Isolation

- Execute batch sizes strictly in ascending order and one process at a time.
- If batch size 1 OOMs, stop the transaction.
- If batch size N OOMs or reaches 90% VRAM, mark N as failed, do not run any
  larger size, and retain the largest smaller passing size as the measured
  capacity ceiling.
- Do not retry an OOM with changed dtype, grid, physics, forcing, solver, or
  thresholds. A retry requires a new versioned technical decision.
- Every lane has a separate status and atomic output namespace. A lane failure
  must not overwrite or mutate another lane.
- A single non-OOM lane failure may allow other lanes to write terminal
  diagnostics, but the whole batch fails because the lanes are identical; stop
  all larger sizes and allow no automatic retry.
- An orphan process, output collision, cross-lane state mutation, or failure to
  isolate stderr is an immediate hard stop.

## Phase P300: One Checkpoint/Resume Validation

Phase P300 is eligible only after Phase T is terminal, batch size 1 passes all
numerical gates, and no unresolved non-OOM consistency or isolation failure
exists. A higher-size OOM alone does not block the batch-size-1 P300 chain once
the GPU is confirmed healthy and cleared.

Run exactly one system-705 lane at batch size 1:

1. start from the unchanged source-derived thickness;
2. repeat only the frozen 1441-1470 equilibrium-centered parent climate;
3. advance to year 150 and write an atomic checkpoint;
4. terminate the process intentionally;
5. validate manifest, state, counter, forcing-phase, and source hashes;
6. resume once from that checkpoint and advance to year 300;
7. write one final year-300 production-format checkpoint and stop.

The year-150 restore must be bitwise exact. Elapsed time, accepted/rejected
step counters, cumulative source, and cumulative mass residual must continue
monotonically with no reset or duplicate history rows. The year-300 state must
pass the Phase T numerical, domain, and mass thresholds. This chain does not
test equilibrium convergence, does not continue to year 600 or 3000, and is
not eligible for forward evaluation.

## Batch Count

- Five short technical batches: one each at batch size 1, 2, 4, 8, and 16.
- One 300-year validation batch: one scientific lane, deliberately executed in
  two process segments around the year-150 checkpoint.
- Total: six batches and seven process segments if no earlier stop rule fires.

## Global Stop Conditions

Stop before launch or stop the affected stage if:

- the canonical validator does not pass or launch authorization is false;
- any frozen input, code, route, or configuration hash differs;
- an output root or fingerprint already exists;
- GPU prelaunch telemetry is missing or an unrelated project GPU process would
  invalidate isolation;
- any numerical consistency, mass, finite/nonnegative, exterior-zero,
  checkpoint, source-immutability, OOM, or failure-isolation rule fails;
- execution would require a parameter, member, batch-size sequence, duration,
  forcing, solver, threshold, or scope change.

No result from this benchmark changes the accepted count or promotes the 2-D
route.

## Exact Authorization Still Required

请明确授权执行版本事务 `gpu-spinup-benchmark-v1`：仅对系统 705 按 batch size 1/2/4/8/16 运行五个 0.1 年技术基准，并在其通过后以 batch size 1 运行唯一一条 300 年、150 年中断续跑的 GPU spin-up 检查链；不授权任何其他模拟。
