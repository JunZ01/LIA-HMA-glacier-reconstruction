# Two-Dimensional GPU Validation Route

Model family: modified planform-2D GPU dynamics.

Purpose: test planform spreading, neighboring-cell interaction, tributary
coupling, numerical conservation, and transferability of the 1-D parameter
priors on a representative subset.

Current canonical evidence:

- discovery Stage B: 13 short-qualification passes;
- systems 2086 and 2534: technical retry decision required;
- no accepted-count effect;
- no adaptive spin-up, 50-year steady, or historical launch authorized.

Results from this route must be reported separately unless a formal model-
family equivalence or disclosed pooled-analysis contract is registered.

Evidence namespace: `two_d_gpu_validation`. Every artifact must identify
`model_family=modified-planform2d-gpu`; 1-D state hashes and 1-D acceptance
labels are never valid 2-D state or acceptance evidence.

Route-specific rules:

- parameter priors may be transferred only as disclosed hypotheses;
- geometry, planform length/separation, mass closure, exterior-zero ice,
  checkpoint restoration, timestep, and free-domain gates remain 2-D-defined;
- systems 2086 and 2534 are unresolved technical failures, not scientific
  failures and not passes;
- system 4588 remains the registered Stage A scientific failure in the frozen
  discovery16 denominator and is not replaced.

Stop on any global rule in `../shared/DECISIONS_AND_STOP_RULES.md`, any attempt
to bypass the exact-contract retry decision, any member-specific rescue, or
any attempt to proceed from short qualification directly to steady or
historical execution. The next legal action is the explicit retry decision or
a read-only audit that directly resolves it.

The next-version benchmark proposal is registered in
`../shared/GPU_SPINUP_BENCHMARK_V1.md` with its machine-readable manifest in
`../shared/gpu_spinup_benchmark_v1.json`. It fixes the sole member to system
705 and does not alter the unresolved status of 2086, 2534, or 4588. The
proposal has zero launch authority until a complete canonical transaction is
explicitly approved.
