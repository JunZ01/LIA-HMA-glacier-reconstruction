# Independent 1-D GPU production candidate v1

This is a fixed-manifest production integration, not a new glacier solver and
not a promotion to the CPU mainline. No two-dimensional or multi-tributary model
is implemented here. The supported geometry remains the existing single
rectangular pseudo-flowline per glacier system.

## Entry point

From the project root:

```text
D:/Anaconda/envs/oggm_env2/python.exe -B Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_lia_gpu_production_v1.py --check
```

The default and `--check` modes are read-only and do not import Torch or run a
model. Add `--plan <frozen-plan.json>` to verify its full input/implementation
closure. There is no unguarded model mode. `--run` additionally requires an exact
authorization file, a reviewed release decision and a matching independent
launch-authorized research branch in the project controller state.

## What is integrated

- The already-tested `active-prefix-v1` float64 dynamics and calendar are reused.
  The original solver files are not edited. Five checked execution-loop
  substitutions select that backend, expose the spinup end/confirmation policy,
  and remove scoring from the GPU process. The step arithmetic is not rewritten.
- A fixed member manifest replaces the old four-group search. There is no
  automatic parameter expansion, beam selection or architectural 512-lane cap.
  Batch size is configurable; increasing it does not change scientific keys.
- Full raw per-member state/results are saved during GPU execution. The GPU
  child must completely exit and its artifacts, member inventory and exit
  evidence must pass verification before the CPU-only scoring child starts.
- The existing six-gate scoring policy is reused. The old 1470-to-2000 direction
  Boolean stays diagnostic only. Scores are provisional, never formal acceptance.
- Durable queue records, single-writer locks, status-write fallback snapshots,
  immutable output and independent terminal recount protect completed work.
- Explicit recovery validates the original plan, frozen raw/checkpoint files,
  unchanged scientific scope and core implementations, and source GPU exit.
  It copies validated bytes to a new independent output; completed models are
  not recomputed. Recovery is separately authorized, never automatic.
- OOM uses `durable-stop-no-auto-retry`: preserve completed results and periodic
  checkpoints, stop the GPU phase, and do not score an incomplete GPU inventory.
  A later explicit recovery can use a smaller configured batch. Other isolated
  member failures retain the inherited stop rule: over 15% technical failures
  after at least 120 terminal members stops that stage. Total storage failure
  cannot guarantee continuation and is reported as a technical failure.

## Scientific policy is explicit

The supported existing GPU policy has 3%/300 m stability windows with explicit
confirmation pairs, float64, a 108000 s maximum timestep, OGGM-style monthly
landings/annual MB cache, original 1470–2000 historical forcing and 5000 m domain
margin. Spin-only delta is removed at the historical transition, as in the
existing state factory; any endpoint ramp is an explicit member parameter.

These are independent GPU policies, not a claim of general CPU OGGM equivalence
or an implicit adoption of the active CPU20 transaction. The supplied candidate
plan runs forward and scoring only, so its disclosed inherited spinup policy
does not authorize or initiate another spinup.

## Concrete next integration candidate

`prepare.py` revalidates and prints a candidate plan; it writes no output and
grants no authority. It selects 375 frozen qualified parents from the existing
512-parent inventory: excludes the original 128 full historical trajectories
and another nine verified complete histories. Deduplication is explicitly
limited to the named independent GPU output roots in the preparation evidence.

Each selected parent already has a 1471 profiling output. Those files lack the
complete annual history and are not valid historical recovery checkpoints.
The proposed first complete 530-year histories therefore start from their
original frozen parents, preserving the earlier profiling evidence. No spinup,
137 completed-history rerun, parameter search, CPU control or accepted-ledger
change is part of this proposal. The initial configurable batch limit is 128;
it is not a hardware ceiling. Output is a new `wgprod-v1` root, with a six-hour
hard total phase budget, GPU allocation ceiling 80%, and one CPU library thread.

## Release boundary

The no-model tests validate software control and storage, not glacier physics or
GPU throughput. The prepared implementation still requires one explicitly
authorized numerical integration acceptance. Prior numerical, long-checkpoint
and status-survival evidence must be listed with a compatibility review; their
existence alone is not a passing release decision. Formal production mode also
requires a recorded final numerical integration pass.

Further CUDA Graph, compression, two-dimensional or tributary-network research
is not a prerequisite for releasing this supported one-dimensional tool.
