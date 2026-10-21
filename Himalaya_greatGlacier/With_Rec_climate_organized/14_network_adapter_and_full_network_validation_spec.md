# LIA network-solver adapter and full-network Level B validation

**Artifact version:** v0, 2026-07-22

**Status:** executable implementation spec for wiring `solve_flowline_network`
into Level B. No code changed while authoring this spec. Supersedes the earlier
(incorrect) "one OGGM flowline -> one FlowlineBranch" plan.

## 1. Why this spec exists

The Level B driver
[`validate_modern_profiles.py`](ice_surface_reconstruction/drivers/validate_modern_profiles.py)
currently solves only the single outlet main flowline per system
(`main_flowline_index` requires exactly one `flows_to is None`; tributaries are
silently ignored). The target systems are multi-tributary great-glacier
complexes, so the accepted operator must solve the whole flowline tree via
`solve_flowline_network`. A single-branch equivalence has already been proven
(69 real main-flowline combos, max surface diff 0.000 m; 2 synthetic
equivalence tests, total suite `23 passed`; helper functions are not tests),
so wiring the network solver cannot change single-flowline baselines.

The naive adapter is wrong and must not be implemented: **one OGGM flowline is
NOT one `FlowlineBranch`.**

## 2. The blocking topology error

`solve_flowline_network`
([core/network.py](ice_surface_reconstruction/core/network.py)) only transfers
surface elevation at a branch's two endpoint nodes; it does not expose interior
node surfaces. But OGGM's `flows_to_indice` points at an **interior** node of
the parent flowline. Therefore the parent flowline must be **split at every
junction index** into segments, so tributaries can inherit the parent surface
at the correct location.

Verified on real system `RGI60-15.01141`:

```
OGGM native:  fl0(nx=15) -> fl2[35]      fl1(nx=8) -> fl2[4]      fl2(nx=91) -> outlet
correct graph: 3 flowlines -> 5 branches (parent in 3 segments + 2 tributaries) -> 2 junctions
```

## 3. Geometry and index conventions (pinned)

Confirmed against `RGI60-15.01141`; these definitions are fixed and must not be
re-derived per implementation:

- **OGGM native index direction:** index 0 = upstream head (highest),
  last index = downstream (parent last point is the terminus, elev 3460 m).
  Index increases downstream.
- **Tributary junction end = its LAST native point** (`surface_h[-1]`, the
  lower end). Its head is native index 0.
- **Driver distance order:** the driver reverses with `[::-1]` to run
  terminus->upstream, so reversed distance index = `n - 1 - native_index`.
  A junction at native `flows_to_indice` maps to reversed index
  `n - 1 - flows_to_indice` (fl2[35]->55, fl2[4]->86 for nx=91).
- **Coordinate units (pinned, do NOT confuse):** OGGM `line.coords` are in
  **grid-cell units**, not metres. Adjacent line points are `dx` cells apart
  (`dx=2` for this system: `map_dx=66 m/cell`, `dx_meter = dx * map_dx = 132 m`).
  A raw coordinate distance is therefore in grid cells; multiply by `map_dx`
  for metres, divide by `dx` for flowline steps. Dividing a coordinate distance
  by `dx_meter` is a units error (grid ÷ metres) and is forbidden.
- **Junction geometric-distance metric (pinned):** Euclidean distance between
  the tributary's LAST native point and the parent's `flows_to_indice` point.
  Report **all three units**, never a single ambiguous "cell":
  - `junction_distance_grid_cells` = raw coordinate distance;
  - `junction_distance_m` = grid cells × `map_dx`;
  - `junction_distance_flowline_steps` = grid cells ÷ `dx`.
  Measured on `RGI60-15.01141`: fl0->fl2[35] = 7.5595 cells = 498.9 m =
  3.78 steps; fl1->fl2[4] = 7.1865 cells = 474.3 m = 3.59 steps. The junction
  ends are ~475-500 m from the parent index, NOT coincident — the junction
  positional error is itself a quantity to record, not a rounding detail.
  On the harder `RGI60-15.06881` these gaps reach 1017-2334 m (median 1271 m).
- **Junction-transfer policy (pinned, v0 = `collapsed-transfer-node`):** despite
  the measured ~500-2300 m gap, v0 sets a **zero-length** connection — the
  tributary end inherits the parent junction surface directly. This is because
  OGGM `flows_to` is a flux-routing relation (`set_flows_to` trims near the
  parent, dynamics distribute flux via a kernel), not a geometric coincidence,
  so the gap is most likely a routing/discretisation artefact rather than a real
  ice segment. The collapse is DECLARED an approximation: the gap and surface/bed
  differences are recorded per tributary (Section 8), and full-network output is
  labelled structurally-approximate — it must NOT claim strict geometric
  continuity. `connector-branch` (a real connecting segment across the gap) is
  the v1 upgrade, triggered by Level B evidence on which junctions actually hurt
  accuracy, not built now. The policy string is stamped into every output.

## 4. Adapter (driver/adapters layer, NOT core)

The adapter converts an OGGM flowline list + bed into a `FlowlineBranch` list.
It carries the OGGM dependency; `core` stays OGGM-free.

Steps:

1. Establish stable node ids: one `outlet` per system; one junction node per
   distinct `(parent_flowline, flows_to_indice)`; **one head node per original
   OGGM flowline** (not per split segment). Two branches meeting the same
   junction MUST receive the identical node id, or the graph will not connect.
2. Split **every** flowline (trunk, intermediate, or leaf) at all junction
   indices of tributaries flowing INTO it, into ordered segments
   `outlet/parent-junction -> junctionA -> junctionB -> head`.
3. **General node rule** (a flowline can be both a tributary and a parent, so
   there is no trunk/tributary special case — verified on 7 of 12 pilot systems,
   e.g. `RGI60-15.06881` has 3 intermediate flowlines fl9/fl10/fl11):
   - the **most-downstream segment** (`lo == 0`) connects downstream to `outlet`
     if `flows_to is None`, else to the parent junction node
     `j:fl{flows_to}:{flows_to_indice}`;
   - every **other segment** (`lo > 0`) connects downstream to THIS flowline's
     own junction node at that cut, `j:fl{self}:{native_index}`;
   - upstream node is the head node if the segment reaches native head, else
     this flowline's junction at that cut.
   The only thing `flows_to` decides is what the `lo == 0` segment attaches to.
4. Convert native indices to reversed distance: `n - 1 - flows_to_indice`.
5. For every emitted branch, record provenance: source flowline number and the
   native index range it covers, so original-point coverage is auditable.

**Split determinism (pinned):**

- Process a parent's junction indices in **descending native index order**
  (downstream -> upstream), so segments are emitted terminus-first.
- **Multiple tributaries at the same parent index share ONE junction node; the
  parent is split there only once.** (Not present in the 12 pilot systems, but
  required for the 191-system production set.)
- A junction at an **adjacent** native index produces a legal **2-point**
  segment — this is allowed and must pass.
- A junction coinciding with a head (native `n-1`) or outlet (native `0`) would
  produce a single-point segment: v0 rejects it with
  `ValueError` classed `adapter-error:endpoint-junction-unsupported`. It never
  occurs in the pilot (0 of 45 junctions), so this is a defensive guard for the
  191-system production set; tests assert this exact status, not a vague
  "degenerate split".
- **Multiple `flows_to is None` (multi-outlet forest):** v0 **fails and logs
  the system** (`status = multi-outlet-unsupported`); it does not attempt a
  forest solve. All 12 pilot systems are single-outlet, so this never triggers
  on the pilot; it is a guard for production.

Pilot topology note: flowline counts range 1 (`RGI60-15.10689`, degenerates to
a single line) to 13 (`RGI60-15.06881`, the strongest multi-level stress test);
no pilot system has two tributaries at one parent index.

## 5. Adapter-only acceptance (before any driver change)

Assert on `RGI60-15.01141` with concrete numbers:

- yields exactly **5 branches, 2 junctions**;
- parent fl2 split into 3 segments; tributaries fl0, fl1 map to the fl2[35] and
  fl2[4] junction nodes;
- each segment's `(flowline number, native index range)` reassembles the
  original OGGM points with **no gap and no unexpected duplicate** except the
  shared junction endpoints, which are recorded explicitly.

Also assert on `RGI60-15.06881` (13 flowlines, 3 intermediate flowlines):

- yields **25 branches, 12 junctions**;
- every non-outlet flowline reassembles fully; the graph connects (every
  junction referenced downstream is produced upstream — no dangling reference);
- with a valid bed the network either solves or returns a physical failure class
  (e.g. `branch-failed:negative-start-thickness`), never a graph-connectivity
  error. (Status DONE: both acceptance glaciers pass; module suite 33 passed.)

## 6. Explicit driver modes

The current driver writes BOTH `archived-capped-iteration` and
`new-exact-perfect-plasticity` rows for the main flowline into one joint CSV in
a single run (`validate_modern_profiles.py` appends both to `rows`). Modes must
not overwrite that joint baseline. Pinned modes:

- `main-only-baseline` — **exactly the current behaviour**: one run producing
  both legacy + exact main-flowline rows, written to the existing CSV
  locations. Reproduces the current baseline number-for-number.
- `full-network` — the whole tree via the adapter, new-exact solver only.
  Writes a **new versioned file**; MUST NOT overwrite the existing 30/37 m
  baseline CSVs (its metric is not directly comparable, Section 7).
- `main-only-network` — `solve_flowline_network` on the single main flowline,
  used as a **test/verification mode only**. Writes an independent versioned or
  temporary file, never the baseline CSV. Its job is to re-prove single-branch
  equivalence, not to produce the baseline.

The archived capped method has no junction-inheritance semantics, so it appears
only inside `main-only-baseline`; there is no fabricated `full-network` legacy
comparison.

**Output contract (pinned, so "never overwrite baseline" is enforced in code):**

- default mode = `main-only-baseline`;
- full-network writes ONLY these new files, never the baseline CSVs:
  - `modern_profile_validation_full_network_v0.csv`
  - `modern_profile_branch_metrics_full_network_v0.csv`
  - `adapter_junction_audit_v0.csv`
  - `network_solution_junction_audit_v0.csv`

## 7. Metric aggregation for full-network (frozen for v0)

Parent splitting duplicates junction endpoints across adjacent segments. One
de-duplication rule governs everything:

- **De-duplicate original OGGM points by `(source_flowline_index,
  native_index)`.** Consequences:
  - a parent split point (e.g. `fl2[35]`) appearing in two adjacent segments
    counts **once**;
  - `fl0[-1]` and `fl2[35]` are **two distinct observations ~499 m apart**
    (Section 3), so **both are kept** — sharing a network node does not merge or
    delete either;
  - no point is dropped merely for being a shared network node.

Frozen v0 aggregation (no longer "choose one"):

- **Glacier-level RMSE:** equal weight over unique OGGM points.
- **Overall summary:** median of per-glacier RMSE.
- **Always also emit per-branch metrics.**
- Width/area-weighting is deferred to a later sensitivity result.

Full-network RMSE is a NEW metric; it is not compared head-to-head with the
historical 30/37 m without stating this rule.

## 8. Junction diagnostics: two separate tables

`JunctionDiagnostic`'s `maximum_surface_mismatch_m` is ~0 by construction in the
success path (the inherited surface IS the upstream start value), so it is a
numerical-continuity QC, not a geometric validation. Emit two tables:

- **`adapter_junction_audit_v0.csv` — static, scenario-independent.** Keyed
  `rgi_id + junction_node_id + tributary_flowline_index` — **one row per
  tributary**, because one physical junction can receive multiple tributaries
  with different distances and surface differences; rows on the same junction
  share `junction_node_id`. Each row: topology (`parent_flowline`,
  `flows_to_indice`), the three junction distances (`_grid_cells`, `_m`,
  `_flowline_steps`), and the modern reference-surface difference (parent point
  vs tributary end). On `RGI60-15.01141`: 33.2 m (fl0) and 213.6 m (fl1). The
  214 m case must never be treated as one spatial point.
- **`network_solution_junction_audit_v0.csv` — per
  `rgi_id + bed_scenario + stress_scenario + junction_node_id +
  tributary_flowline_index`.** Records bed difference, inherited surface,
  tributary start thickness, solve status, and the `junction_transfer_policy`
  string. One row per tributary.

## 9. Network-failure metric semantics

`solve_flowline_network` returns partial results the moment any branch fails.
The driver must therefore, on a non-success network status:

- mark the whole system `status = failed` (carry the network's failure class,
  e.g. `branch-failed:negative-start-thickness`);
- **NOT compute a full-network aggregate RMSE** for that system;
- allow already-solved branch diagnostics to be written, but flagged
  `partial_solution = true`;
- record the failed branch id, solved-branch count, unsolved-branch count, and
  solve order.

This prevents a partial network from being summarised as a complete one.

## 10. Test coverage to add (current is insufficient)

Current network tests (`23 passed`, of which 2 are synthetic single-branch
equivalence tests; helper functions are not counted) only prove single-branch
equivalence. Add:

- parent split at multiple junction indices;
- multi-level tributaries (tributary of a tributary);
- two tributaries into the same parent index;
- junction at an endpoint or adjacent index (degenerate split);
- native-forward-index -> reversed-distance conversion correctness;
- original-point coverage: no gap, only recorded duplicates.

Also record the 3 missing combos: the pilot yields 69, not `12x2x3=72`, because
some systems lack a bed-scenario inversion output (`missing-bed-scenario`).
Keep a reproducible list rather than an unexplained count.

## 11. Revised implementation order

1. Pin junction endpoint + distance metric definitions (Section 3) — done here.
2. Write the pure adapter in a driver/adapters layer; keep OGGM out of `core`.
3. Build stable nodes at outlet, each `flows_to_indice`, and each head
   (one head per original flowline).
4. Split parents at all junction indices (descending native index);
   reverse index = `n - 1 - flows_to_indice`.
5. Store per-segment flowline number and native index range (auditable coverage).
6. Adapter-only acceptance on `RGI60-15.01141`: assert `5 branches / 2 junctions`
   and full point reassembly (Section 5).
7. Add the driver modes (Section 6): `main-only-baseline` reproduces the joint
   baseline in place; `full-network` writes a new versioned file, never
   overwrites old CSVs; `main-only-network` is a test-only mode.
8. Keep `JunctionDiagnostic` as numerical QC; build the two audit tables
   (Section 8) and the network-failure semantics (Section 9).
9. full-network runs the new solver only; no fabricated legacy full-network.

## 12. Out of scope (unchanged)

Ice-divide / boundary classification, harmonic-solver method change, centerline
density sensitivity, and Level C data acquisition remain separate work items.
This spec only makes the network solver usable in Level B correctly.
