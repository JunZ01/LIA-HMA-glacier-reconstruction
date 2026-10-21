# One-Dimensional Production Route

Model family: OGGM flowline dynamics used by the existing accepted curve
population, including registered multi-flowline tributary networks.

Purpose: candidate primary route for increasing the comparable curve
population toward approximately 100 systems.

Design:

- build an adaptive spin-up state library under the shared parameter blocks;
- retain all converged equilibrium lengths;
- keep spin-up and historical forcing as separate matrices;
- evaluate observations only after forward runs;
- use CPU/process benchmarking before selecting production concurrency.

Status: design only. No simulation is authorized.

Evidence namespace: `one_d_production`. All manifests, fingerprints, states,
summaries, and proposed acceptance records must identify
`model_family=oggm-flowline-1d`.

Route-specific rules:

- retain the registered 1-D geometry and observation-domain length definition;
- exclude synthetic safety-tail ice from observation scoring;
- treat adaptive spin-up stability and historical-curve agreement as separate
  gates;
- require source-branch reproduction before any forward-only matrix variation;
- do not infer 2-D spreading, tributary, or planform validity from a 1-D pass.

Stop on any global rule in `../shared/DECISIONS_AND_STOP_RULES.md`, on an
ambiguous main branch/root, on observation-domain overflow or unassessed
length, or when a preregistered bounded family loses its response bracket.
The next legal action is contract design or read-only evidence inventory, not
a model launch.
