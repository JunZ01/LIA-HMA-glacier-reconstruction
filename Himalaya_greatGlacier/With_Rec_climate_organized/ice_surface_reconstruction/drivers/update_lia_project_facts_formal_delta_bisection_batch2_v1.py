"""Record the resolved formal delta-bisection batch-2 screening result."""

from __future__ import annotations

from pathlib import Path

from .build_lia_191_iterative_loop import project_root


HEADER = "Last updated: 2026-08-15 (batch-2 midpoint screening found systems 2939 and 4319)"
SECTION = r"""

## 28. Formal delta-T edge screening batch 2 (2026-08-15)

- The formal inventory was rebuilt after batch 1 without advancing OGGM. It
  contains 2438 deduplicated chains and exactly six formal dual-gate systems:
  `991`, `1077`, `1616`, `2815`, `2902`, and `2928`. The mutually exclusive
  routes are 100 `no_formal_pass`, 52 `total_length_only`, 28
  `small_separation_review`, six `formal_dual_pass`, two
  `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside the controller score sources.
- A read-only Stage-2-physics delta-T reachability audit found no unresolved
  system with an interpolation interval that places both separation-normalized
  endpoints inside 20%. Allowing only a bounded one-sided extrapolation of at
  most 0.125 K identified 12 predicted candidates. This was a proposal screen,
  not a curve acceptance result.
- DEM outlet probes were run for the top eight candidates. Downstream fields
  are complete for all eight, but the existing outlet gates pass only systems
  `79`, `660`, `2827`, `2939`, and `4319`; systems `130`, `1012`, and `1301`
  fail the existing outlet gates and are stopped without an override. The five
  passing systems were rebuilt with spatial-downstream-v2 geometry. All five
  preserve the Core-191 initial geometry and pass the static and 0.1/5-year
  zero-MB gates.
- Adaptive release then executed exactly one proposed delta-T point for each of
  the five systems. All chains are technically clean and observation-domain
  valid. System `79` at `delta_t=+0.125 K` converged with 1.351% stability but
  missed the separation endpoints by 1.642/1.016. System `660` at
  `delta_t=+0.45957938 K` converged with 1.344% stability but missed by
  2.275/2.585. System `2827` at `delta_t=+1.28003246 K` did not converge;
  its preceding-200-year volume change is 220.53%. These three branches are
  stopped.
- System `2939` at `delta_t=+0.26787336 K` passes every formal gate. Its
  forward peak is 13.300 km versus the 13.293 km LIA target and its year-2000
  length is 11.200 km versus the 11.363 km RGI target. Separation errors are
  0.00364/0.08454, total-length errors are 0.0528%/1.436%, and spin-up
  stability is 1.153%.
- System `4319` at `delta_t=+0.14873071 K` also passes every formal gate. Its
  forward peak is 15.800 km versus the 15.701 km LIA target and its year-2000
  length is 13.900 km versus the 13.824 km RGI target. Separation errors are
  0.05296/0.04031, total-length errors are 0.633%/0.547%, and spin-up
  stability is 0.406%.
- Ten declared side points were not run. The failed 79/660/2827 midpoint
  results make their neighboring points uninformative, while 2939 and 4319
  already pass all formal gates and require curve review before freezing. A
  first four-chain attempt produced no usable scientific result because its
  266-character output path exceeded the Windows result-write limit; the exact
  same four parameter points completed under a shorter block/output path.
- Formal tables and separate complete spin-up/forward plots are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_formal_delta_bisection_batch2_v1/`.
  Systems `2939` and `4319` are curve candidates pending user review; they are
  not yet copied into the frozen registry.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = HEADER
            break
    else:
        raise ValueError("facts file has no Last updated line")
    text = "\n".join(lines).rstrip() + "\n"
    marker = "## 28. Formal delta-T edge screening batch 2 (2026-08-15)"
    if marker not in text:
        text += SECTION.strip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    print(HEADER)


if __name__ == "__main__":
    main()
