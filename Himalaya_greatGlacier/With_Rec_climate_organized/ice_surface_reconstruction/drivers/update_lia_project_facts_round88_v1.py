"""Append the verified Round 88 scientific boundary to project facts."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_BEFORE = "91fa9465803ea86a2731529fbc78b161e187bb0cc6c997aa6e6b91c9c4603602"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    project = Path(__file__).resolve()
    while not (project / "LIA_PROJECT_FACTS.md").exists():
        if project.parent == project:
            raise RuntimeError("project-root-not-found")
        project = project.parent
    facts = project / "LIA_PROJECT_FACTS.md"
    before = digest(facts)
    if before != EXPECTED_BEFORE:
        raise RuntimeError(f"facts-hash-conflict:{before}")
    text = facts.read_text(encoding="utf-8")
    if "## 140. Post-final-test climate-reconstruction boundary" in text:
        raise RuntimeError("round88-facts-already-present")
    addition = r'''

## 140. Post-final-test climate-reconstruction boundary (2026-08-16)

- Loop-v2 Round 88 routed only completed summary and manifest files from Rounds 84-87; it read no climate, proxy, station, or locked-period array.
- The rejected five-family temperature candidate cannot be tuned or replaced using the now-consumed 1921-1939 interval and then be presented as independently validated. Confirmatory continuation requires materially new external temperature evidence and a newly preregistered untouched validation target.
- A clearly labeled exploratory temperature reconstruction is possible only after explicit scientific approval; it cannot claim independent validation from 1921-1939.
- Confirmatory precipitation continuation still requires APHRO_MA V1101R1 0.5-degree data or authorized access. A non-equivalent evidence family requires explicit approval and a new frozen contract.
- Decision: hold at the external-evidence and scientific-decision boundary. No automatic parameter/member tuning, production release, or OGGM forcing is authorized.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_88_post_final_test_scientific_router_v1/`.
'''
    facts.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    print(f"before_sha256={before}")
    print(f"after_sha256={digest(facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
