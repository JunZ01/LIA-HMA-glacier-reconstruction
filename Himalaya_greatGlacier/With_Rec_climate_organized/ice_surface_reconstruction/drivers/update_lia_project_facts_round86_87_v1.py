"""Append verified Round 86-87 temperature conclusions to project facts."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_BEFORE = "45aac864046f3bc5d322e2e0ba40c333fadd46cae5471b4abde0c53fb8858c71"


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
    if "## 138. Frozen 1921-1939 temperature final test" in text:
        raise RuntimeError("round86-facts-already-present")
    addition = r'''

## 138. Frozen 1921-1939 temperature final test (2026-08-16)

- The user explicitly authorized temperature unlocking with `允许解锁温度`. Loop-v2 Round 86 recorded that authorization before reading any locked numeric value and executed the Round 79 protocol exactly once.
- The frozen candidate SHA-256 remained `72615401b2c3df6ac7e6dc9e8204aff54aae40b9a295c1f6ba86e206eb864517`. ModE-RA used one exact HTTP 206 record request covering only 1921-1939; all local source hashes remained unchanged. No precipitation locked data was accessed.
- Relative to the frozen four-family control, the five-family candidate passes all six relative gates: median/worst regional RMSE improvements are +3.874%/-2.172%, 75% of 20 regional paths improve, median/worst HMA spatial improvements are +10.707%/+6.983%, and all five spatial paths improve.
- The candidate fails all four absolute regional gates: median/worst regional skills are -13.714%/-99.302%, only 45% of regional paths improve over zero anomaly, and median annual correlation is 0.282. It passes all four absolute HMA spatial gates: median/worst spatial skills are +15.616%/+3.601%, all five paths are positive, and median annual pattern correlation is 0.662.
- Decision: reject the frozen five-family candidate as an externally validated coarse HMA temperature reference for 1921-1939. The locked result must not be used to tune or replace the candidate, and it does not authorize OGGM forcing.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_86_frozen_temperature_final_test_v1/`.

## 139. Frozen-temperature final-test failure mechanism (2026-08-16)

- Loop-v2 Round 87 used only the saved Round 86 artifact; it did not reread any locked local or remote source and did not alter the rejected candidate.
- Against the equal-product HMA observation mean, the five-family candidate has a -0.297 degC mean bias. This mean-level bias accounts for 66.0% of MSE. Centered RMSE is 0.213 degC, the candidate/observation standard-deviation ratio is 0.715, and annual correlation is 0.285.
- Across the five individual HMA products, median bias is -0.203 degC, median standard-deviation ratio is 0.608, and median annual correlation is 0.266. Observational products retain substantial common signal: median/minimum pairwise annual correlations are 0.870/0.528.
- Decision: the dominant failure is an overly cold 1921-1939 regional mean, compounded by muted amplitude and weak annual timing. Observation disagreement is secondary and does not explain the rejection. Retain the Round 86 rejection and return to prelocked temperature-reconstruction design only; the held-out result is failure evidence, not a tuning target.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_87_temperature_final_test_failure_decomposition_v1/`.
'''
    facts.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    print(f"before_sha256={before}")
    print(f"after_sha256={digest(facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
