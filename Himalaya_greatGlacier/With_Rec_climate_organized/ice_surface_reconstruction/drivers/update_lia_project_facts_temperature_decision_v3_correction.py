"""Append the correction to the canonical project facts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 279. Correction to the Hirpora western-causality interpretation (2026-08-19)"
SECTION = r"""

## 279. Correction to the Hirpora western-causality interpretation (2026-08-19)

- The causal wording in Section 278 and `climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v2` is too strong and is superseded here. The Hirpora experiment was numerically and procedurally valid, but its spatial interpretation was not sufficient for causal closure.
- Hirpora is a local June-July Kashmir reconstruction. The tested alpha branches are west-HMA (`lon < 80 E`) regional means formed by transferring ordinary-EnKF branch-minus-baseline increments to a six-member regional curve. Baseline standardization and low-pass smoothing do not make these spatial and seasonal targets equivalent.
- Independent western evidence is heterogeneous. Upper Indus and Hirpora have similar negative 1861-1900 low-frequency trends, while Kumrat has a positive trend. Their 1881-1900 baseline-standardized means are `+1.417`, `-0.113`, and `+2.057 SD`, respectively. The regional project curve is `+2.882 SD`.
- Alpha=0 remains consistent with Hirpora and Upper Indus trend shape, but it does not solve the Hirpora level mismatch: its 1881-1900 absolute mean error is `1.731 SD`. It also remains disqualified by the previously failed frozen uniform-alpha transfer gates.
- Correct decision: the archived regional warming is algorithmically produced by the temperature-operator covariance pathway, and Hirpora supplies evidence against treating that regional signal as universally representative. It does not independently prove that the regional covariance pathway is physically over-amplified everywhere or identify the unique physical cause.
- The western physical cause and validated correction therefore remain unresolved. The next admissible test must compare local or subregional candidate fields with co-located independent records under a frozen spatial contract. No production correction, EnKF replay, or OGGM run is authorized.
"""


def main() -> int:
    text = FACTS.read_text(encoding="utf-8")
    if HEADER in text:
        print("section-already-present")
        return 0
    FACTS.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("section-279-appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
