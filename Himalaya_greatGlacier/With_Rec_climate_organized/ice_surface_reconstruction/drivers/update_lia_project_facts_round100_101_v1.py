"""Append verified Round 100-101 temperature findings."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "e687443b59089decea54078ee0d07b90287d5b137774b1f98c9541c0491bfdf9"


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    facts = root / "LIA_PROJECT_FACTS.md"
    before = facts.read_bytes()
    actual = hashlib.sha256(before).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{actual}")
    addition = r"""

## 153. Western early-temperature proxy contribution closure (2026-08-16)

- Loop-v2 Round 100 used all 50 archived ordinary-EnKF experiment pools and calculated innovation-times-gain contributions separately at every admitted Round 97 station cell before applying the exact station-year centering and network averaging.
- V1 passed algebraic gain closure but failed production reproduction because NumPy advanced indexing reversed the proxy and active-year axes during centering. V2 changed only this implementation, preserved V1 as failed, and retained every scientific input and gate.
- V2 contribution sums reproduce the Round 99 ordinary-EnKF increment curves with correlations 0.9999989 for proxy6 and 0.9999985 for proxy9. Trend errors are 0.344% and 0.022%; algebraic closure is 8.88e-16 degC.
- The wrong trend is broad. Proxy6 has top-10 absolute trend share 30.5%, 62.8 effective contributors, and 12 negative records needed for half of the negative trend. Proxy9 has 25.8%, 82.4 effective contributors, and 18 records, respectively.
- Negative and positive trend contributions are both large and opposing: -8.364 versus +6.028 degC per century in proxy6 and -8.579 versus +5.207 in proxy9. The net wrong trend is the residual of a broad mixed network signal, not a small bad-record subset.
- Decision: close record deletion, top-record ablation, and source-family deletion as repairs. Route only to a network-wide evidence/PSM/chronology decision; do not remove proxies based on contribution rank.
- All source hashes are unchanged, no 1921-1939 value was accessed, and the output figure is nonblank. Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_100_temperature_station_proxy_trend_attribution_v2/.

## 154. Temperature evidence-boundary decision (2026-08-16)

- Loop-v2 Round 101 read only completed summaries, manifests, and the canonical facts file. It read no climate, station, proxy, prior, posterior, or locked-period array.
- The western 1901-1920 failure is verified through robust direct stations, all-six-member sign failure, prior/increment decomposition, and exact full-50-experiment proxy contribution closure.
- Twelve existing-data repair families are closed by prior independent experiments: documented seasonality, current linear PSM temporal stability, expanded LMR network, instrumental product switching, regularized network signal, purged low-frequency signal, one-year alignment, monotonic nonlinearity, newly searched DoD2k chronology, member weighting, prior/localization, and record deletion.
- Decision: automatic tuning of the existing temperature proxy network is not permitted. Confirmatory continuation requires materially new independent temperature evidence or proxy chronology and a newly preregistered untouched validation target. EKF400v2 after authorized WDCC access is one external-evidence option.
- A background-only or cross-validated global increment-shrinkage candidate is permissible only as an explicitly approved exploratory reconstruction. It cannot use the already inspected 1901-1920 interval or consumed 1921-1939 interval as independent validation and cannot authorize production or OGGM forcing.
- The precipitation boundary is unchanged: confirmatory continuation requires APHRO_MA V1101R1 0.5-degree data/authorized access, or explicit approval to use a non-equivalent evidence family.
- Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_101_temperature_evidence_boundary_router_v1/.
"""
    text = before.decode("utf-8")
    if "## 153." in text or "## 154." in text:
        raise RuntimeError("target-sections-already-present")
    facts.write_bytes(before + addition.encode("utf-8"))
    print(hashlib.sha256(facts.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
