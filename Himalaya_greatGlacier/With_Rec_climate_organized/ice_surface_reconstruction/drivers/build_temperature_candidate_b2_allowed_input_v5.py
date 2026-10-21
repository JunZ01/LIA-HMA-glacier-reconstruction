from __future__ import annotations

import json
import time

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import build_temperature_candidate_b2_allowed_input_v3 as v3
import build_temperature_candidate_b2_allowed_input_v4 as v4


v1 = v3.v1
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v5"
ORIGINAL_READ = access.ExactRangeSource.read


def retrying_read(self, *args, **kwargs):
    last = None
    for attempt in range(1, 9):
        try:
            return ORIGINAL_READ(self, *args, **kwargs)
        except Exception as error:
            last = error
            print(f"RETRY range attempt={attempt} error={type(error).__name__}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise last


def main() -> None:
    access.ExactRangeSource.read = retrying_read
    v1.annual_mode = v4.annual_mode
    v1.annual_pmip = v3.annual_pmip
    v1.main()
    path = v1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["v5_change"] = "ten-year ModE-Sim blocks with eight bounded transport retries"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
